# ATS-DETR
# Copyright (c) 2025 Muyao Yuan. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------------------


"""
Implements the knowledge distillation loss
"""

import torch
import torch.nn as nn
from torch.nn import functional as F

import numpy as np

from typing import Tuple, Union

from util.dam import attn_map_to_flat_grid
import math

class PKDLoss(nn.Module):
    """PyTorch version of `PKD: General Distillation Framework for Object
    Detectors via Pearson Correlation Coefficient.

    <https://arxiv.org/abs/2207.02039>`_.

    Args:
        loss_weight (float): Weight of loss. Defaults to 1.0.
        resize_stu (bool): If True, we'll down/up sample the features of the
            student model to the spatial size of those of the teacher model if
            their spatial sizes are different. And vice versa. Defaults to
            True.
    """

    def __init__(self, loss_weight=1.0, resize_stu=True, level_weight=[1,1,1,1], normalization=True):
        super(PKDLoss, self).__init__()
        self.loss_weight = loss_weight
        self.resize_stu = resize_stu
        self.level_weight = level_weight
        self.normalization = normalization

    def norm(self, feat: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Normalize the feature maps to have zero mean and unit variances.

        Args:
            feat (torch.Tensor): The original feature map with shape
                (N, C, H, W).
            masks (torch.Tensor): (N, H, W)
        """
        assert len(feat.shape) == 4
        N, C, H, W = feat.shape
        feat = feat.permute(1, 0, 2, 3).reshape(C, -1) #[C, N*H*W]
        mask = mask.reshape(-1) #[N*H*W]
        # mean and std for valid area
        feat_masked = feat * mask.unsqueeze(0) #[C, N*H*W]
        mean = feat_masked.sum(dim=-1, keepdim=True) / mask.sum() #[C, 1]
        squared_diff = ((feat_masked - mean) ** 2) * mask.unsqueeze(0) # [C, N*H*W]
        std = torch.sqrt(squared_diff.sum(dim=-1, keepdim=True) / mask.sum()) #[C, 1]
        feat = (feat - mean) / (std + 1e-6)
        feat = feat * mask.unsqueeze(0)
        return feat.reshape(C, N, H, W).permute(1, 0, 2, 3)

    def forward(self, preds_S: Union[torch.Tensor, Tuple],
                preds_T: Union[torch.Tensor, Tuple], 
                masks: Union[torch.Tensor, Tuple]) -> torch.Tensor:
        """Forward computation.

        Args:
            preds_S (torch.Tensor | Tuple[torch.Tensor]): The student model
                prediction. If tuple, it should be several tensors with shape
                (N, C, H, W).
            preds_T (torch.Tensor | Tuple[torch.Tensor]): The teacher model
                prediction. If tuple, it should be several tensors with shape
                (N, C, H, W).
            masks (torch.Tensor | Tuple[torch.Tensor]): (N, H, W)

        Return:
            torch.Tensor: The calculated loss value.
        """
        if isinstance(preds_S, torch.Tensor):
            preds_S, preds_T, masks = (preds_S, ), (preds_T, ), (masks, )

        loss = torch.tensor([0.], device=preds_S[0].device)

        for i, (pred_S, pred_T, mask) in enumerate(zip(preds_S, preds_T, masks)):
            size_S, size_T = pred_S.shape[2:], pred_T.shape[2:]
            if size_S[0] != size_T[0]:
                if self.resize_stu:
                    pred_S = F.interpolate(pred_S, size_T, mode='bilinear')
                else:
                    pred_T = F.interpolate(pred_T, size_S, mode='bilinear')
            assert pred_S.shape == pred_T.shape
            
            if self.normalization:
                norm_S, norm_T = self.norm(pred_S, mask), self.norm(pred_T, mask)
            else:
                norm_S = pred_S
                norm_T = pred_T

            # First conduct feature normalization and then calculate the
            # MSE loss. Methematically, it is equivalent to firstly calculate
            # the Pearson Correlation Coefficient (r) between two feature
            # vectors, and then use 1-r as the new feature imitation loss.
            loss += F.mse_loss(norm_S, norm_T) / 2 * self.level_weight[i]

        return loss * self.loss_weight

class DistillationLoss(nn.Module):
    
    def __init__(self, teacher_model, args, postprocessors=None, model_LB=None):
        super().__init__()
        self.teacher_model = teacher_model
        self.model_LB = model_LB
        self.distillation_types = []
        self.weight_dict = {}
        # the distill
        self.level_weight_GD = args.level_weight_GD
        if args.global_distill:
            self.distillation_types += ['global']
            self.weight_dict.update({'loss_global_distillation': args.global_distill_coef})
            self.pkd_loss = PKDLoss(level_weight=self.level_weight_GD, normalization=True)
        if args.adaptive:
            self.postprocessors = postprocessors
            self.distillation_types += ['rate_LB']
            self.weight_dict.update({'loss_learnable_rate_LB': args.learnable_rate_LB_coef})
            self.mAP_threshold = args.mAP_threshold
            self.IoU_threshold = args.IoU_threshold

            self.num_ratio = args.num_ratio
            self.bias = args.bias
            self.ce_loss = nn.CrossEntropyLoss()

    def loss_global_distillation(self, teacher_outputs, outputs, targets=None, LB_outputs=None):
        assert "memory_enc" in teacher_outputs # [bs, L, d_model]
        assert "memory_enc" in outputs 
        assert "mask_flatten" in outputs # True when masked [bs, L]
        assert "spatial_shapes" in outputs
        assert "level_start_index" in outputs

        memory_t = teacher_outputs["memory_enc"]
        memory = outputs["memory_enc"]
        mask_flatten = outputs["mask_flatten"]
        spatial_shapes = outputs['spatial_shapes']
        level_start_index = outputs['level_start_index']

        B_, _, D_ = memory.shape

        features_t = []
        features = []
        masks = []
        for lvl in range(spatial_shapes.shape[0]):
            mask = mask_flatten[:,level_start_index[lvl]:level_start_index[lvl]+(spatial_shapes[lvl].prod())].reshape(B_,*spatial_shapes[lvl]) # [bs, H, W]
            masks.append(~mask)
            feature_t = memory_t[:,level_start_index[lvl]:level_start_index[lvl]+(spatial_shapes[lvl].prod())].reshape(B_,*spatial_shapes[lvl],D_) # [bs, H, W, d_model]
            feature_t = feature_t.permute(0,3,1,2) # [bs, C, H, W]
            features_t.append(feature_t)
            feature = memory[:,level_start_index[lvl]:level_start_index[lvl]+(spatial_shapes[lvl].prod())].reshape(B_,*spatial_shapes[lvl],D_) # [bs, H, W, d_model]
            feature = feature.permute(0,3,1,2) # [bs, C, H, W]
            features.append(feature)

        loss_global_distillation = self.pkd_loss(tuple(features), tuple(features_t), tuple(masks))

        losses = {"loss_global_distillation": loss_global_distillation}

        return losses

    def loss_learnable_rate_LB(self, teacher_outputs, outputs, targets=None, LB_outputs=None):
        rates = outputs["learnable_rate"] # [1, bs] / [1, bs, num_ratio]
    
        losses_ = torch.tensor([0.],device=rates[0].device)
        

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        if not LB_outputs:
            results = self.postprocessors['bbox'](teacher_outputs, orig_target_sizes)
        else:
            results = self.postprocessors['bbox'](LB_outputs, orig_target_sizes)
        mAPs = batch_eval(results, targets, iou_threshold=self.IoU_threshold) #[bs]

        cls_sparse_ratio = torch.zeros_like(mAPs, dtype=int).cuda()
        for bid, mAP in enumerate(mAPs):
            if mAP >= self.bias: # Easy scenes
                cls_sparse_ratio[bid] = 0
            else:
                cls_sparse_ratio[bid] = 1
        losses_ += self.ce_loss(rates[0], cls_sparse_ratio)
    
        loss_learnable_rate = losses_
        
        losses = {"loss_learnable_rate_LB": loss_learnable_rate}

        return losses
    
    def get_loss(self, distillation_type, teacher_outputs, outputs, targets=None, LB_outputs=None):
        distillation_map = {
            'global': self.loss_global_distillation,
            'rate_LB': self.loss_learnable_rate_LB,
        }

        assert distillation_type in distillation_map
        return distillation_map[distillation_type](teacher_outputs, outputs, targets, LB_outputs)
    
    def forward(self, inputs, outputs, targets=None):
        """
        Args:
            inputs: The original inputs that are feed to the teacher model
            outputs: the outputs of the model to be trained.
        """

        # don't backprop throught the teacher
        if self.teacher_model:
            with torch.no_grad():
                teacher_outputs = self.teacher_model(inputs)
        else:
            teacher_outputs = None
        
        if self.model_LB:
            with torch.no_grad():
                LB_outputs = self.model_LB(inputs)
        else:
            LB_outputs = None

        distillation_losses = {}
        for distillation_type in self.distillation_types:
            distillation_losses.update(self.get_loss(distillation_type, teacher_outputs, outputs, targets, LB_outputs))

        return distillation_losses


def build_distillation_loss(teacher_model, args, postprocessors=None, model_LB=None):
    return DistillationLoss(teacher_model, args, postprocessors, model_LB)


from util import box_ops

def calculate_iou(box1, box2):
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    
    x_intersection = torch.max(x1_min, x2_min)
    y_intersection = torch.max(y1_min, y2_min)
    w_intersection = torch.max(torch.tensor(0.0), torch.min(x1_max, x2_max) - x_intersection)
    h_intersection = torch.max(torch.tensor(0.0), torch.min(y1_max, y2_max) - y_intersection)
    
    if w_intersection <= 0 or h_intersection <= 0:
        return torch.tensor(0.0, device=box1.device)
    
    # 计算交并比
    intersection_area = w_intersection * h_intersection
    union_area = (x1_max - x1_min) * (y1_max - y1_min) + (x2_max - x2_min) * (y2_max - y2_min) - intersection_area
    iou = intersection_area / union_area
    
    return iou

def calculate_ap(precision, recall):
    # 计算平均精度（AP）
    ap = torch.tensor(0.0, device=precision.device)
    precision = torch.cat((torch.tensor([1.0], device=precision.device), precision), dim=0)
    recall = torch.cat((torch.tensor([0.0], device=precision.device), recall), dim=0)
    for i in range(len(precision) - 1):
        ap += (recall[i + 1] - recall[i]) * precision[i + 1]
    return ap

def calculate_map(pred_scores, pred_labels, pred_boxes, true_labels, true_boxes, num_classes=91, iou_threshold=0.5):
    
    average_precisions = []
    
    for class_id in range(num_classes):
        class_pred_scores = pred_scores[pred_labels == class_id]
        class_pred_boxes = pred_boxes[pred_labels == class_id]
        
        class_true_labels = true_labels[true_labels == class_id]
        class_true_boxes = true_boxes[true_labels == class_id]
        
        if len(class_true_labels) > 0:
            sorted_indices = torch.argsort(class_pred_scores, descending=True)
            class_pred_scores = class_pred_scores[sorted_indices]

            gt_mask = torch.zeros(len(class_true_labels), dtype=torch.bool, device=pred_scores.device)
            true_positive_mask = torch.zeros(len(class_pred_scores), dtype=torch.bool, device=pred_scores.device)

            for i, pred_box in enumerate(class_pred_boxes):
                max_iou_scores = -1
                max_j = -1
                for j, true_box in enumerate(class_true_boxes):
                    if not gt_mask[j]:
                        iou_scores = calculate_iou(pred_box, true_box)
                        if iou_scores > max_iou_scores:
                            max_iou_scores = iou_scores
                            max_j = j
                if max_iou_scores >= iou_threshold:
                    true_positive_mask[i] = True
                    gt_mask [max_j] = True
            
            cumulative_precision = torch.cumsum(true_positive_mask, dim=0) / (torch.arange(len(sorted_indices), device=pred_scores.device) + 1).float()
            cumulative_recall = torch.cumsum(true_positive_mask, dim=0) / len(class_true_labels)
            
            average_precisions.append(calculate_ap(cumulative_precision, cumulative_recall))
    
    if average_precisions:
        mean_ap = torch.mean(torch.stack(average_precisions))
    else:
        mean_ap = 1.0
    return mean_ap

def batch_eval(results, targets, num_classes=91, iou_threshold=0.5):
    batch_size = len(results)
    mAPs = torch.zeros(batch_size)
    
    for bid in range(batch_size):
        pred_labels = results[bid]['labels']
        pred_scores = results[bid]['scores']
        pred_boxes = results[bid]['boxes']
        true_labels = targets[bid]['labels']
        true_boxes = targets[bid]['boxes']
        orig_size = targets[bid]["orig_size"]
        img_h, img_w = orig_size
        scale_fct = torch.tensor([img_w, img_h, img_w, img_h], device=true_boxes.device).unsqueeze(0)
        true_boxes = box_ops.box_cxcywh_to_xyxy(true_boxes) * scale_fct

        mAP = calculate_map(pred_scores, pred_labels, pred_boxes, true_labels, true_boxes, num_classes, iou_threshold)
        mAPs[bid] = mAP
    
    return mAPs



