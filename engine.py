# ATS-DETR
# Copyright (c) 2025 Muyao Yuan. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------------------
# Modified from Sparse DETR (https://github.com/kakaobrain/sparse-detr)
# Copyright (c) 2021 KakaoBrain. All Rights Reserved.
# ------------------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------------------


"""
Train and eval functions used in main.py
"""
import math
import os
import sys
from typing import Iterable

import torch
import util.misc as utils
from datasets.coco_eval import CocoEvaluator
from datasets.panoptic_eval import PanopticEvaluator
from datasets.data_prefetcher import data_prefetcher

import matplotlib.pyplot as plt

from util.misc import check_unused_parameters


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0, 
                    writer=None, total_iter=0, distillation_criterion=None):
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    metric_logger.add_meter('grad_norm', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    prefetcher = data_prefetcher(data_loader, device, prefetch=True)
    samples, targets = prefetcher.next()

    for i in metric_logger.log_every(range(len(data_loader)), print_freq, header):            
        outputs = model(samples)
        train_sparse_ratio = outputs["sparse_ratio"]
        train_sparse_ratio_avg = torch.mean(train_sparse_ratio)
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        # distillation loss
        if distillation_criterion is not None:
            distill_loss_dict = distillation_criterion(samples, outputs, targets)
            distill_weight_dict = distillation_criterion.weight_dict
            distill_losses = sum(distill_loss_dict[k] * distill_weight_dict[k] for k in distill_loss_dict.keys() if k in distill_weight_dict)
            losses = losses + distill_losses

            # reduce losses for distillation
            distill_loss_dict_reduced = utils.reduce_dict(distill_loss_dict)
            distill_loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                        for k, v in distill_loss_dict_reduced.items()}
            distill_loss_dict_reduced_scaled = {k: v * distill_weight_dict[k]
                                        for k, v in distill_loss_dict_reduced.items() if k in distill_weight_dict}
            distill_losses_reduced_scaled = sum(distill_loss_dict_reduced_scaled.values())

            distill_loss_value = distill_losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)
        if distillation_criterion is not None:
            if not math.isfinite(distill_loss_value):
                print("Distillation loss is {}, stopping training".format(distill_loss_value))
                print(distill_loss_dict_reduced)
                sys.exit(1)
            
        optimizer.zero_grad()
        losses.backward()
        
        if i == 0:
            check_unused_parameters(model, loss_dict, weight_dict)
                
        if max_norm > 0:
            grad_total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        else:
            grad_total_norm = utils.get_total_grad_norm(model.parameters(), max_norm)
            
        if distillation_criterion is not None:
            metric_logger.update(loss=loss_value, distill_loss=distill_loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled, **distill_loss_dict_reduced_scaled, **distill_loss_dict_reduced_unscaled)
        else:
            metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(grad_norm=grad_total_norm)
                    
        optimizer.step()

        if total_iter % (print_freq*10) == 0 and utils.is_main_process():
            writer.add_scalar('train/loss', loss_value, total_iter)
            writer.add_scalar('train/class_error', loss_dict_reduced['class_error'], total_iter)
            writer.add_scalar('lr', optimizer.param_groups[0]["lr"], total_iter)
            if distillation_criterion is not None:
                distill_weight_dict = distillation_criterion.weight_dict
                for k in distill_weight_dict.keys():
                    writer.add_scalar('weight/' + k, distill_weight_dict[k], total_iter)
            writer.add_scalar('train/grad_norm', grad_total_norm, total_iter)
            for key, value in loss_dict_reduced_scaled.items():
                writer.add_scalar('train/'+key, value, total_iter)
            for key, value in loss_dict_reduced_unscaled.items():
                if "corr" in key:
                    writer.add_scalar('train/'+key, value, total_iter)
            if distillation_criterion is not None:
                for key, value in distill_loss_dict_reduced_scaled.items():
                    writer.add_scalar('train/'+key, value, total_iter)
            writer.add_scalar('train/sparse_ratio', train_sparse_ratio_avg, total_iter)

        total_iter += 1
        samples, targets = prefetcher.next()

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}, total_iter


@torch.no_grad()
def evaluate(model, criterion, postprocessors, data_loader, base_ds, device, args, writer=None):
    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'

    iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())
    coco_evaluator = CocoEvaluator(base_ds, iou_types)

    panoptic_evaluator = None
    if 'panoptic' in postprocessors.keys():
        panoptic_evaluator = PanopticEvaluator(
            data_loader.dataset.ann_file,
            data_loader.dataset.ann_folder,
            output_dir=os.path.join(args.output_dir, "panoptic_eval"),
        )

    for step, (samples, targets) in enumerate(metric_logger.log_every(data_loader, 10, header)):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = model(samples)
        test_sparse_ratio = outputs["sparse_ratio"]
        test_sparse_ratio_avg = torch.mean(test_sparse_ratio)
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()),
                             **loss_dict_reduced_scaled,
                             **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])
        metric_logger.update(test_sparse_ratio=test_sparse_ratio_avg)

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessors['bbox'](outputs, orig_target_sizes)
        if 'segm' in postprocessors.keys():
            target_sizes = torch.stack([t["size"] for t in targets], dim=0)
            results = postprocessors['segm'](results, outputs, orig_target_sizes, target_sizes)
        res = {target['image_id'].item(): output for target, output in zip(targets, results)}
        if coco_evaluator is not None:
            coco_evaluator.update(res)

        if panoptic_evaluator is not None:
            res_pano = postprocessors["panoptic"](outputs, target_sizes, orig_target_sizes)
            for i, target in enumerate(targets):
                image_id = target["image_id"].item()
                file_name = f"{image_id:012d}.png"
                res_pano[i]["image_id"] = image_id
                res_pano[i]["file_name"] = file_name

            panoptic_evaluator.update(res_pano)



    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()
    if panoptic_evaluator is not None:
        panoptic_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()
    panoptic_res = None
    if panoptic_evaluator is not None:
        panoptic_res = panoptic_evaluator.summarize()
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    if coco_evaluator is not None:
        if 'bbox' in postprocessors.keys():
            stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
        if 'segm' in postprocessors.keys():
            stats['coco_eval_masks'] = coco_evaluator.coco_eval['segm'].stats.tolist()
    if panoptic_res is not None:
        stats['PQ_all'] = panoptic_res["All"]
        stats['PQ_th'] = panoptic_res["Things"]
        stats['PQ_st'] = panoptic_res["Stuff"]
    return stats, coco_evaluator
