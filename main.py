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


import argparse
import datetime
import json
import random
import time
from tabulate import tabulate
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

import datasets
import util.misc as utils
import datasets.samplers as samplers
from datasets import build_dataset, get_coco_api_from_dataset
from engine import evaluate, train_one_epoch
from models import build_model
from models import build_distillation
from util.benchmark import compute_fps, compute_gflops

from torch.utils.tensorboard import SummaryWriter

def get_args_parser():
    parser = argparse.ArgumentParser('Deformable DETR Detector', add_help=False)
    parser.add_argument('--lr', default=2e-4, type=float)
    parser.add_argument('--lr_backbone_names', default=["backbone.0"], type=str, nargs='+')
    parser.add_argument('--lr_backbone', default=2e-5, type=float)
    parser.add_argument('--lr_linear_proj_names', default=['reference_points', 'sampling_offsets'], type=str, nargs='+')
    parser.add_argument('--lr_linear_proj_mult', default=0.1, type=float)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--lr_drop', default=40, type=int)
    parser.add_argument('--lr_drop_epochs', default=None, type=int, nargs='+')
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')


    parser.add_argument('--sgd', action='store_true')

    # Variants of Deformable DETR
    parser.add_argument('--with_box_refine', default=False, action='store_true')
    parser.add_argument('--two_stage', default=False, action='store_true')

    # Model parameters
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")

    # * Backbone
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")
    parser.add_argument('--position_embedding_scale', default=2 * np.pi, type=float,
                        help="position / size * scale")
    parser.add_argument('--num_feature_levels', default=4, type=int, help='number of feature levels')

    # * Modified architecture
    parser.add_argument('--backbone_from_scratch', default=False, action='store_true')
    parser.add_argument('--finetune_early_layers', default=False, action='store_true')
    parser.add_argument('--scrl_pretrained_path', default='', type=str)

    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=1024, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=300, type=int,
                        help="Number of query slots")
    parser.add_argument('--dec_n_points', default=4, type=int)
    parser.add_argument('--enc_n_points', default=4, type=int)
    
    # * Efficient DETR
    parser.add_argument('--eff_query_init', default=False, action='store_true')
    parser.add_argument('--eff_specific_head', default=False, action='store_true')

    # * Sparse DETR
    parser.add_argument('--use_enc_aux_loss', default=False, action='store_true')
    parser.add_argument('--rho', default=[0.], nargs='+', type=float)
    # adaptive
    parser.add_argument('--adaptive', default=False, action='store_true')
    # distill
    parser.add_argument('--teacher_checkpoint', default='', help='strong teacher from checkpoint')
    parser.add_argument('--LB_checkpoint', default='', help='weak teacher from checkpoint')
    parser.add_argument('--IoU_threshold', default=0.5, type=float)
    parser.add_argument('--global_distill', default=False, action='store_true')
    parser.add_argument('--num_ratio', default=2, type=int)
    parser.add_argument('--min_ratio', default=0., type=float)
    parser.add_argument('--max_ratio', default=1., type=float)
    parser.add_argument('--learnable_rate_LB_coef', default=1, type=float)
    parser.add_argument('--mAP_threshold', default=0.5, type=float)
    parser.add_argument('--bias', default=0.5, type=float)
    # distill loss coefficients
    parser.add_argument('--global_distill_coef', default=1, type=float)
    parser.add_argument('--level_weight_GD', default=[0.25,0.25,0.25,0.25], nargs='+', type=float)
    parser.add_argument('--no_adaption', dest='adaption', action='store_false')

    # * Segmentation
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")

    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")

    # * Matcher
    parser.add_argument('--set_cost_class', default=2, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")

    # * Loss coefficients
    parser.add_argument('--mask_loss_coef', default=1, type=float)
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--cls_loss_coef', default=2, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--mask_prediction_coef', default=1, type=float)
    parser.add_argument('--focal_alpha', default=0.25, type=float)

    # * dataset parameters
    parser.add_argument('--dataset_file', default='coco')
    parser.add_argument('--coco_path', default='./data/coco', type=str)
    parser.add_argument('--coco_panoptic_path', type=str)

    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=2, type=int)
    parser.add_argument('--cache_mode', default=False, action='store_true', help='whether to cache images on memory')
    
    # * benchmark
    parser.add_argument('--approx_benchmark_only', default=False, action='store_true')
    parser.add_argument('--benchmark_only', default=False, action='store_true')
    parser.add_argument('--no_benchmark', dest='benchmark', action='store_false')
    parser.add_argument('--gflops_in_cpu', default=False, action='store_true')

    return parser


def main(args):
    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))

    if args.frozen_weights is not None:
        assert args.masks, "Frozen training is meant for segmentation only"
    print(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model, criterion, postprocessors = build_model(args)
    model.to(device)
    model_without_ddp = model

    if not args.eval and args.global_distill:
        # teacher model & distillation_criterion
        # model_t    
        adaptive_orig = args.adaptive
        args.adaptive = False
        adaption_orig = args.adaption
        args.adaption = False
        rho_orig = args.rho
        args.rho = [args.max_ratio]
        model_t, _, _ = build_model(args)
        args.adaptive = adaptive_orig
        args.adaption = adaption_orig
        args.rho = rho_orig
        model_t.to(device)
        weight_t = torch.load(args.teacher_checkpoint, map_location='cpu')
        model_t.load_state_dict(weight_t['model'], strict=True)
        model_t.eval()

        # model_LB
        adaptive_orig = args.adaptive
        args.adaptive = False
        adaption_orig = args.adaption
        args.adaption = False
        rho_orig = args.rho
        args.rho = [args.min_ratio]
        model_LB, _, _ = build_model(args)
        args.adaptive = adaptive_orig
        args.adaption = adaption_orig
        args.rho = rho_orig
        model_LB.to(device)
        weight_LB = torch.load(args.LB_checkpoint, map_location='cpu')
        model_LB.load_state_dict(weight_LB['model'], strict=True)
        model_LB.eval()
        
        # distillation_criterion
        distillation_criterion = build_distillation(model_t, args, postprocessors, model_LB)
    else:
        distillation_criterion = None
    
    dataset_val_org = build_dataset(image_set='val', args=args)
    
    if args.approx_benchmark_only or args.benchmark_only:
        assert not args.distributed and args.benchmark
    
    if utils.is_main_process() and args.benchmark:
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        if args.benchmark_only:
            gflops, giga_flops_encoder = compute_gflops(model, dataset_val_org, approximated=False, cpu=args.gflops_in_cpu)
        else:
            gflops, giga_flops_encoder = compute_gflops(model, dataset_val_org, approximated=True, cpu=args.gflops_in_cpu)
        fps = compute_fps(model, dataset_val_org, num_iters=20, batch_size=1)
        bfps = compute_fps(model, dataset_val_org, num_iters=20, batch_size=4)
        tab_keys = ["#Params(M)", "GFLOPs", "Encoder GFLOPs", "FPS", "B4FPS"]
        tab_vals = [n_params / 10 ** 6, gflops, giga_flops_encoder, fps, bfps]
        table = tabulate([tab_vals], headers=tab_keys, tablefmt="pipe",
                        floatfmt=".3f", stralign="center", numalign="center")
        print("===== Benchmark (Crude Approx.) =====\n" + table)
        
    if args.approx_benchmark_only or args.benchmark_only:
        import sys; sys.exit()
            
    if args.distributed:
        # wait for benchmark in the main process
        torch.distributed.barrier()
        
    dataset_train = build_dataset(image_set='train', args=args)
    dataset_val = dataset_val_org

    if args.distributed:
        if args.cache_mode:
            sampler_train = samplers.NodeDistributedSampler(dataset_train)
            sampler_val = samplers.NodeDistributedSampler(dataset_val, shuffle=False)
        else:
            sampler_train = samplers.DistributedSampler(dataset_train)
            sampler_val = samplers.DistributedSampler(dataset_val, shuffle=False)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, args.batch_size, drop_last=True)

    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                   collate_fn=utils.collate_fn, num_workers=args.num_workers,
                                   pin_memory=True)
    data_loader_val = DataLoader(dataset_val, args.batch_size, sampler=sampler_val,
                                 drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers,
                                 pin_memory=True)

    args = utils.scale_learning_rate(args) # according to bz update/scale lr
    def match_name_keywords(n, name_keywords):
        out = False
        for b in name_keywords:
            if b in n:
                out = True
                break
        return out

    param_dicts = [
        {
            "params":
                [p for n, p in model_without_ddp.named_parameters()
                 if (not match_name_keywords(n, args.lr_backbone_names) 
                     and not match_name_keywords(n, args.lr_linear_proj_names) 
                     and p.requires_grad)],
            "lr": args.lr,
        },
        {
            "params": [p for n, p in model_without_ddp.named_parameters() 
                       if (match_name_keywords(n, args.lr_backbone_names) 
                           and not match_name_keywords(n, args.lr_linear_proj_names) 
                           and p.requires_grad)],
            "lr": args.lr_backbone,
        },
        {
            "params": [p for n, p in model_without_ddp.named_parameters() 
                       if match_name_keywords(n, args.lr_linear_proj_names) and p.requires_grad],
            "lr": args.lr * args.lr_linear_proj_mult,
        }
    ]
    if args.sgd:
        optimizer = torch.optim.SGD(param_dicts, lr=args.lr, momentum=0.9,
                                    weight_decay=args.weight_decay) # weight decay = L2 Regularization
    else:
        optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                      weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop) # lr=lr*gamma, gamma=0.1

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], 
                                                          find_unused_parameters=True)
        model_without_ddp = model.module

    if args.dataset_file == "coco_panoptic":
        # We also evaluate AP during panoptic training, on original coco DS
        coco_val = datasets.coco.build("val", args)
        base_ds = get_coco_api_from_dataset(coco_val)
    else:
        base_ds = get_coco_api_from_dataset(dataset_val)

    if args.frozen_weights is not None:
        checkpoint = torch.load(args.frozen_weights, map_location='cpu')
        model_without_ddp.detr.load_state_dict(checkpoint['model'])

    output_dir = Path(args.output_dir)
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        missing_keys, unexpected_keys = model_without_ddp.load_state_dict(checkpoint['model'], strict=False)
        unexpected_keys = [k for k in unexpected_keys if not (k.endswith('total_params') or k.endswith('total_ops'))]
        if len(missing_keys) > 0:
            print('Missing Keys: {}'.format(missing_keys))
        if len(unexpected_keys) > 0:
            print('Unexpected Keys: {}'.format(unexpected_keys))
        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            import copy
            p_groups = copy.deepcopy(optimizer.param_groups)
            optimizer.load_state_dict(checkpoint['optimizer'])
            for pg, pg_old in zip(optimizer.param_groups, p_groups):
                pg['lr'] = pg_old['lr']
                pg['initial_lr'] = pg_old['initial_lr']
            print(optimizer.param_groups)
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            # todo: this is a hack for doing experiment that resume from checkpoint 
            # and also modify lr scheduler (e.g., decrease lr in advance).
            args.override_resumed_lr_drop = True
            if args.override_resumed_lr_drop:
                print('Warning: (hack) args.override_resumed_lr_drop is set to True, '
                      'so args.lr_drop would override lr_drop in resumed lr_scheduler.')
                lr_scheduler.step_size = args.lr_drop
                lr_scheduler.base_lrs = list(map(lambda group: group['initial_lr'], optimizer.param_groups))
            lr_scheduler.step(lr_scheduler.last_epoch)
            args.start_epoch = checkpoint['epoch'] + 1
        # check the resumed model
        if not args.eval:
            test_stats, coco_evaluator = evaluate(
                model, criterion, postprocessors, data_loader_val, base_ds, device, args
            )
    
    if args.eval:
        print("Start evaluation")
        test_dir = output_dir.joinpath('test')
        test_dir.mkdir(parents=True, exist_ok=True)
        if utils.is_main_process():
            test_writer = SummaryWriter(test_dir)
        else:
            test_writer = None
        start_time = time.time()
        test_stats, coco_evaluator = evaluate(model, criterion, postprocessors,
                                              data_loader_val, base_ds, device, args, test_writer)
        if args.output_dir:
            utils.save_on_master(coco_evaluator.coco_eval["bbox"].eval, output_dir / "eval.pth")
        print_final_result_on_master(model, dataset_val_org, args, test_stats, start_time)
        return

    if utils.is_main_process():
        writer = SummaryWriter(output_dir)
    else:
        writer = None
    total_iter = 0
    
    print("Start training")
    start_time = time.time()

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            sampler_train.set_epoch(epoch)
        train_stats, total_iter = train_one_epoch(
            model, criterion, data_loader_train, optimizer, device, epoch, args.clip_max_norm, writer, total_iter, distillation_criterion)
        lr_scheduler.step()
            
        if args.output_dir:
            checkpoint_paths = [output_dir / 'checkpoint.pth']
            # extra checkpoint before LR drop and every 5 epochs
            # save every epoch after epoch 45
            if (epoch + 1) % args.lr_drop == 0 or (epoch + 1) % 5 == 0 or epoch >= 45:
                checkpoint_paths.append(output_dir / f'checkpoint{epoch:04}.pth')
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }, checkpoint_path)

        test_stats, coco_evaluator = evaluate(
            model, criterion, postprocessors, data_loader_val, base_ds, device, args
        )

        # write test status
        if utils.is_main_process():
            writer.add_scalar('test/AP', test_stats['coco_eval_bbox'][0], epoch)
            writer.add_scalar('test/AP50', test_stats['coco_eval_bbox'][1], epoch)
            writer.add_scalar('test/AP75', test_stats['coco_eval_bbox'][2], epoch)
            writer.add_scalar('test/APs', test_stats['coco_eval_bbox'][3], epoch)
            writer.add_scalar('test/APm', test_stats['coco_eval_bbox'][4], epoch)
            writer.add_scalar('test/APl', test_stats['coco_eval_bbox'][5], epoch)
            writer.add_scalar('test/class_error', test_stats['class_error'], epoch)
            writer.add_scalar('test/loss', test_stats['loss'], epoch)
            writer.add_scalar('test/loss_ce', test_stats['loss_ce'], epoch)
            writer.add_scalar('test/loss_bbox', test_stats['loss_bbox'], epoch)
            writer.add_scalar('test/loss_giou', test_stats['loss_giou'], epoch)
            writer.add_scalar('test/sparse_ratio', test_stats['test_sparse_ratio'], epoch)
            for key, value in test_stats.items():
                if "corr" in key:
                    writer.add_scalar('test/'+key, value, epoch)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'test_{k}': v for k, v in test_stats.items()},
                     'epoch': epoch}

        if args.output_dir and utils.is_main_process():
            if args.benchmark:
                log_stats.update({'params': n_params, 'gflops': gflops, 'fps': fps, 'bfps': bfps})

            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

            # for evaluation logs
            if coco_evaluator is not None:
                (output_dir / 'eval').mkdir(exist_ok=True)
                if "bbox" in coco_evaluator.coco_eval:
                    filenames = ['latest.pth']
                    if epoch % 50 == 0:
                        filenames.append(f'{epoch:03}.pth')
                    for name in filenames:
                        torch.save(coco_evaluator.coco_eval["bbox"].eval,
                                   output_dir / "eval" / name)
        
    print_final_result_on_master(model, dataset_val_org, args, test_stats, start_time)
    

def print_final_result_on_master(model, dataset_val, args, test_stats, start_time=None):   
    if not utils.is_main_process():
        return False
    
    # training wallclock-time / gpus-hours
    num_gpus = args.world_size if args.distributed else 1
    if start_time is not None:
        total_time = time.time() - start_time
        gpu_hours = total_time / 3600 * num_gpus
        gpu_hours_per_epoch = gpu_hours / args.epochs
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    else:
        total_time_str, gpu_hours, gpu_hours_per_epoch = ["N/A"] * 3

    # make result table
    now = datetime.datetime.now().strftime("%h%d %H:%M")
    tab_keys =  ["Time", "output_dir", "epochs", "bsz", "#GPUs"]
    tab_vals =  [now, Path(args.output_dir), args.epochs, int(args.batch_size * num_gpus), num_gpus]
    tab_keys += ["AP", "AP50", "AP75", "APs", "APm", "APl"]
    tab_vals += [v * 100 for v in test_stats['coco_eval_bbox'][:6]]

    # The average sparse ratio in val set
    tab_keys += ["Avg Rho"]
    tab_vals += [test_stats['test_sparse_ratio']]

    tab_keys += ["E/T", "GPU*hrs", "GPU*hrs/ep"]
    tab_vals += [total_time_str, gpu_hours, gpu_hours_per_epoch]
    
    # add benchmark
    if args.benchmark:
        # adaption layer is not needed during inference
        args.adaption = False
        model_infer, _, _ = build_model(args)
        model_infer.to(args.device)
        if not args.distributed:
            model_weight = model.state_dict()
        else:
            model_weight = model.module.state_dict()
        missing_keys_infer, _ = model_infer.load_state_dict(model_weight, strict=False)
        assert len(missing_keys_infer) == 0, f"missing_keys_infer:{missing_keys_infer}"
        model = model_infer
        gflops, giga_flops_encoder= compute_gflops(model, dataset_val, approximated=False, cpu=args.gflops_in_cpu)
        model.to(args.device)
        fps = compute_fps(model, dataset_val, num_iters=300, batch_size=1)
        tab_keys += ['GFLOPs', 'Encoder GFLOPs', 'FPS']
        tab_vals += [gflops, giga_flops_encoder, fps]
        
    table = tabulate([tab_vals], headers=tab_keys, tablefmt="pipe",
                     floatfmt=".3f", stralign="center", numalign="center")
    
    # dump to the file
    with open("log_result.txt", "a") as f:
        f.write("\n" + table + "\n")
            
    print(f"Save the final result to ./log_result.txt\n{table}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser('ATS-DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
