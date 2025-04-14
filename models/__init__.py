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


from .deformable_detr import build
from .distillation_loss import build_distillation_loss


def build_model(args):
    return build(args)

def build_distillation(teacher_model, args, postprocessors=None, model_LB = None):
    return build_distillation_loss(teacher_model, args, postprocessors, model_LB)
