#!/usr/bin/env bash

set -x

EXP_DIR=exps/r50_ATS_DETR_0.2_0.3
PY_ARGS=${@:1}

python -u main.py \
    --output_dir ${EXP_DIR} \
    --with_box_refine \
    --two_stage \
    --eff_query_init \
    --eff_specific_head \
    --use_enc_aux_loss \
    --adaptive \
    --bias 1.0 \
    --min_ratio 0.2 \
    --max_ratio 0.3 \
    --LB_checkpoint ./teacher_ckpt/r50_teacher_0.2.pth \
    --global_distill \
    --teacher_checkpoint ./teacher_ckpt/r50_teacher_0.3.pth \
    --batch_size 4 \
    ${PY_ARGS}
