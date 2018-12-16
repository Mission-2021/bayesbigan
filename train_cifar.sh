#!/usr/bin/env bash

NOISE="u-100"  # feature/noise (z) distribution is a 50-D uniform

# BiGAN objective
OBJECTIVE="--encode_gen_weight 1 --encode_weight 0 --discrim_weight 0 --joint_discrim_weight 1"

# Latent Regressor (LR) objective
# OBJECTIVE="--encode_gen_weight 0 --encode_weight 1 --discrim_weight 1 --joint_discrim_weight 0"

# Joint Latent Regressor (Joint LR) objective
# OBJECTIVE="--encode_gen_weight 0.25 --encode_weight 1 --discrim_weight 1 --joint_discrim_weight 0"

python -u train_gan.py \
    --num_generator 2 \
    --encode --encode_normalize \
    --dataset cifar3 --crop_size 32 \
    --gen_net_size 32 \
    --feat_net_size 32 \
    --encode_net alexnet_group_padpool \
    --megabatch_gb 0.5 \
    --classifier --classifier_deploy \
    --nolog_gain --nogain --nobias --no_decay_gain \
    --deploy_iters 1 \
    --disp_samples 400 \
    --disp_interval 200 \
    --save_interval 200 \
    --no_disp_one \
    --epochs 400 --decay_epochs 400 \
    --optimizer adam \
    --noise ${NOISE} \
    ${OBJECTIVE} \
    $@
