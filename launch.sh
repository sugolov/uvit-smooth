#!/bin/bash
# Launch 7 smoothing ablations + 1 baseline, one per GPU.
# Usage: bash launch.sh
set -e

export WANDB_ENTITY=hmeng-university-of-toronto

COMMON="--config.sample.n_samples=10000 --config.sample.sample_steps=50 --config.sample.algorithm=dpm_solver"
BASE_CONFIG="configs/cifar10_uvit_small.py"
SMOOTH_CONFIG="configs/cifar10_uvit_small_post_adam.py"

# GPU 0: baseline
CUDA_VISIBLE_DEVICES=0 accelerate launch --num_processes 1 --mixed_precision bf16 \
    train.py --config=$BASE_CONFIG $COMMON &

# GPU 1: alpha=0.05, norm=none
CUDA_VISIBLE_DEVICES=1 accelerate launch --num_processes 1 --mixed_precision bf16 \
    train.py --config=$SMOOTH_CONFIG \
    --config.grad_smooth.alpha=0.05 \
    --config.grad_smooth.normalize=none \
    $COMMON &

# GPU 2: alpha=0.1, norm=none
CUDA_VISIBLE_DEVICES=2 accelerate launch --num_processes 1 --mixed_precision bf16 \
    train.py --config=$SMOOTH_CONFIG \
    --config.grad_smooth.alpha=0.1 \
    --config.grad_smooth.normalize=none \
    $COMMON &

# GPU 3: alpha=0.2, norm=none
CUDA_VISIBLE_DEVICES=3 accelerate launch --num_processes 1 --mixed_precision bf16 \
    train.py --config=$SMOOTH_CONFIG \
    --config.grad_smooth.alpha=0.2 \
    --config.grad_smooth.normalize=none \
    $COMMON &

# GPU 4: alpha=0.05, norm=rescale
CUDA_VISIBLE_DEVICES=4 accelerate launch --num_processes 1 --mixed_precision bf16 \
    train.py --config=$SMOOTH_CONFIG \
    --config.grad_smooth.alpha=0.05 \
    --config.grad_smooth.normalize=rescale \
    $COMMON &

# GPU 5: alpha=0.1, norm=rescale
CUDA_VISIBLE_DEVICES=5 accelerate launch --num_processes 1 --mixed_precision bf16 \
    train.py --config=$SMOOTH_CONFIG \
    --config.grad_smooth.alpha=0.1 \
    --config.grad_smooth.normalize=rescale \
    $COMMON &

# GPU 6: alpha=0.2, norm=rescale
CUDA_VISIBLE_DEVICES=6 accelerate launch --num_processes 1 --mixed_precision bf16 \
    train.py --config=$SMOOTH_CONFIG \
    --config.grad_smooth.alpha=0.2 \
    --config.grad_smooth.normalize=rescale \
    $COMMON &

CUDA_VISIBLE_DEVICES=7 accelerate launch --num_processes 1 --mixed_precision bf16 \
    train.py --config=$SMOOTH_CONFIG \
    --config.grad_smooth.alpha=0.15 \
    --config.grad_smooth.normalize=rescale \
    $COMMON &

echo "Launched 8 runs (1 baseline + 7 smoothed)/"
echo "Monitor: https://wandb.ai/hmeng-university-of-toronto/uvit_cifar10"

wait
echo "All runs complete."