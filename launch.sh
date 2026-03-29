nohup accelerate launch --multi_gpu --num_processes 8 --mixed_precision bf16 \
    train.py --config=configs/cifar10_uvit_small.py > train.log 2>&1 &