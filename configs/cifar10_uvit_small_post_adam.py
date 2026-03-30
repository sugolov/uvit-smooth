"""CIFAR-10 U-ViT-S/2 with post-Adam update smoothing (DeiT-style filters on Adam deltas)."""
import ml_collections
from configs.cifar10_uvit_small import get_config as _base_config


def get_config():
    config = _base_config()

    config.grad_smooth = ml_collections.ConfigDict()
    config.grad_smooth.method = "window"
    config.grad_smooth.post_adam = True
    # Match deit_fork post_adam_smoother default traversal for window/laplacian
    config.grad_smooth.reverse = False
    config.grad_smooth.alpha = 0.1
    config.grad_smooth.proj_only = False
    config.grad_smooth.normalize = "none"
    # Set True if you want only the non–weight-decay part of AdamW updates smoothed
    config.grad_smooth.decouple_weight_decay = True
    # 'separate': smooth in_blocks and out_blocks independently; 'combined': full depth, common params only
    config.grad_smooth.block_grouping = "separate"

    return config