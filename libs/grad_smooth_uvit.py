"""Apply deit_fork-style block gradient smoothing to UViT (in_blocks / out_blocks)."""
from __future__ import annotations

from typing import Any, List, Literal, Optional, Sequence

import torch
import torch.nn as nn

from .grad_filters import _get_params, smoother_simple


def grad_smooth_block_groups(model: nn.Module) -> List[Sequence[nn.Module]]:
    """
    Return transformer block lists that share the same per-block parameter layout
    so `smoother_simple` / post-Adam smoothers can run on each list separately.

    UViT encoder blocks omit skip connections; decoder blocks include skip_linear.
    The middle block is excluded (single block; neighbor filters are not meaningful).
    """
    if hasattr(model, "grad_smooth_block_groups"):
        return list(model.grad_smooth_block_groups())
    if not hasattr(model, "in_blocks") or not hasattr(model, "out_blocks"):
        raise TypeError(
            "model must be UViT (libs.uvit or libs.uvit_t2i) with in_blocks and out_blocks"
        )
    return [tuple(model.in_blocks), tuple(model.out_blocks)]


def _unwrap(model: nn.Module) -> nn.Module:
    return model.module if hasattr(model, "module") else model


def apply_raw_grad_smooth_uvit(
    model: nn.Module,
    gs: Any,
) -> None:
    """
    After backward, smooth gradients within each homogeneous block group.

    gs: config-like dict / FrozenConfigDict with keys:
      method: 'none' | 'window' | 'laplacian' | 'ema'
      alpha, rho, reverse, proj_only, normalize (same as smoother_simple)
    """
    method = gs.get("method", "none") if gs is not None else "none"
    if method == "none":
        return
    m = _unwrap(model)
    for blocks in grad_smooth_block_groups(m):
        if len(blocks) < 2:
            continue
        smoother_simple(
            blocks,
            method=method,
            alpha=float(gs.get("alpha", 0.5)),
            rho=float(gs.get("rho", 0.5)),
            reverse=bool(gs.get("reverse", True)),
            proj_only=bool(gs.get("proj_only", False)),
            normalize=gs.get("normalize", "none"),
        )


def post_adam_smoother_block_groups(
    block_groups: Sequence[Sequence[nn.Module]],
    optimizer: torch.optim.Optimizer,
    method: Literal["window", "laplacian", "ema"] = "window",
    alpha: float = 0.5,
    reverse: bool = False,
    rho: float = 0.5,
    proj_only: bool = False,
    normalize: Optional[Literal["none", "rescale", "normalize_before"]] = "none",
    decouple_weight_decay: bool = False,
) -> None:
    """
    One `optimizer.step()` with update smoothing applied independently within each
    group (same semantics as deit_fork `post_adam_smoother`, but for UViT's split blocks).
    """
    group_hparams = {}
    for group in optimizer.param_groups:
        lr = group.get("lr", None)
        wd = group.get("weight_decay", 0.0)
        for p in group.get("params", []):
            group_hparams[id(p)] = (lr, wd)

    # Pre-step snapshots for every parameter that appears in any block group
    old_by_id = {}
    for blocks in block_groups:
        for b in blocks:
            for param in _get_params(b, proj_only):
                if param.requires_grad and id(param) not in old_by_id:
                    old_by_id[id(param)] = param.data.clone()

    optimizer.step()

    for blocks in block_groups:
        if len(blocks) < 2:
            continue
        L = len(blocks)
        params = [list(_get_params(b, proj_only)) for b in blocks]
        n_params = len(params[0])

        old_params = [[None] * L for _ in range(n_params)]
        for i in range(L):
            for p_idx, param in enumerate(params[i]):
                oid = id(param)
                if oid in old_by_id:
                    old_params[p_idx][i] = old_by_id[oid]

        updates = [[None] * L for _ in range(n_params)]
        wd_updates = [[None] * L for _ in range(n_params)] if decouple_weight_decay else None
        orig_norms = (
            [[None] * L for _ in range(n_params)]
            if normalize in ("rescale", "normalize_before")
            else None
        )

        for i in range(L):
            for p_idx, param in enumerate(params[i]):
                if old_params[p_idx][i] is None:
                    continue
                total_update = param.data - old_params[p_idx][i]

                if decouple_weight_decay:
                    lr, wd = group_hparams.get(id(param), (None, 0.0))
                    if lr is not None and wd and wd != 0.0:
                        wd_delta = old_params[p_idx][i].mul(-lr * wd)
                        wd_updates[p_idx][i] = wd_delta
                        update = total_update - wd_delta
                    else:
                        update = total_update
                else:
                    update = total_update

                updates[p_idx][i] = update.clone()
                if orig_norms is not None:
                    orig_norms[p_idx][i] = update.norm()
                if normalize == "normalize_before" and orig_norms is not None:
                    if orig_norms[p_idx][i] > 0:
                        updates[p_idx][i].div_(orig_norms[p_idx][i])

        smoothed_updates = [[None] * L for _ in range(n_params)]

        if method == "window":
            for i in range(L):
                for p_idx in range(n_params):
                    if updates[p_idx][i] is None:
                        continue
                    left = updates[p_idx][i - 1] if i > 0 else None
                    right = updates[p_idx][i + 1] if i < L - 1 else None
                    w_self = 1 - alpha / 2 if (left is None or right is None) else 1 - alpha
                    smoothed = updates[p_idx][i] * w_self
                    if left is not None:
                        smoothed.add_(left, alpha=alpha / 2)
                    if right is not None:
                        smoothed.add_(right, alpha=alpha / 2)
                    smoothed_updates[p_idx][i] = smoothed

        elif method == "laplacian":
            for i in range(L):
                for p_idx in range(n_params):
                    if updates[p_idx][i] is None:
                        continue
                    left = updates[p_idx][i - 1] if i > 0 else None
                    right = updates[p_idx][i + 1] if i < L - 1 else None
                    w_self = 1 - alpha / 2 if (left is None or right is None) else 1 - alpha
                    smoothed = updates[p_idx][i] * w_self
                    if left is not None:
                        smoothed.add_(left, alpha=-alpha / 2)
                    if right is not None:
                        smoothed.add_(right, alpha=-alpha / 2)
                    smoothed_updates[p_idx][i] = smoothed

        elif method == "ema":
            indices = range(L - 1, -1, -1) if reverse else range(L)
            for p_idx in range(n_params):
                acc = None
                for i in indices:
                    if updates[p_idx][i] is None:
                        continue
                    if acc is None:
                        acc = updates[p_idx][i].clone()
                    else:
                        acc.mul_(rho).add_(updates[p_idx][i] * (1 - rho))
                    smoothed_updates[p_idx][i] = acc.clone()
        else:
            raise ValueError(f"Unknown method: {method}")

        for i in range(L):
            for p_idx in range(n_params):
                if smoothed_updates[p_idx][i] is None:
                    continue
                if normalize == "rescale":
                    orig_norm = orig_norms[p_idx][i] if orig_norms is not None else None
                    if orig_norm is not None and orig_norm > 0:
                        current_norm = smoothed_updates[p_idx][i].norm()
                        if current_norm > 0:
                            smoothed_updates[p_idx][i].mul_(orig_norm / current_norm)
                elif normalize == "normalize_before":
                    orig_norm = orig_norms[p_idx][i] if orig_norms is not None else None
                    if orig_norm is not None and orig_norm > 0:
                        current_norm = smoothed_updates[p_idx][i].norm()
                        if current_norm > 0:
                            smoothed_updates[p_idx][i].mul_(orig_norm / current_norm)

        for i in range(L):
            for p_idx, param in enumerate(params[i]):
                if smoothed_updates[p_idx][i] is not None:
                    param.data.copy_(old_params[p_idx][i])
                    if decouple_weight_decay and wd_updates is not None and wd_updates[p_idx][i] is not None:
                        param.data.add_(wd_updates[p_idx][i])
                    param.data.add_(smoothed_updates[p_idx][i])


def optimizer_step_with_post_adam_uvit(
    optimizer: torch.optim.Optimizer,
    model: nn.Module,
    gs: Any,
) -> None:
    """Run optimizer step; optionally apply post-Adam update smoothing per UViT block group."""
    method = gs.get("method", "none") if gs is not None else "none"
    if gs is None or method == "none":
        optimizer.step()
        return
    m = _unwrap(model)
    groups = grad_smooth_block_groups(m)
    post_adam_smoother_block_groups(
        groups,
        optimizer,
        method=method,
        alpha=float(gs.get("alpha", 0.5)),
        rho=float(gs.get("rho", 0.5)),
        reverse=bool(gs.get("reverse", False)),
        proj_only=bool(gs.get("proj_only", False)),
        normalize=gs.get("normalize", "none"),
        decouple_weight_decay=bool(gs.get("decouple_weight_decay", False)),
    )
