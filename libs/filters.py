import math
import torch
from torch import nn
from typing import Sequence, Literal, Union, Optional

# ---------------------------------------------------------------------
# Gradient‑smoothing utility
# ---------------------------------------------------------------------
def window_smoother(
    residual_blocks: Sequence[torch.nn.Module],
    k: int = 1,
    *,
    device: Union[str, torch.device] = "cuda",
    gamma: float = 0.5,                # weight on the *current* block’s grad
    alpha: float = math.log(2),        # decay rate for distance weighting
    uniform: bool = False,             # if True → equal weights for neighbors
    same_nb_weight: bool = False,      # if True → ignore distance, same weight
    direction: Literal["left", "right", "both"] = "both",
    reverse: bool = False,             # iterate from last block to first
    rescale: bool = False,             # preserve original grad L2‑norm
) -> None:
    """
    Replace each block's .grad with a convex combo of its own gradient and
    those of up to `k` neighboring blocks.

    Parameters
    ----------
    residual_blocks : list[nn.Module]
        Ordered residual (or transformer) blocks whose `.parameters()` share
        the exact layout across blocks.
    k : int
        Number of neighbors to include on each side (depending on `direction`).
    gamma : float
        Weight allocated to the block’s own gradient (0 ≤ gamma ≤ 1).
    alpha : float
        Exponential decay constant for distance‑based weights.
    uniform : bool
        If True, neighbor weights are equal instead of distance‑decayed.
    same_nb_weight : bool
        Overrides distance decay: every neighbor gets identical weight.
    direction : {"left", "right", "both"}
        Which side(s) to sample neighbors from.
    reverse : bool
        Iterate blocks in reverse order (useful for tied/RNN‑style back‑passes).
    rescale : bool
        Renormalise smoothed grad to match the original grad’s L2‑norm.
    """
    N = len(residual_blocks)
    block_iter = reversed(range(N)) if reverse else range(N)

    for i in block_iter:
        # --------------------------------------------------------------
        # 1. Determine neighbor indices for this block
        # --------------------------------------------------------------
        if direction == "left":
            nb_idx = list(range(max(0, i - k), i + 1))
        elif direction == "right":
            nb_idx = list(range(i, min(N, i + k + 1)))
        else:  # "both"
            nb_idx = list(range(max(0, i - k), min(N, i + k + 1)))

        # --------------------------------------------------------------
        # 2. Build weights  (own‑block weight = γ, neighbors = 1‑γ)
        # --------------------------------------------------------------
        if uniform:
            # equal weight for all neighbors regardless of distance
            base = torch.ones(len(nb_idx), device=device)
        else:
            # decay with distance unless same_nb_weight overrides
            dists = torch.tensor([abs(i - j) for j in nb_idx], device=device)
            base  = torch.exp(-alpha * dists)
            if same_nb_weight:
                base[dists > 0] = 1.0  # flatten all neighbor weights

        # own block gets gamma, neighbors share (1‑gamma) proportionally
        own_mask     = torch.tensor([j == i for j in nb_idx], device=device)
        neighbor_sum = base[~own_mask].sum()
        weights      = torch.empty_like(base)

        weights[own_mask]     = gamma
        weights[~own_mask]    = (1.0 - gamma) * base[~own_mask] / neighbor_sum
        weights = weights.to(device)

        # --------------------------------------------------------------
        # 3. Apply weighted average to every parameter tensor
        # --------------------------------------------------------------
        for p_idx, p in enumerate(residual_blocks[i].parameters()):
            if p.grad is None:
                continue

            original_norm = p.grad.norm() if rescale else None
            smoothed      = torch.zeros_like(p.grad)

            for w, j in zip(weights, nb_idx):
                nb_param = list(residual_blocks[j].parameters())[p_idx]
                if nb_param.grad is not None:
                    smoothed.add_(w * nb_param.grad)

            if rescale and smoothed.norm() > 0:
                smoothed.mul_(original_norm / smoothed.norm())

            p.grad.copy_(smoothed)

def momentum_smoother(
    residual_blocks: Sequence[nn.Module],
    beta: float = 0.9,       # momentum coefficient
    reverse: bool = True,    # direction of momentum
    device: Union[str, torch.device] = "cuda",
) -> None:
    """
    For each parameter tensor shared across blocks, replace its .grad with an
    EMA of gradients from earlier blocks in the pass:

        m_t = β · m_{t‑1} + (1‑β) · g_t
        g_t <- m_t

    If `reverse=True` the pass goes from last block to first; otherwise forward.
    """
    # Choose traversal order
    blocks = list(reversed(residual_blocks)) if reverse else list(residual_blocks)

    # One momentum buffer per *parameter index* (assumes same layout across blocks)
    num_params = len(list(blocks[0].parameters()))
    momentum   = [None] * num_params

    for blk in blocks:
        for idx, p in enumerate(blk.parameters()):
            if p.grad is None:
                continue

            g = p.grad.detach()

            if momentum[idx] is None:
                momentum[idx] = g.clone()
            else:
                momentum[idx].mul_(beta).add_(g, alpha=1 - beta)

            # overwrite gradient with smoothed value
            p.grad.copy_(momentum[idx])


# ---------------------------------------------------------------------
# Alternative gradient smoothing methods
# ---------------------------------------------------------------------
"""Gradient smoothing across residual blocks.

Methods:
    window:    g_i = (1-α)*g_i + (α/2)*(g_{i-1} + g_{i+1})
    laplacian: g_i = (1-α)*g_i - (α/2)*(g_{i-1} + g_{i+1})
    ema:       g_i = Σ_{j>=i} ρ^{j-i} * g_j  (or reversed)

Normalization options:
    'none': No normalization
    'rescale': Rescale smoothed gradient to original parameter gradient's norm
    'normalize_before': Normalize all gradients before smoothing, then rescale to original norm
"""


def _get_params(block: nn.Module, proj_only: bool):
    """Yield parameters: only Linear if proj_only, else all."""
    if proj_only:
        for module in block.modules():
            if isinstance(module, nn.Linear):
                for param in module.parameters(recurse=False):
                    yield param
    else:
        for param in block.parameters():
            yield param


def window(blocks: Sequence[torch.nn.Module], alpha: float = 0.5, proj_only: bool = True,
           normalize: Optional[Literal['none', 'rescale', 'normalize_before']] = 'none') -> None:
    """Neighbor averaging (low-pass filter).
    
    Args:
        blocks: Sequence of residual blocks.
        alpha: Smoothing strength.
        proj_only: If True, only smooth Linear layer params.
        normalize: Normalization method ('none', 'rescale', 'normalize_before').
    """
    L = len(blocks)
    
    params = [list(_get_params(b, proj_only)) for b in blocks]
    n_params = len(params[0])
    
    orig = [[None]*L for _ in range(n_params)]
    # Only store norms if we need to normalize before (since we'll lose the original)
    orig_norms = [[None]*L for _ in range(n_params)] if normalize == 'normalize_before' else None
    for i in range(L):
        for p, param in enumerate(params[i]):
            if param.grad is not None:
                orig[p][i] = param.grad.clone()
                if normalize == 'normalize_before':
                    orig_norms[p][i] = param.grad.norm()
                    # Normalize in-place
                    if orig_norms[p][i] > 0:
                        orig[p][i].div_(orig_norms[p][i])
    
    for i in range(L):
        for p, param in enumerate(params[i]):
            if orig[p][i] is None:
                continue
            
            left = orig[p][i-1] if i > 0 else None
            right = orig[p][i+1] if i < L-1 else None
            
            w_self = 1 - alpha/2 if (left is None or right is None) else 1 - alpha
            
            param.grad.copy_(orig[p][i] * w_self)
            if left is not None:
                param.grad.add_(left, alpha=alpha/2)
            if right is not None:
                param.grad.add_(right, alpha=alpha/2)
            
            # Rescale to original norm if requested
            if normalize == 'rescale':
                orig_norm = orig[p][i].norm()
                if orig_norm > 0:
                    current_norm = param.grad.norm()
                    if current_norm > 0:
                        param.grad.mul_(orig_norm / current_norm)
            elif normalize == 'normalize_before' and orig_norms[p][i] > 0:
                current_norm = param.grad.norm()
                if current_norm > 0:
                    param.grad.mul_(orig_norms[p][i] / current_norm)


def laplacian(blocks: Sequence[torch.nn.Module], alpha: float = 0.5, proj_only: bool = True,
              normalize: Optional[Literal['none', 'rescale', 'normalize_before']] = 'none') -> None:
    """Negative neighbor averaging (high-pass / sharpening).
    
    Args:
        blocks: Sequence of residual blocks.
        alpha: Smoothing strength.
        proj_only: If True, only smooth Linear layer params.
        normalize: Normalization method ('none', 'rescale', 'normalize_before').
    """
    L = len(blocks)
    
    params = [list(_get_params(b, proj_only)) for b in blocks]
    n_params = len(params[0])
    
    orig = [[None]*L for _ in range(n_params)]
    # Only store norms if we need to normalize before (since we'll lose the original)
    orig_norms = [[None]*L for _ in range(n_params)] if normalize == 'normalize_before' else None
    for i in range(L):
        for p, param in enumerate(params[i]):
            if param.grad is not None:
                orig[p][i] = param.grad.clone()
                if normalize == 'normalize_before':
                    orig_norms[p][i] = param.grad.norm()
                    # Normalize in-place
                    if orig_norms[p][i] > 0:
                        orig[p][i].div_(orig_norms[p][i])
    
    for i in range(L):
        for p, param in enumerate(params[i]):
            if orig[p][i] is None:
                continue
            
            left = orig[p][i-1] if i > 0 else None
            right = orig[p][i+1] if i < L-1 else None
            
            w_self = 1 - alpha/2 if (left is None or right is None) else 1 - alpha
            
            param.grad.copy_(orig[p][i] * w_self)
            if left is not None:
                param.grad.add_(left, alpha=-alpha/2)
            if right is not None:
                param.grad.add_(right, alpha=-alpha/2)
            
            # Rescale to original norm if requested
            if normalize == 'rescale':
                orig_norm = orig[p][i].norm()
                if orig_norm > 0:
                    current_norm = param.grad.norm()
                    if current_norm > 0:
                        param.grad.mul_(orig_norm / current_norm)
            elif normalize == 'normalize_before' and orig_norms[p][i] > 0:
                current_norm = param.grad.norm()
                if current_norm > 0:
                    param.grad.mul_(orig_norms[p][i] / current_norm)


def ema(blocks: Sequence[torch.nn.Module], rho: float = 0.5, reverse: bool = True, proj_only: bool = True,
        normalize: Optional[Literal['none', 'rescale', 'normalize_before']] = 'none') -> None:
    """Exponential moving average across blocks.
    
    Args:
        blocks: Sequence of residual blocks.
        rho: Decay rate.
        reverse: Whether to reverse the order.
        proj_only: If True, only smooth Linear layer params.
        normalize: Normalization method ('none', 'rescale', 'normalize_before').
    """
    L = len(blocks)
    
    params = [list(_get_params(b, proj_only)) for b in blocks]
    n_params = len(params[0])
    
    orig = [[None]*L for _ in range(n_params)]
    # Only store norms if we need to normalize before (since we'll lose the original)
    orig_norms = [[None]*L for _ in range(n_params)] if normalize == 'normalize_before' else None
    for i in range(L):
        for p, param in enumerate(params[i]):
            if param.grad is not None:
                orig[p][i] = param.grad.clone()
                if normalize == 'normalize_before':
                    orig_norms[p][i] = param.grad.norm()
                    # Normalize in-place
                    if orig_norms[p][i] > 0:
                        orig[p][i].div_(orig_norms[p][i])
    
    indices = range(L-1, -1, -1) if reverse else range(L)
    
    for p in range(n_params):
        acc = None
        for i in indices:
            if orig[p][i] is None:
                continue
            if acc is None:
                acc = orig[p][i].clone()
            else:
                acc.mul_(rho).add_(orig[p][i] * (1 - rho))
            
            # Store smoothed gradient
            params[i][p].grad.copy_(acc)
            
            # Rescale to original norm if requested
            if normalize == 'rescale':
                orig_norm = orig[p][i].norm()
                if orig_norm > 0:
                    current_norm = params[i][p].grad.norm()
                    if current_norm > 0:
                        params[i][p].grad.mul_(orig_norm / current_norm)
            elif normalize == 'normalize_before' and orig_norms[p][i] > 0:
                current_norm = params[i][p].grad.norm()
                if current_norm > 0:
                    params[i][p].grad.mul_(orig_norms[p][i] / current_norm)


def smoother_simple(
    blocks: Sequence[torch.nn.Module],
    method: Literal['none', 'window', 'laplacian', 'ema'] = 'none',
    alpha: float = 0.5,
    rho: float = 0.5,
    reverse: bool = True,
    proj_only: bool = False,
    normalize: Optional[Literal['none', 'rescale', 'normalize_before']] = 'none',
) -> None:
    """Apply gradient smoothing across blocks.
    
    Args:
        blocks: Sequence of residual blocks to smooth gradients for.
        method: Smoothing method to use ('none', 'window', 'laplacian', 'ema').
        alpha: Smoothing strength for window/laplacian methods.
        rho: Decay rate for ema method.
        reverse: Whether to reverse the order for ema method.
        proj_only: If True, only smooth Linear layer params. If False, smooth all.
        normalize: Normalization method ('none', 'rescale', 'normalize_before').
    """
    if method == 'none':
        return
    elif method == 'window':
        window(blocks, alpha=alpha, proj_only=proj_only, normalize=normalize)
    elif method == 'laplacian':
        laplacian(blocks, alpha=alpha, proj_only=proj_only, normalize=normalize)
    elif method == 'ema':
        ema(blocks, rho=rho, reverse=reverse, proj_only=proj_only, normalize=normalize)
    else:
        raise ValueError(f"Unknown method: {method}")


# ---------------------------------------------------------------------
# Post-Adam update smoothing (smoothing applied after Adam computes updates)
# ---------------------------------------------------------------------

def post_adam_smoother(
    blocks: Sequence[torch.nn.Module],
    optimizer: torch.optim.Optimizer,
    method: Literal['window', 'laplacian', 'ema'] = 'window',
    alpha: float = 0.5,
    reverse: bool = False,
    rho: float = 0.5,
    proj_only: bool = False,
    normalize: Optional[Literal['none', 'rescale', 'normalize_before']] = 'none',
    decouple_weight_decay: bool = False,
) -> None:
    """Post-Adam smoother, analogous to the in-gradient smoothers.
    
    Mechanism:
    1) Save current parameters (old)
    2) Run optimizer.step() (updates params)
    3) Compute per-parameter update delta = new - old
    4) Temporarily write delta into .grad for each param
    5) Apply the same filter code used for gradient smoothing across blocks
    6) Restore params to old and apply smoothed deltas
    7) Restore original .grad tensors
    
    Args:
        blocks: Sequence of residual blocks.
        optimizer: The optimizer (should be Adam/AdamW).
        method/alpha/rho/reverse/proj_only/normalize: Same semantics as smoother_simple(...),
            but applied to the parameter delta (update) instead of raw grads.
    """
    L = len(blocks)
    params = [list(_get_params(b, proj_only)) for b in blocks]
    n_params = len(params[0])
    
    # Build mapping from parameter -> (lr, weight_decay) for later decomposition of updates.
    # NOTE: This assumes AdamW-style *decoupled* weight decay (param *= (1 - lr*wd)).
    # If the optimizer is not decoupled, the weight decay effect is mixed into the gradient
    # and cannot be separated cleanly here.
    group_hparams = {}
    for group in optimizer.param_groups:
        lr = group.get('lr', None)
        wd = group.get('weight_decay', 0.0)
        for p in group.get('params', []):
            group_hparams[id(p)] = (lr, wd)

    # Step 1: Store old parameter values
    old_params = [[None]*L for _ in range(n_params)]
    for i in range(L):
        for p_idx, param in enumerate(params[i]):
            if param.requires_grad:
                old_params[p_idx][i] = param.data.clone()

    # Step 2: Let Adam compute updates (this modifies param.data)
    optimizer.step()
    
    # Step 3: Compute updates.
    # If decouple_weight_decay=True, we decompose:
    #   total_delta = (new - old) = wd_delta + adam_delta
    # where wd_delta = -lr*wd*old (AdamW step weight decay), and only adam_delta is smoothed.
    updates = [[None]*L for _ in range(n_params)]  # the part we will smooth
    wd_updates = [[None]*L for _ in range(n_params)] if decouple_weight_decay else None
    # For both 'rescale' and 'normalize_before', we need the original (pre-smoothing) norm
    # of the non-WD (Adam) update.
    orig_norms = [[None]*L for _ in range(n_params)] if normalize in ('rescale', 'normalize_before') else None
    
    for i in range(L):
        for p_idx, param in enumerate(params[i]):
            if old_params[p_idx][i] is not None:
                total_update = param.data - old_params[p_idx][i]

                if decouple_weight_decay:
                    lr, wd = group_hparams.get(id(param), (None, 0.0))
                    # If lr is missing (shouldn't happen), fall back to no decomposition.
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
                if normalize == 'normalize_before' and orig_norms is not None:
                    if orig_norms[p_idx][i] > 0:
                        updates[p_idx][i].div_(orig_norms[p_idx][i])
    
    # Step 4: Smooth updates across blocks
    smoothed_updates = [[None]*L for _ in range(n_params)]
    
    if method == 'window':
        for i in range(L):
            for p_idx in range(n_params):
                if updates[p_idx][i] is None:
                    continue
                
                left = updates[p_idx][i-1] if i > 0 else None
                right = updates[p_idx][i+1] if i < L-1 else None
                
                w_self = 1 - alpha/2 if (left is None or right is None) else 1 - alpha
                smoothed = updates[p_idx][i] * w_self
                
                if left is not None:
                    smoothed.add_(left, alpha=alpha/2)
                if right is not None:
                    smoothed.add_(right, alpha=alpha/2)
                
                smoothed_updates[p_idx][i] = smoothed
                
    elif method == 'laplacian':
        for i in range(L):
            for p_idx in range(n_params):
                if updates[p_idx][i] is None:
                    continue
                
                left = updates[p_idx][i-1] if i > 0 else None
                right = updates[p_idx][i+1] if i < L-1 else None
                
                w_self = 1 - alpha/2 if (left is None or right is None) else 1 - alpha
                smoothed = updates[p_idx][i] * w_self
                
                if left is not None:
                    smoothed.add_(left, alpha=-alpha/2)
                if right is not None:
                    smoothed.add_(right, alpha=-alpha/2)
                
                smoothed_updates[p_idx][i] = smoothed
                
    elif method == 'ema':
        indices = range(L-1, -1, -1) if reverse else range(L)
        
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
    
    # Step 5: Apply normalization if needed
    for i in range(L):
        for p_idx in range(n_params):
            if smoothed_updates[p_idx][i] is None:
                continue
            
            if normalize == 'rescale':
                orig_norm = orig_norms[p_idx][i] if orig_norms is not None else None
                if orig_norm is not None and orig_norm > 0:
                    current_norm = smoothed_updates[p_idx][i].norm()
                    if current_norm > 0:
                        smoothed_updates[p_idx][i].mul_(orig_norm / current_norm)
            elif normalize == 'normalize_before':
                orig_norm = orig_norms[p_idx][i] if orig_norms is not None else None
                if orig_norm is not None and orig_norm > 0:
                    current_norm = smoothed_updates[p_idx][i].norm()
                    if current_norm > 0:
                        smoothed_updates[p_idx][i].mul_(orig_norm / current_norm)
    
    # Step 6: Apply (optional) unsmoothed wd update + smoothed updates
    for i in range(L):
        for p_idx, param in enumerate(params[i]):
            if smoothed_updates[p_idx][i] is not None:
                # Restore old value and apply smoothed update
                param.data.copy_(old_params[p_idx][i])
                if decouple_weight_decay and wd_updates is not None and wd_updates[p_idx][i] is not None:
                    param.data.add_(wd_updates[p_idx][i])
                param.data.add_(smoothed_updates[p_idx][i])


class AdamUpdateSmoother:
    """Wrapper that smooths Adam updates after they are computed but before application.
    
    This class intercepts the optimizer step, lets Adam compute its updates internally,
    then smooths those updates across blocks before applying them.
    
    Usage:
        smoother = AdamUpdateSmoother(blocks, method='window', alpha=0.5)
        # Instead of optimizer.step(), use:
        smoother.step(optimizer)
    """
    
    def __init__(
        self,
        blocks: Sequence[torch.nn.Module],
        method: Literal['window', 'laplacian', 'ema'] = 'window',
        alpha: float = 0.5,
        rho: float = 0.5,
        reverse: bool = True,
        proj_only: bool = True,
        normalize: Optional[Literal['none', 'rescale', 'normalize_before']] = 'none',
    ):
        self.blocks = blocks
        self.method = method
        self.alpha = alpha
        self.rho = rho
        self.reverse = reverse
        self.proj_only = proj_only
        self.normalize = normalize
    
    def step(self, optimizer: torch.optim.Optimizer) -> None:
        """Perform optimizer step with update smoothing.
        
        This method:
        1. Stores current parameter values
        2. Calls optimizer.step() to let Adam compute updates (modifies param.data)
        3. Computes updates as (new_param - old_param)
        4. Smooths the updates across blocks
        5. Restores old values and applies smoothed updates
        
        Note: This replaces the normal optimizer.step() call.
        """
        post_adam_smoother(
            self.blocks,
            optimizer,
            method=self.method,
            alpha=self.alpha,
            rho=self.rho,
            reverse=self.reverse,
            proj_only=self.proj_only,
            normalize=self.normalize,
        )