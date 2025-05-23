"""
Stochastic Rounding version of Schedule‑Free RAdam (RAdamScheduleFreeSR)
=======================================================================

This implementation is adapted from Facebook AI Research’s **schedule‑free** repository:
https://github.com/facebookresearch/schedule_free/blob/main/schedulefree/radam_schedulefree.py

The update rules are identical to the original paper but **every write to model
weights ( y and z ) is performed with stochastic rounding** via
``toolkit.optimizers.optimizer_utils.copy_stochastic`` so that the parameters stay in
bfloat16 / fp16 without accumulating unbiased rounding error. The exponential second
moment is kept in the same dtype as the parameters, but you can switch it to an
``Auto8bitTensor`` for further memory savings – the two‑line change is marked below.

The class is drop‑in‑compatible with the original ``torch.optim.Optimizer`` API.  Call
``optimizer.train()`` before the first forward‑backward pass of every training epoch
and ``optimizer.eval()`` before running validation or saving a checkpoint, exactly as
you would with the upstream schedule‑free optimiser.

Key differences w.r.t. the reference implementation
---------------------------------------------------
* **Stochastic rounding writes** – all calls that used to modify ``p`` or ``state["z"]``
  in‑place now compute an fp32 reference and copy it stochastically.
* **foreach branch removed** – foreach kernels do not support stochastic rounding; we
  fall back to the scalar loop which is still fast on modern GPUs.

If you need gradient accumulation with stochastic rounding (e.g. micro‑batching), call
``toolkit.optimizers.optimizer_utils.stochastic_grad_accummulation(model)`` after every
inner step; this optimiser is agnostic to that detail.

Copyright (c) 2025, Your Name.  MIT License.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, Iterable, Optional, Tuple, Union

import torch
from typing_extensions import TypeAlias

try:
    from torch.optim.optimizer import ParamsT  # PyTorch ≥ 2.2 ships this alias
except ImportError:  # pragma: no cover
    ParamsT: TypeAlias = Union[Iterable[torch.Tensor], Iterable[Dict[str, Any]]]

# Stochastic‑rounding utilities supplied by the repository
from toolkit.optimizers.optimizer_utils import (
    copy_stochastic,
    stochastic_grad_accummulation,
)


class RAdamScheduleFreeSR(torch.optim.Optimizer):
    r"""Schedule‑Free RAdam (stochastic‑rounding edition).

    A warm‑up‑free variant of *Rectified Adam* [Liu et al., 2020] with the adaptive
    re‑parameterisation trick of *Schedule‑Free Optimisation* [Chen et al., 2023].
    This version writes parameters with **stochastic rounding** so that training with
    reduced precision is unbiased.

    The optimizer **must** be toggled between training and evaluation phases by
    calling :py:meth:`train` and :py:meth:`eval` respectively, as in the original
    Schedule‑Free implementation.
    """

    def __init__(
        self,
        params: ParamsT,
        lr: Union[float, torch.Tensor] = 2.5e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        r: float = 0.0,
        weight_lr_power: float = 2.0,
        foreach: Optional[bool] = False,  # foreach kernels disabled – see docstring
        silent_sgd_phase: bool = True,
    ) -> None:
        if foreach:
            raise ValueError(
                "foreach kernels are disabled in RAdamScheduleFreeSR because they "
                "do not expose hooks for stochastic rounding.  Pass foreach=False "
                "(default) or remove the argument."
            )
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            r=r,
            k=0,  # global step counter per param‑group
            train_mode=False,
            weight_sum=0.0,
            lr_max=-1.0,
            scheduled_lr=0.0,
            weight_lr_power=weight_lr_power,
            weight_decay=weight_decay,
            foreach=False,
            silent_sgd_phase=silent_sgd_phase,
        )
        super().__init__(params, defaults)

        self.is_stochastic_rounding_accumulation = False

        # Setup stochastic grad accumulation hooks
        for group in self.param_groups:
            for param in group["params"]:
                if param.requires_grad and param.dtype != torch.float32:
                    self.is_stochastic_rounding_accumulation = True
                    param.register_post_accumulate_grad_hook(
                        stochastic_grad_accummulation
                    )

    # ---------------------------------------------------------------------
    # Public helpers
    # ---------------------------------------------------------------------
    @torch.no_grad()
    def eval(self) -> None:  # pylint: disable=invalid‑name
        """Switch the optimiser to *evaluation* mode (a.k.a. **x‑space**).

        During evaluation we want the parameters ``x`` but the optimiser keeps two
        sets of weights (``y`` and ``z`` in the paper).  The update below follows
        Algorithm 1 of the *Schedule‑Free* paper.  Writes use stochastic rounding.
        """
        for group in self.param_groups:
            if not group["train_mode"]:
                continue
            beta1, _ = group["betas"]
            inv_beta1 = 1.0 / beta1
            for p in group["params"]:
                state = self.state[p]
                z = state.get("z")
                if z is None:
                    continue
                # p ← (1 − 1/β₁)·p  +  (1/β₁)·z
                new_p = p.float().mul(1.0 - inv_beta1).add(z.float(), alpha=inv_beta1)
                copy_stochastic(p, new_p)
                del new_p
            group["train_mode"] = False

    @torch.no_grad()
    def train(self) -> None:  # pylint: disable=invalid‑name
        """Switch the optimiser to *training* mode (a.k.a. **y‑space**)."""
        for group in self.param_groups:
            if group["train_mode"]:
                continue
            beta1, _ = group["betas"]
            for p in group["params"]:
                state = self.state[p]
                z = state.get("z")
                if z is None:
                    continue
                # p ← (1 − β₁)·p  +  β₁·z
                new_p = p.float().mul(1.0 - beta1).add(z.float(), alpha=beta1)
                copy_stochastic(p, new_p)
                del new_p
            group["train_mode"] = True

    # ------------------------------------------------------------------
    # Main optimisation step
    # ------------------------------------------------------------------
    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        """Perform **one** optimisation step.

        A :pyclass:`RuntimeError` is raised if the optimiser has not been switched to
        *training* mode.  This mirrors the behaviour of the upstream code.
        """
        if not self.param_groups[0]["train_mode"]:
            raise RuntimeError(
                "RAdamScheduleFreeSR is in eval mode.  Call optimizer.train() before "
                "back‑propagating and optimizer.eval() before validation/check‑pointing."
            )

        loss: Optional[float] = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            decay = group["weight_decay"]
            silent_sgd_phase = group["silent_sgd_phase"]
            r = group["r"]
            weight_lr_power = group["weight_lr_power"]

            k = group["k"]  # integer step counter (k = 0 before first step)
            step_num = k + 1  # 1‑based index used by the equations

            # ------------------------------------------------------------------
            # Schedule‑free learning‑rate + rectification term (same as baseline)
            # ------------------------------------------------------------------
            beta2_t = beta2**step_num
            bias_correction2 = 1.0 - beta2_t
            rho_inf = 2.0 / (1.0 - beta2) - 1.0
            rho_t = rho_inf - 2.0 * step_num * beta2_t / bias_correction2
            if rho_t > 4.0:
                rect = math.sqrt(
                    (rho_t - 4.0)
                    * (rho_t - 2.0)
                    * rho_inf
                    / ((rho_inf - 4.0) * (rho_inf - 2.0) * rho_t)
                )
            else:
                rect = float(not silent_sgd_phase)

            lr = group["lr"] * rect
            group["scheduled_lr"] = lr
            group["lr_max"] = lr_max = max(lr, group["lr_max"])

            # Weight for x ↔ y interpolation (ckp₁ in the paper)
            weight = (step_num**r) * (lr_max**weight_lr_power)
            weight_sum = group["weight_sum"] + weight
            group["weight_sum"] = weight_sum
            ckp1 = weight / weight_sum if weight_sum != 0.0 else 0.0

            adaptive_y_lr = lr * (beta1 * (1.0 - ckp1) - 1.0)

            # --------------------------------------------------------------
            # Parameter loop (foreach disabled – scalar loop is used)
            # --------------------------------------------------------------
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]

                # --- Lazy state initialisation
                if len(state) == 0:
                    # *z* and *exp_avg_sq* live in reduced precision like *p*
                    state["z"] = torch.clone(p, memory_format=torch.preserve_format)
                    state["exp_avg_sq"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )
                    # ♦ If you would like to keep the second moment in 8‑bit, replace
                    #   the previous line with the two lines below.
                    # exp_avg_sq_fp16 = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # state["exp_avg_sq"] = Auto8bitTensor(exp_avg_sq_fp16)

                buf = state.get("buf_fp32")
                if buf is None:
                    buf = torch.empty_like(p, dtype=torch.float32)
                    state["buf_fp32"] = buf

                z = state["z"]
                exp_avg_sq = state["exp_avg_sq"]

                # --- Second‑moment update (RMS of gradients)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                # --- Gradient normalisation (Adam‑style) or vanilla SGD
                if rho_t > 4.0:
                    buf.copy_(
                        exp_avg_sq, non_blocking=True
                    )  # buf ← exp_avg_sq  (bf16 → fp32)
                    buf.div_(bias_correction2).sqrt_().add_(
                        eps
                    )  # buf now *is* denom (fp32)
                    grad_norm = grad.float() / buf  # fp32
                else:
                    grad_norm = grad.float()

                # --- Weight decay (applied in y‑space)
                if decay != 0.0:
                    grad_norm = grad_norm.add(p, alpha=decay)

                # ----------------------------------------------------------
                # y ‑ update  (done in fp32 then stochastically rounded)
                # ----------------------------------------------------------
                buf.copy_(p, non_blocking=True).mul_(1.0 - ckp1).add_(
                    z, alpha=ckp1
                )  # buf = (1−ckp1)·y + ckp1·z in fp32
                buf.add_(grad_norm, alpha=adaptive_y_lr)  # + lr_y·ĝ
                copy_stochastic(p, buf)  # write back to bf16 with SR

                # ----------------------------------------------------------
                # z ‑ update (SGD‑style)
                # ----------------------------------------------------------
                buf.copy_(z, non_blocking=True)  # buf ← z in fp32
                buf.add_(grad_norm, alpha=-lr)  # buf = z − lr·ĝ
                copy_stochastic(z, buf)

            # bump step counter for the group
            group["k"] = step_num

        return loss
