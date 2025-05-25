"""
Stochastic Rounding version of Schedule-Free RAdam (RAdamScheduleFreeSR)
=======================================================================

This implementation is adapted from Facebook AI Research's **schedule-free** repository:
https://github.com/facebookresearch/schedule_free/blob/main/schedulefree/radam_schedulefree.py


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
    stochastic_grad_accummulation,
)


def copy_stochastic_(target: torch.Tensor, source: torch.Tensor):
    """
    copies source into target using stochastic rounding

    Args:
        target: the target tensor with dtype=bfloat16
        source: the target tensor with dtype=float32
    """
    # create a random 16 bit integer
    result = torch.randint_like(
        source,
        dtype=torch.int32,
        low=0,
        high=(1 << 16),
    )

    # add the random number to the lower 16 bit of the mantissa
    result.add_(source.view(dtype=torch.int32))

    # mask off the lower 16 bit of the mantissa
    result.bitwise_and_(-65536)  # -65536 = FFFF0000 as a signed int32

    # copy the higher 16 bit into the target tensor
    target.copy_(result.view(dtype=torch.float32))

    del result


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
        round_eval_train_switch: bool = False,  # Do not change unless you know what you are doing
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
        self.round_eval_train_switch = round_eval_train_switch

        # # Setup stochastic grad accumulation hooks
        for group in self.param_groups:
            for param in group["params"]:
                if param.requires_grad and param.dtype != torch.float32:
                    self.is_stochastic_rounding_accumulation = True
                    param.register_post_accumulate_grad_hook(
                        stochastic_grad_accummulation
                    )

    def step_hook(self):
        if not self.is_stochastic_rounding_accumulation:
            return
        # Copy over stochastically rounded grads
        for group in self.param_groups:
            for param in group["params"]:
                if param.requires_grad and hasattr(param, "_accum_grad"):
                    param.grad = param._accum_grad
                    del param._accum_grad

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
                if self.round_eval_train_switch:
                    new_p = p.float()
                    new_p.add_(z.float() - new_p, alpha=1 - inv_beta1)
                    copy_stochastic_(p, new_p)
                    del new_p
                else:
                    p.lerp_(z, 1 - inv_beta1)
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
                if self.round_eval_train_switch:
                    new_p = p.float()
                    new_p.add_(z.float() - new_p, alpha=1 - beta1)
                    copy_stochastic_(p, new_p)
                    del new_p
                else:
                    p.lerp_(z, 1 - beta1)
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
        self.step_hook()

        if not self.param_groups[0]["train_mode"]:
            raise RuntimeError(
                "Optimizer is in eval mode. Call optimizer.train() before "
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

            rect: float
            if rho_t > 4.0:
                rect = math.sqrt(
                    (rho_t - 4.0)
                    * (rho_t - 2.0)
                    * rho_inf
                    / ((rho_inf - 4.0) * (rho_inf - 2.0) * rho_t)
                )
            else:
                rect = float(not silent_sgd_phase)

            lr_scheduled = (
                group["lr"] * rect
            )  # DEBUG: Renamed to lr_scheduled for clarity in this scope
            group["scheduled_lr"] = lr_scheduled
            group["lr_max"] = lr_max = max(lr_scheduled, group["lr_max"])

            weight = (step_num**r) * (lr_max**weight_lr_power)
            weight_sum = group["weight_sum"] + weight
            group["weight_sum"] = weight_sum
            ckp1 = weight / weight_sum if weight_sum != 0.0 else 0.0

            adaptive_y_lr = lr_scheduled * (beta1 * (1.0 - ckp1) - 1.0)

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad  # Assuming grad is bf16
                state = self.state[p]

                # --- Lazy state initialisation
                if len(state) == 0:
                    state["z"] = torch.clone(
                        p, memory_format=torch.preserve_format
                    )  # p.dtype (bf16)
                    state["exp_avg_sq"] = torch.zeros_like(
                        p,
                        memory_format=torch.preserve_format,  # p.dtype (bf16)
                    )

                z = state["z"]  # p.dtype (bf16)
                exp_avg_sq = state["exp_avg_sq"]  # p.dtype (bf16)

                # --- Second‑moment update (RMS of gradients) - In-place bf16 ops
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                # --- Gradient normalisation (Adam‑style) or vanilla SGD - All bf16 ops
                # This will now modify 'grad' in-place if rho_t > 4.0, to match reference scalar
                # grad_normalized = grad  # grad_normalized is a reference to grad
                # if rho_t > 4.0:
                #     # Denominator calculation in p.dtype (bf16)
                #     denom = exp_avg_sq.div(bias_correction2).sqrt_().add_(eps)
                #     grad_normalized.div_(denom)  # In-place division
                # # else: grad_normalized is already grad (original)
                if rho_t > 4.0:
                    buf = exp_avg_sq.float()  # convert to fp32 for numerical stability
                    buf.div_(bias_correction2).sqrt_().add_(
                        eps
                    )  # buf now *is* denom (fp32)
                    grad_normalized = grad.float() / buf  # fp32
                else:
                    grad_normalized = grad.float()

                # --- Weight decay (applied in y‑space) - In-place bf16 ops
                if decay != 0.0:
                    grad_normalized.add_(p.float(), alpha=decay)  # In-place, p is bf16

                # ----------------------------------------------------------
                # y - update (parameter p)
                # ----------------------------------------------------------
                buf = p.float()
                buf.mul_(1.0 - ckp1).add_(z.float(), alpha=ckp1)
                buf.add_(grad_normalized, alpha=adaptive_y_lr)
                copy_stochastic_(p, buf)

                # DEBUG: Direct bf16 operations, matching reference scalar path style
                # p.lerp_(z, ckp1)  # p = (1-ckp1)*p + ckp1*z (in-place)
                # p.add_(
                #     grad_normalized, alpha=adaptive_y_lr
                # )  # p = p + adaptive_y_lr * grad_normalized (in-place)

                # ----------------------------------------------------------
                # z - update (SGD‑style) - All bf16 in-place ops
                # ----------------------------------------------------------
                buf = z.float()
                buf.sub_(grad_normalized, alpha=lr_scheduled)
                copy_stochastic_(z, buf)

                # DEBUG: Direct bf16 operation
                # z.sub_(
                #     grad_normalized, alpha=lr_scheduled
                # )  # z = z - lr_scheduled * grad_normalized (in-place)

                del buf

            group["k"] = step_num
        return loss
