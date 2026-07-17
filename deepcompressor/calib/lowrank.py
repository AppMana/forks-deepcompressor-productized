# -*- coding: utf-8 -*-
"""Quantization SVD calibration module."""

from dataclasses import _MISSING_TYPE, MISSING

import torch
import torch.nn as nn

from ..data.cache import TensorCache, TensorsCache
from ..data.common import TensorType
from ..nn.patch.lowrank import LowRankBranch
from ..quantizer.processor import Quantizer
from ..utils import math, tools
from ..utils.config import KeyEnableConfig
from .config import QuantLowRankCalibConfig, SearchBasedCalibObjective
from .search import SearchBasedCalibrator

__all__ = ["QuantLowRankCalibrator", "solve_activation_aware_low_rank"]


@torch.no_grad()
def solve_activation_aware_low_rank(
    weight: torch.Tensor,
    quantized_weight: torch.Tensor,
    covariance: torch.Tensor,
    quantized_cross: torch.Tensor,
    *,
    rank: int,
    damping: float,
    svd_mode: str,
    svd_oversample: int,
    svd_niter: int,
) -> LowRankBranch:
    """Solve the activation-aware reduced-rank W16A16 correction.

    ``covariance`` is E[x.T @ x] and ``quantized_cross`` is
    E[q(x).T @ x]. The target correction therefore includes both W4 and A4
    error: ``W @ x - Q(W) @ q(x)``.
    """

    if weight.ndim < 2 or quantized_weight.shape != weight.shape:
        raise ValueError("Activation-aware weights must have matching matrix shapes")
    weight = weight.view(weight.shape[0], -1)
    quantized_weight = quantized_weight.view(quantized_weight.shape[0], -1)
    in_features = weight.shape[1]
    expected = (in_features, in_features)
    if covariance.shape != expected or quantized_cross.shape != expected:
        raise ValueError(
            f"Activation statistics must both have shape {expected}, got "
            f"{tuple(covariance.shape)} and {tuple(quantized_cross.shape)}"
        )
    if rank <= 0 or rank > min(weight.shape):
        raise ValueError(f"Activation-aware rank must be in 1..{min(weight.shape)}, got {rank}")
    if damping < 0:
        raise ValueError("Activation-aware damping must be non-negative")

    device, output_dtype = weight.device, weight.dtype
    solve_dtype = covariance.dtype
    covariance = covariance.to(device=device, dtype=solve_dtype)
    quantized_cross = quantized_cross.to(device=device, dtype=solve_dtype)
    covariance = (covariance + covariance.mT) * 0.5
    diagonal_scale = covariance.diagonal().mean().abs().clamp_min(torch.finfo(solve_dtype).tiny)
    diagonal = damping * diagonal_scale
    # A tiny numerical floor keeps Cholesky defined when the sampled
    # activation matrix is rank deficient. It is many orders below the
    # configurable statistical damping.
    numerical_floor = torch.finfo(solve_dtype).eps * covariance.shape[0] * diagonal_scale
    regularized = covariance + torch.eye(in_features, device=device, dtype=solve_dtype) * torch.maximum(
        diagonal, numerical_floor
    )
    chol, info = torch.linalg.cholesky_ex(regularized)
    if torch.any(info):
        raise RuntimeError(f"Activation covariance Cholesky failed with info={info.max().item()}")

    # H = E[(W x - Q(W) q(x)) x.T]. If C = L L.T, then H L^-T is
    # the whitened unconstrained regression. Its rank-r SVD is the exact
    # reduced-rank solution under the regularized activation metric.
    weight = weight.to(dtype=solve_dtype)
    quantized_weight = quantized_weight.to(dtype=solve_dtype)
    cross_output_input = weight @ covariance - quantized_weight @ quantized_cross
    whitened = torch.linalg.solve_triangular(chol, cross_output_input.mT, upper=False).mT
    branch = LowRankBranch(
        in_features,
        weight.shape[0],
        rank=rank,
        weight=whitened,
        svd_mode=svd_mode,
        svd_oversample=svd_oversample,
        svd_niter=svd_niter,
    )
    # branch.a currently contains Vh for the whitened solution. Convert
    # Vh @ L^-1 back to the original activation coordinates; branch.b is
    # already U @ S.
    original_dtype = branch.a.weight.dtype
    transformed_a = torch.linalg.solve_triangular(
        chol.mT,
        branch.a.weight.to(dtype=solve_dtype).mT,
        upper=True,
    ).mT
    branch.a.weight.copy_(transformed_a.to(dtype=original_dtype))
    branch.to(device=device, dtype=output_dtype)
    return branch


class QuantLowRankCalibrator(SearchBasedCalibrator[QuantLowRankCalibConfig, LowRankBranch]):
    """The quantization low-rank branch calibrator."""

    def __init__(
        self,
        config: QuantLowRankCalibConfig,
        w_quantizer: Quantizer,
        x_quantizer: Quantizer | None,
        develop_dtype: torch.dtype = torch.float32,
    ) -> None:
        """Initialize the calibrator.

        Args:
            config (`QuantLowRankCalibConfig`):
                The configuration of the quantization low-rank branch calibrator.
            w_quantizer (`Quantizer`):
                The quantizer for weights.
            x_quantizer (`Quantizer` or `None`):
                The quantizer for inputs.
            develop_dtype (`torch.dtype`, *optional*, defaults to `torch.float32`):
                The development data type.
        """
        if isinstance(config, KeyEnableConfig):
            assert config.is_enabled_for(w_quantizer.key), "The calibrator should be enabled for the quantizer."
        else:
            assert config.is_enabled(), "The calibrator should be enabled."
        super().__init__(
            tensor_type=TensorType.Weights,
            config=config,
            w_quantizer=w_quantizer,
            x_quantizer=x_quantizer,
            y_quantizer=None,
            develop_dtype=develop_dtype,
        )
        assert self.needs_quant, "The tensor should be quantized."
        self.num_iters = config.num_iters

    @property
    def population_size(self) -> int:
        """Return the population size of the current iteration."""
        return 1

    @property
    def allows_x_quant_for_wgts(self) -> bool:
        """Whether the calibrator allows input quantization when tensor_type is Weights."""
        return True

    @property
    def allows_w_quant_for_wgts(self) -> bool:
        """Whether the calibrator needs weight quantization when tensor_type is Weights."""
        return True

    def is_done(self) -> bool:
        """Check if the calibration is done."""
        return self.iter >= self.num_iters or self.early_stopped

    def is_last_iter(self) -> bool:
        """Check if the current iteration is the last one."""
        return self.iter == self.num_iters - 1

    def _reset(
        self,
        x_wgts: list[torch.Tensor | nn.Parameter],
        x_acts: TensorsCache | None = None,
        **kwargs,
    ) -> None:  # noqa: C901
        """Reset the calibrator.

        Args:
            x_wgts (`list[torch.Tensor | nn.Parameter]`):
                The weights in x-w computation.
        """
        self.best_branch: LowRankBranch = None
        self.best_error: torch.Tensor = None
        self.error_history: list[tuple[float, float]] = []
        self.early_stopped = False
        if len(x_wgts) > 1 and not self.config.exclusive:
            self.w = torch.cat([wgt.data for wgt in x_wgts], dim=0)
        else:
            assert len(x_wgts) == 1
            self.w = x_wgts[0].data
        self.hat_ws: list[torch.Tensor] = [None] * len(x_wgts)
        self.ocs: list[int] = [wgt.shape[0] for wgt in x_wgts]
        self.activation_cache: TensorCache | None = None
        self.activation_covariance: torch.Tensor | None = None
        self.quantized_cross: torch.Tensor | None = None
        if self.config.activation_aware:
            if not isinstance(x_acts, TensorsCache) or x_acts.num_tensors != 1:
                raise ValueError("Activation-aware low-rank calibration requires exactly one input activation cache")
            self.activation_cache = x_acts.front()
        if self.config.compensate:
            self.qw = torch.cat(
                [
                    self.w_quantizer.quantize(wgt.data, kernel=None, develop_dtype=self.develop_dtype).data
                    for wgt in x_wgts
                ],
                dim=0,
            )
        else:
            self.qw = 0
            if self.config.activation_aware:
                # Preserve SVDQuant's weight-SVD initialization. The first
                # activation-aware iteration improves this initialized W4
                # residual rather than quantizing the original W directly.
                initial = LowRankBranch(
                    self.w.shape[1],
                    self.w.shape[0],
                    rank=self.config.rank,
                    weight=self.w,
                    svd_mode=self.config.svd_mode,
                    svd_oversample=self.config.svd_oversample,
                    svd_niter=self.config.svd_niter,
                )
                self._update_quantized_weights(initial)

    def _update_quantized_weights(self, branch: LowRankBranch) -> None:
        """Quantize the W4 residual represented alongside ``branch``."""

        lw = branch.get_effective_weight().view(self.w.shape)
        rw = self.w - lw
        if len(self.hat_ws) > 1:
            oc_idx = 0
            for idx, oc in enumerate(self.ocs):
                self.hat_ws[idx] = self.w_quantizer.quantize(
                    rw[oc_idx : oc_idx + oc], kernel=None, develop_dtype=self.develop_dtype
                ).data
                oc_idx += oc
            self.qw = torch.cat(self.hat_ws, dim=0)
            if self.objective != SearchBasedCalibObjective.OutputsError:
                oc_idx = 0
                for idx, oc in enumerate(self.ocs):
                    self.hat_ws[idx].add_(lw[oc_idx : oc_idx + oc])
                    oc_idx += oc
        else:
            self.qw = self.w_quantizer.quantize(rw, kernel=None, develop_dtype=self.develop_dtype).data
            self.hat_ws = [self.qw if self.objective == SearchBasedCalibObjective.OutputsError else self.qw + lw]

    def _reshape_activation(self, cache: TensorCache, tensor: torch.Tensor) -> torch.Tensor:
        channels_dim = cache.channels_dim % tensor.ndim
        tensor = tensor.view(-1, *tensor.shape[channels_dim:])
        tensor = cache.reshape(tensor)
        if tensor.ndim != 2 or tensor.shape[1] != self.w.shape[1]:
            raise ValueError(
                f"Activation-aware input expected a 2D matrix with {self.w.shape[1]} columns, got {tuple(tensor.shape)}"
            )
        return tensor

    def _get_activation_statistics(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Accumulate E[x.T x] and E[q(x).T x] once for this group."""

        if self.activation_covariance is not None and self.quantized_cross is not None:
            return self.activation_covariance, self.quantized_cross
        if self.activation_cache is None:
            raise RuntimeError("Activation-aware calibration has no activation cache")
        device = self.w.device
        stats_dtype = self.develop_dtype if self.develop_dtype in {torch.float32, torch.float64} else torch.float32
        in_features = self.w.shape[1]
        covariance = torch.zeros((in_features, in_features), device=device, dtype=stats_dtype)
        quantized_cross = torch.zeros_like(covariance)
        num_rows = 0
        for tensor in self.activation_cache.data:
            tensor = tensor.to(device=device, non_blocking=True)
            quantized = self._process_x_in_xw(tensor, channels_dim=self.activation_cache.channels_dim)
            tensor = self._reshape_activation(self.activation_cache, tensor)
            quantized = self._reshape_activation(self.activation_cache, quantized)
            max_tokens = self.config.activation_num_tokens
            if max_tokens > 0 and tensor.shape[0] > max_tokens:
                indexes = torch.div(
                    torch.arange(max_tokens, device=device) * tensor.shape[0],
                    max_tokens,
                    rounding_mode="floor",
                )
                tensor = tensor.index_select(0, indexes)
                quantized = quantized.index_select(0, indexes)
            tensor = tensor.to(dtype=stats_dtype)
            quantized = quantized.to(dtype=stats_dtype)
            covariance.addmm_(tensor.mT, tensor)
            quantized_cross.addmm_(quantized.mT, tensor)
            num_rows += tensor.shape[0]
        if num_rows == 0:
            raise RuntimeError("Activation-aware calibration selected no activation rows")
        count = torch.tensor(float(num_rows), device=device, dtype=stats_dtype)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(covariance)
            torch.distributed.all_reduce(quantized_cross)
            torch.distributed.all_reduce(count)
        covariance.div_(count)
        quantized_cross.div_(count)
        self.logger.debug("  - activation rows = %d", int(count.item()))
        self.activation_covariance = covariance
        self.quantized_cross = quantized_cross
        return covariance, quantized_cross

    def get_best(self) -> LowRankBranch:
        """Get the best candidate.

        Returns:
            `LowRankBranch`:
                The best candidate.
        """
        return self.best_branch

    def _ask(self) -> LowRankBranch:
        """Ask for the next candidate.

        Returns:
            `LowRankBranch`:
                The next candidate.
        """
        if self.config.activation_aware:
            covariance, quantized_cross = self._get_activation_statistics()
            branch = solve_activation_aware_low_rank(
                self.w,
                self.qw,
                covariance,
                quantized_cross,
                rank=self.config.rank,
                damping=self.config.activation_damping,
                svd_mode=self.config.svd_mode,
                svd_oversample=self.config.svd_oversample,
                svd_niter=self.config.svd_niter,
            )
        else:
            branch = LowRankBranch(
                self.w.shape[1],
                self.w.shape[0],
                rank=self.config.rank,
                weight=self.w - self.qw,
                svd_mode=self.config.svd_mode,
                svd_oversample=self.config.svd_oversample,
                svd_niter=self.config.svd_niter,
            )
        self.wgt_idx = 0
        self._update_quantized_weights(branch)
        return branch

    def _tell(self, error: list[torch.Tensor]) -> None:  # noqa: C901
        """Tell the error of the last candidate and update the best candidate.

        Args:
            errors (list[torch.Tensor]): The error of the last candidate.
        """
        if len(error) > 1:
            error = [sum(error)]
        error = error[0]
        assert isinstance(error, torch.Tensor)
        assert error.numel() == 1, "The error should only have one value."
        if self.best_error is None or error <= self.best_error:
            self.best_error = error
            self.best_branch = self.candidate
        elif self.config.early_stop:
            self.early_stopped = True
        if self.logger.level <= tools.logging.DEBUG:
            self.error_history.append(
                (
                    math.root_(error.to(torch.float64), self.config.degree).item(),
                    math.root_(self.best_error.to(torch.float64), self.config.degree).item(),
                )
            )
            if self.iter % 10 == 9 or self.is_last_iter() or self.early_stopped:
                iter_end = ((self.iter + 10) // 10) * 10
                iter_start = iter_end - 10
                iter_end = min(iter_end, self.iter + 1)
                history = self.error_history[iter_start:iter_end]
                self.logger.debug("  -      iter  = [%s]", ", ".join(f"{i:10d}" for i in range(iter_start, iter_end)))
                self.logger.debug("  -      error = [%s]", ", ".join(f"{e[0]:10.4f}" for e in history))
                self.logger.debug("  - best error = [%s]", ", ".join(f"{e[1]:10.4f}" for e in history))

    def _process_x_in_xw(self, x: torch.Tensor, channels_dim: int | _MISSING_TYPE = MISSING) -> torch.Tensor:
        if not self.needs_x_quant_for_wgts:
            return x
        return self.x_quantizer.quantize(x, channels_dim=channels_dim, develop_dtype=self.develop_dtype).data

    def _process_w_in_xw(self, w: torch.Tensor) -> torch.Tensor:
        hat_w = self.hat_ws[self.wgt_idx]
        self.hat_ws[self.wgt_idx] = None
        self.wgt_idx += 1
        return hat_w if self.needs_w_quant_for_wgts else w

    def _process_y_in_yx(self, y: torch.Tensor, channels_dim: int | _MISSING_TYPE = MISSING) -> torch.Tensor:
        raise RuntimeError("_process_y_in_yx should not be called in QuantSVDCalibrator.")

    def _process_x_in_yx(self, x: torch.Tensor, channels_dim: int | _MISSING_TYPE = MISSING) -> torch.Tensor:
        raise RuntimeError("_process_x_in_yx should not be called in QuantSVDCalibrator.")

    def _process_xw_in_yx(self, w: torch.Tensor) -> torch.Tensor:
        raise RuntimeError("_process_xw_in_yx should not be called in QuantSVDCalibrator.")

    def _process_yw_in_yx(self, w: torch.Tensor) -> torch.Tensor:
        raise RuntimeError("_process_yw_in_yx should not be called in QuantSVDCalibrator.")

    def _process_wgts_centric_mod(
        self, wgts: list[nn.Parameter], mods: list[nn.Module], update_state_dict: bool = True, **kwargs
    ) -> None:
        assert len(self.hat_ws) == len(wgts) == len(mods)
        shared = self.candidate
        if len(self.hat_ws) > 1:
            oc_idx = 0
            for mod, wgt, hat_w in zip(mods, wgts, self.hat_ws, strict=True):
                if update_state_dict:
                    self._state_dict.append((wgt, wgt.data))
                wgt.data = hat_w
                branch = LowRankBranch(wgt.shape[1], wgt.shape[0], rank=self.config.rank)
                branch.a = shared.a
                branch.b.to(dtype=wgt.dtype, device=wgt.device)
                branch.b.weight.copy_(shared.b.weight[oc_idx : oc_idx + wgt.data.shape[0]])
                oc_idx += wgt.data.shape[0]
                self._hooks.append(branch.as_hook().register(mod))
        else:
            if update_state_dict:
                self._state_dict.append((wgts[0], wgts[0].data))
            wgts[0].data = self.hat_ws[0]
            self._hooks.append(shared.as_hook().register(mods))
        if self.needs_x_quant_for_wgts:
            self._hooks.append(self.x_quantizer.as_hook().register(mods))
        self.hat_ws = [None] * len(self.hat_ws)
