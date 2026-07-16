# -*- coding: utf-8 -*-

import torch
import torch.linalg
import torch.nn as nn

from ...utils.hooks import AccumBranchHook, BaseInputPackager, BaseOutputPackager

__all__ = ["LowRankBranch"]


class LowRankBranch(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int,
        alpha: float = 1.0,
        weight: torch.Tensor | None = None,
        svd_mode: str = "exact",
        svd_oversample: int = 8,
        svd_niter: int = 2,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        self.svd_mode = svd_mode
        self.svd_oversample = svd_oversample
        self.svd_niter = svd_niter
        if rank == 0:
            self.a, self.b = None, None
        elif rank < 0:
            self.a, self.b = nn.Linear(in_features, out_features, bias=False), nn.Identity()
        else:
            self.a, self.b = nn.Linear(in_features, rank, bias=False), nn.Linear(rank, out_features, bias=False)
        self.reset_parameters(weight)

    @torch.no_grad()
    def reset_parameters(self, weight: torch.Tensor | None = None) -> None:
        if weight is None:
            if self.rank < 0:
                nn.init.zeros_(self.a.weight)
            elif self.rank > 0:
                nn.init.kaiming_uniform_(self.a.weight)
                nn.init.zeros_(self.b.weight)
            return
        if weight.ndim >= 2:
            assert weight.shape[2:].numel() == 1, "LinearLoRAHook only supports 2D input tensor"
        weight = weight.view(weight.shape[0], -1)
        device, dtype = weight.device, weight.dtype
        self.to(device=device, dtype=dtype)
        out_features, in_features = weight.shape
        assert self.in_features == in_features, "Input features size mismatch"
        assert self.out_features == out_features, "Output features size mismatch"
        if self.rank < 0:
            self.a.weight.data.copy_(weight)
        elif self.rank > 0:
            if self.svd_mode == "exact":
                # Only the leading ``rank`` singular vectors are consumed.
                # Economy SVD is exact for them and avoids unused square
                # matrices for tall and wide diffusion projections.
                u, s, vh = torch.linalg.svd(weight.double(), full_matrices=False)
            elif self.svd_mode == "randomized":
                q = min(min(weight.shape), self.rank + self.svd_oversample)
                devices = [weight.device.index] if weight.is_cuda else []
                with torch.random.fork_rng(devices=devices):
                    torch.manual_seed(0)
                    u, s, v = torch.svd_lowrank(weight.double(), q=q, niter=self.svd_niter)
                vh = v.mH
            else:
                raise ValueError(f"Unsupported SVD mode: {self.svd_mode}")
            # tensor: [oc, ic], u: [oc, oc], s: [oc], vh: [ic, ic]
            # us: [oc, rank], vh: [rank, ic]
            us = u[:, : self.rank] * s[: self.rank]
            vh = vh[: self.rank]
            assert not us.isnan().any(), "NaN in U * S"
            assert not vh.isnan().any(), "NaN in V^T"
            assert not us.isinf().any(), "Inf in U * S"
            assert not vh.isinf().any(), "Inf in V^T"
            self.a.weight.data.copy_(vh.to(dtype))
            self.b.weight.data.copy_(us.to(dtype))

    def get_effective_weight(self) -> torch.Tensor | None:
        if self.rank == 0:
            return None
        elif self.rank < 0:
            return self.a.weight
        else:
            return self.b.weight @ self.a.weight

    def forward(self, input: torch.Tensor) -> torch.Tensor | None:
        if self.a is None:
            return None
        if input.shape[-1] == self.in_features:
            # nn.Linear natively preserves any number of leading dimensions.
            # This includes Anima's [B, T, H, W, C] MLP activations.
            return self.alpha * self.b(self.a(input))
        if input.ndim >= 3 and input.shape[1] == self.in_features:
            # Channel-first convolution activations: [B, C, ...] ->
            # [B, ..., C], apply the linear branch, then restore C to dim 1.
            input = input.movedim(1, -1)
            return (self.alpha * self.b(self.a(input))).movedim(-1, 1)
        raise ValueError(
            f"Low-rank branch expected {self.in_features} input channels in the last or second dimension, "
            f"got shape {tuple(input.shape)}"
        )

    def as_hook(
        self,
        input_packager: BaseInputPackager | None = None,
        output_packager: BaseOutputPackager | None = None,
    ) -> AccumBranchHook:
        """Wrap the module as a branch hook.

        Args:
            input_packager (`BaseInputPackager` or `None`, *optional*, defaults to `None`):
                Input packager.
            output_packager (`BaseOutputPackager` or `None`, *optional*, defaults to `None`):
                Output packager.
        Returns:
            `AccumBranchHook`:
                The branch hook.
        """
        return AccumBranchHook(self, input_packager=input_packager, output_packager=output_packager)
