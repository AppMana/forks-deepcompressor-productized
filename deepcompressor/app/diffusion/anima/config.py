"""Canonical, code-level Anima SVDQuant recipe."""

from __future__ import annotations

from pathlib import Path

import torch

from deepcompressor.app.diffusion.dataset.calib import DiffusionCalibCacheLoaderConfig
from deepcompressor.app.diffusion.quant.config import DiffusionQuantConfig
from deepcompressor.app.diffusion.quant.quantizer.config import (
    DiffusionActivationQuantizerConfig,
    DiffusionWeightQuantizerConfig,
)
from deepcompressor.calib.config import (
    SearchBasedCalibGranularity,
    SearchBasedCalibObjective,
    SearchBasedCalibStrategy,
    SkipBasedDynamicRangeCalibConfig,
    SkipBasedQuantLowRankCalibConfig,
    SkipBasedSmoothCalibConfig,
    SmoothSpanMode,
    SmoothTransfomerConfig,
)
from deepcompressor.data.dtype import QuantDataType

from .struct import AnimaModelStruct

__all__ = ["build_anima_svdquant_config"]


def build_anima_svdquant_config(
    calibration_path: str | Path,
    *,
    rank: int = 32,
    num_samples: int = 100,
    num_iters: int = 100,
    num_workers: int = 0,
    fast: bool = False,
) -> DiffusionQuantConfig:
    """Build the paper-faithful W4A4 + BF16 low-rank PTQ configuration.

    This is deliberately constructed in Python rather than assembled from a
    loose stack of YAML overrides. Rank 32 and rank 128 therefore differ in
    exactly one parameter.
    """

    if rank not in {32, 128}:
        raise ValueError(f"Supported SVDQuant ranks are 32 and 128, got {rank}")
    DiffusionQuantConfig.set_key_map(AnimaModelStruct._get_default_key_map())
    int4 = QuantDataType.from_str("sint4")
    sample_batch_size = 1 if fast else 4
    low_rank = SkipBasedQuantLowRankCalibConfig(
        rank=rank,
        exclusive=False,
        compensate=False,
        early_stop=True,
        degree=2,
        objective=SearchBasedCalibObjective.OutputsError,
        strategy=SearchBasedCalibStrategy.Manual,
        sample_batch_size=sample_batch_size,
        sample_size=-1,
        outputs_device="cpu",
        # The randomized pilot keeps the same iterative residual objective as
        # the exact recipe; only the truncated-SVD solver and smoothing search
        # are cheaper. This makes 1/2/4/... iteration sweeps meaningful.
        num_iters=num_iters,
        svd_mode="randomized" if fast else "exact",
        svd_oversample=8,
        svd_niter=2,
        skips=[],
    )
    weight_range = SkipBasedDynamicRangeCalibConfig(
        degree=2,
        objective=SearchBasedCalibObjective.OutputsError,
        strategy=SearchBasedCalibStrategy.Manual,
        granularity=SearchBasedCalibGranularity.Layer,
        sample_batch_size=sample_batch_size,
        sample_size=-1,
        ratio=1.0,
        skips=[],
    )
    weights = DiffusionWeightQuantizerConfig(
        dtype=int4,
        group_shapes=((1, 64, 1, 1, 1),),
        scale_dtypes=(None,),
        skips=[],
        low_rank=low_rank,
        calib_range=weight_range,
    )
    inputs = DiffusionActivationQuantizerConfig(
        dtype=int4,
        group_shapes=((1, 64, 1, 1, 1),),
        scale_dtypes=(None,),
        static=False,
        allow_unsigned=True,
        skips=[],
        calib_range=None,
    )
    outputs = DiffusionActivationQuantizerConfig(dtype=None, skips=[])
    smooth = SmoothTransfomerConfig(
        proj=SkipBasedSmoothCalibConfig(
            objective=SearchBasedCalibObjective.OutputsError,
            # Manual still computes activation/weight spans and applies the
            # SVDQuant smoothing transform, but evaluates one candidate rather
            # than replaying the block for every grid point.
            strategy=(SearchBasedCalibStrategy.Manual if fast else SearchBasedCalibStrategy.GridSearch),
            granularity=SearchBasedCalibGranularity.Layer,
            spans=[(SmoothSpanMode.AbsMax, SmoothSpanMode.AbsMax)],
            alpha=0.5,
            beta=-1 if fast else -2,
            num_grids=1 if fast else 20,
            allow_low_rank=not fast,
            fuse_when_possible=False,
            sample_batch_size=sample_batch_size,
            sample_size=-1,
            outputs_device="cpu",
            skips=[],
        )
    )
    calibration = DiffusionCalibCacheLoaderConfig(
        data="anima-aesthetic-v1.1",
        path=str(Path(calibration_path).expanduser().resolve()),
        num_samples=num_samples,
        batch_size=sample_batch_size,
        num_workers=num_workers,
    )
    return DiffusionQuantConfig(
        wgts=weights,
        ipts=inputs,
        opts=outputs,
        calib=calibration,
        rotation=None,
        smooth=smooth,
        develop_dtype=torch.float32,
    )
