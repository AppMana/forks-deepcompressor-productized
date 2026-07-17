from types import MethodType, SimpleNamespace

import torch

from deepcompressor.calib.lowrank import (
    QuantLowRankCalibrator,
    _estimate_largest_eigenvalue,
    solve_activation_aware_low_rank,
)
from deepcompressor.nn.patch.lowrank import LowRankBranch


def test_activation_aware_solver_matches_reduced_rank_regression() -> None:
    torch.manual_seed(7)
    num_rows, in_features, out_features, rank = 2048, 12, 9, 3
    inputs = torch.randn(num_rows, in_features) * torch.logspace(-1, 1, in_features)
    quantized_inputs = torch.round(inputs * 3) / 3
    weight = torch.randn(out_features, in_features)
    quantized_weight = torch.round(weight * 2) / 2
    covariance = inputs.mT @ inputs / num_rows
    quantized_cross = quantized_inputs.mT @ inputs / num_rows
    damping = 1e-3

    branch = solve_activation_aware_low_rank(
        weight,
        quantized_weight,
        covariance,
        quantized_cross,
        rank=rank,
        damping=damping,
        svd_mode="exact",
        svd_oversample=8,
        svd_niter=2,
    )

    regularized = (covariance + covariance.mT) * 0.5
    regularized += torch.eye(in_features) * (damping * _estimate_largest_eigenvalue(covariance))
    chol = torch.linalg.cholesky(regularized)
    cross_output_input = weight @ covariance - quantized_weight @ quantized_cross
    whitened = torch.linalg.solve_triangular(chol, cross_output_input.mT, upper=False).mT
    u, s, vh = torch.linalg.svd(whitened.double(), full_matrices=False)
    truncated = (u[:, :rank] * s[:rank]) @ vh[:rank]
    expected = torch.linalg.solve_triangular(chol.double().mT, truncated.mT, upper=True).mT.float()

    assert torch.allclose(branch.get_effective_weight(), expected, atol=3e-5, rtol=3e-5)


def test_activation_aware_solver_reduces_observed_w4a4_error() -> None:
    torch.manual_seed(11)
    num_rows, in_features, out_features, rank = 4096, 16, 10, 2
    inputs = torch.randn(num_rows, in_features) * torch.logspace(-2, 2, in_features)
    quantized_inputs = torch.round(inputs * 2) / 2
    weight = torch.randn(out_features, in_features)
    quantized_weight = torch.round(weight * 2) / 2
    covariance = inputs.mT @ inputs / num_rows
    quantized_cross = quantized_inputs.mT @ inputs / num_rows

    aware = solve_activation_aware_low_rank(
        weight,
        quantized_weight,
        covariance,
        quantized_cross,
        rank=rank,
        damping=1e-4,
        svd_mode="exact",
        svd_oversample=8,
        svd_niter=2,
    ).get_effective_weight()
    weight_only = LowRankBranch(
        in_features,
        out_features,
        rank=rank,
        weight=weight - quantized_weight,
    ).get_effective_weight()
    target = inputs @ weight.mT - quantized_inputs @ quantized_weight.mT
    aware_error = (target - inputs @ aware.mT).square().mean()
    weight_only_error = (target - inputs @ weight_only.mT).square().mean()

    assert aware_error < weight_only_error


def test_activation_aware_solver_adapts_damping_for_numerically_indefinite_covariance() -> None:
    torch.manual_seed(19)
    in_features, out_features, rank = 12, 8, 3
    weight = torch.randn(out_features, in_features)
    quantized_weight = torch.round(weight * 2) / 2
    covariance = torch.eye(in_features)
    # Model FP32 roundoff in a rank-deficient empirical covariance. The
    # requested 1e-4 damping is too small for this final pivot, so a fixed
    # regularizer fails while the adaptive factorization remains well posed.
    covariance[-1, -1] = -5e-4
    quantized_cross = covariance.clone()

    branch = solve_activation_aware_low_rank(
        weight,
        quantized_weight,
        covariance,
        quantized_cross,
        rank=rank,
        damping=1e-4,
        svd_mode="exact",
        svd_oversample=8,
        svd_niter=2,
    )

    assert torch.isfinite(branch.get_effective_weight()).all()


def test_activation_aware_solver_regularizes_rank_deficient_weak_directions() -> None:
    torch.manual_seed(23)
    in_features, out_features, rank = 32, 12, 4
    basis = torch.randn(in_features, 6)
    covariance = basis @ basis.mT
    covariance /= covariance.diagonal().mean()
    weight = torch.randn(out_features, in_features)
    quantized_weight = torch.round(weight * 2) / 2
    quantized_cross = covariance.clone()

    branch = solve_activation_aware_low_rank(
        weight,
        quantized_weight,
        covariance,
        quantized_cross,
        rank=rank,
        damping=1e-3,
        svd_mode="exact",
        svd_oversample=8,
        svd_niter=2,
    )

    effective = branch.get_effective_weight()
    assert torch.isfinite(effective).all()
    assert torch.linalg.matrix_norm(effective) < 2 * torch.linalg.matrix_norm(weight - quantized_weight)


def test_activation_aware_iteration_zero_scores_weight_svd_baseline() -> None:
    calibrator = object.__new__(QuantLowRankCalibrator)
    calibrator.config = SimpleNamespace(activation_aware=True)
    calibrator.iter = 0
    calibrator.wgt_idx = 99
    calibrator.initial_branch = object()
    updated = []

    def update_quantized_weights(self, branch) -> None:
        updated.append(branch)

    def fail_if_statistics_are_requested(self):
        raise AssertionError("candidate zero must not request activation statistics")

    calibrator._update_quantized_weights = MethodType(update_quantized_weights, calibrator)
    calibrator._get_activation_statistics = MethodType(fail_if_statistics_are_requested, calibrator)

    branch = calibrator._ask()

    assert branch is updated[0]
    assert calibrator.initial_branch is None
    assert calibrator.wgt_idx == 0
