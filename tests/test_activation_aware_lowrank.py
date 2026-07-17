import torch

from deepcompressor.calib.lowrank import solve_activation_aware_low_rank
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
    regularized += torch.eye(in_features) * (damping * covariance.diagonal().mean().abs())
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
