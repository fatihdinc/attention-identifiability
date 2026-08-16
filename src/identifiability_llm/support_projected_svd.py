"""Post-hoc SVD baseline restricted to task-supported input subspaces."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .ten_task_attention import D_MODEL
from .ten_task_effective_score import EffectiveScoreBasis


@dataclass(frozen=True)
class SupportProjectedSVD:
    """Compact SVD of M restricted to the observed query/key input supports."""

    matrix: torch.Tensor
    query_vectors: torch.Tensor
    key_vectors: torch.Tensor
    core_u: torch.Tensor
    singular_values: torch.Tensor
    core_vh: torch.Tensor
    query_support_rank: int
    key_support_rank: int
    query_eigenvalue_threshold: float
    key_eigenvalue_threshold: float
    relative_eigenvalue_tolerance: float


def _non_null_input_basis(
    values: torch.Tensor,
    vectors: torch.Tensor,
    *,
    relative_tolerance: float,
) -> tuple[torch.Tensor, int, float]:
    if not 0.0 < relative_tolerance < 1.0:
        raise ValueError("relative_tolerance must be strictly between zero and one")
    values = values.detach().cpu().to(torch.float64)
    vectors = vectors.detach().cpu().to(torch.float64)
    if values.ndim != 1 or vectors.shape != (D_MODEL, D_MODEL):
        raise ValueError("Expected 128 eigenvalues and a 128 x 128 eigenvector matrix")
    maximum = float(values.max().item())
    if maximum <= 0.0:
        raise ValueError("Input Gram has no positive eigenvalues")
    threshold = relative_tolerance * maximum
    rank = int((values > threshold).sum().item())
    if rank == 0:
        raise ValueError("Input Gram support is empty at the requested tolerance")
    return vectors[:, :rank], rank, threshold


def build_support_projected_svd(
    basis: EffectiveScoreBasis,
    *,
    relative_eigenvalue_tolerance: float = 1e-10,
) -> SupportProjectedSVD:
    """Factorize M between the non-null query-input and key-input supports.

    With U_q and U_x spanning those supports, only
    U_q U_q^T M U_x U_x^T can affect task scores.  Its nonzero singular values
    equal those of the compact core U_q^T M U_x, which is factorized here.
    """

    query_vectors, query_rank, query_threshold = _non_null_input_basis(
        basis.query_input_values,
        basis.query_input_vectors,
        relative_tolerance=relative_eigenvalue_tolerance,
    )
    key_vectors, key_rank, key_threshold = _non_null_input_basis(
        basis.key_input_values,
        basis.key_input_vectors,
        relative_tolerance=relative_eigenvalue_tolerance,
    )
    matrix = basis.matrix.detach().cpu().to(torch.float64)
    core = query_vectors.T @ matrix @ key_vectors
    core_u, singular_values, core_vh = torch.linalg.svd(core, full_matrices=False)
    return SupportProjectedSVD(
        matrix=matrix,
        query_vectors=query_vectors,
        key_vectors=key_vectors,
        core_u=core_u,
        singular_values=singular_values,
        core_vh=core_vh,
        query_support_rank=query_rank,
        key_support_rank=key_rank,
        query_eigenvalue_threshold=query_threshold,
        key_eigenvalue_threshold=key_threshold,
        relative_eigenvalue_tolerance=relative_eigenvalue_tolerance,
    )


def reconstruct_support_projected_svd(
    basis: SupportProjectedSVD,
    rank: int,
) -> torch.Tensor:
    """Return the rank-K SVD reconstruction of the task-supported matrix."""

    if rank < 0 or rank > D_MODEL:
        raise ValueError(f"rank must be in [0, {D_MODEL}]")
    retained = min(rank, int(basis.singular_values.numel()))
    if retained == 0:
        return torch.zeros_like(basis.matrix)
    compact = (
        basis.core_u[:, :retained] * basis.singular_values[:retained]
    ) @ basis.core_vh[:retained]
    return basis.query_vectors @ compact @ basis.key_vectors.T


def support_projected_svd_power(basis: SupportProjectedSVD, rank: int) -> float:
    """Cumulative spectral power of the same supported matrix being truncated."""

    if rank < 0 or rank > D_MODEL:
        raise ValueError(f"rank must be in [0, {D_MODEL}]")
    if rank == 0:
        return 0.0
    power = basis.singular_values.square()
    retained = min(rank, int(power.numel()))
    return float((power[:retained].sum() / power.sum()).item())
