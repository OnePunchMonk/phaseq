"""Gradient statistics utilities for PhaseQ.

Provides lightweight functions for computing the gradient topology metrics
that drive the phase detector:

* **Stable rank** – ``||G||_F^2 / ||G||_2^2`` – a continuous relaxation of
  matrix rank that indicates how "spread out" the spectrum is.
* **Grassmannian geodesic distance** – the principal-angle-based distance
  between two *k*-dimensional subspaces, used to measure subspace rotation.
* **Incremental QR factorisation** – streaming rank-revealing QR for
  efficient subspace tracking without full SVD.
* **EMA update** – simple exponential moving average helper.
* **Logging helpers** – optional W&B gradient logging utilities.
"""

from __future__ import annotations

import logging
import math
from typing import Any

import torch

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Stable rank
# ---------------------------------------------------------------------------


@torch.no_grad()
def compute_stable_rank(G: torch.Tensor) -> float:
    """Compute the stable rank of a matrix.

    The stable rank is defined as ``||G||_F^2 / ||G||_2^2`` where
    ``||·||_F`` is the Frobenius norm and ``||·||_2`` is the spectral
    (operator) norm.  Unlike the algebraic rank, stable rank is a smooth
    function of the matrix entries and is robust to small perturbations.

    A high stable rank indicates that many singular values contribute
    significantly (gradient structure is complex / "bursty"), while a low
    stable rank means the gradient is well-approximated by a low-rank matrix.

    Args:
        G: 2-D gradient tensor of shape ``(m, n)``.

    Returns:
        The stable rank as a Python float.  Returns 1.0 if the spectral
        norm is zero (degenerate case).
    """
    if G.ndim == 1:
        return 1.0
    if G.ndim > 2:
        G = G.reshape(G.shape[0], -1)

    fro_sq = torch.sum(G * G).item()
    # Power iteration for the leading singular value (cheap approx of ||G||_2)
    spectral_norm_sq = _power_iteration_sigma_sq(G, n_iters=10)
    if spectral_norm_sq < 1e-12:
        return 1.0
    return fro_sq / spectral_norm_sq


def _power_iteration_sigma_sq(
    A: torch.Tensor, n_iters: int = 10
) -> float:
    """Estimate ``sigma_max(A)^2`` via power iteration on ``A^T A``.

    Args:
        A: 2-D tensor.
        n_iters: Number of power-iteration steps.

    Returns:
        Approximate squared spectral norm.
    """
    m, n = A.shape
    # Start with a random unit vector in the smaller dimension
    if m >= n:
        v = torch.randn(n, 1, device=A.device, dtype=A.dtype)
        v = v / (torch.norm(v) + 1e-12)
        for _ in range(n_iters):
            u = A @ v  # (m, 1)
            v = A.t() @ u  # (n, 1)
            v_norm = torch.norm(v)
            if v_norm < 1e-12:
                return 0.0
            v = v / v_norm
        Av = A @ v
        return torch.sum(Av * Av).item()
    else:
        v = torch.randn(m, 1, device=A.device, dtype=A.dtype)
        v = v / (torch.norm(v) + 1e-12)
        for _ in range(n_iters):
            u = A.t() @ v  # (n, 1)
            v = A @ u  # (m, 1)
            v_norm = torch.norm(v)
            if v_norm < 1e-12:
                return 0.0
            v = v / v_norm
        Atv = A.t() @ v
        return torch.sum(Atv * Atv).item()


# ---------------------------------------------------------------------------
# Grassmannian geodesic distance
# ---------------------------------------------------------------------------


@torch.no_grad()
def grassmannian_distance(U1: torch.Tensor, U2: torch.Tensor) -> float:
    """Compute the geodesic distance on the Grassmannian manifold.

    Given two orthonormal bases ``U1`` and ``U2`` (each of shape ``(d, k)``),
    the Grassmannian distance is:

        ``d_G(U1, U2) = ||Θ||_2``

    where ``Θ = (θ_1, …, θ_k)`` are the *principal angles* between the two
    *k*-dimensional subspaces.  The principal angles are computed from the
    singular values of ``U1^T U2``:

        ``cos(θ_i) = σ_i(U1^T U2)``

    This distance is zero when the subspaces are identical and reaches
    ``sqrt(k) · π/2`` when they are maximally separated.

    Args:
        U1: Orthonormal basis of shape ``(d, k)``.
        U2: Orthonormal basis of shape ``(d, k)``.

    Returns:
        The Grassmannian geodesic distance (scalar).
    """
    if U1.shape != U2.shape:
        raise ValueError(
            f"Subspace bases must have the same shape, got {U1.shape} and {U2.shape}"
        )

    # Compute cosines of principal angles
    M = U1.t() @ U2  # (k, k)
    # Clamp singular values to [0, 1] for numerical safety
    sigmas = torch.linalg.svdvals(M.float()).clamp(0.0, 1.0)
    # Principal angles
    angles = torch.acos(sigmas)
    return torch.norm(angles).item()


@torch.no_grad()
def grassmannian_chordal_distance(U1: torch.Tensor, U2: torch.Tensor) -> float:
    """Compute the chordal distance on the Grassmannian.

    An alternative metric that is cheaper to compute (avoids ``acos``):

        ``d_c(U1, U2) = ||U1 U1^T - U2 U2^T||_F / sqrt(2)``

    This is equivalent to ``sqrt(sum(sin^2(θ_i)))``.

    Args:
        U1: Orthonormal basis ``(d, k)``.
        U2: Orthonormal basis ``(d, k)``.

    Returns:
        Chordal distance (scalar).
    """
    if U1.shape != U2.shape:
        raise ValueError(
            f"Subspace bases must have the same shape, got {U1.shape} and {U2.shape}"
        )
    M = U1.t() @ U2
    sigmas = torch.linalg.svdvals(M.float()).clamp(0.0, 1.0)
    sin_sq = 1.0 - sigmas**2
    return torch.sqrt(sin_sq.sum()).item()


# ---------------------------------------------------------------------------
# Incremental QR factorisation
# ---------------------------------------------------------------------------


@torch.no_grad()
def incremental_qr_update(
    Q: torch.Tensor,
    R: torch.Tensor,
    new_col: torch.Tensor,
    *,
    drop_oldest: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Update a thin QR factorisation with a new column.

    Given an existing factorisation ``A ≈ Q R`` where ``Q`` is ``(d, k)``
    orthonormal and ``R`` is ``(k, k)`` upper-triangular, this function
    appends ``new_col`` and optionally drops the oldest column to maintain
    a fixed-width window.

    The update uses one step of modified Gram-Schmidt:

    1. Project ``new_col`` onto existing ``Q`` columns.
    2. Compute the residual and normalise to get the new basis vector.
    3. Extend ``R`` accordingly.
    4. If ``drop_oldest``, remove the first column (shift left).

    This is dramatically cheaper than a full QR/SVD decomposition and
    provides a streaming, rank-revealing subspace tracker.

    Args:
        Q: Current orthonormal basis ``(d, k)``.
        R: Current upper-triangular factor ``(k, k)``.
        new_col: New gradient snapshot to incorporate ``(d,)`` or ``(d, 1)``.
        drop_oldest: If True, drop the oldest column to keep the basis width
            at ``k``.

    Returns:
        Tuple ``(Q_new, R_new)`` with the updated factorisation.
    """
    new_col = new_col.reshape(-1).to(dtype=Q.dtype, device=Q.device)
    d, k = Q.shape

    # Project new_col onto the current Q
    h = Q.t() @ new_col  # (k,)
    residual = new_col - Q @ h
    rho = torch.norm(residual)

    if rho < 1e-10:
        # new_col lies (nearly) in the existing subspace — re-orthogonalise
        # with a small perturbation to avoid degeneracy
        residual = torch.randn(d, device=Q.device, dtype=Q.dtype)
        residual = residual - Q @ (Q.t() @ residual)
        rho = torch.norm(residual)
        if rho < 1e-12:
            # Extremely degenerate; just return current factorisation
            return Q, R

    q_new = residual / rho

    # Build the extended R: add a new row [0 … 0 | rho] and col h
    # Extended R is (k+1, k+1):
    #   [ R  | h  ]
    #   [ 0  | rho]
    R_ext = torch.zeros(k + 1, k + 1, device=R.device, dtype=R.dtype)
    R_ext[:k, :k] = R
    R_ext[:k, k] = h
    R_ext[k, k] = rho

    # Extend Q
    Q_ext = torch.cat([Q, q_new.unsqueeze(1)], dim=1)  # (d, k+1)

    if drop_oldest and Q_ext.shape[1] > k:
        # Drop the first column
        Q_ext = Q_ext[:, 1:]
        R_ext = R_ext[1:, 1:]
        # Re-orthogonalise via economy QR to avoid drift
        if Q_ext.shape[1] <= Q_ext.shape[0]:
            Q_ext, R_ext = torch.linalg.qr(Q_ext, mode="reduced")

    return Q_ext, R_ext


@torch.no_grad()
def qr_subspace(G: torch.Tensor, k: int) -> torch.Tensor:
    """Extract a *k*-dimensional subspace basis from a gradient matrix.

    Uses randomised range-finding (one pass of ``G @ Ω`` with Gaussian
    ``Ω``) followed by economy QR for efficiency.  This is far cheaper
    than a full SVD when ``k ≪ min(m, n)``.

    Args:
        G: 2-D gradient tensor ``(m, n)``.
        k: Desired subspace dimension.

    Returns:
        Orthonormal basis ``Q`` of shape ``(m, k)``.
    """
    m, n = G.shape
    k = min(k, m, n)
    # Random projection
    Omega = torch.randn(n, k, device=G.device, dtype=G.dtype)
    Y = G @ Omega  # (m, k)
    Q, _ = torch.linalg.qr(Y, mode="reduced")
    return Q[:, :k]


# ---------------------------------------------------------------------------
# EMA helper
# ---------------------------------------------------------------------------


def ema_update(running: float, new: float, alpha: float) -> float:
    """Exponential moving average update.

    Computes ``(1 - α) * running + α * new``.

    Args:
        running: Current EMA value.
        new: New observation.
        alpha: Smoothing factor in ``(0, 1]``.  Larger values track faster.

    Returns:
        Updated EMA value.
    """
    return (1.0 - alpha) * running + alpha * new


def ema_update_tensor(
    running: torch.Tensor, new: torch.Tensor, alpha: float
) -> torch.Tensor:
    """Element-wise EMA update for tensors (in-place on ``running``).

    Args:
        running: Current EMA tensor (updated in-place).
        new: New observation tensor.
        alpha: Smoothing factor.

    Returns:
        Reference to ``running`` after the in-place update.
    """
    running.mul_(1.0 - alpha).add_(new, alpha=alpha)
    return running


# ---------------------------------------------------------------------------
# Gradient projection (GaLore-style)
# ---------------------------------------------------------------------------


@torch.no_grad()
def compute_projection_matrices(
    G: torch.Tensor, rank: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute left and right projection matrices for GaLore-style compression.

    Performs a truncated SVD of the gradient:

        ``G ≈ U_r Σ_r V_r^T``

    and returns ``(U_r, V_r)`` which can be used to project the gradient
    into a low-rank subspace:

        ``G_proj = U_r^T G V_r``  (shape ``(r, r)``)

    Args:
        G: 2-D gradient tensor ``(m, n)``.
        rank: Target rank ``r``.

    Returns:
        Tuple ``(U_r, V_r)`` where ``U_r`` is ``(m, r)`` and ``V_r`` is
        ``(n, r)``.
    """
    m, n = G.shape
    rank = min(rank, m, n)

    if rank >= min(m, n):
        # No projection needed
        U = torch.eye(m, device=G.device, dtype=G.dtype)
        V = torch.eye(n, device=G.device, dtype=G.dtype)
        return U, V

    # Use randomised SVD for efficiency
    U, S, Vt = _randomized_svd(G.float(), rank)
    return U.to(G.dtype), Vt.t().to(G.dtype)


def _randomized_svd(
    A: torch.Tensor, k: int, n_oversamples: int = 10, n_power_iter: int = 2
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Randomised truncated SVD.

    Implements the Halko-Martinsson-Tropp algorithm for computing the top-*k*
    singular triplets of a matrix efficiently.

    Args:
        A: Input matrix ``(m, n)`` in float32.
        k: Number of singular values / vectors to compute.
        n_oversamples: Oversampling parameter for the random projection.
        n_power_iter: Number of power iteration steps for improved accuracy.

    Returns:
        Tuple ``(U, S, Vt)`` — truncated left singular vectors ``(m, k)``,
        singular values ``(k,)``, and right singular vectors ``(k, n)``.
    """
    m, n = A.shape
    k = min(k, m, n)
    p = min(k + n_oversamples, min(m, n))

    # Stage A: form an approximate orthonormal basis for the range of A
    Omega = torch.randn(n, p, device=A.device, dtype=A.dtype)
    Y = A @ Omega
    for _ in range(n_power_iter):
        Y = A @ (A.t() @ Y)
    Q, _ = torch.linalg.qr(Y, mode="reduced")

    # Stage B: SVD of the small matrix B = Q^T A
    B = Q.t() @ A  # (p, n)
    U_hat, S, Vt = torch.linalg.svd(B, full_matrices=False)
    U = Q @ U_hat

    return U[:, :k], S[:k], Vt[:k, :]


# ---------------------------------------------------------------------------
# INT8 moment quantisation helpers
# ---------------------------------------------------------------------------


@torch.no_grad()
def quantize_to_int8(
    tensor: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantise a float tensor to INT8 with per-channel scaling.

    Uses symmetric quantisation: ``q = round(x / scale)`` where
    ``scale = max(|x|) / 127`` per row.

    Args:
        tensor: Float tensor of any shape.  Quantised along the last dim.

    Returns:
        Tuple ``(quantized_int8, scale, zero_point)`` where
        ``quantized_int8`` has dtype ``torch.int8``, ``scale`` and
        ``zero_point`` are float tensors used for dequantisation.
    """
    flat = tensor.reshape(-1, tensor.shape[-1]) if tensor.ndim > 1 else tensor.unsqueeze(0)
    absmax = flat.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
    scale = absmax / 127.0
    quantized = (flat / scale).round().clamp(-128, 127).to(torch.int8)
    zero_point = torch.zeros_like(scale)
    quantized = quantized.reshape(tensor.shape)
    scale = scale.reshape(tensor.shape[:-1] + (1,)) if tensor.ndim > 1 else scale.squeeze(0)
    zero_point = zero_point.reshape_as(scale)
    return quantized, scale, zero_point


@torch.no_grad()
def dequantize_from_int8(
    quantized: torch.Tensor,
    scale: torch.Tensor,
    zero_point: torch.Tensor,
    target_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Dequantise an INT8 tensor back to float.

    Args:
        quantized: INT8 tensor.
        scale: Per-channel scale factor.
        zero_point: Per-channel zero point (unused for symmetric quant).
        target_dtype: Desired output dtype.

    Returns:
        Dequantised float tensor.
    """
    return (quantized.to(target_dtype) - zero_point.to(target_dtype)) * scale.to(target_dtype)


# ---------------------------------------------------------------------------
# Gradient logging
# ---------------------------------------------------------------------------


def log_gradient_stats(
    stats: dict[str, Any],
    step: int,
    prefix: str = "phaseq",
    use_wandb: bool = True,
) -> None:
    """Log gradient statistics to W&B and/or the Python logger.

    Args:
        stats: Dictionary of metric name → value.
        step: Current training step.
        prefix: Prefix for metric names.
        use_wandb: Whether to attempt logging to W&B.
    """
    prefixed: dict[str, Any] = {f"{prefix}/{k}": v for k, v in stats.items()}

    if use_wandb:
        try:
            import wandb  # type: ignore[import-untyped]

            if wandb.run is not None:
                wandb.log(prefixed, step=step)
        except ImportError:
            pass

    # Always log to the standard logger at DEBUG level
    logger.debug("step=%d | %s", step, prefixed)


def compute_gradient_summary(model: torch.nn.Module) -> dict[str, float]:
    """Compute a summary of gradient norms across the model.

    Args:
        model: The model with ``.grad`` attributes populated.

    Returns:
        Dictionary with keys like ``grad_norm/total``,
        ``grad_norm/<param_name>``, ``grad_sparsity/<param_name>``.
    """
    summary: dict[str, float] = {}
    total_norm_sq = 0.0
    for name, param in model.named_parameters():
        if param.grad is not None:
            g = param.grad.data
            norm = g.norm().item()
            summary[f"grad_norm/{name}"] = norm
            total_norm_sq += norm**2
            # Sparsity: fraction of near-zero elements
            sparsity = (g.abs() < 1e-7).float().mean().item()
            summary[f"grad_sparsity/{name}"] = sparsity
    summary["grad_norm/total"] = math.sqrt(total_norm_sq)
    return summary
