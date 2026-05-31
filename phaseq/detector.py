"""Grassmannian Phase Detector for PhaseQ.

Monitors per-layer gradient topology during continued pre-training and
classifies each layer's gradient regime into one of three phases:

* **BURST** – high stable rank, fast subspace rotation.
* **DECAY** – declining stable rank, slowing rotation.
* **STABLE** – low stable rank, near-zero rotation.

The detector maintains an EMA-smoothed history of two signals per tracked
layer:

1. **Stable rank** ``sr(G) = ||G||_F^2 / ||G||_2^2``.
2. **Subspace rotation rate** – the Grassmannian geodesic distance between
   consecutive *k*-dimensional gradient subspaces, obtained via cheap
   incremental QR factorisation (no full SVD).

Phase boundaries are defined by two thresholds ``τ₁`` and ``τ₂`` on the
smoothed rotation rate:

* ``rotation ≥ τ₁`` → **BURST**
* ``τ₂ ≤ rotation < τ₁`` → **DECAY**
* ``rotation < τ₂`` → **STABLE**
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import torch

from .config import Phase, PhaseQConfig
from .utils import (
    compute_stable_rank,
    ema_update,
    grassmannian_distance,
    qr_subspace,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Per-layer tracking state
# ---------------------------------------------------------------------------


@dataclass
class LayerStats:
    """Internal tracking state for a single parameter / layer.

    This is not user-facing — it is managed entirely by
    :class:`GrassmannianPhaseDetector`.

    Attributes:
        name: Parameter name.
        phase: Current detected phase.
        prev_subspace: Previous subspace basis ``(d, k)`` for rotation calc.
        ema_rotation: EMA-smoothed Grassmannian rotation rate.
        ema_stable_rank: EMA-smoothed stable rank.
        raw_rotation: Most recent raw rotation value.
        raw_stable_rank: Most recent raw stable rank value.
        step_count: Number of stat updates performed.
        shape: Shape of the parameter tensor.
        phase_history: List of ``(step, phase)`` transitions.
    """

    name: str = ""
    phase: Phase = Phase.BURST
    prev_subspace: torch.Tensor | None = None
    ema_rotation: float = 1.0  # start high → BURST
    ema_stable_rank: float = 1.0
    raw_rotation: float = 1.0
    raw_stable_rank: float = 1.0
    step_count: int = 0
    shape: tuple[int, ...] = ()
    phase_history: list[tuple[int, Phase]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Phase detector
# ---------------------------------------------------------------------------


class GrassmannianPhaseDetector:
    """Detects gradient topology phases for each tracked layer.

    Usage::

        detector = GrassmannianPhaseDetector(config)
        detector.register_layers(model)

        # Inside the training loop, after backward():
        for name, param in model.named_parameters():
            if param.grad is not None:
                detector.update(name, param.grad, global_step)

        # Query phase for a specific layer
        phase = detector.get_phase("model.layers.0.self_attn.q_proj.weight")

    Args:
        config: PhaseQ configuration containing tau1, tau2, ema_alpha,
            subspace_rank, and warmup_steps.
    """

    def __init__(self, config: PhaseQConfig) -> None:
        self.config = config
        self._layer_stats: dict[str, LayerStats] = {}
        self._global_step: int = 0

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register_layers(
        self,
        model: torch.nn.Module,
        *,
        min_dim: int = 128,
        include_patterns: list[str] | None = None,
    ) -> list[str]:
        """Auto-register all 2-D parameters large enough to track.

        Args:
            model: The model whose parameters will be tracked.
            min_dim: Minimum smaller dimension for a parameter to be tracked.
                Tiny parameters (embeddings aside) do not have meaningful
                gradient subspaces.
            include_patterns: Optional list of regex patterns; only matching
                parameter names are registered.  ``None`` means all eligible
                parameters.

        Returns:
            List of registered parameter names.
        """
        import re

        registered: list[str] = []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if param.ndim < 2:
                continue
            effective_shape = (param.shape[0], param.numel() // param.shape[0])
            if min(effective_shape) < min_dim:
                continue
            if include_patterns is not None:
                if not any(re.search(p, name) for p in include_patterns):
                    continue
            self._register_layer(name, param.shape)
            registered.append(name)

        logger.info("GrassmannianPhaseDetector: registered %d layers", len(registered))
        return registered

    def _register_layer(self, name: str, shape: tuple[int, ...]) -> None:
        """Register a single layer for tracking."""
        self._layer_stats[name] = LayerStats(
            name=name,
            shape=shape,
            phase=Phase.BURST,
            phase_history=[(0, Phase.BURST)],
        )

    def register_layer(self, name: str, shape: tuple[int, ...]) -> None:
        """Manually register a single layer for tracking.

        Args:
            name: Parameter name.
            shape: Shape of the parameter tensor.
        """
        self._register_layer(name, shape)

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    @torch.no_grad()
    def update(
        self,
        name: str,
        grad: torch.Tensor,
        step: int,
    ) -> Phase | None:
        """Incorporate a new gradient observation for a layer.

        This method:
        1. Computes the stable rank of ``grad``.
        2. Extracts a *k*-dimensional subspace from ``grad``.
        3. Computes the Grassmannian distance to the previous subspace.
        4. Updates the EMA-smoothed statistics.
        5. Determines the current phase based on thresholds.

        Args:
            name: Parameter name (must have been registered).
            grad: Gradient tensor for this parameter.
            step: Current global training step.

        Returns:
            The detected :class:`~phaseq.config.Phase` for this layer, or
            ``None`` if the layer is not registered.
        """
        if name not in self._layer_stats:
            return None

        self._global_step = step
        stats = self._layer_stats[name]
        stats.step_count += 1

        # Reshape to 2-D
        G = grad.data
        if G.ndim == 1:
            return stats.phase
        if G.ndim > 2:
            G = G.reshape(G.shape[0], -1)

        # 1. Stable rank
        sr = compute_stable_rank(G)
        stats.raw_stable_rank = sr
        stats.ema_stable_rank = ema_update(
            stats.ema_stable_rank, sr, self.config.ema_alpha
        )

        # 2. Subspace extraction
        k = min(self.config.subspace_rank, *G.shape)
        current_subspace = qr_subspace(G, k)

        # 3. Rotation rate
        if stats.prev_subspace is not None and stats.prev_subspace.shape == current_subspace.shape:
            rotation = grassmannian_distance(stats.prev_subspace, current_subspace)
            # Normalise by sqrt(k) * pi/2 so the value is in [0, 1]
            import math

            max_dist = math.sqrt(k) * math.pi / 2.0
            rotation = rotation / max_dist if max_dist > 0 else rotation
        else:
            rotation = 1.0  # first observation → assume maximum rotation

        stats.raw_rotation = rotation
        stats.ema_rotation = ema_update(
            stats.ema_rotation, rotation, self.config.ema_alpha
        )
        stats.prev_subspace = current_subspace

        # 4. Phase determination (with warmup guard)
        if step < self.config.warmup_steps:
            new_phase = Phase.BURST
        else:
            new_phase = self._classify_phase(stats.ema_rotation)

        # 5. Record transition
        if new_phase != stats.phase:
            stats.phase_history.append((step, new_phase))
            logger.info(
                "Phase transition: layer=%s  step=%d  %s → %s  "
                "(rotation=%.4f, stable_rank=%.2f)",
                name,
                step,
                stats.phase.value,
                new_phase.value,
                stats.ema_rotation,
                stats.ema_stable_rank,
            )
            stats.phase = new_phase

        return stats.phase

    def _classify_phase(self, rotation: float) -> Phase:
        """Classify phase from the smoothed rotation rate.

        Args:
            rotation: EMA-smoothed normalised Grassmannian rotation rate.

        Returns:
            The detected phase.
        """
        if rotation >= self.config.tau1:
            return Phase.BURST
        elif rotation >= self.config.tau2:
            return Phase.DECAY
        else:
            return Phase.STABLE

    # ------------------------------------------------------------------
    # Batch update (convenience)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def update_all(
        self,
        model: torch.nn.Module,
        step: int,
    ) -> dict[str, Phase]:
        """Update all registered layers from the model's current gradients.

        Args:
            model: Model with ``.grad`` populated (after ``backward()``).
            step: Current global training step.

        Returns:
            Dict mapping parameter name → detected phase.
        """
        phases: dict[str, Phase] = {}
        for name, param in model.named_parameters():
            if name in self._layer_stats and param.grad is not None:
                phase = self.update(name, param.grad, step)
                if phase is not None:
                    phases[name] = phase
        return phases

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_phase(self, name: str) -> Phase:
        """Get the current phase of a layer.

        Args:
            name: Parameter name.

        Returns:
            Current phase.  Returns ``Phase.BURST`` for unknown layers.
        """
        if name in self._layer_stats:
            return self._layer_stats[name].phase
        return Phase.BURST

    def get_layer_stats(self, name: str) -> LayerStats | None:
        """Get the full tracking state for a layer.

        Args:
            name: Parameter name.

        Returns:
            :class:`LayerStats` or ``None`` if the layer is not registered.
        """
        return self._layer_stats.get(name)

    def get_all_phases(self) -> dict[str, Phase]:
        """Get phases for all registered layers.

        Returns:
            Dict mapping parameter name → current phase.
        """
        return {name: stats.phase for name, stats in self._layer_stats.items()}

    def get_global_phase(self) -> Phase:
        """Determine the dominant phase across all layers.

        Uses majority voting: the phase assigned to the most layers wins.

        Returns:
            The dominant phase.
        """
        if not self._layer_stats:
            return Phase.BURST

        counts = {Phase.BURST: 0, Phase.DECAY: 0, Phase.STABLE: 0}
        for stats in self._layer_stats.values():
            counts[stats.phase] += 1

        return max(counts, key=lambda p: counts[p])

    # ------------------------------------------------------------------
    # Statistics for logging
    # ------------------------------------------------------------------

    def get_stats_summary(self) -> dict[str, Any]:
        """Aggregate statistics across all tracked layers for logging.

        Returns:
            Dictionary suitable for passing to
            :func:`~phaseq.utils.log_gradient_stats`.
        """
        if not self._layer_stats:
            return {}

        rotations = []
        stable_ranks = []
        phase_counts = {Phase.BURST: 0, Phase.DECAY: 0, Phase.STABLE: 0}

        for stats in self._layer_stats.values():
            rotations.append(stats.ema_rotation)
            stable_ranks.append(stats.ema_stable_rank)
            phase_counts[stats.phase] += 1

        n = len(rotations)
        summary: dict[str, Any] = {
            "rotation/mean": sum(rotations) / n,
            "rotation/max": max(rotations),
            "rotation/min": min(rotations),
            "stable_rank/mean": sum(stable_ranks) / n,
            "stable_rank/max": max(stable_ranks),
            "stable_rank/min": min(stable_ranks),
            "phase_count/burst": phase_counts[Phase.BURST],
            "phase_count/decay": phase_counts[Phase.DECAY],
            "phase_count/stable": phase_counts[Phase.STABLE],
            "global_phase": self.get_global_phase().value,
        }
        return summary

    @property
    def registered_layers(self) -> list[str]:
        """List of registered parameter names."""
        return list(self._layer_stats.keys())

    @property
    def num_layers(self) -> int:
        """Number of registered layers."""
        return len(self._layer_stats)
