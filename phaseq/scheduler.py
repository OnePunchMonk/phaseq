"""Per-Layer Rank Scheduler for PhaseQ.

Maps phase detector output to concrete rank and compression configurations
for each layer / layer group.  Handles smooth transitions between phases
using exponential interpolation over a configurable window.

Phase → Configuration mapping:

* **BURST**: Full rank, no projection, BF16 weights, FP32 moments.
* **DECAY**: GaLore-style projection with adaptive rank (decreases as
  rotation slows), BF16 weights, INT8 moments.
* **STABLE**: Maximum compression (Q-GaLore level), minimum rank, NF4
  weights, INT8 moments.

Transitions are not instantaneous.  When a layer moves from one phase to
another, the scheduler interpolates the rank over ``transition_steps``
using an exponential schedule to prevent training instability.
"""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass, field
from typing import Any

import torch

from .config import (
    BURST_CONFIG,
    DECAY_CONFIG,
    STABLE_CONFIG,
    LayerGroupConfig,
    Phase,
    PhaseConfig,
    PhaseQConfig,
)
from .detector import GrassmannianPhaseDetector

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Per-layer schedule state
# ---------------------------------------------------------------------------


@dataclass
class LayerSchedule:
    """Runtime schedule state for a single parameter.

    Attributes:
        name: Parameter name.
        group_name: Name of the layer group this parameter belongs to.
        shape: Parameter shape.
        max_rank: Maximum rank for this parameter.
        current_rank: Currently active rank.
        target_rank: Rank being transitioned toward.
        current_phase: Detected phase for this layer.
        previous_phase: Phase before the most recent transition.
        transition_start_step: Step at which the current transition began.
        transition_start_rank: Rank at the start of the current transition.
        active_config: The ``PhaseConfig`` currently governing this layer.
        error_feedback_buffer: Accumulated projection error for smooth
            transition (LDAdam-inspired).
    """

    name: str = ""
    group_name: str = "default"
    shape: tuple[int, ...] = ()
    max_rank: int = 0
    current_rank: int = 0
    target_rank: int = 0
    current_phase: Phase = Phase.BURST
    previous_phase: Phase = Phase.BURST
    transition_start_step: int = 0
    transition_start_rank: int = 0
    active_config: PhaseConfig = field(default_factory=lambda: BURST_CONFIG)
    error_feedback_buffer: torch.Tensor | None = None


# ---------------------------------------------------------------------------
# Rank scheduler
# ---------------------------------------------------------------------------


class PerLayerRankScheduler:
    """Manages per-layer rank and compression settings.

    Usage::

        scheduler = PerLayerRankScheduler(config, detector)
        scheduler.register_layers(model)

        # In the training loop, after detector.update_all():
        scheduler.step(global_step)

        # Query current settings for a parameter:
        schedule = scheduler.get_schedule("model.layers.0.mlp.gate_proj.weight")
        rank = schedule.current_rank
        phase_config = schedule.active_config

    Args:
        config: PhaseQ configuration.
        detector: A :class:`~phaseq.detector.GrassmannianPhaseDetector`
            that has already registered the same layers.
    """

    def __init__(
        self,
        config: PhaseQConfig,
        detector: GrassmannianPhaseDetector,
    ) -> None:
        self.config = config
        self.detector = detector
        self._schedules: dict[str, LayerSchedule] = {}
        self._layer_group_map: dict[str, LayerGroupConfig] = {}

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register_layers(self, model: torch.nn.Module) -> list[str]:
        """Register all layers that the detector is tracking.

        Assigns each layer to a layer group based on the group's
        ``layer_name_patterns``.  Layers that don't match any group are
        placed in a catch-all ``"default"`` group.

        Args:
            model: The model (same one passed to the detector).

        Returns:
            List of registered parameter names.
        """
        registered: list[str] = []
        groups = self.config.layer_groups or [LayerGroupConfig()]

        for name, param in model.named_parameters():
            if name not in self.detector.registered_layers:
                continue

            # Find matching group
            group = self._find_group(name, groups)
            self._layer_group_map[name] = group

            # Compute max rank
            if param.ndim >= 2:
                effective_shape = (param.shape[0], param.numel() // param.shape[0])
                max_rank = min(effective_shape)
            else:
                max_rank = param.numel()

            if group.max_rank > 0:
                max_rank = min(max_rank, group.max_rank)

            initial_rank = max_rank  # BURST → full rank

            schedule = LayerSchedule(
                name=name,
                group_name=group.name,
                shape=tuple(param.shape),
                max_rank=max_rank,
                current_rank=initial_rank,
                target_rank=initial_rank,
                current_phase=Phase.BURST,
                active_config=group.phase_configs.get(Phase.BURST, BURST_CONFIG),
            )
            self._schedules[name] = schedule
            registered.append(name)

        logger.info("PerLayerRankScheduler: registered %d layers", len(registered))
        return registered

    @staticmethod
    def _find_group(
        name: str, groups: list[LayerGroupConfig]
    ) -> LayerGroupConfig:
        """Find the first matching layer group for a parameter name."""
        for group in groups:
            for pattern in group.layer_name_patterns:
                if re.search(pattern, name):
                    return group
        return LayerGroupConfig()  # default catch-all

    # ------------------------------------------------------------------
    # Step (called every training step)
    # ------------------------------------------------------------------

    def step(self, global_step: int) -> None:
        """Advance the scheduler by one step.

        Reads the current phase from the detector for each layer, computes
        the target rank, and interpolates the active rank if a transition is
        in progress.

        Args:
            global_step: Current training step.
        """
        for name, schedule in self._schedules.items():
            detected_phase = self.detector.get_phase(name)

            # Check for phase change
            if detected_phase != schedule.current_phase:
                self._start_transition(name, schedule, detected_phase, global_step)

            # Interpolate rank during transition
            self._interpolate_rank(schedule, global_step)

            # Update active config
            group = self._layer_group_map.get(name, LayerGroupConfig())
            schedule.active_config = group.phase_configs.get(
                schedule.current_phase,
                self._default_config_for_phase(schedule.current_phase),
            )

    def _start_transition(
        self,
        name: str,
        schedule: LayerSchedule,
        new_phase: Phase,
        step: int,
    ) -> None:
        """Begin a smooth transition from the current phase to a new one."""
        schedule.previous_phase = schedule.current_phase
        schedule.current_phase = new_phase
        schedule.transition_start_step = step
        schedule.transition_start_rank = schedule.current_rank

        # Compute target rank based on phase
        target_rank = self._compute_target_rank(schedule, new_phase)
        schedule.target_rank = target_rank

        logger.debug(
            "Transition started: layer=%s  %s → %s  rank %d → %d  step=%d",
            name,
            schedule.previous_phase.value,
            new_phase.value,
            schedule.current_rank,
            target_rank,
            step,
        )

    def _compute_target_rank(
        self, schedule: LayerSchedule, phase: Phase
    ) -> int:
        """Compute the target rank for a given phase.

        In DECAY, the target rank also adapts based on the current rotation
        rate from the detector (lower rotation → lower rank).
        """
        group = self._layer_group_map.get(schedule.name, LayerGroupConfig())
        phase_config = group.phase_configs.get(
            phase, self._default_config_for_phase(phase)
        )
        base_rank = max(1, int(schedule.max_rank * phase_config.rank_fraction))

        if phase == Phase.DECAY:
            # Adapt rank based on rotation rate within the DECAY band
            layer_stats = self.detector.get_layer_stats(schedule.name)
            if layer_stats is not None:
                tau1 = self.config.tau1
                tau2 = self.config.tau2
                rotation = layer_stats.ema_rotation
                # Linear interpolation: rotation=tau1 → full DECAY rank,
                # rotation=tau2 → STABLE rank
                if tau1 > tau2:
                    t = max(0.0, min(1.0, (rotation - tau2) / (tau1 - tau2)))
                else:
                    t = 0.0
                stable_rank = max(
                    1,
                    int(
                        schedule.max_rank
                        * group.phase_configs.get(
                            Phase.STABLE, STABLE_CONFIG
                        ).rank_fraction
                    ),
                )
                base_rank = max(1, int(stable_rank + t * (base_rank - stable_rank)))

        return base_rank

    def _interpolate_rank(
        self, schedule: LayerSchedule, global_step: int
    ) -> None:
        """Smoothly interpolate rank during a transition.

        Uses an exponential schedule: ``rank(t) = target + (start - target) * exp(-5t)``
        where ``t ∈ [0, 1]`` is the transition progress.
        """
        if schedule.current_rank == schedule.target_rank:
            return

        steps_since = global_step - schedule.transition_start_step
        T = max(1, self.config.transition_steps)
        t = min(1.0, steps_since / T)

        # Exponential decay toward target
        alpha = 1.0 - math.exp(-5.0 * t)
        interpolated = schedule.transition_start_rank + alpha * (
            schedule.target_rank - schedule.transition_start_rank
        )
        schedule.current_rank = max(1, int(round(interpolated)))

        # Snap to target at end of transition
        if t >= 1.0:
            schedule.current_rank = schedule.target_rank

    @staticmethod
    def _default_config_for_phase(phase: Phase) -> PhaseConfig:
        """Return a default PhaseConfig for a given phase."""
        defaults = {
            Phase.BURST: BURST_CONFIG,
            Phase.DECAY: DECAY_CONFIG,
            Phase.STABLE: STABLE_CONFIG,
        }
        return defaults.get(phase, BURST_CONFIG)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_schedule(self, name: str) -> LayerSchedule | None:
        """Get the current schedule for a parameter.

        Args:
            name: Parameter name.

        Returns:
            :class:`LayerSchedule` or ``None``.
        """
        return self._schedules.get(name)

    def get_all_schedules(self) -> dict[str, LayerSchedule]:
        """Return all layer schedules."""
        return dict(self._schedules)

    def get_stats_summary(self) -> dict[str, Any]:
        """Aggregate scheduling statistics for logging.

        Returns:
            Dictionary suitable for W&B logging.
        """
        if not self._schedules:
            return {}

        ranks = [s.current_rank for s in self._schedules.values()]
        max_ranks = [s.max_rank for s in self._schedules.values()]
        compressions = [
            1.0 - (s.current_rank / s.max_rank) if s.max_rank > 0 else 0.0
            for s in self._schedules.values()
        ]

        return {
            "rank/mean": sum(ranks) / len(ranks),
            "rank/max": max(ranks),
            "rank/min": min(ranks),
            "compression/mean": sum(compressions) / len(compressions),
            "compression/max": max(compressions),
            "memory_ratio": sum(ranks) / max(sum(max_ranks), 1),
        }
