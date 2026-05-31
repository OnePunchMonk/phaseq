"""PhaseQ configuration dataclasses.

Defines all configuration structures for the PhaseQ optimizer, including
phase-specific optimizer settings, per-layer-group configs, and the main
``PhaseQConfig`` that ties everything together.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any

import torch


class Phase(enum.Enum):
    """Training phase detected by the Grassmannian phase detector.

    Attributes:
        BURST: High gradient rank and fast subspace rotation.  Corresponds to
            the initial adaptation period of continued pre-training where the
            model rapidly adjusts to the new domain.
        DECAY: Declining gradient rank with slowing subspace rotation.  The
            model is consolidating learned representations.
        STABLE: Low gradient rank and near-zero subspace rotation.  Gradient
            structure has converged; maximum compression is safe.
    """

    BURST = "burst"
    DECAY = "decay"
    STABLE = "stable"


class WeightDType(enum.Enum):
    """Weight storage data type for a given phase."""

    FP32 = "fp32"
    BF16 = "bf16"
    FP16 = "fp16"
    NF4 = "nf4"
    INT8 = "int8"

    def to_torch_dtype(self) -> torch.dtype | None:
        """Convert to a ``torch.dtype``.

        Returns:
            The corresponding ``torch.dtype``, or ``None`` for quantised
            types (NF4, INT8) that require special handling.
        """
        mapping: dict[WeightDType, torch.dtype | None] = {
            WeightDType.FP32: torch.float32,
            WeightDType.BF16: torch.bfloat16,
            WeightDType.FP16: torch.float16,
            WeightDType.NF4: None,
            WeightDType.INT8: None,
        }
        return mapping[self]


class MomentDType(enum.Enum):
    """Moment (optimizer state) storage data type."""

    FP32 = "fp32"
    BF16 = "bf16"
    FP16 = "fp16"
    INT8 = "int8"

    def to_torch_dtype(self) -> torch.dtype | None:
        """Convert to a ``torch.dtype``.

        Returns:
            The corresponding ``torch.dtype``, or ``None`` for INT8 which
            requires special quantisation.
        """
        mapping: dict[MomentDType, torch.dtype | None] = {
            MomentDType.FP32: torch.float32,
            MomentDType.BF16: torch.bfloat16,
            MomentDType.FP16: torch.float16,
            MomentDType.INT8: None,
        }
        return mapping[self]


# ---------------------------------------------------------------------------
# Per-phase optimizer settings
# ---------------------------------------------------------------------------


@dataclass
class PhaseConfig:
    """Optimizer settings for a specific training phase.

    Args:
        phase: The training phase this config applies to.
        rank_fraction: Fraction of the maximum rank to use for gradient
            projection.  1.0 means full rank (no projection).
        weight_dtype: Data type for storing model weights in this phase.
        moment_dtype: Data type for storing first / second moment estimates.
        use_projection: Whether to apply GaLore-style gradient projection.
        projection_update_freq: How often (in steps) to recompute the
            projection matrix.
        error_feedback: Whether to use exponential error feedback for
            smooth transitions (LDAdam-inspired).
    """

    phase: Phase = Phase.BURST
    rank_fraction: float = 1.0
    weight_dtype: WeightDType = WeightDType.BF16
    moment_dtype: MomentDType = MomentDType.FP32
    use_projection: bool = False
    projection_update_freq: int = 200
    error_feedback: bool = False


# Convenience pre-built phase configs

BURST_CONFIG = PhaseConfig(
    phase=Phase.BURST,
    rank_fraction=1.0,
    weight_dtype=WeightDType.BF16,
    moment_dtype=MomentDType.FP32,
    use_projection=False,
    projection_update_freq=200,
    error_feedback=False,
)

DECAY_CONFIG = PhaseConfig(
    phase=Phase.DECAY,
    rank_fraction=0.5,
    weight_dtype=WeightDType.BF16,
    moment_dtype=MomentDType.INT8,
    use_projection=True,
    projection_update_freq=200,
    error_feedback=True,
)

STABLE_CONFIG = PhaseConfig(
    phase=Phase.STABLE,
    rank_fraction=0.125,
    weight_dtype=WeightDType.NF4,
    moment_dtype=MomentDType.INT8,
    use_projection=True,
    projection_update_freq=500,
    error_feedback=True,
)


# ---------------------------------------------------------------------------
# Per-layer-group configuration
# ---------------------------------------------------------------------------


@dataclass
class LayerGroupConfig:
    """Configuration for a group of layers that share a schedule.

    Args:
        name: Human-readable group name (e.g. ``"attention"``, ``"mlp"``).
        layer_name_patterns: List of regex patterns matching parameter names
            that belong to this group.
        current_phase: The detected phase for this group.
        phase_configs: Mapping from ``Phase`` to ``PhaseConfig``.
        max_rank: Maximum allowed rank for gradient projection.  Set to the
            smaller dimension of the parameter by default (``0`` means auto).
        current_rank: Currently active rank.
        target_rank: Rank the scheduler is transitioning toward.
        transition_progress: Progress of the current transition (0.0 → 1.0).
    """

    name: str = "default"
    layer_name_patterns: list[str] = field(default_factory=lambda: [".*"])
    current_phase: Phase = Phase.BURST
    phase_configs: dict[Phase, PhaseConfig] = field(
        default_factory=lambda: {
            Phase.BURST: BURST_CONFIG,
            Phase.DECAY: DECAY_CONFIG,
            Phase.STABLE: STABLE_CONFIG,
        }
    )
    max_rank: int = 0  # 0 = auto-detect from parameter shape
    current_rank: int = 0
    target_rank: int = 0
    transition_progress: float = 1.0


# ---------------------------------------------------------------------------
# Main PhaseQ configuration
# ---------------------------------------------------------------------------


@dataclass
class PhaseQConfig:
    """Top-level configuration for the PhaseQ optimizer.

    Args:
        tau1: Threshold for the Burst → Decay phase transition.  When the
            EMA-smoothed Grassmannian rotation rate drops below ``tau1``, the
            detector signals DECAY.
        tau2: Threshold for the Decay → Stable phase transition.  When the
            EMA-smoothed Grassmannian rotation rate drops below ``tau2``, the
            detector signals STABLE.
        ema_alpha: Smoothing factor for the exponential moving average of
            gradient statistics.  Larger values give more weight to recent
            observations.
        subspace_rank: Rank of the subspace used for Grassmannian tracking
            in the phase detector.
        transition_steps: Number of optimizer steps over which to smoothly
            transition rank and quantisation settings between phases.
        warmup_steps: Number of initial steps before the phase detector
            begins operating (all layers stay in BURST during warmup).
        stat_compute_freq: How often (in steps) to recompute gradient
            statistics and update the phase detector.
        layer_groups: Per-layer-group configurations.  If empty, all
            parameters are placed in a single default group.
        lr: Base learning rate for AdamW.
        betas: AdamW beta coefficients.
        eps: AdamW epsilon for numerical stability.
        weight_decay: AdamW weight decay coefficient.
        max_grad_norm: Maximum gradient norm for clipping (0 = disabled).
        log_stats: Whether to log gradient statistics and phase transitions.
        log_prefix: Prefix for logged metric names.
    """

    # Phase detection
    tau1: float = 0.3
    tau2: float = 0.05
    ema_alpha: float = 0.1
    subspace_rank: int = 16
    transition_steps: int = 100
    warmup_steps: int = 50
    stat_compute_freq: int = 10

    # Layer groups
    layer_groups: list[LayerGroupConfig] = field(default_factory=list)

    # AdamW hyper-parameters
    lr: float = 1e-4
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0

    # Logging
    log_stats: bool = True
    log_prefix: str = "phaseq"

    def to_dict(self) -> dict[str, Any]:
        """Serialise the config to a plain dictionary (for logging)."""
        result: dict[str, Any] = {}
        for k, v in self.__dict__.items():
            if isinstance(v, list):
                result[k] = [
                    item.__dict__ if hasattr(item, "__dict__") else item for item in v
                ]
            elif isinstance(v, tuple):
                result[k] = list(v)
            elif isinstance(v, enum.Enum):
                result[k] = v.value
            else:
                result[k] = v
        return result

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> PhaseQConfig:
        """Create a ``PhaseQConfig`` from a plain dictionary.

        Args:
            d: Dictionary of config values (e.g. loaded from YAML).

        Returns:
            A new ``PhaseQConfig`` instance.
        """
        layer_groups_raw = d.pop("layer_groups", [])
        layer_groups: list[LayerGroupConfig] = []
        for lg in layer_groups_raw:
            if isinstance(lg, dict):
                # Convert nested phase_configs
                if "phase_configs" in lg:
                    pc_raw = lg.pop("phase_configs")
                    pc: dict[Phase, PhaseConfig] = {}
                    for phase_key, phase_cfg in pc_raw.items():
                        phase_enum = Phase(phase_key) if isinstance(phase_key, str) else phase_key
                        if isinstance(phase_cfg, dict):
                            phase_cfg["phase"] = phase_enum
                            if "weight_dtype" in phase_cfg:
                                phase_cfg["weight_dtype"] = WeightDType(phase_cfg["weight_dtype"])
                            if "moment_dtype" in phase_cfg:
                                phase_cfg["moment_dtype"] = MomentDType(phase_cfg["moment_dtype"])
                            pc[phase_enum] = PhaseConfig(**phase_cfg)
                        else:
                            pc[phase_enum] = phase_cfg
                    lg["phase_configs"] = pc
                if "current_phase" in lg and isinstance(lg["current_phase"], str):
                    lg["current_phase"] = Phase(lg["current_phase"])
                layer_groups.append(LayerGroupConfig(**lg))
            else:
                layer_groups.append(lg)

        if "betas" in d and isinstance(d["betas"], list):
            d["betas"] = tuple(d["betas"])

        return cls(layer_groups=layer_groups, **d)
