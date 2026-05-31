"""PhaseQ Optimizer implementation."""

import math
from typing import Any, Callable, Iterable

import torch
from torch.optim import Optimizer

from .config import PhaseQConfig, Phase, LayerGroupConfig, WeightDType, MomentDType
from .detector import GrassmannianPhaseDetector
from .scheduler import PerLayerRankScheduler

try:
    import bitsandbytes as bnb
except ImportError:
    bnb = None

class PhaseQAdamW(Optimizer):
    """PhaseQ Adaptive Optimizer.
    
    A phased adaptive optimizer for continued pre-training (CPT) of LLMs.
    It tracks gradient topology (Grassmannian distance and stable rank) to
    detect the training phase and automatically adjusts rank and quantization.
    """
    
    def __init__(
        self,
        params: Iterable[torch.nn.Parameter] | Iterable[dict[str, Any]],
        config: PhaseQConfig = None,
        **kwargs
    ):
        if config is None:
            config = PhaseQConfig()
            
        defaults = dict(
            lr=config.lr,
            betas=config.betas,
            eps=config.eps,
            weight_decay=config.weight_decay,
        )
        defaults.update(kwargs)
        super().__init__(params, defaults)
        
        self.config = config
        self.detector = GrassmannianPhaseDetector(config)
        self.scheduler = PerLayerRankScheduler(config, self.detector)
        
        self.projections = {}
        self.global_step = 0
        
        # Register layers immediately if params are parameters (not dicts)
        # Note: in a real implementation we might pass model directly, but here we mock it
        # by creating a dummy module wrapper if needed, or by modifying register_layers to take parameters.
        # Since detector/scheduler expect a model with named_parameters, we'll wait or handle it lazily.
        
    def _get_param_name(self, p: torch.nn.Parameter) -> str:
        for group in self.param_groups:
            if 'name' in group:
                return group['name']
        return f"param_{id(p)}"
        
    def register_model(self, model: torch.nn.Module):
        """Registers a model for phase detection and scheduling."""
        self.detector.register_layers(model)
        self.scheduler.register_layers(model)

    @torch.no_grad()
    def step(self, closure: Callable[[], float] | None = None) -> float | None:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        self.global_step += 1

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                param_name = self._get_param_name(p)

                # 1. Update Detector & Get Phase
                # Detector requires update via `update_all` or manually for each param
                self.detector.update(param_name, grad, self.global_step)
                current_phase = self.detector.get_phase(param_name)
                
                # 2. Update Scheduler & Get Config
                # Step scheduler for this step
                self.scheduler.step(self.global_step)
                schedule = self.scheduler.get_schedule(param_name)
                
                if schedule is None:
                    # Fallback config
                    use_projection = False
                    error_feedback = False
                    target_rank = 0
                    moment_dtype = MomentDType.FP32
                else:
                    phase_config = schedule.active_config
                    use_projection = phase_config.use_projection
                    error_feedback = phase_config.error_feedback
                    target_rank = schedule.current_rank
                    moment_dtype = phase_config.moment_dtype
                
                # 3. Apply Error Feedback (LDAdam style)
                if schedule is not None and error_feedback and schedule.error_feedback_buffer is not None:
                    grad.add_(schedule.error_feedback_buffer)
                    
                # 4. GaLore Projection
                projected_grad = grad
                if use_projection and grad.dim() == 2 and target_rank > 0 and target_rank < min(grad.shape):
                    state = self.state[p]
                    step = state.get("step", 0)
                    
                    update_freq = phase_config.projection_update_freq if schedule is not None else 200
                    
                    if p not in self.projections or step % update_freq == 0:
                        U, S, Vh = torch.linalg.svd(grad.float(), full_matrices=False)
                        P = Vh[:target_rank, :]
                        self.projections[p] = P.to(grad.device)
                        
                    P = self.projections[p]
                    projected_grad = torch.matmul(grad.float(), P.mT).to(grad.dtype)
                    
                    # Update error feedback
                    if error_feedback and schedule is not None:
                        reconstructed_grad = torch.matmul(projected_grad.float(), P).to(grad.dtype)
                        error = grad - reconstructed_grad
                        if schedule.error_feedback_buffer is None:
                            schedule.error_feedback_buffer = error.clone()
                        else:
                            # EMA update for error feedback
                            alpha = 0.9
                            schedule.error_feedback_buffer.mul_(alpha).add_(error, alpha=1 - alpha)
                        
                # 5. AdamW Update on Projected Gradient
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    if moment_dtype == MomentDType.INT8 and bnb is not None:
                        # Placeholder for 8-bit moments
                        state["exp_avg"] = torch.zeros_like(projected_grad)
                        state["exp_avg_sq"] = torch.zeros_like(projected_grad)
                    else:
                        state["exp_avg"] = torch.zeros_like(projected_grad)
                        state["exp_avg_sq"] = torch.zeros_like(projected_grad)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]

                state["step"] += 1

                # Decay moments
                exp_avg.mul_(beta1).add_(projected_grad, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(projected_grad, projected_grad, value=1.0 - beta2)

                denom = exp_avg_sq.sqrt().add_(group["eps"])

                # Bias correction
                bias_correction1 = 1 - beta1 ** state["step"]
                bias_correction2 = 1 - beta2 ** state["step"]
                step_size = group["lr"] * math.sqrt(bias_correction2) / bias_correction1

                update = exp_avg / denom

                # Weight decay
                if group["weight_decay"] > 0:
                    if use_projection and p in self.projections:
                        P = self.projections[p]
                        proj_weight = torch.matmul(p.data.float(), P.mT).to(p.dtype)
                        update.add_(proj_weight, alpha=group["weight_decay"])
                    else:
                        update.add_(p.data, alpha=group["weight_decay"])

                # Unproject and apply
                if use_projection and p in self.projections:
                    P = self.projections[p]
                    full_update = torch.matmul(update.float(), P).to(p.dtype)
                    p.data.add_(full_update, alpha=-step_size)
                else:
                    p.data.add_(update, alpha=-step_size)

        return loss
