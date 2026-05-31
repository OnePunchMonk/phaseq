"""HuggingFace Trainer hooks for PhaseQ Optimizer."""

from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl
import logging
import torch

from .optimizer import PhaseQAdamW

logger = logging.getLogger(__name__)

class PhaseQTrainerCallback(TrainerCallback):
    """
    A HuggingFace Trainer callback to integrate the PhaseQ optimizer.
    
    This callback ensures that the optimizer's internal step counter is updated,
    and logs phase transitions and gradient statistics to Weights & Biases or the
    Trainer's logging system.
    """
    
    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs
    ):
        """
        Called at the end of each training step.
        """
        optimizer = kwargs.get("optimizer")
        
        if optimizer is not None and isinstance(optimizer, PhaseQAdamW):
            # We can advance the scheduler if it has a step method
            # Note: PhaseQAdamW step already calls the detector and scheduler
            
            # Log metrics
            if state.global_step % args.logging_steps == 0:
                stats_to_log = {}
                
                # Get summary from detector
                if hasattr(optimizer.detector, "get_stats_summary"):
                    det_summary = optimizer.detector.get_stats_summary()
                    for k, v in det_summary.items():
                        stats_to_log[f"phaseq_detector/{k}"] = v
                        
                # Get summary from scheduler
                if hasattr(optimizer.scheduler, "get_stats_summary"):
                    sch_summary = optimizer.scheduler.get_stats_summary()
                    for k, v in sch_summary.items():
                        stats_to_log[f"phaseq_scheduler/{k}"] = v
                
                if stats_to_log:
                    # Inject logs back into trainer or log directly to wandb
                    try:
                        import wandb
                        if wandb.run is not None:
                            wandb.log(stats_to_log, step=state.global_step)
                    except ImportError:
                        pass
                        
                    # Also log via standard logging
                    logger.debug(f"PhaseQ Stats at step {state.global_step}: {stats_to_log}")
