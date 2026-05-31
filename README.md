# PhaseQ: Phased Adaptive Optimizer for Continued Pre-Training

PhaseQ is a novel adaptive optimizer designed specifically for Continued Pre-Training (CPT) of Large Language Models (LLMs). It detects gradient topology shifts during training using lightweight Grassmannian tracking and adapts rank, quantization, and per-layer memory allocation accordingly.

## The Core Insight

During Continued Pre-Training, gradients exhibit a distinct two-phase structure:
1. **Burst Phase**: Rapid adaptation to the new domain, characterized by high stable rank and fast gradient subspace rotation.
2. **Consolidation Phase**: Stabilization of learned representations, characterized by declining stable rank and slowing subspace rotation.

Static optimizers (like AdamW or fixed-rank GaLore) miss these dynamics. PhaseQ dynamically tracks these phases and adjusts optimization strategies per layer.

## Architecture

- **Grassmannian Phase Detector**: Monitors stable rank and Grassmannian geodesic distance using incremental QR factorization.
- **Per-Layer Rank Scheduler**: Maps detected phases to rank configurations (Full rank -> GaLore-style compression -> Q-GaLore minimum rank).
- **PhaseQAdamW Optimizer**: A drop-in replacement for standard optimizers that implements the dynamic compression logic.

## Usage

```python
from phaseq import PhaseQAdamW, PhaseQConfig
from phaseq.hooks import PhaseQTrainerCallback

# 1. Setup config
config = PhaseQConfig(
    lr=1e-4,
    tau1=0.3,
    tau2=0.05
)

# 2. Inject names into params for tracking
for name, param in model.named_parameters():
    param.name = name

param_groups = [{"params": [p], "name": n} for n, p in model.named_parameters() if p.requires_grad]

# 3. Initialize optimizer
optimizer = PhaseQAdamW(param_groups, config=config)
optimizer.register_model(model)

# 4. Integrate with HuggingFace Trainer
trainer = Trainer(
    model=model,
    optimizers=(optimizer, None),
    callbacks=[PhaseQTrainerCallback()]
)
trainer.train()
```

## Running Experiments

```bash
python experiments/run_phaseq.py --model_name gpt2 --dataset_name wikitext
```

## Next Steps to Continue Building

- [ ] **Extensive CPT Baselines**: Run full Pareto-frontier comparisons against recent optimizers like APOLLO, Fira, and SCALE on a medical or code corpus.
- [ ] **Grassmannian Hyperparameter Tuning**: Tune the $	au_1$ and $	au_2$ velocity thresholds across different model architectures (e.g., Llama-3, Qwen) to find universal defaults.
- [ ] **Scaling Laws**: Validate the phase transition timing predictions against the CPT scaling laws literature.
- [ ] **MoE Integration**: Extend the per-layer rank scheduler to handle Mixture of Experts (MoE) routing layers.
