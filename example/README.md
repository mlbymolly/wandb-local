# Fashion-MNIST CNN — W&B Example

A minimal script demonstrating core Weights & Biases features with a Keras CNN trained on Fashion-MNIST.

**What it shows:**
- `wandb.init()` with hyperparameter config
- `WandbMetricsLogger` for automatic epoch-level metric tracking
- `WandbModelCheckpoint` for saving best model weights
- Dataset and model logging as W&B Artifacts
- Sweep-ready structure (hyperparams via `wandb.config`)

## Quickstart

```bash
pip install -r requirements.txt
wandb login
python tutorial.py
```

Then open your W&B dashboard to see metrics, artifacts, and model checkpoints.

## Running a Hyperparameter Sweep

```bash
wandb sweep sweep.yaml
wandb agent <sweep-id>
```

See the [W&B Sweeps docs](https://docs.wandb.ai/guides/sweeps) for sweep configuration options.
