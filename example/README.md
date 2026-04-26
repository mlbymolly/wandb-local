# Fashion-MNIST Examples — W&B

Two runnable scripts demonstrating core Weights & Biases features.

---

## `tutorial.py` — Scikit-learn MLP

A scikit-learn `MLPClassifier` trained on Fashion-MNIST showing:
- `wandb.init()` with hyperparameter config
- Per-epoch `wandb.log()` for loss and accuracy curves
- Dataset and model logging as W&B Artifacts
- Sweep-ready structure (hyperparams via `wandb.config`)

```bash
pip install -r requirements.txt
wandb login
python tutorial.py
```

### Hyperparameter Sweep

```bash
wandb sweep sweep.yaml
wandb agent <sweep-id>
```

---

## `distributed_training.py` — PyTorch DDP Log Organization

A PyTorch DistributedDataParallel training example showing how to organize W&B logs across multiple workers:
- `wandb.init(group=...)` to cluster all ranks under one experiment in the UI
- Per-rank metrics (`rank_N/train_loss`) logged from every worker
- Global aggregated metrics (`global/train_loss`) logged from rank 0 only via `dist.all_reduce`
- Validation and `run.summary` written by rank 0 to avoid duplicate entries

```bash
# Simulate 2 workers locally (no GPU required)
python distributed_training.py

# Or launch with torchrun
torchrun --nproc_per_node=2 distributed_training.py
```

---

See the [W&B Docs](https://docs.wandb.ai) for full API reference.
