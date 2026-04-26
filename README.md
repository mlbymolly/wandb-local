# Weights & Biases POC

A proof-of-concept demonstrating [Weights & Biases](https://wandb.ai) for ML experiment tracking, artifact management, and hyperparameter optimization — designed to run on GCP.

## What's included

| Path | Description |
|------|-------------|
| `example/` | Runnable Keras CNN script with W&B integration |
| `Effective_MLOps_W&B.ipynb` | End-to-end MLOps notebook with PyTorch + W&B (experiment tracking, artifacts, sweeps) |

## Key W&B Features Demonstrated

- **Experiment Tracking** — metrics, loss curves, and system stats logged automatically each epoch
- **Artifacts** — dataset and model versioning with lineage tracking
- **Sweeps** — Bayesian hyperparameter search across configurable search spaces
- **Model Registry** — aliasing and promoting best models (`latest`, `best`)

## Quick Start

### 1. Install dependencies

```bash
pip install -r example/requirements.txt
```

### 2. Authenticate

```bash
wandb login
```

### 3. Run the example

```bash
cd example
python tutorial.py
```

### 4. (Optional) Run a hyperparameter sweep

```bash
cd example
wandb sweep sweep.yaml
wandb agent <sweep-id>
```

## Running on GCP

### Vertex AI

Wrap `tutorial.py` in a Vertex AI custom training job. W&B runs from Vertex AI the same way as locally — set the `WANDB_API_KEY` environment variable as a secret and pass it through:

```bash
gcloud ai custom-jobs create \
  --region=us-central1 \
  --display-name=wandb-poc \
  --worker-pool-spec=machine-type=n1-standard-4,replica-count=1,\
container-image-uri=gcr.io/deeplearning-platform-release/tf2-gpu.2-13,\
  --environment-variables=WANDB_API_KEY=$WANDB_API_KEY
```

### Compute Engine / Cloud Run

Set `WANDB_API_KEY` as an environment variable or Secret Manager secret. No other GCP-specific configuration is needed — W&B communicates outbound over HTTPS.

## Resources

- [W&B Docs](https://docs.wandb.ai)
- [W&B + GCP Integration Guide](https://docs.wandb.ai/guides/integrations/gcp)
- [Vertex AI Custom Training](https://cloud.google.com/vertex-ai/docs/training/overview)
