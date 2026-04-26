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

### Ray

W&B works inside Ray remote functions and actors — each worker calls `wandb.init()` independently. Use the `group` parameter to tie all workers from a single job together in the W&B UI.

#### Install

```bash
pip install wandb ray
```

#### Ray Core (remote functions)

```python
import ray
import wandb

@ray.remote
def train_worker(config, group_id):
    run = wandb.init(
        project="my-project",
        group=group_id,       # groups all workers from this job
        job_type="train",
        config=config,
    )
    # ... training logic ...
    wandb.log({"loss": 0.42, "accuracy": 0.95})
    run.finish()

ray.init()
group_id = wandb.util.generate_id()  # shared across all workers
futures = [train_worker.remote({"lr": 0.01}, group_id) for _ in range(4)]
ray.get(futures)
```

#### Ray Tune

Ray Tune has a built-in W&B callback that handles `wandb.init()` and metric logging automatically:

```python
from ray import tune
from ray.air.integrations.wandb import WandbLoggerCallback

def trainable(config):
    for epoch in range(config["epochs"]):
        loss = train_one_epoch(config)
        tune.report(loss=loss)

tuner = tune.Tuner(
    trainable,
    param_space={"lr": tune.grid_search([0.001, 0.01]), "epochs": 10},
    run_config=tune.RunConfig(
        callbacks=[WandbLoggerCallback(project="my-project")]
    ),
)
tuner.fit()
```

#### Authentication

Each Ray worker needs `WANDB_API_KEY` in its environment. Pass it when submitting the job:

```bash
# Ray job submission
ray job submit --working-dir . \
  --runtime-env-json '{"env_vars": {"WANDB_API_KEY": "'$WANDB_API_KEY'"}}' \
  -- python train.py
```

Or set it in a `runtime_env` dict when calling `ray.init()`:

```python
ray.init(runtime_env={"env_vars": {"WANDB_API_KEY": os.environ["WANDB_API_KEY"]}})
```

#### Running Ray jobs on GCP

For Ray clusters on GCP (KubeRay on GKE, or Ray on Vertex AI), store the API key in Secret Manager and inject it as an environment variable into your worker pod spec or job config — same pattern as the Vertex AI section above.

## Resources

- [W&B Docs](https://docs.wandb.ai)
- [W&B + GCP Integration Guide](https://docs.wandb.ai/guides/integrations/gcp)
- [Vertex AI Custom Training](https://cloud.google.com/vertex-ai/docs/training/overview)
