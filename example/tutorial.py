"""
Weights & Biases POC — Fashion-MNIST MLP with scikit-learn
Demonstrates: experiment tracking, hyperparameter config, artifact logging,
and W&B Sweeps. Run with: python tutorial.py
"""

import gzip
import os
import urllib.request
import numpy as np
import joblib
import wandb
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import train_test_split

# Default hyperparameters — overridden by a W&B Sweep when run via wandb agent
DEFAULTS = dict(
    layer_1_size=16,
    layer_2_size=32,
    hidden_layer_size=128,
    learn_rate=0.01,
    momentum=0.9,
    epochs=8,
    batch_size=64,
)

LABELS = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]


# Mirrors that host the original Fashion-MNIST IDX files. We try them in order
# so a single mirror outage does not break training.
_FASHION_MNIST_MIRRORS = [
    "https://storage.googleapis.com/tensorflow/tf-keras-datasets/",
    "https://ossci-datasets.s3.amazonaws.com/mnist/",
    "https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion/",
]
_FASHION_MNIST_FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}
DATA_DIR = os.environ.get("FASHION_MNIST_DIR", "data/fashion-mnist")


def _download(filename: str, dest_dir: str) -> str:
    """Download `filename` to `dest_dir` from the first reachable mirror."""
    os.makedirs(dest_dir, exist_ok=True)
    dest_path = os.path.join(dest_dir, filename)
    if os.path.exists(dest_path) and os.path.getsize(dest_path) > 0:
        return dest_path

    last_err: Exception | None = None
    for base_url in _FASHION_MNIST_MIRRORS:
        url = base_url + filename
        try:
            print(f"Downloading {url}")
            urllib.request.urlretrieve(url, dest_path)
            return dest_path
        except Exception as exc:  # pragma: no cover - network paths
            last_err = exc
            print(f"  failed: {exc}")
            if os.path.exists(dest_path):
                os.remove(dest_path)
    raise RuntimeError(f"Could not download {filename} from any mirror") from last_err


def _read_idx_images(path: str) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        magic = int.from_bytes(f.read(4), "big")
        if magic != 2051:
            raise ValueError(f"Bad magic {magic} in image file {path}")
        n = int.from_bytes(f.read(4), "big")
        rows = int.from_bytes(f.read(4), "big")
        cols = int.from_bytes(f.read(4), "big")
        buf = f.read(n * rows * cols)
    return np.frombuffer(buf, dtype=np.uint8).reshape(n, rows * cols)


def _read_idx_labels(path: str) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        magic = int.from_bytes(f.read(4), "big")
        if magic != 2049:
            raise ValueError(f"Bad magic {magic} in label file {path}")
        n = int.from_bytes(f.read(4), "big")
        buf = f.read(n)
    return np.frombuffer(buf, dtype=np.uint8)


def load_data(subsample_ratio: float = 0.167, data_dir: str = DATA_DIR):
    """Load Fashion-MNIST from local IDX files (downloaded on first use)."""
    paths = {
        key: _download(name, data_dir)
        for key, name in _FASHION_MNIST_FILES.items()
    }

    x_train = _read_idx_images(paths["train_images"]).astype(np.float32) / 255.0
    y_train = _read_idx_labels(paths["train_labels"]).astype(int)
    x_test = _read_idx_images(paths["test_images"]).astype(np.float32) / 255.0
    y_test = _read_idx_labels(paths["test_labels"]).astype(int)

    rng = np.random.default_rng(42)
    mask = rng.random(len(x_train)) < subsample_ratio
    return (x_train[mask], y_train[mask]), (x_test, y_test)


def train():
    run = wandb.init(config=DEFAULTS)
    cfg = run.config

    (x_train, y_train), (x_test, y_test) = load_data()

    # Log dataset as a W&B Artifact
    dataset_artifact = wandb.Artifact("fashion-mnist-subset", type="dataset")
    dataset_artifact.add(
        wandb.Table(
            columns=["split", "samples"],
            data=[["train", len(x_train)], ["test", len(x_test)]],
        ),
        "summary",
    )
    run.log_artifact(dataset_artifact)

    clf = MLPClassifier(
        hidden_layer_sizes=(cfg.layer_1_size, cfg.layer_2_size, cfg.hidden_layer_size),
        learning_rate_init=cfg.learn_rate,
        momentum=cfg.momentum,
        batch_size=cfg.batch_size,
        solver="sgd",
        max_iter=1,
        warm_start=True,
        random_state=42,
    )

    # Train one epoch at a time to log per-epoch metrics
    for epoch in range(cfg.epochs):
        clf.fit(x_train, y_train)
        val_proba = clf.predict_proba(x_test)
        val_preds = clf.predict(x_test)
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": clf.loss_,
            "val_loss": log_loss(y_test, val_proba),
            "val_accuracy": accuracy_score(y_test, val_preds),
        })

    test_acc = accuracy_score(y_test, clf.predict(x_test))
    run.summary.update({"test_loss": clf.loss_, "test_accuracy": test_acc})

    # Save model as a W&B Artifact
    os.makedirs("models", exist_ok=True)
    model_path = "models/model.joblib"
    joblib.dump(clf, model_path)
    model_artifact = wandb.Artifact(f"fashion-mnist-mlp-{run.id}", type="model")
    model_artifact.add_file(model_path)
    run.log_artifact(model_artifact, aliases=["latest", "best"])

    run.finish()


if __name__ == "__main__":
    train()
