"""
Distributed Training Log Organization with W&B
Demonstrates how to organize logs across multiple workers using:
  - wandb.init(group=...) to cluster all ranks under one experiment
  - rank-0-only global metrics (aggregated loss/accuracy)
  - per-rank local metrics for debugging worker-level behavior
  - proper run lifecycle across all processes

Run via torchrun:
  torchrun --nproc_per_node=2 distributed_training.py

Or simulate locally without GPUs (spawns workers via multiprocessing):
  python distributed_training.py
"""

import os
import gzip
import urllib.request
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, TensorDataset, DistributedSampler
import wandb

# ── Data ─────────────────────────────────────────────────────────────────────

_MIRRORS = [
    "https://storage.googleapis.com/tensorflow/tf-keras-datasets/",
    "https://ossci-datasets.s3.amazonaws.com/mnist/",
    "https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion/",
]
_FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images":  "t10k-images-idx3-ubyte.gz",
    "test_labels":  "t10k-labels-idx1-ubyte.gz",
}
DATA_DIR = os.environ.get("FASHION_MNIST_DIR", "data/fashion-mnist")


def _download(filename: str, dest_dir: str) -> str:
    os.makedirs(dest_dir, exist_ok=True)
    dest = os.path.join(dest_dir, filename)
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        return dest
    for base in _MIRRORS:
        try:
            urllib.request.urlretrieve(base + filename, dest)
            return dest
        except Exception:
            if os.path.exists(dest):
                os.remove(dest)
    raise RuntimeError(f"Could not download {filename}")


def _read_images(path: str) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        f.read(4)
        n = int.from_bytes(f.read(4), "big")
        rows = int.from_bytes(f.read(4), "big")
        cols = int.from_bytes(f.read(4), "big")
        buf = f.read(n * rows * cols)
    return np.frombuffer(buf, dtype=np.uint8).reshape(n, rows * cols)


def _read_labels(path: str) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        f.read(4)
        n = int.from_bytes(f.read(4), "big")
        buf = f.read(n)
    return np.frombuffer(buf, dtype=np.uint8)


def load_tensors():
    paths = {k: _download(v, DATA_DIR) for k, v in _FILES.items()}
    x_train = torch.tensor(_read_images(paths["train_images"]) / 255.0, dtype=torch.float32)
    y_train = torch.tensor(_read_labels(paths["train_labels"]), dtype=torch.long)
    x_test  = torch.tensor(_read_images(paths["test_images"])  / 255.0, dtype=torch.float32)
    y_test  = torch.tensor(_read_labels(paths["test_labels"]),  dtype=torch.long)
    return TensorDataset(x_train, y_train), TensorDataset(x_test, y_test)


# ── Model ─────────────────────────────────────────────────────────────────────

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(784, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        return self.net(x)


# ── Training worker ───────────────────────────────────────────────────────────

def train_worker(rank: int, world_size: int, group_name: str):
    """One training process. Each rank gets its own W&B run inside the group."""

    # ── Init process group (gloo works on CPU; use nccl for GPU) ──────────────
    dist.init_process_group(
        backend="gloo",
        init_method="env://",
        world_size=world_size,
        rank=rank,
    )

    # ── W&B: all ranks share a group so the UI clusters them together ─────────
    #
    # group     → one experiment name shared by every rank
    # job_type  → "worker" for all ranks; rank 0 could be "chief" if preferred
    # name      → per-run label makes individual rank runs easy to spot
    #
    run = wandb.init(
        project="distributed-training-demo",
        group=group_name,
        job_type="worker",
        name=f"rank-{rank}",
        config={
            "world_size": world_size,
            "rank": rank,
            "epochs": 3,
            "batch_size": 128,
            "lr": 1e-3,
        },
    )
    cfg = run.config

    # ── Data: each rank sees a disjoint shard via DistributedSampler ──────────
    train_ds, test_ds = load_tensors()
    sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, sampler=sampler)
    test_loader  = DataLoader(test_ds,  batch_size=256, shuffle=False)

    # ── Model + DDP wrapper ───────────────────────────────────────────────────
    device = torch.device("cpu")
    model = MLP().to(device)
    model = torch.nn.parallel.DistributedDataParallel(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(cfg.epochs):
        sampler.set_epoch(epoch)
        model.train()
        local_loss = 0.0
        local_correct = 0
        local_total = 0

        for x, y in train_loader:
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            local_loss += loss.item() * len(y)
            local_correct += (logits.argmax(1) == y).sum().item()
            local_total += len(y)

        local_avg_loss = local_loss / local_total
        local_acc = local_correct / local_total

        # ── Per-rank metrics: log from every worker ───────────────────────────
        # Prefix with rank so all workers can log the same key without conflict.
        run.log({
            f"rank_{rank}/train_loss": local_avg_loss,
            f"rank_{rank}/train_acc":  local_acc,
            "epoch": epoch + 1,
        })

        # ── Global metrics: aggregate across ranks, log only from rank 0 ──────
        # Use all_reduce to sum scalars, then divide by world_size on rank 0.
        loss_tensor = torch.tensor(local_avg_loss)
        acc_tensor  = torch.tensor(local_acc)
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(acc_tensor,  op=dist.ReduceOp.SUM)

        if rank == 0:
            run.log({
                "global/train_loss": (loss_tensor / world_size).item(),
                "global/train_acc":  (acc_tensor  / world_size).item(),
                "epoch": epoch + 1,
            })

    # ── Validation: run on rank 0 only to avoid duplicate logging ─────────────
    if rank == 0:
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for x, y in test_loader:
                preds = model(x).argmax(1)
                correct += (preds == y).sum().item()
                total += len(y)
        val_acc = correct / total
        run.summary["val_accuracy"] = val_acc
        print(f"[rank 0] val accuracy: {val_acc:.4f}")

    dist.destroy_process_group()
    run.finish()


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    world_size = int(os.environ.get("WORLD_SIZE", 2))
    rank       = int(os.environ.get("RANK", -1))
    group_name = os.environ.get("WANDB_RUN_GROUP", f"dist-run-{wandb.util.generate_id()}")

    if rank >= 0:
        # Launched by torchrun — each process calls this directly
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29500")
        train_worker(rank, world_size, group_name)
    else:
        # Simulate locally: spawn `world_size` processes via multiprocessing
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = "29500"
        mp.spawn(
            fn=lambda rank: train_worker(rank, world_size, group_name),
            nprocs=world_size,
            join=True,
        )
        print(f"\nAll runs logged under W&B group: {group_name}")


if __name__ == "__main__":
    main()
