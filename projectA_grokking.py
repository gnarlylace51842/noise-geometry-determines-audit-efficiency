"""
projectA_run_long_adamw.py

Copy-paste this entire file and run:

    python3 projectA_run_long_adamw.py

What it does:
- Modular addition task: y = (a + b) mod p, with p=97
- Train set: random pairs (size n_train)
- Test set: exhaustive grid of all p^2 pairs (9409), stable + paper-ready
- One-hot input: onehot(a) || onehot(b) -> MLP -> 97-way classification
- Uses Apple MPS if available, forces float32
- Runs:
  (1) sanity checks
  (2) overfit-one-batch test (proves gradients/updates are OK)
  (3) LONG AdamW run with weight_decay=2e-2 for 200k epochs

Saves:
- runs_projectA_long/adamw_none_eta0.0_seed0_history.json
- runs_projectA_long/adamw_none_eta0.0_seed0_summary.json

After baseline works (or at least runs), you can uncomment noise runs at the bottom.
"""

import os
import json
import time
import random
from typing import List, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# -----------------------------
# Repro + Device
# -----------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# -----------------------------
# Data
# -----------------------------
def make_modadd_data_exhaustive_test(p: int, n_train: int, seed: int):
    rng = np.random.default_rng(seed)

    train_pairs = rng.integers(0, p, size=(n_train, 2), dtype=np.int64)
    train_labels = (train_pairs[:, 0] + train_pairs[:, 1]) % p

    a = np.arange(p, dtype=np.int64)
    grid_a, grid_b = np.meshgrid(a, a, indexing="ij")
    test_pairs = np.stack([grid_a.reshape(-1), grid_b.reshape(-1)], axis=1).astype(np.int64)
    test_labels = (test_pairs[:, 0] + test_pairs[:, 1]) % p

    return train_pairs, train_labels, test_pairs, test_labels


class ModAddDatasetOneHot(Dataset):
    def __init__(self, p: int, pairs: np.ndarray, labels: np.ndarray):
        self.p = int(p)
        self.pairs = pairs.astype(np.int64)
        self.labels = labels.astype(np.int64)

    def __len__(self):
        return int(self.labels.shape[0])

    def __getitem__(self, idx: int):
        a, b = self.pairs[idx]
        x = np.zeros(2 * self.p, dtype=np.float32)
        x[a] = 1.0
        x[self.p + b] = 1.0
        y = int(self.labels[idx])
        return torch.from_numpy(x), torch.tensor(y, dtype=torch.long)


# -----------------------------
# Noise (kept here for later; baseline uses none)
# -----------------------------
def corrupt_uniform(labels: np.ndarray, p: int, eta: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    y = labels.copy()
    n = len(y)
    k = int(round(eta * n))
    if k <= 0:
        return y
    idx = rng.choice(n, size=k, replace=False)
    for i in idx:
        true = int(y[i])
        new = int(rng.integers(0, p - 1))
        if new >= true:
            new += 1
        y[i] = new
    return y

def corrupt_class_conditional(labels: np.ndarray, p: int, eta: float, target_classes: List[int], boost: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    y = labels.copy()
    n = len(y)
    k = int(round(eta * n))
    if k <= 0:
        return y

    weights = np.ones(n, dtype=np.float64)
    mask = np.isin(y, np.array(target_classes, dtype=np.int64))
    weights[mask] *= float(boost)
    weights /= weights.sum()

    idx = rng.choice(n, size=k, replace=False, p=weights)
    for i in idx:
        true = int(y[i])
        new = int(rng.integers(0, p - 1))
        if new >= true:
            new += 1
        y[i] = new
    return y

def corrupt_input_dependent(pairs: np.ndarray, labels: np.ndarray, p: int, eta: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    y = labels.copy()
    n = len(y)
    k = int(round(eta * n))
    if k <= 0:
        return y

    a_vals = pairs[:, 0]
    region_idx = np.where(a_vals < (p // 4))[0]

    if len(region_idx) >= k:
        idx = rng.choice(region_idx, size=k, replace=False)
    else:
        chosen = list(region_idx)
        remaining = k - len(chosen)
        other = np.setdiff1d(np.arange(n), region_idx)
        chosen += list(rng.choice(other, size=remaining, replace=False))
        idx = np.array(chosen, dtype=np.int64)

    for i in idx:
        true = int(y[i])
        new = int(rng.integers(0, p - 1))
        if new >= true:
            new += 1
        y[i] = new
    return y


# -----------------------------
# Model
# -----------------------------
class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int, out_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.relu(self.fc1(x)))


# -----------------------------
# Helpers
# -----------------------------
def sanity_checks(p: int, train_pairs, train_labels, test_pairs, test_labels):
    print("\n=== SANITY CHECKS ===")
    print("p =", p)
    print("train_pairs shape:", train_pairs.shape, "test_pairs shape:", test_pairs.shape)
    print("train label min/max:", int(train_labels.min()), int(train_labels.max()))
    print("test  label min/max:", int(test_labels.min()), int(test_labels.max()))
    print("unique train labels:", int(len(np.unique(train_labels))))
    print("unique test  labels:", int(len(np.unique(test_labels))))
    vals, counts = np.unique(test_labels, return_counts=True)
    maj = int(vals[counts.argmax()])
    maj_acc = float((test_labels == maj).mean())
    print("majority class:", maj, "majority baseline acc:", maj_acc)
    print("=== END SANITY CHECKS ===\n")


@torch.no_grad()
def eval_acc(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    for x, y in loader:
        x = x.to(device, dtype=torch.float32)
        y = y.to(device)
        pred = model(x).argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.numel()
    return correct / total


@torch.no_grad()
def weight_norm(model: nn.Module) -> float:
    s = 0.0
    for p in model.parameters():
        s += p.float().norm().item()
    return float(s)


def overfit_one_batch(model: nn.Module, train_loader: DataLoader, device: torch.device):
    print("\n=== OVERFIT ONE BATCH TEST ===")
    model.train()
    x, y = next(iter(train_loader))
    x = x.to(device, dtype=torch.float32)
    y = y.to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=1e-2, weight_decay=0.0)
    checkpoints = {0, 10, 50, 100, 200, 500, 1000}

    for step in range(1500):
        opt.zero_grad(set_to_none=True)
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        opt.step()

        if step in checkpoints:
            with torch.no_grad():
                pred = logits.argmax(dim=1)
                acc = (pred == y).float().mean().item()
            print(f"step={step:4d} loss={loss.item():.4f} acc={acc:.3f} wnorm={weight_norm(model):.2f}")
            if acc > 0.99:
                print("PASS: model can overfit a batch.")
                print("=== END OVERFIT TEST ===\n")
                return

    print("WARNING: did not reach ~100% batch accuracy in overfit test.")
    print("=== END OVERFIT TEST ===\n")


def compute_metrics(history: Dict[str, List[float]], thr: float = 0.95, consec: int = 5):
    epochs = np.array(history["epoch"], dtype=np.int64)
    tr = np.array(history["train_acc"], dtype=np.float64)
    te = np.array(history["test_acc"], dtype=np.float64)

    tg = None
    for i in range(0, len(te) - consec + 1):
        if np.all(te[i:i + consec] >= thr):
            tg = int(epochs[i])
            break
    gap_area = float(np.sum(np.maximum(0.0, tr - te)))
    return tg, gap_area


# -----------------------------
# Main run
# -----------------------------
def run_one_long_adamw(
    out_dir: str,
    seed: int,
    noise_type: str,
    eta: float,
    p: int,
    n_train: int,
    hidden: int,
    batch_size: int,
    epochs: int,
    eval_every: int,
    lr: float,
    weight_decay: float,
):
    os.makedirs(out_dir, exist_ok=True)
    set_seed(seed)
    device = get_device()
    print("Device:", device)

    train_pairs, train_labels, test_pairs, test_labels = make_modadd_data_exhaustive_test(p=p, n_train=n_train, seed=seed)
    sanity_checks(p, train_pairs, train_labels, test_pairs, test_labels)

    # Apply corruption only to training labels (baseline uses "none")
    if noise_type == "none":
        y_train = train_labels
    elif noise_type == "uniform":
        y_train = corrupt_uniform(train_labels, p=p, eta=eta, seed=seed + 123)
    elif noise_type == "class_cond":
        y_train = corrupt_class_conditional(train_labels, p=p, eta=eta, target_classes=list(range(10)), boost=3.0, seed=seed + 123)
    elif noise_type == "input_dep":
        y_train = corrupt_input_dependent(train_pairs, train_labels, p=p, eta=eta, seed=seed + 123)
    else:
        raise ValueError("Unknown noise_type")

    train_ds = ModAddDatasetOneHot(p, train_pairs, y_train)
    test_ds  = ModAddDatasetOneHot(p, test_pairs, test_labels)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False, drop_last=False)

    model = MLP(in_dim=2 * p, hidden=hidden, out_dim=p).to(device)

    # Logits shape sanity
    with torch.no_grad():
        xb, yb = next(iter(train_loader))
        xb = xb.to(device, dtype=torch.float32)
        print("logits shape:", tuple(model(xb).shape), "(should be [batch, p] = [*,", p, "])")

    # Overfit test (prove updates work)
    overfit_one_batch(model, train_loader, device)

    # Re-init model for fair training
    model = MLP(in_dim=2 * p, hidden=hidden, out_dim=p).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    history = {"epoch": [], "loss": [], "train_acc": [], "test_acc": [], "weight_norm": []}
    running_loss = 0.0
    num_batches = 0
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        for x, y in train_loader:
            x = x.to(device, dtype=torch.float32)
            y = y.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            opt.step()

            running_loss += loss.item()
            num_batches += 1

        if epoch == 1 or epoch % eval_every == 0:
            avg_loss = running_loss / max(1, num_batches)
            running_loss = 0.0
            num_batches = 0

            tr = eval_acc(model, train_loader, device)
            te = eval_acc(model, test_loader, device)
            wn = weight_norm(model)

            history["epoch"].append(epoch)
            history["loss"].append(float(avg_loss))
            history["train_acc"].append(float(tr))
            history["test_acc"].append(float(te))
            history["weight_norm"].append(float(wn))

            print(f"epoch={epoch:7d} loss={avg_loss:.4f} train={tr:.3f} test={te:.3f} wnorm={wn:.2f}")

    tg, gap_area = compute_metrics(history, thr=0.95, consec=5)

    summary = {
        "seed": seed,
        "noise_type": noise_type,
        "eta": float(eta),
        "p": p,
        "n_train": n_train,
        "hidden": hidden,
        "batch_size": batch_size,
        "epochs": epochs,
        "eval_every": eval_every,
        "optimizer": "AdamW",
        "lr": lr,
        "weight_decay": weight_decay,
        "time_to_grok": tg,
        "gap_area": gap_area,
        "wall_time_sec": time.time() - t0,
    }

    tag = f"adamw_{noise_type}_eta{eta}_seed{seed}"
    with open(os.path.join(out_dir, f"{tag}_history.json"), "w") as f:
        json.dump(history, f)
    with open(os.path.join(out_dir, f"{tag}_summary.json"), "w") as f:
        json.dump(summary, f)

    print("Saved:", tag, "time_to_grok:", tg, "gap_area:", gap_area)
    return summary


if __name__ == "__main__":
    # ---------- BASELINE LONG RUN CONFIG ----------
    OUT_DIR = "runs_projectA_long"
    P = 97
    SEED = 0

    # Start from the regime you already know learns:
    N_TRAIN = 5000
    HIDDEN = 256
    BATCH_SIZE = 512

    # The changes you requested:
    EPOCHS = 200000
    EVAL_EVERY = 2000
    LR = 1e-3
    WEIGHT_DECAY = 2e-2  # slightly stronger than before

    # Baseline first
    RUNS = [
        {"noise_type": "none", "eta": 0.0},

        # Uncomment AFTER baseline finishes:
        # {"noise_type": "uniform",   "eta": 0.10},
        # {"noise_type": "class_cond","eta": 0.10},
        # {"noise_type": "input_dep","eta": 0.10},
    ]

    for r in RUNS:
        run_one_long_adamw(
            out_dir=OUT_DIR,
            seed=SEED,
            noise_type=r["noise_type"],
            eta=float(r["eta"]),
            p=P,
            n_train=N_TRAIN,
            hidden=HIDDEN,
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            eval_every=EVAL_EVERY,
            lr=LR,
            weight_decay=WEIGHT_DECAY,
        )