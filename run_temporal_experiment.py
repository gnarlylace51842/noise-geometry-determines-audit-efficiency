"""
run_temporal_experiment.py — Temporal / path-dependence experiments for grokking.

Tests whether grokking recovery depends on optimization history, not just the
final state of the training dataset. This addresses the hysteresis question:
can two models trained on identical final datasets grok differently depending
on the order in which noise was introduced and corrected?

CONDITIONS:
  clean  — baseline: clean labels throughout (reference for grokking timing)
  A      — noisy labels throughout, no correction (baseline for suppression)
  B      — clean labels → inject noise at transition_epoch, continue training
           Question: does grokking persist despite later corruption?
  C      — noisy labels → correct ALL corruptions at transition_epoch, continue
           Question: can grokking recover without restarting optimization?
  D      — noisy labels → correct ALL corruptions at transition_epoch, reinitialize
           Question: is recovery dependent on removing optimization history?

KEY COMPARISON:
  C vs D: identical corrected datasets, different optimization history.
  If C ≠ D, this is evidence for hysteresis / path-dependence in grokking.

Usage:
  # Smoke test (seed 0 only, input_dep, all conditions):
  python run_temporal_experiment.py --conditions clean A B C D \\
    --transition-epoch 50000 --eta 0.04 --seeds 0 --out-dir runs_temporal_smoke

  # Full experiment (seeds 0-2, both noise types):
  python run_temporal_experiment.py --conditions clean A B C D \\
    --transition-epoch 50000 --eta 0.04 --seeds 0 1 2 \\
    --noise-types input_dep uniform --out-dir runs_temporal_full
"""

import argparse
import csv
import json
import os
import random
import time
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


# ─────────────────────────────────────────────────────────────────────────────
# Utilities  (shared with run_audit_experiment.py by copy — keeps files standalone)
# ─────────────────────────────────────────────────────────────────────────────

def configure_torch() -> None:
    torch.set_default_dtype(torch.float32)
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def fmt_eta(eta: float) -> str:
    s = f"{eta:.4f}".rstrip("0").rstrip(".")
    return s if s else "0"


# ─────────────────────────────────────────────────────────────────────────────
# Data
# ─────────────────────────────────────────────────────────────────────────────

def make_modadd_data(
    p: int, n_train: int, seed: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    train_pairs = rng.integers(0, p, size=(n_train, 2), dtype=np.int64)
    train_labels = (train_pairs[:, 0] + train_pairs[:, 1]) % p
    a = np.arange(p, dtype=np.int64)
    ga, gb = np.meshgrid(a, a, indexing="ij")
    test_pairs = np.stack([ga.ravel(), gb.ravel()], axis=1).astype(np.int64)
    test_labels = (test_pairs[:, 0] + test_pairs[:, 1]) % p
    return train_pairs, train_labels, test_pairs, test_labels


class ModAddDataset(Dataset):
    def __init__(self, p: int, pairs: np.ndarray, labels: np.ndarray):
        n = len(labels)
        x = np.zeros((n, 2 * p), dtype=np.float32)
        x[np.arange(n), pairs[:, 0]] = 1.0
        x[np.arange(n), p + pairs[:, 1]] = 1.0
        self.features = torch.from_numpy(x)
        self.labels = torch.from_numpy(labels.astype(np.int64))

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], self.labels[idx]


# ─────────────────────────────────────────────────────────────────────────────
# Noise
# ─────────────────────────────────────────────────────────────────────────────

def _rand_wrong(rng: np.random.Generator, true: int, p: int) -> int:
    v = int(rng.integers(0, p - 1))
    return v + 1 if v >= true else v


def corrupt_uniform(
    labels: np.ndarray, p: int, eta: float, seed: int
) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    y = labels.copy()
    k = int(round(eta * len(y)))
    mask = np.zeros(len(y), dtype=bool)
    if k <= 0:
        return y, mask
    idx = rng.choice(len(y), size=k, replace=False)
    for i in idx:
        y[i] = _rand_wrong(rng, int(labels[i]), p)
    mask[idx] = True
    return y, mask


def corrupt_input_dep(
    pairs: np.ndarray, labels: np.ndarray, p: int, eta: float, seed: int
) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    y = labels.copy()
    k = int(round(eta * len(y)))
    mask = np.zeros(len(y), dtype=bool)
    if k <= 0:
        return y, mask
    region = np.where(pairs[:, 0] < (p // 4))[0]
    if len(region) >= k:
        idx = rng.choice(region, size=k, replace=False)
    else:
        other = np.setdiff1d(np.arange(len(y)), region)
        idx = np.concatenate([region, rng.choice(other, size=k - len(region), replace=False)])
    for i in idx:
        y[i] = _rand_wrong(rng, int(labels[i]), p)
    mask[idx] = True
    return y, mask


def apply_noise(
    noise_type: str, pairs: np.ndarray, labels: np.ndarray, p: int, eta: float, seed: int
) -> Tuple[np.ndarray, np.ndarray]:
    if eta <= 0.0 or noise_type == "none":
        return labels.copy(), np.zeros(len(labels), dtype=bool)
    if noise_type == "uniform":
        return corrupt_uniform(labels, p, eta, seed)
    if noise_type == "input_dep":
        return corrupt_input_dep(pairs, labels, p, eta, seed)
    raise ValueError(f"Unknown noise_type: {noise_type!r}")


# ─────────────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────────────

class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int, out_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.relu(self.fc1(x)))


def make_model_and_opt(p: int, hidden: int, lr: float, wd: float, device: torch.device, seed: int):
    set_seed(seed)
    model = MLP(2 * p, hidden, p).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    return model, opt


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def eval_acc(model: nn.Module, dataset: Dataset, device: torch.device, batch_size: int) -> float:
    model.eval()
    correct = total = 0
    for x, y in DataLoader(dataset, batch_size=batch_size, shuffle=False):
        x, y = x.to(device), y.to(device)
        correct += (model(x).argmax(1) == y).sum().item()
        total += y.numel()
    return correct / total


def time_to_thresh(
    epochs: List[int], test_acc: List[float], thr: float, consec: int = 1
) -> Optional[int]:
    te = np.asarray(test_acc)
    ep = np.asarray(epochs)
    for i in range(len(te) - consec + 1):
        if np.all(te[i: i + consec] >= thr):
            return int(ep[i])
    return None


def gap_area(train_acc: List[float], test_acc: List[float]) -> float:
    return float(np.sum(np.maximum(0.0, np.asarray(train_acc) - np.asarray(test_acc))))


# ─────────────────────────────────────────────────────────────────────────────
# Config + tag
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TemporalConfig:
    out_dir: str
    seed: int
    condition: str         # "clean" | "A" | "B" | "C" | "D"
    noise_type: str        # "input_dep" | "uniform"
    eta: float
    transition_epoch: int  # B: noise injection epoch; C/D: label correction epoch
    p: int = 97
    n_train: int = 5000
    hidden: int = 256
    batch_size: int = 512
    epochs: int = 200_000
    eval_every: int = 2_000
    lr: float = 1e-3
    weight_decay: float = 2e-2
    grok_threshold: float = 0.95
    grok_consecutive: int = 5
    stop_on_grok: bool = True
    resume: bool = True
    force_restart: bool = False


def make_tag(c: TemporalConfig) -> str:
    return (
        f"temporal_{c.condition}_{c.noise_type}_eta{fmt_eta(c.eta)}"
        f"_t{c.transition_epoch}_seed{c.seed}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Persistence
# ─────────────────────────────────────────────────────────────────────────────

CSV_FIELDS = [
    "tag", "condition", "seed", "noise_type", "eta", "transition_epoch",
    "p", "n_train", "hidden", "batch_size",
    "epochs_ran", "epochs_planned", "eval_every", "lr", "weight_decay",
    "grok_threshold", "grok_consecutive",
    "num_corrupted",
    "time_to_grok", "time_to_0.80", "time_to_0.90",
    "final_test_acc", "final_train_acc", "gap_area",
    "phase1_final_test_acc", "phase1_grokked",
    "wall_time_sec", "stopped_early", "status",
]


def save_json(path: str, obj) -> None:
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def load_json(path: str) -> Dict:
    with open(path) as f:
        return json.load(f)


def upsert_csv(out_dir: str, row: Dict) -> None:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "summary.csv")
    rows: List[Dict] = []
    if os.path.exists(path):
        with open(path, newline="") as f:
            r = csv.DictReader(f)
            if list(r.fieldnames or []) == CSV_FIELDS:
                rows = list(r)
    flat = {k: row.get(k) for k in CSV_FIELDS}
    replaced = False
    for i, existing in enumerate(rows):
        if existing.get("tag") == row.get("tag"):
            rows[i] = flat
            replaced = True
            break
    if not replaced:
        rows.append(flat)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        w.writerows(rows)


def build_summary(
    c: TemporalConfig,
    history: Dict,
    num_corrupted: int,
    epochs_ran: int,
    wall: float,
    stopped_early: bool,
    status: str,
    phase1_final_test: Optional[float],
    phase1_grokked: bool,
) -> Dict:
    ep = history["epoch"]
    tr = history["train_acc"]
    te = history["test_acc"]
    return {
        "tag": make_tag(c),
        "condition": c.condition,
        "seed": c.seed,
        "noise_type": c.noise_type,
        "eta": c.eta,
        "transition_epoch": c.transition_epoch,
        "p": c.p,
        "n_train": c.n_train,
        "hidden": c.hidden,
        "batch_size": c.batch_size,
        "epochs_ran": epochs_ran,
        "epochs_planned": c.epochs,
        "eval_every": c.eval_every,
        "lr": c.lr,
        "weight_decay": c.weight_decay,
        "grok_threshold": c.grok_threshold,
        "grok_consecutive": c.grok_consecutive,
        "num_corrupted": num_corrupted,
        "time_to_grok": time_to_thresh(ep, te, c.grok_threshold, c.grok_consecutive),
        "time_to_0.80": time_to_thresh(ep, te, 0.80, 1),
        "time_to_0.90": time_to_thresh(ep, te, 0.90, 1),
        "final_test_acc": te[-1] if te else None,
        "final_train_acc": tr[-1] if tr else None,
        "gap_area": gap_area(tr, te) if tr and te else None,
        "phase1_final_test_acc": phase1_final_test,
        "phase1_grokked": phase1_grokked,
        "wall_time_sec": wall,
        "stopped_early": stopped_early,
        "status": status,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main training run
# ─────────────────────────────────────────────────────────────────────────────

def run_one(c: TemporalConfig) -> Dict:
    configure_torch()
    os.makedirs(c.out_dir, exist_ok=True)
    tag = make_tag(c)
    h_path = os.path.join(c.out_dir, f"{tag}_history.json")
    s_path = os.path.join(c.out_dir, f"{tag}_summary.json")

    if c.resume and not c.force_restart and os.path.exists(s_path):
        s = load_json(s_path)
        if s.get("status") == "completed":
            upsert_csv(c.out_dir, s)
            print(f"Skip (completed): {tag}")
            return s

    device = get_device()
    print(f"\n=== {tag}  device={device} ===")
    t_start = time.time()

    train_pairs, true_labels, test_pairs, test_labels = make_modadd_data(c.p, c.n_train, c.seed)
    noisy_labels, corrupted_mask = apply_noise(c.noise_type, train_pairs, true_labels, c.p, c.eta, seed=c.seed + 123)
    num_corrupted = int(corrupted_mask.sum())

    test_ds = ModAddDataset(c.p, test_pairs, test_labels)

    # ── Dataset assignment per condition ──────────────────────────────────────
    #
    # clean / A: single dataset throughout (no transition)
    # B:  phase1=clean,  phase2=noisy   (transition_epoch = noise injection point)
    # C:  phase1=noisy,  phase2=clean   (transition_epoch = correction point, keep opt)
    # D:  phase1=noisy,  phase2=clean   (transition_epoch = correction point, reinit)
    #
    if c.condition == "clean":
        phase1_ds = ModAddDataset(c.p, train_pairs, true_labels)
        phase2_ds = None
    elif c.condition == "A":
        phase1_ds = ModAddDataset(c.p, train_pairs, noisy_labels)
        phase2_ds = None
    elif c.condition == "B":
        phase1_ds = ModAddDataset(c.p, train_pairs, true_labels)
        phase2_ds = ModAddDataset(c.p, train_pairs, noisy_labels)
    elif c.condition in ("C", "D"):
        phase1_ds = ModAddDataset(c.p, train_pairs, noisy_labels)
        phase2_ds = ModAddDataset(c.p, train_pairs, true_labels)
    else:
        raise ValueError(f"Unknown condition: {c.condition!r}")

    model, opt = make_model_and_opt(c.p, c.hidden, c.lr, c.weight_decay, device, c.seed)
    history: Dict = {"epoch": [], "loss": [], "train_acc": [], "test_acc": [], "phase": []}

    stopped_early = False
    phase1_final_test: Optional[float] = None
    phase1_grokked = False
    final_epoch = 0
    running_loss = 0.0
    num_batches = 0

    for epoch in range(1, c.epochs + 1):
        # ── Phase transition ──────────────────────────────────────────────────
        in_phase2 = (phase2_ds is not None) and (epoch > c.transition_epoch)

        if phase2_ds is not None and epoch == c.transition_epoch + 1:
            # Record end-of-phase-1 state
            phase1_final_test = history["test_acc"][-1] if history["test_acc"] else None
            tg1 = time_to_thresh(history["epoch"], history["test_acc"], c.grok_threshold, c.grok_consecutive)
            phase1_grokked = tg1 is not None
            if c.condition == "B":
                print(f"  [epoch {epoch}] Injecting noise (condition B). Phase-1 grokked={phase1_grokked}")
            else:
                print(f"  [epoch {epoch}] Correcting all labels (condition {c.condition}). Phase-1 grokked={phase1_grokked}")
            if c.condition == "D":
                # Reinitialize model and optimizer — discard all optimization history
                model, opt = make_model_and_opt(c.p, c.hidden, c.lr, c.weight_decay, device, c.seed + 77777)
                print(f"  [condition D] Model and optimizer reinitialized.")

        train_ds = phase2_ds if in_phase2 else phase1_ds
        current_phase = 2 if in_phase2 else 1

        g = torch.Generator()
        g.manual_seed(c.seed * 1_000_003 + epoch)
        train_loader = DataLoader(train_ds, batch_size=c.batch_size, shuffle=True, generator=g)

        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(x), y)
            loss.backward()
            opt.step()
            running_loss += loss.item()
            num_batches += 1

        should_eval = (epoch == 1) or (epoch % c.eval_every == 0) or (epoch == c.epochs)
        if not should_eval:
            continue

        avg_loss = running_loss / max(1, num_batches)
        running_loss = 0.0
        num_batches = 0

        tr_acc = eval_acc(model, train_ds, device, c.batch_size)
        te_acc = eval_acc(model, test_ds, device, c.batch_size)

        history["epoch"].append(int(epoch))
        history["loss"].append(float(avg_loss))
        history["train_acc"].append(float(tr_acc))
        history["test_acc"].append(float(te_acc))
        history["phase"].append(int(current_phase))
        final_epoch = epoch

        print(f"  epoch={epoch:7d}  loss={avg_loss:.4f}  train={tr_acc:.3f}  test={te_acc:.3f}  phase={current_phase}")

        save_json(h_path, history)

        tg = time_to_thresh(history["epoch"], history["test_acc"], c.grok_threshold, c.grok_consecutive)
        # For condition B: don't stop early until we're past the transition (noise injection)
        # epoch, otherwise stop_on_grok kills the run before noise is ever applied.
        past_transition = (phase2_ds is None) or (epoch > c.transition_epoch)
        if c.stop_on_grok and tg is not None and past_transition:
            stopped_early = True
            print(f"  Grokked at epoch {tg}. Stopping early.")
            break

    wall = time.time() - t_start
    summary = build_summary(
        c, history, num_corrupted, final_epoch, wall,
        stopped_early, "completed", phase1_final_test, phase1_grokked,
    )
    save_json(h_path, history)
    save_json(s_path, summary)
    upsert_csv(c.out_dir, summary)

    print(
        f"  Done: {tag}  time_to_grok={summary['time_to_grok']}"
        f"  final_test={summary['final_test_acc']:.3f}"
        f"  phase1_grokked={phase1_grokked}"
    )
    return summary


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Temporal path-dependence experiment for grokking.")
    p.add_argument("--out-dir", default="runs_temporal")
    p.add_argument("--p", type=int, default=97)
    p.add_argument("--n-train", type=int, default=5000)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--epochs", type=int, default=200_000)
    p.add_argument("--eval-every", type=int, default=2_000)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=2e-2)
    p.add_argument("--grok-threshold", type=float, default=0.95)
    p.add_argument("--grok-consecutive", type=int, default=5)
    p.add_argument("--no-stop-on-grok", action="store_false", dest="stop_on_grok")
    p.set_defaults(stop_on_grok=True)
    p.add_argument("--force-restart", action="store_true")
    p.add_argument("--no-resume", action="store_true")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--eta", type=float, default=0.04)
    p.add_argument("--transition-epoch", type=int, default=50_000,
                   help="B: epoch at which noise is injected. C/D: epoch at which labels are corrected.")
    p.add_argument(
        "--noise-types", nargs="+",
        choices=["input_dep", "uniform"],
        default=["input_dep", "uniform"],
    )
    p.add_argument(
        "--conditions", nargs="+",
        choices=["clean", "A", "B", "C", "D"],
        default=["clean", "A", "C", "D"],
        help="clean=clean baseline, A=noisy baseline, B=clean→noisy, C=noisy→clean(continue), D=noisy→clean(reinit)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    configure_torch()
    os.makedirs(args.out_dir, exist_ok=True)

    total = len(args.seeds) * len(args.noise_types) * len(args.conditions)
    print(f"Scheduled runs: {total}")
    print(f"Conditions: {args.conditions}")
    print(f"Transition epoch: {args.transition_epoch}")
    print(f"  B meaning: clean→noisy at epoch {args.transition_epoch}")
    print(f"  C/D meaning: noisy→clean (corrected) at epoch {args.transition_epoch}")

    completed = 0
    for seed in args.seeds:
        for noise_type in args.noise_types:
            for condition in args.conditions:
                c = TemporalConfig(
                    out_dir=args.out_dir,
                    seed=seed,
                    condition=condition,
                    noise_type=noise_type,
                    eta=args.eta,
                    transition_epoch=args.transition_epoch,
                    p=args.p,
                    n_train=args.n_train,
                    hidden=args.hidden,
                    batch_size=args.batch_size,
                    epochs=args.epochs,
                    eval_every=args.eval_every,
                    lr=args.lr,
                    weight_decay=args.weight_decay,
                    grok_threshold=args.grok_threshold,
                    grok_consecutive=args.grok_consecutive,
                    stop_on_grok=args.stop_on_grok,
                    resume=not args.no_resume,
                    force_restart=args.force_restart,
                )
                run_one(c)
                completed += 1
                print(f"Progress: {completed}/{total}")

    print(f"\nAll done. Summary: {os.path.join(args.out_dir, 'summary.csv')}")


if __name__ == "__main__":
    main()
