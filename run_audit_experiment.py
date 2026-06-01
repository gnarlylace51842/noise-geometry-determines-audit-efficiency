"""
run_audit_experiment.py  —  Realistic audit-budget grokking recovery experiment.

AUDIT SEMANTICS (the key fix vs run_relabeling_experiment.py):
  Policies select B examples from ALL n_train, not from the known-corrupted subset.
  This mirrors a real annotation-review workflow: you don't know which labels are wrong.

  After auditing B examples:
    num_corrected  = |audited ∩ corrupted|   — how many bad labels you found
    hit_rate       = num_corrected / B        — audit efficiency

POLICIES:
  none          — no audit (baseline)
  random_audit  — pick B uniformly at random from all n_train          (naive baseline)
  region_audit  — pick B from input region a < p//4                    (domain knowledge)
  loss_audit    — brief warm-up training, pick top-B highest-loss       (data-centric triage)

NOISE TYPES:
  input_dep  — all η·n corruptions concentrated in a < p//4  (structured geometry)
  uniform    — η·n corruptions spread uniformly               (control condition)

KEY QUANTITATIVE PREDICTIONS  (input_dep η=0.04, n_train=5000, p=97):
  region ≈ 1237 examples, all 200 corruptions inside it.
  random_audit hit_rate ≈ 200/5000 =  4 %
  region_audit hit_rate ≈ 200/1237 = 16 %   (4× better!)
  Under uniform noise: region_audit ≈ random_audit ≈ 4 % → validates geometry claim.

Usage:
  # Pilot — seed 0, input_dep + uniform, eta=0.04, all policies, budgets 0–800:
  python run_audit_experiment.py --out-dir runs_audit_pilot --seeds 0 --etas 0.04

  # With loss_audit:
  python run_audit_experiment.py --policies none random_audit region_audit loss_audit

  # Multi-seed full experiment:
  python run_audit_experiment.py --seeds 0 1 2 --etas 0.04 0.06 --out-dir runs_audit_full
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
# Utilities
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
    """Pre-computes one-hot features for fast iteration."""

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
# Noise generation  (returns noisy_labels + corrupted_mask)
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
    """All η·n corruptions concentrated in region a < p//4."""
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
    """Returns (noisy_labels, corrupted_mask)."""
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


# ─────────────────────────────────────────────────────────────────────────────
# Audit policies  ← REALISTIC: select from ALL n_train, not just corrupted set
# ─────────────────────────────────────────────────────────────────────────────

def audit_random(n_train: int, budget: int, seed: int) -> np.ndarray:
    """Uniform random audit: B examples chosen from all n_train (no domain knowledge)."""
    if budget <= 0:
        return np.empty(0, dtype=np.int64)
    rng = np.random.default_rng(seed)
    k = min(budget, n_train)
    return np.sort(rng.choice(n_train, size=k, replace=False).astype(np.int64))


def audit_region(pairs: np.ndarray, p: int, budget: int, seed: int) -> np.ndarray:
    """Region audit: inspect B examples, prioritizing the suspected high-error zone (a < p//4).

    Fills the budget from the region first; if B > region_size, fills remainder from
    outside the region uniformly. This ensures the budget is always respected exactly.

    Principled justification: an analyst notices elevated validation loss for small-a
    inputs, or a domain expert flags this region as prone to mislabeling.
    Under input_dep noise (all corruptions in region): hit_rate ≈ 4× random_audit.
    Under uniform noise: hit_rate ≈ random_audit → validates geometry claim.
    """
    if budget <= 0:
        return np.empty(0, dtype=np.int64)
    n_train = len(pairs)
    region = np.where(pairs[:, 0] < (p // 4))[0].astype(np.int64)
    rng = np.random.default_rng(seed)

    chosen: List[int] = []
    # Exhaust region first
    from_region = min(budget, len(region))
    if from_region > 0:
        chosen.extend(rng.choice(region, size=from_region, replace=False).tolist())
    # Fill remainder from non-region if budget > region_size
    remaining = budget - from_region
    if remaining > 0:
        non_region = np.setdiff1d(np.arange(n_train), region, assume_unique=True)
        fill = min(remaining, len(non_region))
        if fill > 0:
            chosen.extend(rng.choice(non_region, size=fill, replace=False).tolist())

    return np.sort(np.array(chosen, dtype=np.int64))


@torch.no_grad()
def _per_example_losses(
    model: nn.Module, dataset: Dataset, device: torch.device, batch_size: int
) -> np.ndarray:
    model.eval()
    parts: List[torch.Tensor] = []
    for x, y in DataLoader(dataset, batch_size=batch_size, shuffle=False):
        x, y = x.to(device), y.to(device)
        parts.append(F.cross_entropy(model(x), y, reduction="none").cpu())
    return torch.cat(parts).numpy()


@torch.no_grad()
def _per_example_probs(
    model: nn.Module, dataset: Dataset, device: torch.device, batch_size: int
) -> np.ndarray:
    """Returns softmax probabilities, shape (n, num_classes)."""
    model.eval()
    parts: List[torch.Tensor] = []
    for x, y in DataLoader(dataset, batch_size=batch_size, shuffle=False):
        x = x.to(device)
        parts.append(torch.softmax(model(x), dim=1).cpu())
    return torch.cat(parts).numpy()


def _compute_warmup_losses(
    pairs: np.ndarray,
    noisy_labels: np.ndarray,
    p: int,
    hidden: int,
    warmup_epochs: int,
    batch_size: int,
    lr: float,
    wd: float,
    device: torch.device,
    seed: int,
) -> np.ndarray:
    """Train a fresh model for warmup_epochs on noisy data; return per-example loss.
    RNG state is saved/restored so this does not affect the main run's randomness.
    """
    py_st = random.getstate()
    np_st = np.random.get_state()
    pt_st = torch.get_rng_state()
    try:
        set_seed(seed)
        ds = ModAddDataset(p, pairs, noisy_labels)
        model = MLP(2 * p, hidden, p).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
        for ep in range(warmup_epochs):
            g = torch.Generator()
            g.manual_seed(seed * 999_983 + ep)
            for x, y in DataLoader(ds, batch_size=batch_size, shuffle=True, generator=g):
                x, y = x.to(device), y.to(device)
                opt.zero_grad(set_to_none=True)
                F.cross_entropy(model(x), y).backward()
                opt.step()
        return _per_example_losses(model, ds, device, batch_size)
    finally:
        random.setstate(py_st)
        np.random.set_state(np_st)
        torch.set_rng_state(pt_st.cpu() if isinstance(pt_st, torch.Tensor) else pt_st)


def audit_loss(losses: np.ndarray, budget: int) -> np.ndarray:
    """Loss triage: top-B highest-loss examples from ALL training examples.

    Requires a warmup model trained briefly on noisy data. Purely data-driven;
    no domain knowledge required — the most 'realistic' policy for production ML.
    """
    if budget <= 0:
        return np.empty(0, dtype=np.int64)
    k = min(budget, len(losses))
    order = np.argsort(-losses, kind="mergesort")
    return np.sort(order[:k].astype(np.int64))


def _compute_warmup_with_forgetting(
    pairs: np.ndarray,
    noisy_labels: np.ndarray,
    p: int,
    hidden: int,
    warmup_epochs: int,
    batch_size: int,
    lr: float,
    wd: float,
    device: torch.device,
    seed: int,
    forgetting_eval_every: int = 500,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Train a warmup model; return (losses, entropy, forgetting_counts).

    forgetting_count[i] = number of evals where example i flipped correct→incorrect.
    RNG state is saved/restored so this does not perturb the main run.
    """
    py_st = random.getstate()
    np_st = np.random.get_state()
    pt_st = torch.get_rng_state()
    try:
        set_seed(seed)
        ds = ModAddDataset(p, pairs, noisy_labels)
        n = len(noisy_labels)
        model = MLP(2 * p, hidden, p).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

        forgetting_counts = np.zeros(n, dtype=np.int32)
        prev_correct: Optional[np.ndarray] = None

        for ep in range(warmup_epochs):
            g = torch.Generator()
            g.manual_seed(seed * 999_983 + ep)
            for x, y in DataLoader(ds, batch_size=batch_size, shuffle=True, generator=g):
                x, y = x.to(device), y.to(device)
                opt.zero_grad(set_to_none=True)
                F.cross_entropy(model(x), y).backward()
                opt.step()

            if (ep + 1) % forgetting_eval_every == 0 or ep == warmup_epochs - 1:
                with torch.no_grad():
                    parts_c: List[torch.Tensor] = []
                    for x, y in DataLoader(ds, batch_size=batch_size, shuffle=False):
                        x, y = x.to(device), y.to(device)
                        parts_c.append((model(x).argmax(1) == y).cpu())
                curr_correct = torch.cat(parts_c).numpy()
                if prev_correct is not None:
                    forgetting_counts += (prev_correct & ~curr_correct).astype(np.int32)
                prev_correct = curr_correct

        losses = _per_example_losses(model, ds, device, batch_size)
        probs = _per_example_probs(model, ds, device, batch_size)
        entropy = -np.sum(probs * np.log(probs + 1e-8), axis=1)
        return losses, entropy, forgetting_counts
    finally:
        random.setstate(py_st)
        np.random.set_state(np_st)
        torch.set_rng_state(pt_st.cpu() if isinstance(pt_st, torch.Tensor) else pt_st)


def audit_uncertainty(entropy: np.ndarray, budget: int) -> np.ndarray:
    """Uncertainty triage: top-B highest-entropy (most uncertain) examples.

    Uses softmax entropy from a warmup model. Unlike raw loss, entropy is bounded
    and invariant to the scale of the logits — a different inductive bias for finding
    corrupted examples.
    """
    if budget <= 0:
        return np.empty(0, dtype=np.int64)
    k = min(budget, len(entropy))
    return np.sort(np.argsort(-entropy, kind="mergesort")[:k].astype(np.int64))


def audit_forgetting(forgetting_counts: np.ndarray, budget: int) -> np.ndarray:
    """Forgetting triage: top-B examples by number of forgetting events during warmup.

    A forgetting event is when an example flips from correctly to incorrectly predicted
    across consecutive evaluations. Corrupted examples tend to be learned and forgotten
    repeatedly because they conflict with the true data distribution.
    """
    if budget <= 0:
        return np.empty(0, dtype=np.int64)
    k = min(budget, len(forgetting_counts))
    return np.sort(np.argsort(-forgetting_counts, kind="mergesort")[:k].astype(np.int64))


# ─────────────────────────────────────────────────────────────────────────────
# AuditResult
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class AuditResult:
    final_labels: np.ndarray       # labels the main model trains on
    corrupted_mask: np.ndarray     # ground truth: which were corrupted (for eval ONLY)
    audited_indices: np.ndarray    # the B examples we selected for inspection
    corrected_indices: np.ndarray  # audited ∩ corrupted — examples actually fixed

    @property
    def num_corrupted(self) -> int:
        return int(self.corrupted_mask.sum())

    @property
    def num_audited(self) -> int:
        return len(self.audited_indices)

    @property
    def num_corrected(self) -> int:
        return len(self.corrected_indices)

    @property
    def hit_rate(self) -> float:
        """Fraction of audited examples that were actually corrupted."""
        return self.num_corrected / max(1, self.num_audited)

    @property
    def correction_rate(self) -> float:
        """Fraction of total corruption that was fixed by this audit."""
        return self.num_corrected / max(1, self.num_corrupted)

    def stats(self) -> Dict:
        """Scalar stats for CSV / summary JSON."""
        return {
            "num_corrupted": self.num_corrupted,
            "num_audited": self.num_audited,
            "num_corrected": self.num_corrected,
            "hit_rate": round(self.hit_rate, 6),
            "correction_rate": round(self.correction_rate, 6),
        }

    def serialize(self) -> Dict:
        """Full serialization for checkpoint (includes index arrays)."""
        return {
            "final_labels": self.final_labels.tolist(),
            "corrupted_mask": self.corrupted_mask.tolist(),
            "audited_indices": self.audited_indices.tolist(),
            "corrected_indices": self.corrected_indices.tolist(),
        }

    @classmethod
    def deserialize(cls, d: Dict) -> "AuditResult":
        return cls(
            final_labels=np.array(d["final_labels"], dtype=np.int64),
            corrupted_mask=np.array(d["corrupted_mask"], dtype=bool),
            audited_indices=np.array(d["audited_indices"], dtype=np.int64),
            corrected_indices=np.array(d["corrected_indices"], dtype=np.int64),
        )


# ─────────────────────────────────────────────────────────────────────────────
# Prepare audit (apply noise → select B → correct → build AuditResult)
# ─────────────────────────────────────────────────────────────────────────────

def prepare_audit(
    policy: str,
    budget: int,
    pairs: np.ndarray,
    true_labels: np.ndarray,
    p: int,
    n_train: int,
    seed: int,
    hidden: int,
    warmup_epochs: int,
    batch_size: int,
    lr: float,
    wd: float,
    noise_type: str,
    eta: float,
    device: torch.device,
) -> AuditResult:
    noisy_labels, corrupted_mask = apply_noise(noise_type, pairs, true_labels, p, eta, seed=seed + 123)

    if policy == "none" or budget <= 0:
        audited = np.empty(0, dtype=np.int64)
    elif policy == "random_audit":
        audited = audit_random(n_train, budget, seed=seed + 456)
    elif policy == "region_audit":
        audited = audit_region(pairs, p, budget, seed=seed + 789)
    elif policy == "loss_audit":
        losses = _compute_warmup_losses(
            pairs, noisy_labels, p, hidden, warmup_epochs, batch_size, lr, wd, device, seed=seed + 20_000
        )
        audited = audit_loss(losses, budget)
    elif policy in ("uncertainty_audit", "forgetting_audit"):
        _, entropy, forgetting_counts = _compute_warmup_with_forgetting(
            pairs, noisy_labels, p, hidden, warmup_epochs, batch_size, lr, wd, device, seed=seed + 20_000
        )
        if policy == "uncertainty_audit":
            audited = audit_uncertainty(entropy, budget)
        else:
            audited = audit_forgetting(forgetting_counts, budget)
    else:
        raise ValueError(f"Unknown policy: {policy!r}")

    # Apply corrections (no-op for clean examples — only corrupted ones change)
    final_labels = noisy_labels.copy()
    if len(audited) > 0:
        final_labels[audited] = true_labels[audited]

    corrupted_idx = np.where(corrupted_mask)[0].astype(np.int64)
    corrected = np.intersect1d(audited, corrupted_idx)

    return AuditResult(
        final_labels=final_labels,
        corrupted_mask=corrupted_mask,
        audited_indices=audited,
        corrected_indices=corrected,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

def time_to_thresh(
    epochs: List[int], test_acc: List[float], thr: float, consec: int = 1
) -> Optional[int]:
    te = np.asarray(test_acc)
    ep = np.asarray(epochs)
    for i in range(len(te) - consec + 1):
        if np.all(te[i : i + consec] >= thr):
            return int(ep[i])
    return None


def gap_area(train_acc: List[float], test_acc: List[float]) -> float:
    return float(np.sum(np.maximum(0.0, np.asarray(train_acc) - np.asarray(test_acc))))


def auc_test(test_acc: List[float]) -> float:
    return float(np.sum(np.asarray(test_acc)))


# ─────────────────────────────────────────────────────────────────────────────
# RunConfig + tag
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RunConfig:
    out_dir: str
    seed: int
    noise_type: str   # "input_dep" | "uniform"
    eta: float
    policy: str       # "none" | "random_audit" | "region_audit" | "loss_audit"
    budget: int       # B (0 = no-audit baseline, same as policy "none")
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
    loss_warmup_epochs: int = 2_000
    stop_on_grok: bool = True
    resume: bool = True
    force_restart: bool = False


def make_tag(c: RunConfig) -> str:
    pol = "none" if (c.policy == "none" or c.budget <= 0) else f"{c.policy}_B{c.budget}"
    return f"audit_{c.noise_type}_eta{fmt_eta(c.eta)}_{pol}_seed{c.seed}"


def _cfg_key(c: RunConfig) -> Dict:
    return {k: getattr(c, k) for k in [
        "seed", "noise_type", "eta", "policy", "budget",
        "p", "n_train", "hidden", "batch_size", "epochs",
        "eval_every", "lr", "weight_decay",
        "grok_threshold", "grok_consecutive", "loss_warmup_epochs",
    ]}


# ─────────────────────────────────────────────────────────────────────────────
# Persistence helpers
# ─────────────────────────────────────────────────────────────────────────────

def save_json(path: str, obj: Dict) -> None:
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def load_json(path: str) -> Dict:
    with open(path) as f:
        return json.load(f)


def summary_matches(summary: Dict, c: RunConfig) -> bool:
    key = _cfg_key(c)
    return all(summary.get(k) == v for k, v in key.items())


def ckpt_matches(ckpt: Dict, c: RunConfig) -> bool:
    return all(ckpt.get("config_key", {}).get(k) == v for k, v in _cfg_key(c).items())


def save_checkpoint(
    path: str, c: RunConfig, model: nn.Module, opt: torch.optim.Optimizer,
    history: Dict, audit: AuditResult, last_epoch: int, elapsed: float,
) -> None:
    torch.save({
        "config_key": _cfg_key(c),
        "model_state": model.state_dict(),
        "optimizer_state": opt.state_dict(),
        "history": history,
        "audit": audit.serialize(),
        "last_epoch": int(last_epoch),
        "elapsed": float(elapsed),
        "py_rng": random.getstate(),
        "np_rng": np.random.get_state(),
        "pt_rng": torch.get_rng_state(),
    }, path)


def load_checkpoint(path: str, device: torch.device) -> Dict:
    return torch.load(path, map_location=device, weights_only=False)


# ─────────────────────────────────────────────────────────────────────────────
# CSV output
# ─────────────────────────────────────────────────────────────────────────────

CSV_FIELDS = [
    "tag", "seed", "noise_type", "eta", "policy", "budget",
    "p", "n_train", "hidden", "batch_size",
    "epochs_ran", "epochs_planned", "eval_every", "lr", "weight_decay",
    "grok_threshold", "grok_consecutive", "loss_warmup_epochs",
    "num_corrupted", "num_audited", "num_corrected",
    "hit_rate", "correction_rate",
    "time_to_grok", "time_to_0.80", "time_to_0.90",
    "final_test_acc", "final_train_acc", "gap_area", "auc_test",
    "wall_time_sec", "stopped_early", "status",
]


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
    c: RunConfig, history: Dict, audit: AuditResult,
    epochs_ran: int, wall: float, stopped_early: bool, status: str,
) -> Dict:
    ep = history["epoch"]
    tr = history["train_acc"]
    te = history["test_acc"]
    return {
        "tag": make_tag(c),
        "seed": c.seed,
        "noise_type": c.noise_type,
        "eta": c.eta,
        "policy": c.policy,
        "budget": c.budget,
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
        "loss_warmup_epochs": c.loss_warmup_epochs,
        **audit.stats(),
        "time_to_grok": time_to_thresh(ep, te, c.grok_threshold, c.grok_consecutive),
        "time_to_0.80": time_to_thresh(ep, te, 0.80, 1),
        "time_to_0.90": time_to_thresh(ep, te, 0.90, 1),
        "final_test_acc": te[-1] if te else None,
        "final_train_acc": tr[-1] if tr else None,
        "gap_area": gap_area(tr, te) if tr and te else None,
        "auc_test": auc_test(te) if te else None,
        "wall_time_sec": wall,
        "stopped_early": stopped_early,
        "status": status,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Eval helper
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


# ─────────────────────────────────────────────────────────────────────────────
# Main training run
# ─────────────────────────────────────────────────────────────────────────────

def run_one(c: RunConfig) -> Dict:
    configure_torch()
    os.makedirs(c.out_dir, exist_ok=True)
    tag = make_tag(c)
    h_path = os.path.join(c.out_dir, f"{tag}_history.json")
    s_path = os.path.join(c.out_dir, f"{tag}_summary.json")
    ck_path = os.path.join(c.out_dir, f"{tag}_checkpoint.pt")

    # Skip completed runs (resume mode)
    if c.resume and not c.force_restart and os.path.exists(s_path):
        s = load_json(s_path)
        if s.get("status") == "completed" and summary_matches(s, c):
            upsert_csv(c.out_dir, s)
            print(f"Skip (completed): {tag}")
            return s

    device = get_device()
    print(f"\n=== {tag}  device={device} ===")
    t_start = time.time()

    train_pairs, true_labels, test_pairs, test_labels = make_modadd_data(c.p, c.n_train, c.seed)

    # Load checkpoint (if resuming)
    ckpt = None
    if c.resume and not c.force_restart and os.path.exists(ck_path):
        ckpt = load_checkpoint(ck_path, device)
        if not ckpt_matches(ckpt, c):
            print("  Ignoring incompatible checkpoint.")
            ckpt = None

    # Prepare labels (restore from checkpoint or run audit policy)
    if ckpt is not None and "audit" in ckpt:
        audit = AuditResult.deserialize(ckpt["audit"])
    else:
        print(f"  Preparing audit: policy={c.policy} budget={c.budget} ...")
        audit = prepare_audit(
            policy=c.policy,
            budget=c.budget,
            pairs=train_pairs,
            true_labels=true_labels,
            p=c.p,
            n_train=c.n_train,
            seed=c.seed,
            hidden=c.hidden,
            warmup_epochs=c.loss_warmup_epochs,
            batch_size=c.batch_size,
            lr=c.lr,
            wd=c.weight_decay,
            noise_type=c.noise_type,
            eta=c.eta,
            device=device,
        )

    print(
        f"  Audit: corrupted={audit.num_corrupted}"
        f"  audited={audit.num_audited}"
        f"  corrected={audit.num_corrected}"
        f"  hit_rate={audit.hit_rate:.3f}"
        f"  correction_rate={audit.correction_rate:.3f}"
    )

    train_ds = ModAddDataset(c.p, train_pairs, audit.final_labels)
    test_ds  = ModAddDataset(c.p, test_pairs, test_labels)

    set_seed(c.seed)
    model = MLP(2 * c.p, c.hidden, c.p).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=c.lr, weight_decay=c.weight_decay)

    history: Dict = {"epoch": [], "loss": [], "train_acc": [], "test_acc": []}
    start_epoch = 1
    elapsed_before = 0.0
    stopped_early = False

    if ckpt is not None:
        model.load_state_dict(ckpt["model_state"])
        opt.load_state_dict(ckpt["optimizer_state"])
        history = ckpt["history"]
        start_epoch = int(ckpt["last_epoch"]) + 1
        elapsed_before = float(ckpt.get("elapsed", 0.0))
        if "py_rng" in ckpt:
            random.setstate(ckpt["py_rng"])
        if "np_rng" in ckpt:
            np.random.set_state(ckpt["np_rng"])
        if "pt_rng" in ckpt:
            torch.set_rng_state(ckpt["pt_rng"].cpu())
        print(f"  Resuming from epoch {start_epoch}")

        # Check if already grokked in restored history
        if c.stop_on_grok and history["epoch"]:
            tg = time_to_thresh(history["epoch"], history["test_acc"], c.grok_threshold, c.grok_consecutive)
            if tg is not None:
                summary = build_summary(c, history, audit, history["epoch"][-1], elapsed_before, True, "completed")
                save_json(h_path, history)
                save_json(s_path, summary)
                upsert_csv(c.out_dir, summary)
                print(f"  Already grokked at {tg}; skipping.")
                return summary

    running_loss = 0.0
    num_batches = 0
    final_epoch = history["epoch"][-1] if history["epoch"] else 0

    for epoch in range(start_epoch, c.epochs + 1):
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
        te_acc = eval_acc(model, test_ds,  device, c.batch_size)

        history["epoch"].append(int(epoch))
        history["loss"].append(float(avg_loss))
        history["train_acc"].append(float(tr_acc))
        history["test_acc"].append(float(te_acc))
        final_epoch = epoch

        print(f"  epoch={epoch:7d}  loss={avg_loss:.4f}  train={tr_acc:.3f}  test={te_acc:.3f}")

        elapsed_total = elapsed_before + (time.time() - t_start)
        save_json(h_path, history)
        save_checkpoint(ck_path, c, model, opt, history, audit, epoch, elapsed_total)

        tg = time_to_thresh(history["epoch"], history["test_acc"], c.grok_threshold, c.grok_consecutive)
        if c.stop_on_grok and tg is not None:
            stopped_early = True
            print(f"  Grokked at epoch {tg}. Stopping early.")
            break

    wall = elapsed_before + (time.time() - t_start)
    summary = build_summary(c, history, audit, final_epoch, wall, stopped_early, "completed")
    save_json(h_path, history)
    save_json(s_path, summary)
    upsert_csv(c.out_dir, summary)

    print(
        f"  Done: {tag}  time_to_grok={summary['time_to_grok']}"
        f"  final_test={summary['final_test_acc']:.3f}"
        f"  hit_rate={audit.hit_rate:.3f}"
        f"  corrected={audit.num_corrected}/{audit.num_corrupted}"
    )
    return summary


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Audit-budget grokking recovery experiment.")
    p.add_argument("--out-dir", default="runs_audit_experiment")
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
    p.add_argument("--loss-warmup-epochs", type=int, default=2_000)
    p.add_argument("--no-stop-on-grok", action="store_false", dest="stop_on_grok")
    p.set_defaults(stop_on_grok=True)
    p.add_argument("--force-restart", action="store_true")
    p.add_argument("--no-resume", action="store_true")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--etas", type=float, nargs="+", default=[0.04])
    p.add_argument("--budgets", type=int, nargs="+", default=[0, 100, 200, 400, 800])
    p.add_argument(
        "--noise-types", nargs="+",
        choices=["input_dep", "uniform"],
        default=["input_dep", "uniform"],
    )
    p.add_argument(
        "--policies", nargs="+",
        choices=["none", "random_audit", "region_audit", "loss_audit", "uncertainty_audit", "forgetting_audit"],
        default=["none", "random_audit", "region_audit"],
    )
    return p.parse_args()


def resolve_policy_budget_pairs(policies: List[str], budgets: List[int]) -> List[Tuple[str, int]]:
    pairs: List[Tuple[str, int]] = []
    positive = [b for b in budgets if b > 0]
    for pol in policies:
        if pol == "none":
            pairs.append(("none", 0))
        else:
            for b in positive:
                pairs.append((pol, b))
    return pairs


def main() -> None:
    args = parse_args()
    configure_torch()
    os.makedirs(args.out_dir, exist_ok=True)

    policy_budget_pairs = resolve_policy_budget_pairs(args.policies, args.budgets)
    total = len(args.seeds) * len(args.etas) * len(args.noise_types) * len(policy_budget_pairs)
    print(f"Scheduled runs: {total}")

    completed = 0
    for seed in args.seeds:
        for eta in args.etas:
            for noise_type in args.noise_types:
                for policy, budget in policy_budget_pairs:
                    c = RunConfig(
                        out_dir=args.out_dir,
                        seed=int(seed),
                        noise_type=noise_type,
                        eta=float(eta),
                        policy=policy,
                        budget=int(budget),
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
                        loss_warmup_epochs=args.loss_warmup_epochs,
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
