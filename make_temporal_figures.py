"""
make_temporal_figures.py — Figures for temporal path-dependence experiments.

Usage:
  python3 make_temporal_figures.py --in-dir runs_temporal_full --out-dir figures_temporal
"""

import argparse
import json
import os
import sys
from typing import Dict, List, Optional

import numpy as np

try:
    import pandas as pd
except ImportError:
    sys.exit("pandas required: pip install pandas")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
except ImportError:
    sys.exit("matplotlib required: pip install matplotlib")

try:
    import seaborn as sns
    sns.set_theme(style="whitegrid", font_scale=1.1)
except ImportError:
    plt.style.use("seaborn-v0_8-whitegrid")


# ─────────────────────────────────────────────────────────────────────────────
# Style
# ─────────────────────────────────────────────────────────────────────────────

COND_COLORS = {
    "clean": "#2196F3",
    "A":     "#555555",
    "B":     "#FF9800",
    "C":     "#4CAF50",
    "D":     "#E91E63",
}
COND_LABELS = {
    "clean": "Clean baseline",
    "A":     "A: Noisy throughout",
    "B":     "B: Clean → noisy at 50k",
    "C":     "C: Noisy → clean at 50k (continue)",
    "D":     "D: Noisy → clean at 50k (reinitialize)",
}
COND_LS = {
    "clean": "-",
    "A":     "--",
    "B":     "-.",
    "C":     "-",
    "D":     ":",
}
NOISE_LABELS = {
    "input_dep": "Input-dep. noise",
    "uniform":   "Uniform noise",
}


# ─────────────────────────────────────────────────────────────────────────────
# I/O helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_summary(in_dir: str) -> pd.DataFrame:
    path = os.path.join(in_dir, "summary.csv")
    if not os.path.exists(path):
        sys.exit(f"summary.csv not found in {in_dir}")
    df = pd.read_csv(path)
    for col in ["time_to_grok", "time_to_0.80", "time_to_0.90",
                "final_test_acc", "final_train_acc", "wall_time_sec",
                "transition_epoch", "num_corrupted"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def load_history(in_dir: str, tag: str) -> Optional[Dict]:
    path = os.path.join(in_dir, f"{tag}_history.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def savefig(fig, out_dir: str, name: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 1 — Learning curves: test acc vs epoch for all conditions
# ─────────────────────────────────────────────────────────────────────────────

def fig_learning_curves(df: pd.DataFrame, in_dir: str, out_dir: str,
                         noise_type: str = "input_dep", seed: int = 0) -> None:
    subset = df[(df["noise_type"] == noise_type) & (df["seed"] == seed)]
    if subset.empty:
        print(f"fig1: no data for noise_type={noise_type} seed={seed}")
        return

    conditions = [c for c in ["clean", "A", "B", "C", "D"] if c in subset["condition"].values]
    fig, ax = plt.subplots(figsize=(10, 5))

    for cond in conditions:
        row = subset[subset["condition"] == cond].iloc[0]
        hist = load_history(in_dir, row["tag"])
        if hist is None:
            continue
        epochs = np.array(hist["epoch"]) / 1000
        test_acc = np.array(hist["test_acc"])

        ax.plot(epochs, test_acc,
                color=COND_COLORS.get(cond, "gray"),
                ls=COND_LS.get(cond, "-"),
                lw=2.2,
                label=COND_LABELS.get(cond, cond))

    # Mark transition epoch
    t_epoch = df["transition_epoch"].dropna().iloc[0] / 1000 if not df["transition_epoch"].dropna().empty else 50
    ax.axvline(t_epoch, color="gray", lw=1.2, ls=":", alpha=0.7, label=f"Transition (epoch {int(t_epoch)}k)")
    ax.axhline(0.95, color="red", lw=1, ls=":", alpha=0.6, label="Grok threshold (0.95)")

    ax.set_xlabel("Epoch (×1000)", fontsize=12)
    ax.set_ylabel("Test accuracy", fontsize=12)
    ax.set_ylim(-0.02, 1.05)
    ax.set_title(f"Temporal conditions — {NOISE_LABELS.get(noise_type, noise_type)}, seed {seed}", fontsize=12)
    ax.legend(fontsize=9, loc="upper left")
    fig.tight_layout()
    savefig(fig, out_dir, f"fig1_learning_curves_{noise_type}_seed{seed}.png")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2 — C vs D: time to grok across seeds and noise types
# ─────────────────────────────────────────────────────────────────────────────

def fig_hysteresis(df: pd.DataFrame, out_dir: str) -> None:
    subset = df[df["condition"].isin(["C", "D"])].copy()
    if subset.empty:
        print("fig2: no C/D data")
        return

    noise_types = [n for n in ["input_dep", "uniform"] if n in subset["noise_type"].values]
    seeds = sorted(subset["seed"].unique())

    fig, axes = plt.subplots(1, len(noise_types), figsize=(7 * len(noise_types), 5), sharey=True)
    if len(noise_types) == 1:
        axes = [axes]

    for ax, noise_type in zip(axes, noise_types):
        sub = subset[subset["noise_type"] == noise_type]
        c_times = []
        d_times = []
        labels = []

        for seed in seeds:
            c_row = sub[(sub["condition"] == "C") & (sub["seed"] == seed)]
            d_row = sub[(sub["condition"] == "D") & (sub["seed"] == seed)]
            c_t = c_row["time_to_grok"].values[0] / 1000 if not c_row.empty else np.nan
            d_t = d_row["time_to_grok"].values[0] / 1000 if not d_row.empty else np.nan
            c_times.append(c_t)
            d_times.append(d_t)
            labels.append(f"seed {seed}")

        x = np.arange(len(seeds))
        width = 0.35
        bars_c = ax.bar(x - width / 2, c_times, width, label=COND_LABELS["C"],
                        color=COND_COLORS["C"], alpha=0.85)
        bars_d = ax.bar(x + width / 2, d_times, width, label=COND_LABELS["D"],
                        color=COND_COLORS["D"], alpha=0.85)

        # Annotate advantage
        for i, (ct, dt) in enumerate(zip(c_times, d_times)):
            if not np.isnan(ct) and not np.isnan(dt):
                adv = dt - ct
                ax.annotate(f"−{adv:.0f}k", xy=(x[i], ct), xytext=(x[i], ct - 3),
                            ha="center", fontsize=8, color="darkgreen",
                            arrowprops=dict(arrowstyle="->", color="darkgreen", lw=0.8))

        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_xlabel("Seed")
        ax.set_title(NOISE_LABELS.get(noise_type, noise_type), fontsize=11)
        if ax is axes[0]:
            ax.set_ylabel("Epochs to grok (×1000)")
        ax.legend(fontsize=8)

    fig.suptitle(
        "Hysteresis in grokking recovery: C (continue) vs D (reinitialize)\n"
        "Identical corrected datasets — C groks faster because optimization history is preserved",
        fontsize=11,
    )
    fig.tight_layout()
    savefig(fig, out_dir, "fig2_hysteresis_C_vs_D.png")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 3 — All conditions: mean time to grok (bar chart, both noise types)
# ─────────────────────────────────────────────────────────────────────────────

def fig_all_conditions_summary(df: pd.DataFrame, out_dir: str) -> None:
    conditions = [c for c in ["clean", "A", "B", "C", "D"] if c in df["condition"].values]
    noise_types = [n for n in ["input_dep", "uniform"] if n in df["noise_type"].values]

    fig, axes = plt.subplots(1, len(noise_types), figsize=(7 * len(noise_types), 5), sharey=True)
    if len(noise_types) == 1:
        axes = [axes]

    for ax, noise_type in zip(axes, noise_types):
        sub = df[df["noise_type"] == noise_type]
        grp = sub.groupby("condition")["time_to_grok"].agg(["mean", "std", "count"]).reindex(conditions)

        means = grp["mean"].values / 1000
        stds = grp["std"].fillna(0).values / 1000
        grokked_counts = sub.groupby("condition")["time_to_grok"].apply(lambda x: x.notna().sum()).reindex(conditions)
        total_counts = sub.groupby("condition")["time_to_grok"].count().reindex(conditions)

        x = np.arange(len(conditions))
        colors = [COND_COLORS.get(c, "gray") for c in conditions]

        bars = ax.bar(x, means, yerr=stds, capsize=4, color=colors, alpha=0.85)

        for i, (cond, bar) in enumerate(zip(conditions, bars)):
            g = int(grokked_counts.get(cond, 0))
            t = int(total_counts.get(cond, 0))
            h = bar.get_height()
            label = f"{g}/{t}" if not np.isnan(h) and h > 0 else f"0/{t}"
            ax.text(bar.get_x() + bar.get_width() / 2, max(h, 2) + 1,
                    label, ha="center", va="bottom", fontsize=9, fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels([c for c in conditions], fontsize=10)
        ax.set_xlabel("Condition")
        ax.set_title(NOISE_LABELS.get(noise_type, noise_type), fontsize=11)
        if ax is axes[0]:
            ax.set_ylabel("Mean time to grok (×1000 epochs)")
        ax.set_ylim(0, ax.get_ylim()[1] * 1.15)

    fig.suptitle(
        "Time to grokking across all temporal conditions\n"
        "(bar height = mean; label = seeds that grokked / total seeds)",
        fontsize=11,
    )
    fig.tight_layout()
    savefig(fig, out_dir, "fig3_all_conditions_summary.png")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 4 — Condition B: does grokking persist after noise injection?
# ─────────────────────────────────────────────────────────────────────────────

def fig_condition_B_persistence(df: pd.DataFrame, in_dir: str, out_dir: str) -> None:
    subset = df[df["condition"].isin(["clean", "B"])]
    if subset.empty:
        print("fig4: no clean/B data")
        return

    noise_types = [n for n in ["input_dep", "uniform"] if n in subset["noise_type"].values]
    seeds = sorted(subset["seed"].unique())

    fig, axes = plt.subplots(len(noise_types), len(seeds),
                             figsize=(5 * len(seeds), 4 * len(noise_types)),
                             sharey=True, sharex=True)
    if len(noise_types) == 1:
        axes = [axes]
    if len(seeds) == 1:
        axes = [[ax] for ax in axes]

    transition_k = df["transition_epoch"].dropna().iloc[0] / 1000 if not df["transition_epoch"].dropna().empty else 50

    for r, noise_type in enumerate(noise_types):
        for c, seed in enumerate(seeds):
            ax = axes[r][c]
            for cond in ["clean", "B"]:
                row = subset[(subset["noise_type"] == noise_type) &
                             (subset["seed"] == seed) &
                             (subset["condition"] == cond)]
                if row.empty:
                    continue
                hist = load_history(in_dir, row.iloc[0]["tag"])
                if hist is None:
                    continue
                epochs = np.array(hist["epoch"]) / 1000
                test_acc = np.array(hist["test_acc"])
                ax.plot(epochs, test_acc,
                        color=COND_COLORS.get(cond, "gray"),
                        ls=COND_LS.get(cond, "-"),
                        lw=2, label=COND_LABELS.get(cond, cond))

            ax.axvline(transition_k, color="gray", lw=1, ls=":", alpha=0.6)
            ax.axhline(0.95, color="red", lw=0.8, ls=":", alpha=0.5)
            ax.set_ylim(-0.02, 1.05)
            ax.set_title(f"{NOISE_LABELS.get(noise_type, noise_type)}, seed {seed}", fontsize=9)
            if c == 0:
                ax.set_ylabel("Test accuracy", fontsize=10)
            if r == len(noise_types) - 1:
                ax.set_xlabel("Epoch (×1000)", fontsize=10)
            if r == 0 and c == 0:
                ax.legend(fontsize=8)

    fig.suptitle(
        "Condition B: grokking persistence after noise injection at epoch 50k\n"
        "Dashed vertical = noise injection; red dotted = grok threshold",
        fontsize=11,
    )
    fig.tight_layout()
    savefig(fig, out_dir, "fig4_condition_B_persistence.png")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--in-dir", default="runs_temporal_full")
    p.add_argument("--out-dir", default="figures_temporal")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--noise-type", default="input_dep")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    df = load_summary(args.in_dir)

    print(f"Loaded {len(df)} rows from {args.in_dir}/summary.csv")
    print(f"Conditions: {sorted(df['condition'].unique())}")
    print(f"Noise types: {sorted(df['noise_type'].unique())}")
    print(f"Seeds: {sorted(df['seed'].unique())}")
    print()

    for noise_type in df["noise_type"].unique():
        for seed in sorted(df["seed"].unique()):
            fig_learning_curves(df, args.in_dir, args.out_dir, noise_type=noise_type, seed=seed)

    fig_hysteresis(df, args.out_dir)
    fig_all_conditions_summary(df, args.out_dir)
    fig_condition_B_persistence(df, args.in_dir, args.out_dir)

    print(f"\nAll figures saved to: {args.out_dir}/")


if __name__ == "__main__":
    main()
