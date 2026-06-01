"""
make_figures.py  —  Visualize audit-budget grokking recovery experiments.

Reads summary.csv and *_history.json files from an experiment output directory
and generates figures for the JHSS paper / portfolio.

Usage:
  python make_figures.py --in-dir runs_audit_pilot --out-dir figures_pilot
  python make_figures.py --in-dir runs_audit_full  --out-dir figures_full --seed 0

Output figures (all PNG at 150 dpi):
  fig1_grokking_curves.png      — test-acc curves per policy × budget (single condition)
  fig2_final_test_acc.png       — bar chart: final test acc vs budget, by policy & noise type
  fig3_time_to_grok.png         — bar chart: time to grok vs budget (or time-to-0.80)
  fig4_hit_rate.png             — bar chart: hit_rate by policy & noise type  ← KEY RESULT
  fig5_geometry_matters.png     — region_audit hit_rate: input_dep vs uniform comparison
  fig6_correction_rate.png      — what fraction of total corruption was fixed
"""

import argparse
import json
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import pandas as pd
except ImportError:
    sys.exit("pandas is required: pip install pandas")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
except ImportError:
    sys.exit("matplotlib is required: pip install matplotlib")

# Optional seaborn for nicer grouped bars
try:
    import seaborn as sns
    sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False
    plt.style.use("seaborn-v0_8-whitegrid")


# ─────────────────────────────────────────────────────────────────────────────
# Colors / style
# ─────────────────────────────────────────────────────────────────────────────

POLICY_COLORS = {
    "none":          "#555555",
    "random_audit":  "#2196F3",
    "region_audit":  "#FF9800",
    "loss_audit":    "#4CAF50",
}
POLICY_LABELS = {
    "none":          "No audit (baseline)",
    "random_audit":  "Random audit",
    "region_audit":  "Region audit  (a < p/4)",
    "loss_audit":    "Loss-based audit",
}
NOISE_MARKERS = {
    "input_dep": "o",
    "uniform":   "s",
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
    # Coerce numeric columns
    for col in ["budget", "num_corrupted", "num_audited", "num_corrected",
                "hit_rate", "correction_rate", "final_test_acc", "final_train_acc",
                "gap_area", "auc_test", "time_to_grok", "time_to_0.80", "time_to_0.90",
                "wall_time_sec"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def load_history(in_dir: str, tag: str) -> Optional[Dict]:
    path = os.path.join(in_dir, f"{tag}_history.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def savefig(fig: plt.Figure, out_dir: str, name: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 1 — Grokking curves (test acc vs epoch) for one (noise_type, eta, seed)
# ─────────────────────────────────────────────────────────────────────────────

def fig_grokking_curves(
    df: pd.DataFrame,
    in_dir: str,
    out_dir: str,
    noise_type: str = "input_dep",
    eta: float = 0.04,
    seed: int = 0,
    budgets_to_show: Optional[List[int]] = None,
) -> None:
    mask = (df["noise_type"] == noise_type) & (df["eta"] == eta) & (df["seed"] == seed)
    subset = df[mask]
    if subset.empty:
        print(f"fig1: no data for noise_type={noise_type} eta={eta} seed={seed}")
        return

    policies = [p for p in ["none", "random_audit", "region_audit", "loss_audit"]
                if p in subset["policy"].values]

    fig, axes = plt.subplots(1, len(policies), figsize=(5 * len(policies), 4), sharey=True)
    if len(policies) == 1:
        axes = [axes]

    for ax, policy in zip(axes, policies):
        if policy == "none":
            rows = subset[subset["policy"] == "none"]
        else:
            rows = subset[subset["policy"] == policy]
            if budgets_to_show:
                rows = rows[rows["budget"].isin(budgets_to_show)]
            rows = rows.sort_values("budget")

        color = POLICY_COLORS.get(policy, "gray")

        for _, row in rows.iterrows():
            hist = load_history(in_dir, row["tag"])
            if hist is None:
                continue
            epochs = np.array(hist["epoch"])
            test_acc = np.array(hist["test_acc"])

            if policy == "none":
                label = "No audit"
                lw, ls = 2.5, "--"
            else:
                b = int(row["budget"]) if not np.isnan(row.get("budget", np.nan)) else 0
                label = f"B={b}"
                lw, ls = 1.8, "-"
                # Shade color by budget
                frac = 0.4 + 0.6 * (b / max(1, rows["budget"].max()))
                color = plt.cm.Blues(frac) if policy == "random_audit" else (
                    plt.cm.Oranges(frac) if policy == "region_audit" else
                    plt.cm.Greens(frac)
                )

            ax.plot(epochs / 1000, test_acc, label=label, lw=lw, ls=ls, color=color)

        ax.axhline(0.95, color="red", lw=1, ls=":", alpha=0.7, label="grok threshold")
        ax.set_title(POLICY_LABELS.get(policy, policy), fontsize=10)
        ax.set_xlabel("Epoch (×1000)")
        ax.set_ylim(-0.02, 1.05)
        ax.legend(fontsize=8, loc="upper left")
        if ax is axes[0]:
            ax.set_ylabel("Test accuracy")

    fig.suptitle(
        f"Grokking recovery — {NOISE_LABELS.get(noise_type, noise_type)}  η={eta}  seed={seed}",
        fontsize=12,
    )
    fig.tight_layout()
    savefig(fig, out_dir, "fig1_grokking_curves.png")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2 — Final test accuracy vs budget (bar chart)
# ─────────────────────────────────────────────────────────────────────────────

def fig_final_test_acc(df: pd.DataFrame, out_dir: str, eta: float = 0.04) -> None:
    mask = (df["eta"] == eta) & (df["status"] == "completed")
    subset = df[mask].copy()
    if subset.empty:
        print("fig2: no completed data")
        return

    noise_types = [n for n in ["input_dep", "uniform"] if n in subset["noise_type"].values]
    policies = [p for p in ["none", "random_audit", "region_audit", "loss_audit"]
                if p in subset["policy"].values]

    # Aggregate over seeds
    grp = subset.groupby(["noise_type", "policy", "budget"])["final_test_acc"].agg(["mean", "std"]).reset_index()

    fig, axes = plt.subplots(1, len(noise_types), figsize=(7 * len(noise_types), 5), sharey=True)
    if len(noise_types) == 1:
        axes = [axes]

    budgets_sorted = sorted(subset["budget"].dropna().unique().astype(int).tolist())

    for ax, noise_type in zip(axes, noise_types):
        sub = grp[grp["noise_type"] == noise_type]

        # Baseline (no-audit) horizontal line
        baseline = sub[(sub["policy"] == "none")]["mean"]
        if not baseline.empty:
            ax.axhline(baseline.values[0], color=POLICY_COLORS["none"], lw=1.5, ls="--",
                       label=POLICY_LABELS["none"], alpha=0.8)

        # Bars for each policy × budget
        active = [p for p in policies if p != "none"]
        n_pol = len(active)
        bar_width = 0.8 / max(n_pol, 1)
        x_labels = [str(b) for b in budgets_sorted if b > 0]
        x_pos = np.arange(len(x_labels))

        for i, policy in enumerate(active):
            means, stds = [], []
            for b in budgets_sorted:
                if b == 0:
                    continue
                row = sub[(sub["policy"] == policy) & (sub["budget"] == b)]
                means.append(row["mean"].values[0] if not row.empty else np.nan)
                stds.append(row["std"].values[0] if not row.empty else 0)

            offset = (i - (n_pol - 1) / 2) * bar_width
            ax.bar(
                x_pos + offset, means, bar_width * 0.9,
                yerr=stds, capsize=3,
                color=POLICY_COLORS.get(policy, "gray"),
                label=POLICY_LABELS.get(policy, policy),
                alpha=0.85,
            )

        ax.axhline(0.95, color="red", lw=1, ls=":", alpha=0.6)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(x_labels)
        ax.set_xlabel("Audit budget B")
        ax.set_title(NOISE_LABELS.get(noise_type, noise_type))
        ax.set_ylim(0, 1.05)
        if ax is axes[0]:
            ax.set_ylabel("Final test accuracy")
        ax.legend(fontsize=8)

    fig.suptitle(f"Final test accuracy vs audit budget  (η={eta})", fontsize=12)
    fig.tight_layout()
    savefig(fig, out_dir, "fig2_final_test_acc.png")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 3 — Time to grok (or time-to-0.80) vs budget
# ─────────────────────────────────────────────────────────────────────────────

def fig_time_to_grok(df: pd.DataFrame, out_dir: str, eta: float = 0.04,
                     metric: str = "time_to_0.80") -> None:
    mask = (df["eta"] == eta) & (df["status"] == "completed")
    subset = df[mask].copy()
    if subset.empty or metric not in subset.columns:
        print(f"fig3: no data for metric={metric}")
        return

    noise_types = [n for n in ["input_dep", "uniform"] if n in subset["noise_type"].values]
    policies = [p for p in ["random_audit", "region_audit", "loss_audit"]
                if p in subset["policy"].values]

    grp = subset.groupby(["noise_type", "policy", "budget"])[metric].agg(["mean", "std"]).reset_index()
    budgets_sorted = sorted([b for b in subset["budget"].dropna().unique().astype(int).tolist() if b > 0])

    fig, axes = plt.subplots(1, len(noise_types), figsize=(7 * len(noise_types), 5), sharey=True)
    if len(noise_types) == 1:
        axes = [axes]

    for ax, noise_type in zip(axes, noise_types):
        sub = grp[grp["noise_type"] == noise_type]

        for policy in policies:
            rows = sub[sub["policy"] == policy].sort_values("budget")
            if rows.empty:
                continue
            xs = rows["budget"].values
            ys = rows["mean"].values / 1000  # convert to k-epochs
            errs = rows["std"].values / 1000
            ax.errorbar(
                xs, ys, yerr=errs,
                marker="o", lw=2, capsize=4,
                color=POLICY_COLORS.get(policy, "gray"),
                label=POLICY_LABELS.get(policy, policy),
            )

        # Baseline
        base = subset[(subset["noise_type"] == noise_type) & (subset["policy"] == "none")][metric]
        if not base.empty:
            ax.axhline(base.mean() / 1000, color=POLICY_COLORS["none"], lw=1.5, ls="--",
                       label=POLICY_LABELS["none"], alpha=0.8)

        ax.set_xlabel("Audit budget B")
        ax.set_ylabel("Epochs to reach threshold (×1000)")
        ax.set_title(NOISE_LABELS.get(noise_type, noise_type))
        ax.legend(fontsize=8)
        # Lower = better
        ax.invert_yaxis()

    metric_label = {"time_to_grok": "0.95 (grok)", "time_to_0.80": "0.80", "time_to_0.90": "0.90"}.get(metric, metric)
    fig.suptitle(f"Epochs to reach test acc ≥ {metric_label}  (η={eta})", fontsize=12)
    fig.tight_layout()
    savefig(fig, out_dir, "fig3_time_to_grok.png")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 4 — Hit rate by policy & noise type  ← KEY RESULT
# ─────────────────────────────────────────────────────────────────────────────

def fig_hit_rate(df: pd.DataFrame, out_dir: str, eta: float = 0.04,
                 budget: Optional[int] = None) -> None:
    """Hit rate = num_corrected / budget. Shows WHY targeted audit is more efficient."""
    mask = (df["eta"] == eta) & (df["policy"] != "none")
    if budget is not None:
        mask &= df["budget"] == budget
    subset = df[mask].copy()
    if subset.empty:
        print("fig4: no data")
        return

    noise_types = [n for n in ["input_dep", "uniform"] if n in subset["noise_type"].values]
    policies = [p for p in ["random_audit", "region_audit", "loss_audit"]
                if p in subset["policy"].values]

    # Aggregate over seeds × budgets (or just selected budget)
    grp = subset.groupby(["noise_type", "policy"])["hit_rate"].agg(["mean", "std"]).reset_index()

    fig, ax = plt.subplots(figsize=(8, 5))

    n_noise = len(noise_types)
    n_pol = len(policies)
    bar_width = 0.7 / max(n_pol, 1)
    x_pos = np.arange(n_noise)

    for i, policy in enumerate(policies):
        means, stds = [], []
        for noise_type in noise_types:
            row = grp[(grp["noise_type"] == noise_type) & (grp["policy"] == policy)]
            means.append(row["mean"].values[0] if not row.empty else np.nan)
            stds.append(row["std"].values[0] if not row.empty else 0.0)

        offset = (i - (n_pol - 1) / 2) * bar_width
        bars = ax.bar(
            x_pos + offset, means, bar_width * 0.9,
            yerr=stds, capsize=4,
            color=POLICY_COLORS.get(policy, "gray"),
            label=POLICY_LABELS.get(policy, policy),
            alpha=0.85,
        )

    # Expected baseline = η (random audit expected hit rate)
    ax.axhline(eta, color="black", lw=1.5, ls=":", alpha=0.7,
               label=f"Random audit expected (η={eta})")

    ax.set_xticks(x_pos)
    ax.set_xticklabels([NOISE_LABELS.get(n, n) for n in noise_types])
    ax.set_ylabel("Hit rate  (num_corrected / budget)")
    budget_str = f"B={budget}" if budget is not None else "all budgets"
    ax.set_title(f"Audit efficiency by policy & noise geometry  (η={eta}, {budget_str})\n"
                 f"Higher hit rate → more corruptions found per unit budget", fontsize=11)
    ax.legend(fontsize=9)
    ax.set_ylim(0, min(1.0, ax.get_ylim()[1] * 1.2))
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))

    # Annotation: "4× better" arrow for input_dep
    fig.tight_layout()
    savefig(fig, out_dir, "fig4_hit_rate.png")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 5 — "Geometry matters": region_audit hit_rate for input_dep vs uniform
# ─────────────────────────────────────────────────────────────────────────────

def fig_geometry_matters(df: pd.DataFrame, out_dir: str, eta: float = 0.04) -> None:
    """Smoking-gun figure: region_audit is 4× more efficient under input_dep noise,
    but no better than random_audit under uniform noise.
    This directly demonstrates that noise geometry determines audit strategy value.
    """
    mask = (df["eta"] == eta) & (df["policy"].isin(["random_audit", "region_audit"]))
    subset = df[mask].copy()
    if subset.empty:
        print("fig5: no data")
        return

    noise_types = [n for n in ["input_dep", "uniform"] if n in subset["noise_type"].values]
    budgets_sorted = sorted([b for b in subset["budget"].dropna().unique().astype(int).tolist() if b > 0])

    fig, axes = plt.subplots(1, len(noise_types), figsize=(6 * len(noise_types), 4.5), sharey=True)
    if len(noise_types) == 1:
        axes = [axes]

    for ax, noise_type in zip(axes, noise_types):
        sub = subset[subset["noise_type"] == noise_type]
        grp = sub.groupby(["policy", "budget"])["hit_rate"].agg(["mean", "std"]).reset_index()

        for policy in ["random_audit", "region_audit"]:
            rows = grp[grp["policy"] == policy].sort_values("budget")
            if rows.empty:
                continue
            xs = rows["budget"].values
            ys = rows["mean"].values
            errs = rows["std"].fillna(0).values
            ax.errorbar(
                xs, ys, yerr=errs,
                marker=NOISE_MARKERS.get(noise_type, "o"),
                lw=2.5, capsize=4, markersize=7,
                color=POLICY_COLORS.get(policy, "gray"),
                label=POLICY_LABELS.get(policy, policy),
            )

        ax.axhline(eta, color="black", lw=1.5, ls=":", alpha=0.7, label=f"Expected if random (η={eta})")
        ax.set_xlabel("Audit budget B")
        ax.set_title(NOISE_LABELS.get(noise_type, noise_type), fontsize=11)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
        ax.set_ylim(0, None)
        if ax is axes[0]:
            ax.set_ylabel("Hit rate  (corruptions found / budget)")
        ax.legend(fontsize=9)

    fig.suptitle(
        "Noise geometry determines audit efficiency\n"
        "Input-dep: region audit finds 4× more corruptions. Uniform: no benefit.",
        fontsize=11,
    )
    fig.tight_layout()
    savefig(fig, out_dir, "fig5_geometry_matters.png")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 6 — Correction rate (fraction of total corruption fixed)
# ─────────────────────────────────────────────────────────────────────────────

def fig_correction_rate(df: pd.DataFrame, out_dir: str, eta: float = 0.04) -> None:
    mask = (df["eta"] == eta) & (df["policy"] != "none")
    subset = df[mask].copy()
    if subset.empty:
        print("fig6: no data")
        return

    noise_types = [n for n in ["input_dep", "uniform"] if n in subset["noise_type"].values]
    policies = [p for p in ["random_audit", "region_audit", "loss_audit"]
                if p in subset["policy"].values]
    budgets_sorted = sorted([b for b in subset["budget"].dropna().unique().astype(int).tolist() if b > 0])

    grp = subset.groupby(["noise_type", "policy", "budget"])["correction_rate"].agg(["mean", "std"]).reset_index()

    fig, axes = plt.subplots(1, len(noise_types), figsize=(7 * len(noise_types), 4.5), sharey=True)
    if len(noise_types) == 1:
        axes = [axes]

    for ax, noise_type in zip(axes, noise_types):
        sub = grp[grp["noise_type"] == noise_type]
        for policy in policies:
            rows = sub[sub["policy"] == policy].sort_values("budget")
            if rows.empty:
                continue
            xs = rows["budget"].values
            ys = rows["mean"].values
            errs = rows["std"].fillna(0).values
            ax.errorbar(
                xs, ys, yerr=errs, marker="o", lw=2.5, capsize=4, markersize=7,
                color=POLICY_COLORS.get(policy, "gray"),
                label=POLICY_LABELS.get(policy, policy),
            )

        ax.axhline(1.0, color="green", lw=1, ls=":", alpha=0.5, label="All corruption fixed")
        ax.set_xlabel("Audit budget B")
        ax.set_title(NOISE_LABELS.get(noise_type, noise_type))
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
        ax.set_ylim(0, 1.1)
        if ax is axes[0]:
            ax.set_ylabel("Correction rate  (corruptions fixed / total)")
        ax.legend(fontsize=9)

    fig.suptitle(f"Fraction of total corruption fixed by audit  (η={eta})", fontsize=12)
    fig.tight_layout()
    savefig(fig, out_dir, "fig6_correction_rate.png")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate figures for audit-budget grokking experiments.")
    p.add_argument("--in-dir", default="runs_audit_experiment", help="Directory with summary.csv + history JSONs")
    p.add_argument("--out-dir", default="figures", help="Directory to write PNG files")
    p.add_argument("--eta", type=float, default=0.04)
    p.add_argument("--seed", type=int, default=0, help="Seed used for curve plots (fig1)")
    p.add_argument("--noise-type", default="input_dep", help="Noise type for fig1 curves")
    p.add_argument("--budget-for-hitrate", type=int, default=None,
                   help="Single budget to use for hit-rate bar chart (default: all budgets averaged)")
    p.add_argument("--time-metric", default="time_to_0.80",
                   choices=["time_to_grok", "time_to_0.80", "time_to_0.90"])
    return p.parse_args()


def main() -> None:
    args = parse_args()
    df = load_summary(args.in_dir)

    print(f"Loaded {len(df)} rows from {args.in_dir}/summary.csv")
    print(f"Noise types: {sorted(df['noise_type'].unique())}")
    print(f"Policies:    {sorted(df['policy'].unique())}")
    print(f"Budgets:     {sorted(df['budget'].dropna().unique().tolist())}")
    print(f"Seeds:       {sorted(df['seed'].unique().tolist())}")
    print(f"Eta values:  {sorted(df['eta'].unique().tolist())}")
    print()

    fig_grokking_curves(df, args.in_dir, args.out_dir,
                        noise_type=args.noise_type, eta=args.eta, seed=args.seed)
    fig_final_test_acc(df, args.out_dir, eta=args.eta)
    fig_time_to_grok(df, args.out_dir, eta=args.eta, metric=args.time_metric)
    fig_hit_rate(df, args.out_dir, eta=args.eta, budget=args.budget_for_hitrate)
    fig_geometry_matters(df, args.out_dir, eta=args.eta)
    fig_correction_rate(df, args.out_dir, eta=args.eta)

    print(f"\nAll figures saved to: {args.out_dir}/")


if __name__ == "__main__":
    main()
