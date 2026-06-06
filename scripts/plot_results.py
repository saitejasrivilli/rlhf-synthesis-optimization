#!/usr/bin/env python3
"""
Training curve visualization for all experiments.

Reads result JSON files from results/ and produces:
  - results/training_curves.png   — reward + loss + KL over iterations (LLM-PPO)
  - results/mlp_curves.png        — epoch reward + loss breakdown (MLP-PPO)
  - results/comparison.png        — bar chart comparing all methods

Usage:
    python scripts/plot_results.py
    python scripts/plot_results.py --output_dir results/plots
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = Path(__file__).parent.parent / "results"


def load_json(path):
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def moving_average(values, window=5):
    if len(values) < window:
        return list(range(len(values))), values
    ma = np.convolve(values, np.ones(window) / window, mode="valid").tolist()
    return list(range(window - 1, len(values))), ma


# ---------------------------------------------------------------------------
# LLM-PPO training curves
# ---------------------------------------------------------------------------

def plot_llm_ppo(data, out_path: Path):
    if not data or "training" not in data:
        return
    history = data["training"].get("history", [])
    if not history:
        return

    iters   = [h["iteration"]   for h in history]
    rewards = [h["reward_mean"] for h in history]
    losses  = [h["loss_total"]  for h in history]
    kls     = [h["kl"]          for h in history]

    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    fig.suptitle("LLM-PPO Training — Qwen2.5-7B + LoRA (r=8)", fontsize=14, fontweight="bold")

    # Reward
    axes[0].plot(iters, rewards, color="#4e79a7", alpha=0.35, linewidth=1, label="per-iter")
    x_ma, y_ma = moving_average(rewards, 5)
    if y_ma:
        axes[0].plot([iters[i] for i in x_ma], y_ma,
                     color="#4e79a7", linewidth=2.5, label="MA-5")
    axes[0].axhline(0.603, color="gray", linestyle="--", linewidth=1, label="rule baseline")
    axes[0].set_ylabel("Avg Reward")
    axes[0].legend(fontsize=9)
    axes[0].set_ylim(0.4, 0.85)
    axes[0].grid(True, alpha=0.3)

    # Loss
    axes[1].plot(iters, losses, color="#e15759", alpha=0.4, linewidth=1)
    x_ma, y_ma = moving_average(losses, 5)
    if y_ma:
        axes[1].plot([iters[i] for i in x_ma], y_ma, color="#e15759", linewidth=2.5)
    axes[1].set_ylabel("PPO Loss")
    axes[1].grid(True, alpha=0.3)

    # KL
    axes[2].plot(iters, kls, color="#76b7b2", linewidth=1.5)
    axes[2].axhline(0, color="black", linewidth=0.8, linestyle=":")
    axes[2].set_ylabel("KL Divergence")
    axes[2].set_xlabel("Iteration")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# MLP-PPO epoch curves
# ---------------------------------------------------------------------------

def plot_mlp_ppo(data, out_path: Path):
    if not data:
        return
    training      = data.get("training_metrics", {})
    epoch_rewards = training.get("epoch_rewards", [])
    epoch_losses  = training.get("epoch_losses",  [])
    pol_losses    = training.get("epoch_policy_losses", [])
    val_losses    = training.get("epoch_value_losses",  [])

    if not epoch_rewards:
        return

    epochs = list(range(1, len(epoch_rewards) + 1))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("MLP-PPO Training (real state vectors)", fontsize=13, fontweight="bold")

    ax1.plot(epochs, epoch_rewards, "o-", color="#4e79a7", linewidth=2, markersize=6)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Avg Reward")
    ax1.set_title("Reward per Epoch")
    ax1.set_ylim(0, 1.05)
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, epoch_losses, "o-", color="#e15759", linewidth=2, label="total", markersize=5)
    if pol_losses:
        ax2.plot(epochs, pol_losses, "s--", color="#f28e2b", linewidth=1.5, label="policy", markersize=4)
    if val_losses:
        ax2.plot(epochs, val_losses, "^--", color="#76b7b2", linewidth=1.5, label="value",  markersize=4)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.set_title("Loss Breakdown")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# Method comparison bar chart
# ---------------------------------------------------------------------------

def plot_comparison(benchmark, dpo_data, out_path: Path):
    rows = {}
    if benchmark:
        for name, stats in benchmark.items():
            if isinstance(stats, dict) and stats.get("mean") is not None:
                rows[name] = stats

    if dpo_data and "evaluation" in dpo_data:
        ev = dpo_data["evaluation"]
        rows["DPO (Qwen2.5-7B)"] = {
            "mean":           ev.get("average_reward", 0),
            "pct_above_0.6":  ev.get("successful_trajectories", 0),
            "improvement_pct":ev.get("improvement", 0),
        }

    if not rows:
        return

    names  = list(rows.keys())
    means  = [rows[n].get("mean", 0) for n in names]
    pct60  = [rows[n].get("pct_above_0.6", 0) for n in names]
    colors = ["#4e79a7", "#f28e2b", "#e15759", "#76b7b2", "#59a14f"][:len(names)]

    x = np.arange(len(names))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("RLHF Synthesis — Method Comparison", fontsize=13, fontweight="bold")

    bars = ax1.bar(x, means, width=0.5, color=colors, edgecolor="white", linewidth=1.2)
    ax1.axhline(0.5, color="gray", linestyle="--", linewidth=1, label="0.5 baseline")
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=20, ha="right", fontsize=9)
    ax1.set_ylabel("Avg Reward")
    ax1.set_ylim(0, 1.0)
    ax1.set_title("Average Reward (test set)")
    ax1.legend(fontsize=9)
    ax1.grid(axis="y", alpha=0.3)
    for bar, val in zip(bars, means):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f"{val:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    bars2 = ax2.bar(x, pct60, width=0.5, color=colors, edgecolor="white", linewidth=1.2)
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, rotation=20, ha="right", fontsize=9)
    ax2.set_ylabel("Trajectories > 0.60 reward (%)")
    ax2.set_ylim(0, 100)
    ax2.set_title("High-Quality Synthesis Rate")
    ax2.grid(axis="y", alpha=0.3)
    for bar, val in zip(bars2, pct60):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                 f"{val:.0f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--output_dir", default=str(RESULTS_DIR))
    return p.parse_args()


def main():
    args     = parse_args()
    out_dir  = Path(args.output_dir)

    llm_data = load_json(RESULTS_DIR / "rlhf_llm_distributed.json")
    mlp_data = load_json(RESULTS_DIR / "rlhf_results.json")
    bench    = load_json(RESULTS_DIR / "benchmark_results.json")
    dpo_data = load_json(RESULTS_DIR / "dpo_results.json")

    plot_llm_ppo(llm_data,      out_dir / "training_curves.png")
    plot_mlp_ppo(mlp_data,      out_dir / "mlp_curves.png")
    plot_comparison(bench, dpo_data, out_dir / "comparison.png")

    print("\nAll plots saved to:", out_dir)


if __name__ == "__main__":
    main()
