#!/usr/bin/env python3
"""
Benchmark harness: compare MLP-PPO baseline vs LLM-PPO policy.

Usage:
    # Run full benchmark (trains both models):
    python scripts/benchmark.py --data_file data/trajectories_improvable.jsonl

    # Evaluate from saved checkpoints (skip training):
    python scripts/benchmark.py --mlp_ckpt models/best_policy.pth \
                                 --llm_ckpt models/llm_policy/best \
                                 --eval_only

Output:
    results/benchmark_results.json   — machine-readable
    stdout                           — results table
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.reward.learned_reward_model import LearnedRewardModel
from src.reward.real_reward_model import RealRewardModel
from src.training.real_ppo_trainer import RealPPOPolicy, RealPPOTrainer
from src.evaluation.real_rlhf_evaluator import evaluate_rlhf_policy, save_rlhf_results
from src.paths import get_results_dir, initialize_directories

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_trajectories(path: Path) -> List[Dict]:
    data = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def _reward_stats(rewards: List[float]) -> Dict:
    n = len(rewards)
    mean = sum(rewards) / n
    std  = (sum((r - mean) ** 2 for r in rewards) / n) ** 0.5
    baseline = 0.5
    return {
        "mean":             round(mean, 4),
        "std":              round(std, 4),
        "max":              round(max(rewards), 4),
        "min":              round(min(rewards), 4),
        "pct_above_0.6":   round(sum(r > 0.60 for r in rewards) / n * 100, 1),
        "pct_above_0.75":  round(sum(r > 0.75 for r in rewards) / n * 100, 1),
        "improvement_pct": round((mean - baseline) / baseline * 100, 2),
    }


# ---------------------------------------------------------------------------
# Rule-based baseline
# ---------------------------------------------------------------------------

def run_rule_baseline(test_traj: List[Dict]) -> Dict:
    logger.info("[Baseline] Rule-based reward model (no training)")
    rm = RealRewardModel()
    rewards = rm.score_batch(test_traj)
    return _reward_stats(rewards)


# ---------------------------------------------------------------------------
# MLP-PPO baseline
# ---------------------------------------------------------------------------

def run_mlp_ppo(
    train_traj: List[Dict],
    test_traj:  List[Dict],
    ckpt:       str | None,
    device:     str,
    fast:       bool = False,
) -> Dict:
    logger.info("[MLP-PPO] Training actor-critic MLP with PPO")
    rm = RealRewardModel()
    policy = RealPPOPolicy(state_dim=128, action_dim=32, hidden_dim=256).to(device)

    if ckpt and Path(ckpt).exists():
        policy.load_state_dict(torch.load(ckpt, map_location=device))
        logger.info(f"  Loaded MLP checkpoint from {ckpt}")
        train_time = 0.0
    else:
        trainer = RealPPOTrainer(policy, rm, learning_rate=1e-4, device=device)
        t0 = time.time()
        trainer.train(train_traj, epochs=2 if fast else 5)
        train_time = time.time() - t0
        logger.info(f"  Training time: {train_time:.1f}s")

    rewards = rm.score_batch(test_traj)
    stats = _reward_stats(rewards)
    stats["train_time_s"] = round(train_time, 1)
    return stats


# ---------------------------------------------------------------------------
# LLM-PPO (Llama-2 7B + LoRA)
# ---------------------------------------------------------------------------

def run_llm_ppo(
    train_traj:  List[Dict],
    test_traj:   List[Dict],
    ckpt:        str | None,
    model_name:  str,
    device:      str,
    num_iters:   int = 50,
    fast:        bool = False,
) -> Dict:
    logger.info(f"[LLM-PPO] Llama-2 7B + LoRA | model={model_name}")

    try:
        from src.policy.llm_synthesis_policy import LLMSynthesisPolicy
        from src.training.llm_ppo_trainer import LLMPPOConfig, LLMPPOTrainer
    except ImportError as e:
        logger.warning(f"LLM policy unavailable: {e} — skipping")
        return {"error": str(e), "mean": None}

    rm = LearnedRewardModel()

    if ckpt and Path(ckpt).exists():
        policy = LLMSynthesisPolicy.from_pretrained(ckpt, base_model=model_name)

        train_time = 0.0
        logger.info(f"  Loaded LLM checkpoint from {ckpt}")
    else:
        policy = LLMSynthesisPolicy(
            model_name=model_name, lora_r=8, lora_alpha=16, device_map={"": device}
        )

        import copy
        ref_policy = copy.deepcopy(policy)
        for p in ref_policy.parameters():
            p.requires_grad_(False)

        config = LLMPPOConfig(
            model_name=model_name,
            num_iterations=10 if fast else num_iters,
            batch_size=2 if fast else 4,
            ppo_epochs=1 if fast else 3,
            learning_rate=1e-5,
            output_dir="models/llm_policy_benchmark",
        )

        trainer = LLMPPOTrainer(
            policy=policy, ref_policy=ref_policy,
            reward_model=rm, config=config, device=device,
        )

        t0 = time.time()
        trainer.train(train_traj)
        train_time = time.time() - t0
        logger.info(f"  Training time: {train_time:.1f}s")

    rewards = rm.score_batch(test_traj)
    stats = _reward_stats(rewards)
    stats["train_time_s"] = round(train_time, 1)
    return stats


# ---------------------------------------------------------------------------
# Results table
# ---------------------------------------------------------------------------

def print_table(results: Dict[str, Dict]):
    COLS = ["mean", "std", "max", "pct_above_0.6", "pct_above_0.75", "improvement_pct", "train_time_s"]
    HEADS = ["Model", "Avg Reward", "Std", "Max", ">0.6 (%)", ">0.75 (%)", "Δ vs baseline (%)", "Train (s)"]

    col_widths = [max(len(h), 20) for h in HEADS]
    col_widths[0] = 30

    def row(cells):
        return "  ".join(str(c).ljust(w) for c, w in zip(cells, col_widths))

    print("\n" + "=" * (sum(col_widths) + 2 * len(col_widths)))
    print("  BENCHMARK RESULTS — RLHF Synthesis Optimization")
    print("=" * (sum(col_widths) + 2 * len(col_widths)))
    print(row(HEADS))
    print("-" * (sum(col_widths) + 2 * len(col_widths)))

    for model_name, stats in results.items():
        if stats.get("error"):
            cells = [model_name] + ["N/A"] * (len(HEADS) - 1)
        else:
            cells = [model_name] + [stats.get(c, "N/A") for c in COLS]
        print(row(cells))

    print("=" * (sum(col_widths) + 2 * len(col_widths)) + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_file",  default="data/trajectories_improvable.jsonl")
    p.add_argument("--model_name", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--mlp_ckpt",   default=None, help="Path to saved MLP policy .pth")
    p.add_argument("--llm_ckpt",   default=None, help="Path to saved LLM LoRA checkpoint dir")
    p.add_argument("--eval_only",  action="store_true", help="Skip training, only evaluate checkpoints")
    p.add_argument("--num_iters",  type=int, default=50, help="LLM PPO iterations")
    p.add_argument("--fast",       action="store_true", help="Quick smoke-test (fewer epochs)")
    p.add_argument("--skip_llm",   action="store_true", help="Skip LLM-PPO (if GPU memory limited)")
    return p.parse_args()


def main():
    args = parse_args()
    initialize_directories()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")

    data_path = Path(args.data_file)
    if not data_path.is_absolute():
        data_path = Path(__file__).parent.parent / data_path

    trajectories = load_trajectories(data_path)
    split = int(len(trajectories) * 0.8)
    train_traj = trajectories[:split]
    test_traj  = trajectories[split:]
    logger.info(f"Data: {len(train_traj)} train / {len(test_traj)} test")

    results: Dict[str, Dict] = {}

    # 1. Rule-based baseline
    results["Rule-based baseline"] = run_rule_baseline(test_traj)

    # 2. MLP-PPO
    results["MLP-PPO (128→256→256)"] = run_mlp_ppo(
        train_traj, test_traj,
        ckpt=args.mlp_ckpt if args.eval_only else None,
        device=device,
        fast=args.fast,
    )

    # 3. LLM-PPO
    if not args.skip_llm:
        results[f"LLM-PPO (Llama-2-7B + LoRA r={8})"] = run_llm_ppo(
            train_traj, test_traj,
            ckpt=args.llm_ckpt if args.eval_only else None,
            model_name=args.model_name,
            device=device,
            num_iters=args.num_iters,
            fast=args.fast,
        )

    print_table(results)

    out_path = get_results_dir() / "benchmark_results.json"
    save_rlhf_results(results, out_path)
    logger.info(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
