#!/usr/bin/env python3
"""
Molecule generalization test: train on N-1 molecules, evaluate on held-out molecule.

This tests whether the policy learns transferable chemistry, not just molecule-specific
reward hacking. Run once per molecule (leave-one-out cross-validation).

Usage:
    python scripts/generalization_test.py \
        --data_file data/trajectories_real_ord.jsonl \
        --held_out Ketoprofen

    # All 5 molecules (takes ~5x longer)
    python scripts/generalization_test.py --all_molecules
"""
import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.reward.real_reward_model import RealRewardModel
from src.paths import initialize_directories, get_results_dir

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

MOLECULES = ["Aspirin", "Ibuprofen", "Naproxen", "Paracetamol", "Ketoprofen"]


def load_trajectories(path):
    data = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def split_by_molecule(trajectories, held_out: str):
    train = [t for t in trajectories if t.get("molecule") != held_out]
    test  = [t for t in trajectories if t.get("molecule") == held_out]
    return train, test


def run_mlp_ppo(train_traj, test_traj, device, epochs=5):
    """Run MLP-PPO on train molecules, evaluate on held-out."""
    from src.training.real_ppo_trainer import RealPPOPolicy, RealPPOTrainer

    policy       = RealPPOPolicy(state_dim=128, action_dim=8, hidden_dim=256)
    reward_model = RealRewardModel()
    trainer      = RealPPOTrainer(
        policy=policy, reward_model=reward_model,
        learning_rate=1e-4, device=device,
    )
    train_metrics = trainer.train(train_traj, epochs=epochs, batch_size=16)

    # Evaluate on held-out molecule
    test_rewards = [reward_model.score_trajectory(t) for t in test_traj]
    baseline     = sum(test_rewards) / len(test_rewards) if test_rewards else 0.0
    final_reward = train_metrics.get("final_reward", 0.0)
    improvement  = ((final_reward - baseline) / (baseline + 1e-8)) * 100

    return {
        "method":       "MLP-PPO",
        "train_reward": final_reward,
        "test_reward":  float(sum(test_rewards) / len(test_rewards)) if test_rewards else 0.0,
        "improvement":  improvement,
        "n_test":       len(test_traj),
    }


def run_dpo(train_traj, test_traj, device, model_name, sft_ckpt=None):
    """Run DPO on train molecules, evaluate on held-out."""
    from src.policy.llm_synthesis_policy import LLMSynthesisPolicy
    from src.reward.preference_reward_model import build_preference_pairs
    from src.training.dpo_trainer import DPOConfig, DPOTrainer
    from src.reward.real_reward_model import RealRewardModel

    if sft_ckpt and Path(sft_ckpt).exists():
        policy = LLMSynthesisPolicy.from_pretrained(sft_ckpt, base_model=model_name)
    else:
        policy = LLMSynthesisPolicy(model_name=model_name, device_map={"": device})

    pairs = build_preference_pairs(train_traj, pairs_per_molecule=15)
    config = DPOConfig(beta=0.1, learning_rate=1e-5, num_epochs=2, batch_size=2)
    trainer = DPOTrainer(policy=policy, config=config, device=device)
    dpo_metrics = trainer.train(pairs)

    reward_model = RealRewardModel()
    test_rewards = [reward_model.score_trajectory(t) for t in test_traj]
    avg_test     = float(sum(test_rewards) / len(test_rewards)) if test_rewards else 0.0
    baseline     = 0.517  # rule-based baseline

    return {
        "method":            "DPO",
        "best_reward_margin": dpo_metrics.get("best_reward_margin", 0.0),
        "test_reward":       avg_test,
        "improvement":       ((avg_test - baseline) / baseline) * 100,
        "n_test":            len(test_traj),
    }


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_file",    default="data/trajectories_real_ord.jsonl")
    p.add_argument("--held_out",     default=None,
                   help="Single molecule to hold out (e.g. Ketoprofen)")
    p.add_argument("--all_molecules", action="store_true",
                   help="Run leave-one-out over all 5 molecules")
    p.add_argument("--method",       choices=["mlp", "dpo", "both"], default="mlp")
    p.add_argument("--model_name",   default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--sft_ckpt",     default=None)
    p.add_argument("--epochs",       type=int, default=5)
    return p.parse_args()


def main():
    args = parse_args()

    os.environ.setdefault("HF_HOME",            "/storage/gxg8313/saiteja/hf_cache")
    os.environ.setdefault("TRANSFORMERS_CACHE", "/storage/gxg8313/saiteja/hf_cache")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    initialize_directories()

    data_path = Path(args.data_file)
    if not data_path.is_absolute():
        data_path = Path(__file__).parent.parent / data_path
    trajectories = load_trajectories(data_path)

    held_out_list = MOLECULES if args.all_molecules else [args.held_out or "Ketoprofen"]

    all_results = []

    for held_out in held_out_list:
        train_traj, test_traj = split_by_molecule(trajectories, held_out)
        logger.info(f"\n{'='*60}")
        logger.info(f"Held-out molecule: {held_out}")
        logger.info(f"  Train: {len(train_traj)} | Test: {len(test_traj)}")
        logger.info(f"{'='*60}")

        result = {"held_out": held_out, "n_train": len(train_traj), "n_test": len(test_traj)}

        if args.method in ("mlp", "both"):
            mlp_result = run_mlp_ppo(train_traj, test_traj, device, epochs=args.epochs)
            result["mlp_ppo"] = mlp_result
            logger.info(f"  MLP-PPO test reward: {mlp_result['test_reward']:.4f} "
                        f"({mlp_result['improvement']:+.1f}%)")

        if args.method in ("dpo", "both"):
            dpo_result = run_dpo(train_traj, test_traj, device, args.model_name, args.sft_ckpt)
            result["dpo"] = dpo_result
            logger.info(f"  DPO test reward:     {dpo_result['test_reward']:.4f} "
                        f"({dpo_result['improvement']:+.1f}%)")

        all_results.append(result)

    # Summary table
    logger.info("\n" + "="*70)
    logger.info("GENERALIZATION RESULTS (held-out molecule)")
    logger.info(f"{'Held-out':15} {'Method':10} {'Test Reward':12} {'Improvement':12}")
    logger.info("-"*50)
    for r in all_results:
        mol = r["held_out"]
        for method_key in ("mlp_ppo", "dpo"):
            if method_key in r:
                m = r[method_key]
                logger.info(f"{mol:15} {m['method']:10} {m['test_reward']:.4f}        {m['improvement']:+.1f}%")
    logger.info("="*70)

    output = {
        "timestamp":    datetime.now().isoformat(),
        "experiment":   "molecule_generalization",
        "held_out_molecules": held_out_list,
        "results":      all_results,
    }

    results_file = get_results_dir() / "generalization_results.json"
    with open(results_file, "w") as f:
        json.dump(output, f, indent=2)
    logger.info(f"Results saved to {results_file}")


if __name__ == "__main__":
    main()
