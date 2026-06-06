#!/usr/bin/env python3
"""
Launch a W&B hyperparameter sweep for DPO training.

Sweeps β, learning_rate, num_epochs, batch_size, pairs_per_mol via Bayesian
optimization over 20 trials to find the best DPO configuration.

Usage:
    # Create sweep and run 1 trial (for testing)
    python scripts/run_sweep.py --count 1

    # Full 20-trial sweep
    python scripts/run_sweep.py --count 20

    # Resume an existing sweep
    python scripts/run_sweep.py --sweep_id <sweep-id> --count 5
"""
import argparse
import json
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

SWEEP_CONFIG = {
    "program": "scripts/train_dpo.py",
    "method":  "bayes",
    "metric":  {"goal": "maximize", "name": "test_avg_reward"},
    "parameters": {
        "beta":          {"distribution": "log_uniform_values", "min": 0.01,  "max": 0.5},
        "learning_rate": {"distribution": "log_uniform_values", "min": 1e-6,  "max": 1e-4},
        "num_epochs":    {"values": [2, 3, 5]},
        "batch_size":    {"values": [2, 4]},
        "pairs_per_mol": {"values": [10, 20, 40]},
    },
}


def run_dpo_trial():
    """Called by wandb agent for each sweep trial."""
    import wandb
    import torch
    from src.policy.llm_synthesis_policy import LLMSynthesisPolicy
    from src.reward.preference_reward_model import build_preference_pairs
    from src.reward.real_reward_model import RealRewardModel
    from src.training.dpo_trainer import DPOConfig, DPOTrainer
    from src.evaluation.real_rlhf_evaluator import evaluate_rlhf_policy
    from src.paths import initialize_directories

    os.environ.setdefault("HF_HOME",             "/storage/gxg8313/saiteja/hf_cache")
    os.environ.setdefault("TRANSFORMERS_CACHE",  "/storage/gxg8313/saiteja/hf_cache")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    with wandb.init() as run:
        cfg = run.config
        device = "cuda" if torch.cuda.is_available() else "cpu"
        initialize_directories()

        data_path = Path(__file__).parent.parent / "data/trajectories_real_ord.jsonl"
        trajectories = []
        with open(data_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    trajectories.append(json.loads(line))

        split      = int(len(trajectories) * 0.8)
        train_traj = trajectories[:split]
        test_traj  = trajectories[split:]

        sft_ckpt = Path(__file__).parent.parent / "models/sft_policy/best"
        if sft_ckpt.exists():
            policy = LLMSynthesisPolicy.from_pretrained(
                str(sft_ckpt), base_model="Qwen/Qwen2.5-7B-Instruct"
            )
        else:
            policy = LLMSynthesisPolicy(
                model_name="Qwen/Qwen2.5-7B-Instruct", device_map={"": device}
            )

        pairs  = build_preference_pairs(train_traj, pairs_per_molecule=cfg.pairs_per_mol)
        config = DPOConfig(
            beta=cfg.beta,
            learning_rate=cfg.learning_rate,
            num_epochs=cfg.num_epochs,
            batch_size=cfg.batch_size,
        )

        trainer     = DPOTrainer(policy=policy, config=config, device=device)
        dpo_results = trainer.train(pairs, use_wandb=True)

        reward_model = RealRewardModel()
        eval_results = evaluate_rlhf_policy(test_traj, reward_model)

        wandb.log({
            "test_avg_reward":  eval_results["average_reward"],
            "test_improvement": eval_results["improvement"],
            "best_margin":      dpo_results.get("best_reward_margin", 0),
        })

        logger.info(
            f"Trial done | β={cfg.beta:.3f} | lr={cfg.learning_rate:.2e} | "
            f"test_reward={eval_results['average_reward']:.4f}"
        )


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--sweep_id", default=None,
                   help="Existing sweep ID to resume. If None, creates a new sweep.")
    p.add_argument("--project",  default="rlhf-synthesis-sweep")
    p.add_argument("--count",    type=int, default=20,
                   help="Number of sweep trials to run")
    return p.parse_args()


def main():
    args = parse_args()

    try:
        import wandb
    except ImportError:
        logger.error("pip install wandb")
        sys.exit(1)

    if args.sweep_id:
        sweep_id = f"{args.project}/{args.sweep_id}"
        logger.info(f"Resuming sweep: {sweep_id}")
    else:
        sweep_id = wandb.sweep(SWEEP_CONFIG, project=args.project)
        logger.info(f"Created sweep: {sweep_id}")
        logger.info(f"  View at: https://wandb.ai/{wandb.api.default_entity}/{args.project}/sweeps/{sweep_id}")

    wandb.agent(sweep_id, function=run_dpo_trial, count=args.count)


if __name__ == "__main__":
    main()
