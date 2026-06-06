#!/usr/bin/env python3
"""
DPO training entry point for synthesis policy.

Pipeline:
  1. Load preference pairs from trajectory data
  2. (Optional) Start from SFT checkpoint
  3. Train with DPO loss (no separate reward model)
  4. Evaluate on held-out test set

Usage:
    # From pretrained base model
    python scripts/train_dpo.py \
        --data_file data/trajectories_real_ord.jsonl \
        --num_epochs 3 --output_dir models/dpo_policy

    # From SFT checkpoint (recommended)
    python scripts/train_dpo.py \
        --sft_ckpt models/sft_policy/best \
        --data_file data/trajectories_real_ord.jsonl
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

from src.policy.llm_synthesis_policy import LLMSynthesisPolicy
from src.reward.preference_reward_model import build_preference_pairs, PreferenceRewardModel
from src.reward.real_reward_model import RealRewardModel
from src.training.dpo_trainer import DPOConfig, DPOTrainer
from src.evaluation.real_rlhf_evaluator import evaluate_rlhf_policy, save_rlhf_results
from src.paths import initialize_directories, get_results_dir

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def load_trajectories(path):
    data = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name",     default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--sft_ckpt",       default=None,
                   help="Path to SFT checkpoint (recommended starting point)")
    p.add_argument("--data_file",      default="data/trajectories_real_ord.jsonl")
    p.add_argument("--beta",           type=float, default=0.1)
    p.add_argument("--num_epochs",     type=int,   default=3)
    p.add_argument("--batch_size",     type=int,   default=4)
    p.add_argument("--learning_rate",  type=float, default=1e-5)
    p.add_argument("--pairs_per_mol",  type=int,   default=20,
                   help="Preference pairs per molecule")
    p.add_argument("--output_dir",     default="models/dpo_policy")
    p.add_argument("--lora_r",         type=int,   default=8)
    p.add_argument("--lora_alpha",     type=int,   default=16)
    p.add_argument("--use_wandb",      action="store_true")
    return p.parse_args()


def main():
    args = parse_args()

    os.environ.setdefault("HF_HOME",              "/storage/gxg8313/saiteja/hf_cache")
    os.environ.setdefault("TRANSFORMERS_CACHE",   "/storage/gxg8313/saiteja/hf_cache")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    initialize_directories()

    if args.use_wandb:
        import wandb
        wandb.init(
            project="rlhf-synthesis",
            name=f"dpo-{datetime.now().strftime('%m%d-%H%M')}",
            config=vars(args),
        )

    # -----------------------------------------------------------------------
    # Data + preference pairs
    # -----------------------------------------------------------------------
    data_path = Path(args.data_file)
    if not data_path.is_absolute():
        data_path = Path(__file__).parent.parent / data_path

    trajectories = load_trajectories(data_path)
    split        = int(len(trajectories) * 0.8)
    train_traj   = trajectories[:split]
    test_traj    = trajectories[split:]

    pairs = build_preference_pairs(train_traj, pairs_per_molecule=args.pairs_per_mol)
    logger.info(f"Dataset: {len(train_traj)} train | {len(test_traj)} test | {len(pairs)} pairs")

    # -----------------------------------------------------------------------
    # Policy
    # -----------------------------------------------------------------------
    if args.sft_ckpt and Path(args.sft_ckpt).exists():
        logger.info(f"Loading from SFT checkpoint: {args.sft_ckpt}")
        policy = LLMSynthesisPolicy.from_pretrained(args.sft_ckpt, base_model=args.model_name)
    else:
        logger.info(f"Loading base model: {args.model_name}")
        policy = LLMSynthesisPolicy(
            model_name=args.model_name,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            device_map={"": device},
        )

    # -----------------------------------------------------------------------
    # DPO training
    # -----------------------------------------------------------------------
    config = DPOConfig(
        beta=args.beta,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        output_dir=args.output_dir,
    )

    trainer = DPOTrainer(policy=policy, config=config, device=device)
    dpo_results = trainer.train(pairs, use_wandb=args.use_wandb)

    # -----------------------------------------------------------------------
    # Evaluation
    # -----------------------------------------------------------------------
    reward_model = RealRewardModel()
    eval_results = evaluate_rlhf_policy(test_traj, reward_model)

    output = {
        "timestamp":   datetime.now().isoformat(),
        "method":      "DPO",
        "model":       args.model_name,
        "sft_ckpt":    args.sft_ckpt,
        "beta":        args.beta,
        "dataset": {
            "total": len(trajectories),
            "train": len(train_traj),
            "test":  len(test_traj),
            "pairs": len(pairs),
        },
        "dpo_training": dpo_results,
        "evaluation":   eval_results,
    }

    results_file = get_results_dir() / "dpo_results.json"
    save_rlhf_results(output, results_file)

    if args.use_wandb:
        import wandb
        wandb.log({"test_avg_reward": eval_results["average_reward"],
                   "test_improvement": eval_results["improvement"]})
        wandb.finish()

    logger.info("=" * 60)
    logger.info("DPO TRAINING COMPLETE")
    logger.info(f"  Best reward margin : {dpo_results['best_reward_margin']:.4f}")
    logger.info(f"  Test avg reward    : {eval_results['average_reward']:.4f}")
    logger.info(f"  Test improvement   : {eval_results['improvement']:+.1f}%")
    logger.info(f"  Checkpoint         : {args.output_dir}/best")
    logger.info(f"  Results            : {results_file}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
