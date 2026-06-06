#!/usr/bin/env python3
"""
GRPO training for synthesis policy.

GRPO vs PPO:
  - Generates G completions per prompt (default G=4)
  - Advantages = group-normalized rewards (no value head, no critic)
  - More stable training, lower variance than single-sample PPO

Usage:
    # From SFT checkpoint (recommended)
    CUDA_VISIBLE_DEVICES=0 python scripts/train_grpo.py \
        --sft_ckpt models/sft_policy/best \
        --data_file data/trajectories_real_ord.jsonl \
        --group_size 4 --num_iterations 200

    # With W&B
    python scripts/train_grpo.py --use_wandb --wandb_project rlhf-synthesis
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
from src.reward.real_reward_model import RealRewardModel
from src.training.grpo_trainer import GRPOConfig, GRPOTrainer
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
    p.add_argument("--model_name",      default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--sft_ckpt",        default=None)
    p.add_argument("--data_file",       default="data/trajectories_real_ord.jsonl")
    p.add_argument("--group_size",      type=int,   default=4)
    p.add_argument("--beta",            type=float, default=0.04)
    p.add_argument("--clip_ratio",      type=float, default=0.2)
    p.add_argument("--num_iterations",  type=int,   default=200)
    p.add_argument("--batch_size",      type=int,   default=2)
    p.add_argument("--learning_rate",   type=float, default=5e-6)
    p.add_argument("--max_new_tokens",  type=int,   default=128)
    p.add_argument("--temperature",     type=float, default=0.9)
    p.add_argument("--output_dir",      default="models/grpo_policy")
    p.add_argument("--lora_r",          type=int,   default=8)
    p.add_argument("--lora_alpha",      type=int,   default=16)
    p.add_argument("--use_wandb",       action="store_true")
    p.add_argument("--wandb_project",   default="rlhf-synthesis")
    return p.parse_args()


def main():
    args = parse_args()

    os.environ.setdefault("HF_HOME",             "/storage/gxg8313/saiteja/hf_cache")
    os.environ.setdefault("TRANSFORMERS_CACHE",  "/storage/gxg8313/saiteja/hf_cache")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    initialize_directories()

    if args.use_wandb:
        import wandb
        wandb.init(
            project=args.wandb_project,
            name=f"grpo-G{args.group_size}-{datetime.now().strftime('%m%d-%H%M')}",
            config=vars(args),
        )

    data_path = Path(args.data_file)
    if not data_path.is_absolute():
        data_path = Path(__file__).parent.parent / data_path

    trajectories = load_trajectories(data_path)
    split        = int(len(trajectories) * 0.8)
    train_traj   = trajectories[:split]
    test_traj    = trajectories[split:]
    logger.info(f"Data: {len(train_traj)} train | {len(test_traj)} test")

    # Policy
    if args.sft_ckpt and Path(args.sft_ckpt).exists():
        logger.info(f"Loading SFT checkpoint: {args.sft_ckpt}")
        policy = LLMSynthesisPolicy.from_pretrained(args.sft_ckpt, base_model=args.model_name)
    else:
        policy = LLMSynthesisPolicy(
            model_name=args.model_name,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            device_map={"": device},
        )

    reward_model = RealRewardModel()

    config = GRPOConfig(
        model_name=args.model_name,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        group_size=args.group_size,
        beta=args.beta,
        clip_ratio=args.clip_ratio,
        num_iterations=args.num_iterations,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        output_dir=args.output_dir,
    )

    trainer = GRPOTrainer(
        policy=policy,
        reward_model=reward_model,
        config=config,
        device=device,
    )

    grpo_results = trainer.train(train_traj)

    eval_results = evaluate_rlhf_policy(test_traj, reward_model)

    output = {
        "timestamp":    datetime.now().isoformat(),
        "method":       "GRPO",
        "model":        args.model_name,
        "sft_ckpt":     args.sft_ckpt,
        "group_size":   args.group_size,
        "beta":         args.beta,
        "dataset": {
            "total": len(trajectories),
            "train": len(train_traj),
            "test":  len(test_traj),
        },
        "training":   grpo_results,
        "evaluation": eval_results,
    }

    results_file = get_results_dir() / "grpo_results.json"
    save_rlhf_results(output, results_file)

    if args.use_wandb:
        import wandb
        wandb.log({"test/avg_reward": eval_results["average_reward"],
                   "test/improvement": eval_results["improvement"]})
        wandb.finish()

    logger.info("=" * 60)
    logger.info("GRPO TRAINING COMPLETE")
    logger.info(f"  Group size G       : {args.group_size}")
    logger.info(f"  Best train reward  : {grpo_results['best_reward']:.4f}")
    logger.info(f"  Test avg reward    : {eval_results['average_reward']:.4f}")
    logger.info(f"  Test improvement   : {eval_results['improvement']:+.1f}%")
    logger.info(f"  Checkpoint         : {args.output_dir}/best")
    logger.info(f"  Results            : {results_file}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
