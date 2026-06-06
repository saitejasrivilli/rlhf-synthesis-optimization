#!/usr/bin/env python3
"""
Closed-loop hill climbing: rejection sampling + Agent GRPO over R rounds.

Each round:
  1. Generate G rollouts per problem with the current policy
  2. Keep rollouts with reward >= threshold (rejection sampling)
  3. Augment the training dataset with kept trajectories
  4. Run Agent GRPO on the augmented dataset for N iterations
  5. Eval on held-out set; checkpoint if accuracy improves

Usage
-----
# Quick smoke test (2 rounds, 5 train problems, 10 GRPO iters/round):
CUDA_VISIBLE_DEVICES=0 python scripts/hill_climb.py \
    --data_file data/gsm8k_train_tool_dataset.jsonl \
    --num_rounds 2 \
    --grpo_iters_per_round 10 \
    --num_train_problems 10

# Full run:
CUDA_VISIBLE_DEVICES=0 python scripts/hill_climb.py \
    --data_file data/gsm8k_train_tool_dataset.jsonl \
    --num_rounds 5 \
    --rollouts_per_problem 4 \
    --grpo_iters_per_round 50 \
    --reward_threshold 0.5 \
    --output_dir models/hill_climb_5r \
    2>&1 | tee logs/hill_climb.log
"""
import argparse
import json
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

os.environ.setdefault("TRANSFORMERS_CACHE", "/storage/gxg8313/saiteja/hf_cache")
os.environ.setdefault("HF_HOME",            "/storage/gxg8313/saiteja/hf_cache")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_file",            default="data/gsm8k_train_tool_dataset.jsonl")
    p.add_argument("--output_dir",           default="models/hill_climb")
    p.add_argument("--model_name",           default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--num_rounds",           type=int,   default=5)
    p.add_argument("--rollouts_per_problem", type=int,   default=4)
    p.add_argument("--reward_threshold",     type=float, default=0.5)
    p.add_argument("--top_k_per_problem",    type=int,   default=2)
    p.add_argument("--grpo_iters_per_round", type=int,   default=50)
    p.add_argument("--grpo_group_size",      type=int,   default=4)
    p.add_argument("--grpo_batch_size",      type=int,   default=2)
    p.add_argument("--max_new_tokens",       type=int,   default=200)
    p.add_argument("--eval_problems",        type=int,   default=20)
    p.add_argument("--num_train_problems",   type=int,   default=0,
                   help="Limit training set size (0 = use all)")
    p.add_argument("--use_wandb",            action="store_true",
                   help="Log to W&B (must call wandb.init() before training)")
    return p.parse_args()


def main():
    args = parse_args()

    from src.policy.llm_synthesis_policy import LLMSynthesisPolicy
    from src.training.hill_climber import HillClimber, HillClimbConfig

    # Optionally slice dataset for smoke tests
    if args.num_train_problems > 0:
        original = args.data_file
        rows = []
        with open(original) as f:
            for line in f:
                rows.append(line)
        sliced = Path(args.output_dir) / "sliced_train.jsonl"
        sliced.parent.mkdir(parents=True, exist_ok=True)
        n = min(args.num_train_problems + args.eval_problems, len(rows))
        with open(sliced, "w") as f:
            f.writelines(rows[:n])
        data_file = str(sliced)
        logger.info("Using sliced dataset: %d problems", n)
    else:
        data_file = args.data_file

    logger.info("Loading policy: %s", args.model_name)
    policy = LLMSynthesisPolicy(model_name=args.model_name)

    cfg = HillClimbConfig(
        data_file            = data_file,
        output_dir           = args.output_dir,
        num_rounds           = args.num_rounds,
        rollouts_per_problem = args.rollouts_per_problem,
        reward_threshold     = args.reward_threshold,
        top_k_per_problem    = args.top_k_per_problem,
        grpo_iters_per_round = args.grpo_iters_per_round,
        grpo_group_size      = args.grpo_group_size,
        grpo_batch_size      = args.grpo_batch_size,
        max_new_tokens       = args.max_new_tokens,
        eval_problems        = args.eval_problems,
    )

    climber = HillClimber(policy, cfg)
    results = climber.run()

    print("\n=== Hill Climb Results ===")
    print(f"{'Round':<8} {'Accuracy':<12} {'Tool Use':<12} {'Avg Reward':<14} {'Dataset'}")
    print("-" * 60)
    for r in results:
        print(f"{r['round']:<8} {r['accuracy']:<12.3f} {r['tool_use']:<12.3f} "
              f"{r['avg_reward']:<14.3f} {r['dataset_size']}")

    if len(results) >= 2:
        delta = results[-1]["accuracy"] - results[0]["accuracy"]
        print(f"\nAccuracy: {results[0]['accuracy']:.3f} → {results[-1]['accuracy']:.3f} "
              f"({delta:+.3f} over {args.num_rounds} rounds)")
        print(f"Dataset:  {results[0]['dataset_size']} → {results[-1]['dataset_size']} problems")


if __name__ == "__main__":
    main()
