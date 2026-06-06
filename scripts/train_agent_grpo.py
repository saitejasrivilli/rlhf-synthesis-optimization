#!/usr/bin/env python3
"""
Train an LLM agent with GRPO + verifiable rewards on GSM8K.

This extends the chemistry GRPO pipeline (train_grpo.py) with:
  1. Multi-turn rollouts: model calls python_executor tool, gets result, continues
  2. Verifiable rewards: reward=1 if <final_answer> matches ground truth
  3. No learned reward model — pure RLVR

Architecture:
  Qwen2.5-7B-Instruct + LoRA (r=8) → AgentGRPOTrainer
  G=4 completions per problem → group-relative advantages → clipped policy gradient

Usage:
    # Download dataset first
    python data/create_gsm8k_tool_dataset.py

    # Train (single GPU, 100 iterations for a pilot run)
    CUDA_VISIBLE_DEVICES=0 python scripts/train_agent_grpo.py \\
        --data_file data/gsm8k_train_tool_dataset.jsonl \\
        --num_iterations 100 \\
        --group_size 4

    # With W&B
    python scripts/train_agent_grpo.py --use_wandb --wandb_project rlhf-synthesis

    # Resume from SFT checkpoint
    python scripts/train_agent_grpo.py --sft_ckpt models/sft_policy/best
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
from src.training.grpo_trainer import GRPOConfig
from src.training.agent_grpo_trainer import AgentGRPOTrainer
from src.paths import initialize_directories, get_results_dir

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def load_problems(path: str) -> list:
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
    p.add_argument("--sft_ckpt",       default=None)
    p.add_argument("--data_file",      default="data/gsm8k_train_tool_dataset.jsonl")
    p.add_argument("--group_size",     type=int,   default=4)
    p.add_argument("--beta",           type=float, default=0.04)
    p.add_argument("--clip_ratio",     type=float, default=0.2)
    p.add_argument("--num_iterations", type=int,   default=200)
    p.add_argument("--batch_size",     type=int,   default=2)
    p.add_argument("--learning_rate",  type=float, default=5e-6)
    p.add_argument("--max_new_tokens", type=int,   default=256)
    p.add_argument("--temperature",    type=float, default=0.9)
    p.add_argument("--lora_r",         type=int,   default=8)
    p.add_argument("--lora_alpha",     type=int,   default=16)
    p.add_argument("--output_dir",     default="models/agent_grpo")
    p.add_argument("--use_wandb",      action="store_true")
    p.add_argument("--wandb_project",  default="rlhf-synthesis")
    return p.parse_args()


def evaluate(problems: list, policy, n: int = 50) -> dict:
    """Quick accuracy eval: does the model reach the correct final answer?"""
    from src.reward.verifiable_reward import compute_reward
    from src.training.agent_grpo_trainer import run_agent_rollout, build_agent_prompt

    sample   = problems[:n]
    correct  = 0
    used_tool = 0

    policy.eval()
    with torch.no_grad():
        for item in sample:
            prompt = build_agent_prompt(item["problem"])
            _, _, output = run_agent_rollout(
                policy, prompt, max_new_tokens=256, temperature=0.0
            )
            r = compute_reward(output, item["answer"])
            if r >= 1.0:
                correct += 1
            if "<tool_call>" in output:
                used_tool += 1

    return {
        "accuracy":      correct / len(sample),
        "tool_use_rate": used_tool / len(sample),
        "n":             len(sample),
    }


def main():
    args = parse_args()

    os.environ.setdefault("HF_HOME",                "/storage/gxg8313/saiteja/hf_cache")
    os.environ.setdefault("TRANSFORMERS_CACHE",     "/storage/gxg8313/saiteja/hf_cache")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    initialize_directories()

    if args.use_wandb:
        import wandb
        wandb.init(
            project=args.wandb_project,
            name=f"agent-grpo-G{args.group_size}-{datetime.now().strftime('%m%d-%H%M')}",
            config=vars(args),
        )

    data_path = Path(args.data_file)
    if not data_path.is_absolute():
        data_path = Path(__file__).parent.parent / data_path

    if not data_path.exists():
        logger.error(
            f"Dataset not found: {data_path}\n"
            "Run: python data/create_gsm8k_tool_dataset.py"
        )
        sys.exit(1)

    problems = load_problems(str(data_path))
    split    = int(len(problems) * 0.9)
    train_p  = problems[:split]
    test_p   = problems[split:]
    logger.info(f"Dataset: {len(train_p)} train | {len(test_p)} test")

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

    trainer = AgentGRPOTrainer(policy=policy, config=config, device=device)

    # Baseline accuracy before training
    logger.info("Evaluating baseline (pre-training)...")
    baseline = evaluate(test_p, policy, n=min(50, len(test_p)))
    logger.info(
        f"Baseline: accuracy={baseline['accuracy']:.3f} "
        f"| tool_use={baseline['tool_use_rate']:.3f}"
    )

    train_results = trainer.train(train_p)

    logger.info("Evaluating after training...")
    final_eval = evaluate(test_p, policy, n=min(50, len(test_p)))
    improvement = (final_eval["accuracy"] - baseline["accuracy"]) * 100

    output = {
        "timestamp":  datetime.now().isoformat(),
        "method":     "AgentGRPO",
        "model":      args.model_name,
        "dataset":    str(data_path),
        "group_size": args.group_size,
        "training":   train_results,
        "baseline":   baseline,
        "final_eval": final_eval,
        "improvement_pct": improvement,
    }

    results_file = get_results_dir() / "agent_grpo_results.json"
    results_file.parent.mkdir(parents=True, exist_ok=True)
    with open(results_file, "w") as f:
        json.dump(output, f, indent=2)

    if args.use_wandb:
        import wandb
        wandb.log({
            "eval/accuracy":      final_eval["accuracy"],
            "eval/tool_use_rate": final_eval["tool_use_rate"],
            "eval/improvement":   improvement,
        })
        wandb.finish()

    logger.info("=" * 60)
    logger.info("AGENT GRPO TRAINING COMPLETE")
    logger.info(f"  Group size G        : {args.group_size}")
    logger.info(f"  Best train reward   : {train_results['best_reward']:.4f}")
    logger.info(f"  Baseline accuracy   : {baseline['accuracy']:.3f}")
    logger.info(f"  Final accuracy      : {final_eval['accuracy']:.3f}")
    logger.info(f"  Improvement         : {improvement:+.1f}%")
    logger.info(f"  Tool use rate       : {final_eval['tool_use_rate']:.3f}")
    logger.info(f"  Checkpoint          : {args.output_dir}/best")
    logger.info(f"  Results             : {results_file}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
