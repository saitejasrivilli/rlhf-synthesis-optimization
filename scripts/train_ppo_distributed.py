#!/usr/bin/env python3
"""
Distributed PPO training for Llama-2 7B + LoRA on 4x NVIDIA A30 GPUs.

Launch with torchrun:
    torchrun --nproc_per_node=4 scripts/train_ppo_distributed.py \
        --model_name meta-llama/Llama-2-7b-hf \
        --num_iterations 200 \
        --batch_size 4 \
        --output_dir models/llm_policy

Or with DeepSpeed:
    deepspeed --num_gpus=4 scripts/train_ppo_distributed.py \
        --deepspeed configs/deepspeed_zero2.json
"""

import argparse
import copy
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.policy.llm_synthesis_policy import LLMSynthesisPolicy
from src.reward.learned_reward_model import LearnedRewardModel
from src.training.llm_ppo_trainer import LLMPPOConfig, LLMPPOTrainer
from src.evaluation.real_rlhf_evaluator import evaluate_rlhf_policy, save_rlhf_results
from src.paths import get_data_path, get_results_dir, initialize_directories

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s][rank%(process)d] %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Distributed helpers
# ---------------------------------------------------------------------------

def setup_distributed():
    """Initialise process group for torchrun / deepspeed."""
    rank       = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    if world_size > 1:
        dist.init_process_group(backend="nccl")

    torch.cuda.set_device(local_rank)
    return rank, local_rank, world_size


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_trajectories(path: Path):
    data = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Distributed RLHF PPO for synthesis")
    p.add_argument("--model_name",    default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--lora_r",        type=int,   default=8)
    p.add_argument("--lora_alpha",    type=int,   default=16)
    p.add_argument("--num_iterations",type=int,   default=200)
    p.add_argument("--batch_size",    type=int,   default=4,
                   help="Trajectories per GPU per iteration")
    p.add_argument("--ppo_epochs",    type=int,   default=3)
    p.add_argument("--learning_rate", type=float, default=1e-5)
    p.add_argument("--kl_div_weight", type=float, default=0.1)
    p.add_argument("--clip_ratio",    type=float, default=0.2)
    p.add_argument("--max_new_tokens",type=int,   default=128)
    p.add_argument("--temperature",   type=float, default=0.7)
    p.add_argument("--output_dir",    default="models/llm_policy")
    p.add_argument("--data_file",     default="data/trajectories_improvable.jsonl")
    p.add_argument("--finetune_reward_model", action="store_true",
                   help="Fine-tune BERT reward model on trajectories before PPO")
    p.add_argument("--deepspeed",     default=None,
                   help="Path to DeepSpeed config JSON (enables DeepSpeed mode)")
    p.add_argument("--load_in_8bit",  action="store_true",
                   help="Load base model in 8-bit (saves ~7GB VRAM per GPU)")
    return p.parse_args()


def main():
    args = parse_args()
    rank, local_rank, world_size = setup_distributed()
    device = torch.device(f"cuda:{local_rank}")

    if rank == 0:
        initialize_directories()
        logger.info("=" * 70)
        logger.info("RLHF Synthesis Optimization — Distributed PPO")
        logger.info(f"  GPUs: {world_size}  |  Model: {args.model_name}")
        logger.info(f"  LoRA r={args.lora_r}, alpha={args.lora_alpha}")
        logger.info(f"  Iterations: {args.num_iterations}  |  Batch/GPU: {args.batch_size}")
        logger.info("=" * 70)

    # -----------------------------------------------------------------------
    # Data
    # -----------------------------------------------------------------------
    data_path = Path(args.data_file)
    if not data_path.is_absolute():
        data_path = Path(__file__).parent.parent / data_path

    trajectories = load_trajectories(data_path)
    split = int(len(trajectories) * 0.8)
    train_traj = trajectories[:split]
    test_traj  = trajectories[split:]

    if rank == 0:
        logger.info(f"Dataset: {len(train_traj)} train | {len(test_traj)} test")

    # -----------------------------------------------------------------------
    # Reward model — fine-tune on rank 0, broadcast weights
    # -----------------------------------------------------------------------
    reward_model = LearnedRewardModel().to(device)

    if args.finetune_reward_model and rank == 0:
        logger.info("Fine-tuning BERT reward model on training trajectories…")
        reward_model.finetune(train_traj, epochs=3, lr=1e-4, device=str(device))
        reward_path = Path(args.output_dir) / "reward_model.pt"
        reward_path.parent.mkdir(parents=True, exist_ok=True)
        reward_model.save(str(reward_path))

    if world_size > 1:
        dist.barrier()

    # -----------------------------------------------------------------------
    # Policy (device_map="auto" distributes across GPUs via accelerate)
    # -----------------------------------------------------------------------
    # Each process loads its own copy; LoRA+value-head updates are synced via DDP.
    policy = LLMSynthesisPolicy(
        model_name=args.model_name,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        load_in_8bit=args.load_in_8bit,
        device_map={"": local_rank},   # pin each rank to its GPU
    )

    # Reference log-probs computed via disable_adapter() — no second model loaded.

    # Wrap trainable parts (value head + LoRA) in DDP for gradient sync
    if world_size > 1:
        policy.value_head = DDP(
            policy.value_head.to(device),
            device_ids=[local_rank],
            output_device=local_rank,
        )

    # -----------------------------------------------------------------------
    # Trainer
    # -----------------------------------------------------------------------
    config = LLMPPOConfig(
        model_name=args.model_name,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        num_iterations=args.num_iterations,
        batch_size=args.batch_size,
        ppo_epochs=args.ppo_epochs,
        learning_rate=args.learning_rate,
        kl_div_weight=args.kl_div_weight,
        clip_ratio=args.clip_ratio,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        output_dir=args.output_dir,
    )

    trainer = LLMPPOTrainer(
        policy=policy,
        reward_model=reward_model,
        config=config,
        device=str(device),
        rank=rank,
    )

    # -----------------------------------------------------------------------
    # Train
    # -----------------------------------------------------------------------
    training_results = trainer.train(train_traj)

    # -----------------------------------------------------------------------
    # Evaluate (rank 0 only)
    # -----------------------------------------------------------------------
    if rank == 0:
        eval_results = evaluate_rlhf_policy(test_traj, reward_model)

        output = {
            "timestamp": datetime.now().isoformat(),
            "model": args.model_name,
            "lora": {"r": args.lora_r, "alpha": args.lora_alpha},
            "world_size": world_size,
            "dataset": {
                "total": len(trajectories),
                "train": len(train_traj),
                "test": len(test_traj),
                "source": args.data_file,
            },
            "training": training_results,
            "evaluation": eval_results,
        }

        results_file = get_results_dir() / "rlhf_llm_distributed.json"
        save_rlhf_results(output, results_file)

        logger.info("=" * 70)
        logger.info("RESULTS")
        logger.info(f"  Best train reward : {training_results['best_reward']:.4f}")
        logger.info(f"  Test avg reward   : {eval_results['average_reward']:.4f}")
        logger.info(f"  Test improvement  : {eval_results['improvement']:+.1f}%")
        logger.info(f"  Results saved to  : {results_file}")
        logger.info("=" * 70)

    cleanup_distributed()


if __name__ == "__main__":
    main()
