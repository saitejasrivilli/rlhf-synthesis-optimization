#!/usr/bin/env python3
"""
Supervised Fine-Tuning (SFT) warmup before RLHF/DPO.

Standard RLHF pipeline: SFT → Reward Model → PPO  (or SFT → DPO)

This script fine-tunes the LLM on high-yield trajectories (yield > threshold)
using standard next-token cross-entropy loss. The SFT checkpoint then serves
as the starting point for PPO/DPO, giving the policy a strong prior over
high-quality synthesis conditions.

Why SFT before PPO?
  - Prevents early training instability (policy starts near-optimal)
  - Reduces KL divergence between initial and reference policy
  - Empirically reduces PPO iterations needed by ~50%

Usage:
    python scripts/sft_warmup.py \
        --data_file data/trajectories_real_ord.jsonl \
        --yield_threshold 0.85 \
        --num_epochs 3 \
        --output_dir models/sft_policy
"""
import argparse
import json
import logging
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.paths import initialize_directories, get_results_dir
from src.reward.real_reward_model import RealRewardModel

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


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


def filter_high_yield(trajectories, threshold: float):
    rm = RealRewardModel()
    good = [t for t in trajectories if rm.score_trajectory(t) >= threshold]
    logger.info(
        f"SFT dataset: {len(good)}/{len(trajectories)} trajectories "
        f"above reward threshold {threshold}"
    )
    return good


def make_sft_pairs(trajectories):
    """Build (prompt, target) pairs for SFT."""
    from src.policy.llm_synthesis_policy import LLMSynthesisPolicy  # for prompt format
    pairs = []
    for traj in trajectories:
        prompt = (
            f"### Optimize pharmaceutical synthesis conditions\n\n"
            f"Molecule: {traj.get('molecule', 'Unknown')}\n"
            f"CAS: {traj.get('cas_number', 'N/A')}\n"
            f"Objective: maximize yield and selectivity, minimize hazard and steps\n\n"
            f"### Optimal conditions (JSON):\n"
        )
        params   = traj.get("parameters", traj)
        outcomes = traj.get("outcomes", {})
        cond = {
            "temperature_celsius":   params.get("temperature_celsius",  traj.get("temperature_celsius",  80.0)),
            "time_hours":            params.get("time_hours",            traj.get("time_hours",            4.0)),
            "catalyst_loading_M":    params.get("catalyst_loading_M",   traj.get("catalyst_loading_M",   0.05)),
            "solvent_ratio_ml_mmol": params.get("solvent_ratio_ml_mmol",traj.get("solvent_ratio_ml_mmol", 3.0)),
        }
        target = json.dumps(cond)
        pairs.append((prompt, target))
    return pairs


# ---------------------------------------------------------------------------
# SFT training loop
# ---------------------------------------------------------------------------

def sft_train(policy, pairs, config):
    import random

    trainable = [p for p in policy.model.parameters() if p.requires_grad]
    opt    = torch.optim.AdamW(trainable, lr=config.learning_rate)
    scaler = torch.amp.GradScaler("cuda")
    tok    = policy.tokenizer

    history = []
    best_loss = float("inf")
    output_path = Path(config.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    device = next(policy.model.parameters()).device

    for epoch in range(config.num_epochs):
        random.shuffle(pairs)
        ep_loss = 0.0
        n_batches = 0

        for i in range(0, len(pairs), config.batch_size):
            batch = pairs[i : i + config.batch_size]
            prompts  = [p for p, _ in batch]
            targets  = [t for _, t in batch]
            combined = [p + t for p, t in batch]

            enc = tok(
                combined,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            ).to(device)

            labels = enc["input_ids"].clone()
            for j, (prompt, _) in enumerate(batch):
                prompt_len = len(tok(prompt, add_special_tokens=False).input_ids)
                labels[j, :prompt_len] = -100   # mask prompt tokens

            policy.model.train()
            with torch.amp.autocast("cuda"):
                out  = policy.model(**enc, labels=labels)
                loss = out.loss

            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            scaler.step(opt)
            scaler.update()
            opt.zero_grad()

            ep_loss   += loss.item()
            n_batches += 1

        avg_loss = ep_loss / max(1, n_batches)
        history.append(avg_loss)
        logger.info(f"SFT Epoch {epoch+1}/{config.num_epochs}: loss={avg_loss:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            policy.save_pretrained(str(output_path / "best"))

    return history, best_loss


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name",       default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--data_file",        default="data/trajectories_real_ord.jsonl")
    p.add_argument("--yield_threshold",  type=float, default=0.80,
                   help="Minimum reward score to include in SFT dataset")
    p.add_argument("--num_epochs",       type=int,   default=3)
    p.add_argument("--batch_size",       type=int,   default=4)
    p.add_argument("--learning_rate",    type=float, default=2e-5)
    p.add_argument("--output_dir",       default="models/sft_policy")
    p.add_argument("--lora_r",           type=int,   default=8)
    p.add_argument("--lora_alpha",       type=int,   default=16)
    return p.parse_args()


def main():
    args   = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    initialize_directories()

    import os
    os.environ.setdefault("HF_HOME", "/storage/gxg8313/saiteja/hf_cache")

    # ---- Load data ----
    data_path = Path(args.data_file)
    if not data_path.is_absolute():
        data_path = Path(__file__).parent.parent / data_path

    all_trajs = load_trajectories(data_path)
    high_trajs = filter_high_yield(all_trajs, args.yield_threshold)

    if not high_trajs:
        logger.error(f"No trajectories above threshold {args.yield_threshold}. Try lowering it.")
        return

    pairs = make_sft_pairs(high_trajs)
    logger.info(f"SFT training on {len(pairs)} (prompt, target) pairs")

    # ---- Load policy ----
    from src.policy.llm_synthesis_policy import LLMSynthesisPolicy

    policy = LLMSynthesisPolicy(
        model_name=args.model_name,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        device_map={"": device},
    )

    # ---- Train ----
    class Cfg:
        learning_rate = args.learning_rate
        num_epochs    = args.num_epochs
        batch_size    = args.batch_size
        output_dir    = args.output_dir

    history, best_loss = sft_train(policy, pairs, Cfg())

    # ---- Save results ----
    results = {
        "model":            args.model_name,
        "data_file":        str(data_path),
        "yield_threshold":  args.yield_threshold,
        "num_sft_pairs":    len(pairs),
        "num_epochs":       args.num_epochs,
        "loss_history":     history,
        "best_loss":        best_loss,
        "checkpoint":       str(Path(args.output_dir) / "best"),
    }

    out = get_results_dir() / "sft_results.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)

    logger.info("=" * 60)
    logger.info("SFT WARMUP COMPLETE")
    logger.info(f"  Pairs trained on : {len(pairs)}")
    logger.info(f"  Best loss        : {best_loss:.4f}")
    logger.info(f"  Checkpoint       : {args.output_dir}/best")
    logger.info(f"  Results          : {out}")
    logger.info("=" * 60)
    logger.info("Next step → run PPO/DPO starting from this checkpoint:")
    logger.info(f"  python scripts/train_ppo_distributed.py --model_name {args.output_dir}/best")
    logger.info(f"  python scripts/train_dpo.py --sft_ckpt {args.output_dir}/best")


if __name__ == "__main__":
    main()
