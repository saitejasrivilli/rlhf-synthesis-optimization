#!/usr/bin/env python3
"""
DAPO training for synthesis policy.

DAPO (Decoupled Clip and Dynamic Sampling Policy Optimization) improves GRPO with:
  1. Asymmetric clip bounds: clip_low=0.2, clip_high=0.28 — separate lower/upper clipping
     prevents large probability ratios from dominating while still allowing upward learning
  2. Token-level policy gradient: loss computed per token, not per sequence, which
     reduces variance on long responses and prevents length bias
  3. Zero-advantage filtering: skip groups where reward.std() < 1e-6 — all-same-reward
     groups contribute no learning signal and only add noise
  4. Entropy bonus: +coeff * entropy on response tokens — encourages diversity and
     prevents premature mode collapse observed in GRPO

Reference: DAPO paper (ByteDance / Seed, 2025)

Usage:
    CUDA_VISIBLE_DEVICES=0 python scripts/train_dapo.py \\
        --sft_ckpt models/sft_policy/best \\
        --data_file data/trajectories_real_ord.jsonl \\
        --group_size 4 --num_iterations 200

    # With W&B
    python scripts/train_dapo.py --use_wandb --wandb_project rlhf-synthesis
"""
import argparse
import json
import logging
import os
import random
import sys
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.policy.llm_synthesis_policy import LLMSynthesisPolicy
from src.reward.real_reward_model import RealRewardModel
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
    p.add_argument("--beta",            type=float, default=0.04,  help="KL penalty weight")
    p.add_argument("--clip_low",        type=float, default=0.2,   help="DAPO asymmetric lower clip")
    p.add_argument("--clip_high",       type=float, default=0.28,  help="DAPO asymmetric upper clip")
    p.add_argument("--entropy_coeff",   type=float, default=0.001, help="Entropy bonus coefficient")
    p.add_argument("--num_iterations",  type=int,   default=200)
    p.add_argument("--batch_size",      type=int,   default=2)
    p.add_argument("--learning_rate",   type=float, default=5e-6)
    p.add_argument("--max_new_tokens",  type=int,   default=128)
    p.add_argument("--temperature",     type=float, default=0.9)
    p.add_argument("--output_dir",      default="models/dapo_policy")
    p.add_argument("--lora_r",          type=int,   default=8)
    p.add_argument("--lora_alpha",      type=int,   default=16)
    p.add_argument("--log_interval",    type=int,   default=10)
    p.add_argument("--save_interval",   type=int,   default=50)
    p.add_argument("--use_wandb",       action="store_true")
    p.add_argument("--wandb_project",   default="rlhf-synthesis")
    return p.parse_args()


def _model_device(policy):
    return next(policy.model.parameters()).device


def _generate_group(policy, prompt, group_size, max_new_tokens, temperature):
    """Generate group_size completions for a single prompt. Returns (query_ids, response_ids)."""
    enc = policy.tokenizer(
        [prompt] * group_size,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    ).to(_model_device(policy))

    with torch.no_grad():
        out = policy.model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=0.95,
            pad_token_id=policy.tokenizer.eos_token_id,
        )

    q_len = enc["input_ids"].shape[1]
    return enc["input_ids"], out[:, q_len:]   # [G, Q], [G, R]


def _token_logprobs_and_entropy(policy, query_ids, response_ids):
    """
    Compute per-token log-probs and per-token entropy over response tokens.

    Returns:
        token_lp:      [G, R]  — log-prob of each response token
        token_entropy: [G, R]  — entropy H = -sum(p * log p) at each position
    """
    full_ids  = torch.cat([query_ids, response_ids], dim=1)   # [G, Q+R]
    attn_mask = (full_ids != policy.tokenizer.pad_token_id).long()

    out = policy.model(input_ids=full_ids, attention_mask=attn_mask)

    logits = out.logits[:, :-1, :].float()    # [G, Q+R-1, V]
    labels = full_ids[:, 1:]                   # [G, Q+R-1]
    G, T   = labels.shape

    log_probs = F.log_softmax(logits, dim=-1)  # [G, T, V]

    # Per-token log-prob of the actual token
    token_lp = log_probs.gather(2, labels.unsqueeze(-1)).squeeze(-1)  # [G, T]

    # Per-token entropy: H = -sum(p * log_p)
    probs         = log_probs.exp()
    token_entropy = -(probs * log_probs).sum(dim=-1)                   # [G, T]

    q_len = query_ids.shape[1]
    return token_lp[:, q_len - 1:], token_entropy[:, q_len - 1:]      # [G, R]


def _ref_token_logprobs(policy, query_ids, response_ids):
    """Log-probs from the frozen reference (base model, adapters disabled)."""
    full_ids  = torch.cat([query_ids, response_ids], dim=1)
    attn_mask = (full_ids != policy.tokenizer.pad_token_id).long()

    with torch.no_grad(), policy.model.disable_adapter():
        out = policy.model(input_ids=full_ids, attention_mask=attn_mask)

    logits = out.logits[:, :-1, :].float()
    labels = full_ids[:, 1:]
    G, T   = labels.shape
    token_lp = F.log_softmax(logits, dim=-1).gather(
        2, labels.unsqueeze(-1)
    ).squeeze(-1)

    q_len = query_ids.shape[1]
    return token_lp[:, q_len - 1:]   # [G, R]


def dapo_step(
    policy,
    reward_model,
    trajectories,
    optimizer,
    scaler,
    cfg,
    device,
):
    """
    Single DAPO training step.

    Returns metrics dict with mean_reward, entropy, groups_skipped, kl_div, loss.
    """
    G              = cfg["group_size"]
    clip_low       = cfg["clip_low"]
    clip_high      = cfg["clip_high"]
    beta           = cfg["beta"]
    entropy_coeff  = cfg["entropy_coeff"]
    max_new_tokens = cfg["max_new_tokens"]
    temperature    = cfg["temperature"]

    prompts = [policy.build_prompt(t) for t in trajectories]

    # ---- 1. Sample G completions per prompt (no_grad) ----
    policy.eval()
    all_q_ids, all_r_ids, all_old_lp = [], [], []

    with torch.no_grad():
        for prompt in prompts:
            q_ids, r_ids = _generate_group(policy, prompt, G, max_new_tokens, temperature)
            old_lp, _    = _token_logprobs_and_entropy(policy, q_ids, r_ids)
            all_q_ids.append(q_ids)
            all_r_ids.append(r_ids)
            all_old_lp.append(old_lp)

    # ---- 2. Reference log-probs ----
    all_ref_lp = []
    with torch.no_grad():
        for q_ids, r_ids in zip(all_q_ids, all_r_ids):
            ref_lp = _ref_token_logprobs(policy, q_ids, r_ids)
            all_ref_lp.append(ref_lp)

    # ---- 3. Rewards and group-relative advantages ----
    all_rewards    = []
    all_advantages = []
    groups_skipped = 0

    for i, traj in enumerate(trajectories):
        r_ids      = all_r_ids[i]
        conditions = policy.decode_conditions(r_ids)
        rewards    = torch.tensor(
            [reward_model.score_trajectory({**traj, **c}) for c in conditions],
            dtype=torch.float32, device=device,
        )

        # DAPO: skip zero-advantage groups (all rewards identical)
        if rewards.std() < 1e-6:
            groups_skipped += 1
            all_rewards.append(rewards)
            all_advantages.append(None)
            continue

        adv = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        all_rewards.append(rewards)
        all_advantages.append(adv)

    # ---- 4. Policy update ----
    policy.train()
    total_loss    = 0.0
    total_reward  = 0.0
    total_entropy = 0.0
    total_kl      = 0.0
    n_updated     = 0

    optimizer.zero_grad()

    for i in range(len(trajectories)):
        if all_advantages[i] is None:
            total_reward += all_rewards[i].mean().item()
            continue

        q_ids  = all_q_ids[i]
        r_ids  = all_r_ids[i]
        old_lp = all_old_lp[i].detach()    # [G, R]
        ref_lp = all_ref_lp[i].detach()    # [G, R]
        adv    = all_advantages[i].detach() # [G]

        with torch.amp.autocast("cuda"):
            new_lp, entropy = _token_logprobs_and_entropy(policy, q_ids, r_ids)

            # Token-level probability ratio
            log_ratio  = new_lp - old_lp              # [G, R]
            ratio      = torch.exp(log_ratio)          # [G, R]

            # DAPO asymmetric clip: separate lower and upper bounds
            adv_exp = adv.unsqueeze(-1).expand_as(ratio)   # [G, R]

            surr1  = ratio * adv_exp
            # clip_high for positive adv, clip_low for negative adv (asymmetric)
            ratio_clipped = torch.where(
                adv_exp >= 0,
                torch.clamp(ratio, 1.0 - clip_low,  1.0 + clip_high),
                torch.clamp(ratio, 1.0 - clip_high, 1.0 + clip_low),
            )
            surr2  = ratio_clipped * adv_exp

            # Token-level policy loss (mean over tokens, then over group)
            policy_loss = -torch.min(surr1, surr2).mean()

            # Entropy bonus (mean per-token entropy, averaged over group)
            entropy_bonus = entropy_coeff * entropy.mean()

            # KL penalty: token-level, averaged
            kl = (ref_lp - new_lp).mean().clamp(min=0)

            loss = policy_loss - entropy_bonus + beta * kl

        scaler.scale(loss).backward()

        total_loss    += loss.item()
        total_reward  += all_rewards[i].mean().item()
        total_entropy += entropy.mean().item()
        total_kl      += kl.item()
        n_updated     += 1

    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(
        [p for p in policy.model.parameters() if p.requires_grad], 1.0
    )
    scaler.step(optimizer)
    scaler.update()

    n = len(trajectories)
    denom = max(n_updated, 1)
    return {
        "loss":           total_loss    / denom,
        "mean_reward":    total_reward  / n,
        "entropy":        total_entropy / denom,
        "kl_div":         total_kl      / denom,
        "groups_skipped": groups_skipped,
    }


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
            project=args.wandb_project,
            name=f"dapo-G{args.group_size}-{datetime.now().strftime('%m%d-%H%M')}",
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

    trainable = [p for p in policy.model.parameters() if p.requires_grad]
    optimizer  = torch.optim.AdamW(trainable, lr=args.learning_rate)
    scaler     = torch.amp.GradScaler("cuda")

    cfg = {
        "group_size":      args.group_size,
        "clip_low":        args.clip_low,
        "clip_high":       args.clip_high,
        "beta":            args.beta,
        "entropy_coeff":   args.entropy_coeff,
        "max_new_tokens":  args.max_new_tokens,
        "temperature":     args.temperature,
    }

    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    history     = []
    best_reward = -1.0

    logger.info(
        f"DAPO training | {args.num_iterations} iters | G={args.group_size} | "
        f"clip=[{args.clip_low},{args.clip_high}] | entropy_coeff={args.entropy_coeff}"
    )

    for i in range(args.num_iterations):
        batch = random.sample(train_traj, min(args.batch_size, len(train_traj)))

        stats = dapo_step(
            policy, reward_model, batch, optimizer, scaler, cfg, device
        )
        stats["iteration"] = i + 1
        history.append(stats)

        if args.use_wandb:
            import wandb
            wandb.log(
                {
                    "dapo/mean_reward":    stats["mean_reward"],
                    "dapo/entropy":        stats["entropy"],
                    "dapo/groups_skipped": stats["groups_skipped"],
                    "dapo/kl_div":         stats["kl_div"],
                    "dapo/loss":           stats["loss"],
                },
                step=i + 1,
            )

        if (i + 1) % args.log_interval == 0:
            logger.info(
                "[%d/%d] reward=%.4f | entropy=%.4f | skipped=%d | kl=%.5f | loss=%.4f",
                i + 1, args.num_iterations,
                stats["mean_reward"], stats["entropy"],
                stats["groups_skipped"], stats["kl_div"], stats["loss"],
            )

        if stats["mean_reward"] > best_reward:
            best_reward = stats["mean_reward"]
            policy.save_pretrained(str(output_path / "best"))

        if (i + 1) % args.save_interval == 0:
            policy.save_pretrained(str(output_path / f"ckpt_{i+1:05d}"))

    logger.info(f"DAPO training complete | best_reward={best_reward:.4f}")

    eval_results = evaluate_rlhf_policy(test_traj, reward_model)

    output = {
        "timestamp":   datetime.now().isoformat(),
        "method":      "DAPO",
        "model":       args.model_name,
        "sft_ckpt":    args.sft_ckpt,
        "group_size":  args.group_size,
        "clip_low":    args.clip_low,
        "clip_high":   args.clip_high,
        "entropy_coeff": args.entropy_coeff,
        "beta":        args.beta,
        "dataset": {
            "total": len(trajectories),
            "train": len(train_traj),
            "test":  len(test_traj),
        },
        "training": {
            "best_reward":    best_reward,
            "num_iterations": args.num_iterations,
            "history":        history,
        },
        "evaluation": eval_results,
    }

    results_file = get_results_dir() / "dapo_results.json"
    save_rlhf_results(output, results_file)

    if args.use_wandb:
        import wandb
        wandb.log({
            "test/avg_reward":  eval_results["average_reward"],
            "test/improvement": eval_results["improvement"],
        })
        wandb.finish()

    logger.info("=" * 60)
    logger.info("DAPO TRAINING COMPLETE")
    logger.info(f"  Group size G       : {args.group_size}")
    logger.info(f"  Clip [low, high]   : [{args.clip_low}, {args.clip_high}]")
    logger.info(f"  Entropy coeff      : {args.entropy_coeff}")
    logger.info(f"  Best train reward  : {best_reward:.4f}")
    logger.info(f"  Test avg reward    : {eval_results['average_reward']:.4f}")
    logger.info(f"  Test improvement   : {eval_results['improvement']:+.1f}%")
    logger.info(f"  Checkpoint         : {args.output_dir}/best")
    logger.info(f"  Results            : {results_file}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
