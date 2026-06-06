#!/usr/bin/env python3
"""
GRPO training with Process Reward Model (PRM) step-level rewards.

Replaces the binary verifiable reward (1/0) with a dense step-level signal:
  - Each reasoning step that passes numeric verification: +0.10–0.15
  - Correct final answer: +1.0
  - Discounted sum (γ=0.9): earlier steps weighted higher

This matches the Scale AI / frontier-lab approach of using process rewards
to train models that show their work rather than shortcutting to answers.

Usage:
  CUDA_VISIBLE_DEVICES=0 python scripts/train_prm_grpo.py \
    --model Qwen/Qwen2.5-7B-Instruct \
    --num_iters 100 \
    --group_size 4 \
    --gamma 0.9 \
    --step_weight 0.5 \
    --output_dir models/prm_grpo
"""
import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

os.environ.setdefault("TRANSFORMERS_CACHE", "/storage/gxg8313/saiteja/hf_cache")
os.environ.setdefault("HF_HOME",            "/storage/gxg8313/saiteja/hf_cache")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model",          default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--data_file",      default="data/gsm8k_train_tool_dataset.jsonl")
    p.add_argument("--num_problems",   type=int,   default=20)
    p.add_argument("--num_iters",      type=int,   default=60)
    p.add_argument("--group_size",     type=int,   default=4,   help="G rollouts per problem")
    p.add_argument("--batch_size",     type=int,   default=2)
    p.add_argument("--lr",             type=float, default=5e-6)
    p.add_argument("--clip_ratio",     type=float, default=0.2)
    p.add_argument("--kl_coeff",       type=float, default=0.04)
    p.add_argument("--max_new_tokens", type=int,   default=256)
    p.add_argument("--gamma",          type=float, default=0.9,  help="PRM discount factor")
    p.add_argument("--step_weight",    type=float, default=0.5,  help="Weight on step rewards vs final")
    p.add_argument("--output_dir",     default="models/prm_grpo")
    p.add_argument("--log_interval",   type=int,   default=10)
    p.add_argument("--lora_r",         type=int,   default=8)
    return p.parse_args()


def load_problems(data_file: str, n: int):
    problems = []
    with open(data_file) as f:
        for line in f:
            row = json.loads(line)
            problems.append(row)
            if len(problems) >= n:
                break
    return problems


def build_prompt(problem: str) -> str:
    return (
        "<|im_start|>system\nYou are a mathematical reasoning assistant. "
        "Think step by step. Show each calculation explicitly. "
        "Use <think>...</think> for reasoning and "
        "<final_answer>NUMBER</final_answer> for your answer.<|im_end|>\n"
        f"<|im_start|>user\n{problem}<|im_end|>\n"
        "<|im_start|>assistant\n<think>"
    )


def run_grpo_with_prm(args):
    from src.policy.llm_synthesis_policy import LLMSynthesisPolicy
    from src.training.grpo_trainer import GRPOConfig
    from src.reward.process_reward_model import compute_prm_reward, parse_steps
    from src.reward.verifiable_reward import compute_reward

    logger.info("Loading policy: %s", args.model)
    policy = LLMSynthesisPolicy(args.model, lora_r=args.lora_r)
    policy.model.gradient_checkpointing_enable()
    policy.model.enable_input_require_grads()

    problems = load_problems(args.data_file, args.num_problems)
    logger.info("Loaded %d problems", len(problems))

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    optimizer = torch.optim.AdamW(
        [p for p in policy.model.parameters() if p.requires_grad],
        lr=args.lr,
    )

    history = []
    best_reward = 0.0
    best_step_frac = 0.0

    for iteration in range(1, args.num_iters + 1):
        torch.cuda.empty_cache()
        import random
        batch_problems = random.sample(problems, min(args.batch_size, len(problems)))

        all_rewards, all_step_fracs = [], []
        policy_losses = []

        for item in batch_problems:
            prompt = build_prompt(item["problem"])
            answer = item["answer"]

            # Generate G rollouts
            responses = []
            for _ in range(args.group_size):
                _, _, resp = _generate_one(policy, prompt, args.max_new_tokens)
                responses.append(resp)

            # PRM rewards
            rewards = []
            step_fracs = []
            for resp in responses:
                total, step_scores = compute_prm_reward(
                    resp, answer,
                    gamma=args.gamma,
                    step_weight=args.step_weight,
                )
                rewards.append(total)
                final_r = step_scores[-1]
                step_r  = sum(step_scores[:-1])
                step_fracs.append(step_r / (total + 1e-8))

            all_rewards.extend(rewards)
            all_step_fracs.extend(step_fracs)

            # GRPO advantage: group-normalize
            r_tensor = torch.tensor(rewards, dtype=torch.float32)
            mean_r = r_tensor.mean()
            std_r  = r_tensor.std() + 1e-8
            advantages = ((r_tensor - mean_r) / std_r).tolist()

            # Policy gradient update for each rollout
            for resp, adv in zip(responses, advantages):
                loss = _pg_loss(policy, prompt, resp, adv, args.kl_coeff)
                if loss is not None:
                    loss.backward()
                    policy_losses.append(loss.item())

        if policy_losses:
            torch.nn.utils.clip_grad_norm_(
                [p for p in policy.model.parameters() if p.requires_grad], 1.0
            )
            optimizer.step()
            optimizer.zero_grad()

        avg_reward   = sum(all_rewards) / len(all_rewards) if all_rewards else 0
        avg_step_frac = sum(all_step_fracs) / len(all_step_fracs) if all_step_fracs else 0
        avg_loss     = sum(policy_losses) / len(policy_losses) if policy_losses else 0

        if avg_reward > best_reward:
            best_reward    = avg_reward
            best_step_frac = avg_step_frac
            policy.save_pretrained(str(out_dir / "best"))

        history.append({
            "iter": iteration,
            "avg_reward": round(avg_reward, 4),
            "step_reward_frac": round(avg_step_frac, 4),
            "loss": round(avg_loss, 6),
        })

        if iteration % args.log_interval == 0:
            logger.info(
                "[%d/%d] reward=%.4f (step_frac=%.2f) | loss=%.6f | best=%.4f",
                iteration, args.num_iters,
                avg_reward, avg_step_frac, avg_loss, best_reward,
            )

    # Save results
    results = {
        "model":          args.model,
        "num_iters":      args.num_iters,
        "gamma":          args.gamma,
        "step_weight":    args.step_weight,
        "best_reward":    best_reward,
        "best_step_frac": best_step_frac,
        "history":        history,
    }
    with open(out_dir / "prm_grpo_results.json", "w") as f:
        json.dump(results, f, indent=2)

    logger.info(
        "PRM GRPO done | best_reward=%.4f | step_frac=%.2f | saved to %s",
        best_reward, best_step_frac, out_dir,
    )
    return results


def _generate_one(policy, prompt, max_new_tokens):
    """Generate a single response; returns (prompt_ids, response_ids, text)."""
    enc = policy.tokenizer(prompt, return_tensors="pt").to(policy.device)
    with torch.no_grad():
        out = policy.model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.9,
            top_p=0.95,
            pad_token_id=policy.tokenizer.eos_token_id,
        )
    prompt_len = enc["input_ids"].shape[1]
    response_ids = out[:, prompt_len:]
    text = policy.tokenizer.decode(response_ids[0], skip_special_tokens=True)
    return enc["input_ids"], response_ids, text


def _pg_loss(policy, prompt, response, advantage, kl_coeff):
    """Compute policy-gradient loss for one (prompt, response) pair."""
    enc = policy.tokenizer(prompt, return_tensors="pt").to(policy.device)
    resp_ids = policy.tokenizer(
        response, return_tensors="pt", add_special_tokens=False
    ).input_ids.to(policy.device)

    if resp_ids.shape[1] == 0:
        return None

    try:
        lp = policy.get_logprobs(enc["input_ids"], resp_ids)
        seq_lp = lp.sum(dim=1)          # [1]
        loss = -advantage * seq_lp.squeeze()
        return loss
    except Exception:
        return None


def main():
    args = parse_args()
    results = run_grpo_with_prm(args)
    print("\n=== PRM GRPO Results ===")
    print(f"Best reward:     {results['best_reward']:.4f}")
    print(f"Step frac:       {results['best_step_frac']:.2f}  "
          f"(fraction of reward from step-level PRM signal)")
    print(f"Iters:           {results['num_iters']}")
    print(f"γ (discount):    {results['gamma']}")
    print(f"Step weight:     {results['step_weight']}")


if __name__ == "__main__":
    main()
