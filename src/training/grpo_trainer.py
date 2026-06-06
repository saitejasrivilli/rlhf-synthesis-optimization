"""
GRPO (Group Relative Policy Optimization) trainer for synthesis.

Key difference vs PPO:
- Generate G completions per prompt in each iteration
- Advantages = group-normalized rewards (no value head)
- Simpler, more stable for LLMs (no critic collapse)

Reference: DeepSeekMath / DeepSeek-R1 GRPO paper.
"""
import logging
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


@dataclass
class GRPOConfig:
    model_name: str = "Qwen/Qwen2.5-7B-Instruct"
    lora_r: int = 8
    lora_alpha: int = 16

    # GRPO-specific
    group_size: int = 4             # G: completions per prompt
    beta: float = 0.04              # KL penalty weight
    clip_ratio: float = 0.2         # PPO-style clip

    # Training loop
    num_iterations: int = 200
    batch_size: int = 2             # prompts per iteration (total = batch*G)
    gradient_accumulation_steps: int = 4

    # Optimiser
    learning_rate: float = 5e-6
    max_grad_norm: float = 1.0

    # Generation
    max_new_tokens: int = 128
    temperature: float = 0.9        # higher than PPO — diversity matters for GRPO

    # Logging
    log_interval: int = 10
    save_interval: int = 50
    output_dir: str = "models/grpo_policy"


class GRPOTrainer:
    """
    GRPO: for each prompt, generate G completions, compute group-relative
    advantages, then apply clipped policy gradient + KL penalty.

    Advantages_i = (reward_i - mean(rewards)) / (std(rewards) + 1e-8)

    No value head → no critic collapse. Memory cost: G forward passes per step
    (all can be batched together).
    """

    def __init__(
        self,
        policy,             # LLMSynthesisPolicy
        reward_model,       # any model with score_trajectory()
        config: GRPOConfig,
        device: str = "cuda",
        rank: int = 0,
    ):
        self.policy       = policy
        self.reward_model = reward_model
        self.config       = config
        self.device       = device
        self.rank         = rank

        trainable = [p for p in self.policy.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(trainable, lr=config.learning_rate)
        self.scaler    = torch.amp.GradScaler("cuda")

        try:
            import wandb
            self._wandb = wandb if wandb.run is not None else None
        except ImportError:
            self._wandb = None

        logger.info(
            f"GRPOTrainer | G={config.group_size} | β={config.beta} | "
            f"clip={config.clip_ratio} | lr={config.learning_rate} | "
            f"wandb={'on' if self._wandb else 'off'}"
        )

    # ------------------------------------------------------------------
    # Core GRPO step
    # ------------------------------------------------------------------

    def train_step(self, trajectories: List[Dict]) -> Dict[str, float]:
        """
        For each prompt in the batch, generate G completions,
        score them, compute group-relative advantages, update policy.
        """
        G   = self.config.group_size
        cfg = self.config

        prompts = [self.policy.build_prompt(t) for t in trajectories]

        # ---- 1. Generate G completions per prompt ----
        self.policy.eval()
        with torch.no_grad():
            all_query_ids    = []
            all_response_ids = []
            all_old_lp       = []

            for prompt in prompts:
                q_ids, r_ids = self.policy.generate(
                    [prompt] * G,
                    max_new_tokens=cfg.max_new_tokens,
                    temperature=cfg.temperature,
                )
                old_lp, _ = self.policy.get_logprobs_and_values(q_ids, r_ids)
                all_query_ids.append(q_ids)       # [G, Q]
                all_response_ids.append(r_ids)    # [G, R]
                all_old_lp.append(old_lp)         # [G, R]

        # ---- 2. Reference log-probs ----
        all_ref_lp = []
        with torch.no_grad(), self.policy.model.disable_adapter():
            for q_ids, r_ids in zip(all_query_ids, all_response_ids):
                ref_lp, _ = self.policy.get_logprobs_and_values(q_ids, r_ids)
                all_ref_lp.append(ref_lp)

        # ---- 3. Rewards & group-relative advantages ----
        all_rewards    = []
        all_advantages = []

        for i, traj in enumerate(trajectories):
            r_ids = all_response_ids[i]
            conditions = self.policy.decode_conditions(r_ids)
            rewards = torch.tensor(
                [self.reward_model.score_trajectory({**traj, **c}) for c in conditions],
                dtype=torch.float32, device=self.device,
            )
            adv = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
            all_rewards.append(rewards)
            all_advantages.append(adv)

        # ---- 4. Policy update ----
        self.policy.train()
        total_loss   = 0.0
        total_reward = 0.0
        total_kl     = 0.0

        self.optimizer.zero_grad()

        for i in range(len(trajectories)):
            q_ids  = all_query_ids[i]
            r_ids  = all_response_ids[i]
            old_lp = all_old_lp[i]
            ref_lp = all_ref_lp[i]
            adv    = all_advantages[i].detach()

            with torch.amp.autocast("cuda"):
                new_lp, _ = self.policy.get_logprobs_and_values(q_ids, r_ids)

                # Sequence-level ratio
                log_ratio = (new_lp - old_lp.detach()).mean(dim=-1)  # [G]
                ratio     = torch.exp(log_ratio)

                # Clipped surrogate
                surr1      = ratio * adv
                surr2      = torch.clamp(ratio, 1 - cfg.clip_ratio, 1 + cfg.clip_ratio) * adv
                policy_loss = -torch.min(surr1, surr2).mean()

                # KL penalty (ref vs new)
                kl = (ref_lp.detach() - new_lp).mean(dim=-1).mean().clamp(min=0)

                loss = (policy_loss + cfg.beta * kl) / cfg.gradient_accumulation_steps

            self.scaler.scale(loss).backward()

            total_loss   += loss.item() * cfg.gradient_accumulation_steps
            total_reward += all_rewards[i].mean().item()
            total_kl     += kl.item()

        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(
            [p for p in self.policy.parameters() if p.requires_grad],
            cfg.max_grad_norm,
        )
        self.scaler.step(self.optimizer)
        self.scaler.update()

        n = len(trajectories)
        return {
            "loss":        total_loss   / n,
            "reward_mean": total_reward / n,
            "kl":          total_kl     / n,
        }

    # ------------------------------------------------------------------
    # Full training loop
    # ------------------------------------------------------------------

    def train(self, trajectories: List[Dict]) -> Dict:
        output_path = Path(self.config.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        history    = []
        best_reward = -1.0

        logger.info(
            f"GRPO training | {self.config.num_iterations} iters "
            f"| G={self.config.group_size} | batch={self.config.batch_size}"
        )

        for i in range(self.config.num_iterations):
            batch = random.sample(
                trajectories, min(self.config.batch_size, len(trajectories))
            )
            stats = self.train_step(batch)
            stats["iteration"] = i + 1
            history.append(stats)

            if self._wandb and self.rank == 0:
                self._wandb.log(
                    {"grpo/reward": stats["reward_mean"],
                     "grpo/loss":   stats["loss"],
                     "grpo/kl":     stats["kl"]},
                    step=i + 1,
                )

            if self.rank == 0 and (i + 1) % self.config.log_interval == 0:
                logger.info(
                    f"[{i+1}/{self.config.num_iterations}] "
                    f"reward={stats['reward_mean']:.4f} | "
                    f"loss={stats['loss']:.4f} | kl={stats['kl']:.5f}"
                )

            if stats["reward_mean"] > best_reward:
                best_reward = stats["reward_mean"]
                if self.rank == 0:
                    self.policy.save_pretrained(str(output_path / "best"))

            if self.rank == 0 and (i + 1) % self.config.save_interval == 0:
                self.policy.save_pretrained(str(output_path / f"ckpt_{i+1:05d}"))

        logger.info(f"GRPO complete | best_reward={best_reward:.4f}")
        return {
            "best_reward":    best_reward,
            "num_iterations": self.config.num_iterations,
            "group_size":     self.config.group_size,
            "history":        history,
        }
