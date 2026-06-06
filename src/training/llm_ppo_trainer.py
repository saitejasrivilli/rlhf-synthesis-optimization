"""
PPO trainer for Llama-2 7B + LoRA synthesis policy.

Training loop per iteration:
  1. Sample molecule batch
  2. Generate conditions (LLM forward, no grad)
  3. Score with reward model
  4. Compute advantages = reward - baseline
  5. PPO mini-batch update (ppo_epochs inner loops)
     - Clipped surrogate loss
     - Value MSE loss
     - KL penalty vs frozen reference policy
     - Entropy bonus
  6. Gradient clip + Adam step on LoRA + value head params only
"""
import copy
import logging
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


@dataclass
class LLMPPOConfig:
    # Model
    model_name: str = "Qwen/Qwen2.5-7B-Instruct"
    lora_r: int = 8
    lora_alpha: int = 16

    # Training loop
    num_iterations: int = 200
    batch_size: int = 4              # trajectories per iteration (per GPU)
    ppo_epochs: int = 3              # inner PPO update epochs
    gradient_accumulation_steps: int = 4

    # Optimiser
    learning_rate: float = 1e-5
    max_grad_norm: float = 1.0

    # PPO hyper-params
    clip_ratio: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    kl_div_weight: float = 0.1

    # Generation
    max_new_tokens: int = 128
    temperature: float = 0.7

    # Logging / checkpointing
    log_interval: int = 10
    save_interval: int = 50
    output_dir: str = "models/llm_policy"


class LLMPPOTrainer:
    """
    PPO trainer that updates the LoRA weights and value head of LLMSynthesisPolicy.

    Reference log-probs are computed by disabling LoRA adapters on the same model
    (PEFT disable_adapter context), so no second model copy is needed — memory-efficient.
    """

    def __init__(
        self,
        policy,         # LLMSynthesisPolicy (trainable)
        reward_model,   # LearnedRewardModel or RealRewardModel
        config: LLMPPOConfig,
        device: str = "cuda",
        rank: int = 0,  # for distributed logging
        ref_policy=None,  # kept for API compatibility; ignored
    ):
        self.policy = policy
        self.reward_model = reward_model
        self.config = config
        self.device = device
        self.rank = rank

        trainable = (
            [p for p in self.policy.model.parameters() if p.requires_grad]
            + list(self.policy.value_head.parameters())
        )
        self.optimizer = torch.optim.Adam(trainable, lr=config.learning_rate)
        self.scaler    = torch.amp.GradScaler("cuda")

        # W&B — log if a run is already initialised externally
        try:
            import wandb
            self._wandb = wandb if wandb.run is not None else None
        except ImportError:
            self._wandb = None

        logger.info(
            f"LLMPPOTrainer | device={device} | lr={config.learning_rate} "
            f"| clip={config.clip_ratio} | kl_weight={config.kl_div_weight} "
            f"| wandb={'on' if self._wandb else 'off'}"
        )

    # ------------------------------------------------------------------
    # Reward computation
    # ------------------------------------------------------------------

    def _compute_rewards(
        self, conditions: List[Dict], trajectories: List[Dict]
    ) -> torch.Tensor:
        rewards = []
        for cond, traj in zip(conditions, trajectories):
            merged = {**traj, **cond}
            r = self.reward_model.score_trajectory(merged)
            rewards.append(r)
        return torch.tensor(rewards, dtype=torch.float32, device=self.device)

    # ------------------------------------------------------------------
    # PPO loss
    # ------------------------------------------------------------------

    def _ppo_loss(
        self,
        log_probs: torch.Tensor,       # [B, R]
        old_log_probs: torch.Tensor,   # [B, R]
        advantages: torch.Tensor,      # [B]
        values: torch.Tensor,          # [B]
        returns: torch.Tensor,         # [B]
        ref_log_probs: torch.Tensor,   # [B, R]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:

        # Sequence-level ratio (mean over response tokens)
        log_ratio = (log_probs - old_log_probs).mean(dim=-1)  # [B]
        ratio = torch.exp(log_ratio)

        # Clipped surrogate
        adv = advantages.detach()
        surr1 = ratio * adv
        surr2 = torch.clamp(ratio, 1 - self.config.clip_ratio, 1 + self.config.clip_ratio) * adv
        policy_loss = -torch.min(surr1, surr2).mean()

        # Value loss
        value_loss = nn.functional.mse_loss(values, returns.detach())

        # KL penalty (forward KL approximation)
        kl = (old_log_probs.detach() - ref_log_probs.detach()).mean(dim=-1).mean()
        kl_loss = self.config.kl_div_weight * kl.clamp(min=0)

        # Entropy bonus (negative entropy of log-probs distribution)
        entropy = -log_probs.mean()

        total = (
            policy_loss
            + self.config.value_coef * value_loss
            + kl_loss
            - self.config.entropy_coef * entropy
        )

        stats = {
            "loss_total":  total.item(),
            "loss_policy": policy_loss.item(),
            "loss_value":  value_loss.item(),
            "kl":          kl.item(),
            "entropy":     entropy.item(),
        }
        return total, stats

    # ------------------------------------------------------------------
    # Single training step
    # ------------------------------------------------------------------

    def train_step(self, trajectories: List[Dict]) -> Dict[str, float]:
        queries = [self.policy.build_prompt(t) for t in trajectories]

        # --- 1. Generate (no grad) ---
        self.policy.eval()
        with torch.no_grad():
            query_ids, response_ids = self.policy.generate(
                queries,
                max_new_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature,
            )
            old_lp, old_values = self.policy.get_logprobs_and_values(
                query_ids, response_ids
            )

        # --- 2. Reference log-probs (base model, LoRA disabled) ---
        with torch.no_grad(), self.policy.model.disable_adapter():
            ref_lp, _ = self.policy.get_logprobs_and_values(query_ids, response_ids)

        # --- 3. Reward ---
        conditions = self.policy.decode_conditions(response_ids)
        rewards = self._compute_rewards(conditions, trajectories)
        returns = rewards

        # --- 4. Advantages ---
        advantages = returns - old_values.detach()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # --- 5. PPO inner epochs ---
        self.policy.train()
        accum_stats: Dict[str, float] = {}

        for inner in range(self.config.ppo_epochs):
            with torch.cuda.amp.autocast():
                lp, values = self.policy.get_logprobs_and_values(query_ids, response_ids)
                loss, stats = self._ppo_loss(
                    lp, old_lp.detach(), advantages, values, returns, ref_lp
                )
                loss = loss / self.config.gradient_accumulation_steps

            self.scaler.scale(loss).backward()

            if (inner + 1) % self.config.gradient_accumulation_steps == 0 or (
                inner + 1 == self.config.ppo_epochs
            ):
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    [p for p in self.policy.parameters() if p.requires_grad],
                    self.config.max_grad_norm,
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

            for k, v in stats.items():
                accum_stats[k] = accum_stats.get(k, 0.0) + v / self.config.ppo_epochs

        accum_stats["reward_mean"] = rewards.mean().item()
        accum_stats["reward_std"]  = rewards.std().item()
        return accum_stats

    # ------------------------------------------------------------------
    # Full training loop
    # ------------------------------------------------------------------

    def train(self, trajectories: List[Dict]) -> Dict:
        output_path = Path(self.config.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        history = []
        best_reward = -1.0

        logger.info(
            f"LLM PPO training | {self.config.num_iterations} iters "
            f"| batch={self.config.batch_size} | ppo_epochs={self.config.ppo_epochs}"
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
                    {
                        "reward_mean":  stats["reward_mean"],
                        "reward_std":   stats["reward_std"],
                        "loss_total":   stats["loss_total"],
                        "loss_policy":  stats["loss_policy"],
                        "loss_value":   stats["loss_value"],
                        "kl":           stats["kl"],
                        "entropy":      stats["entropy"],
                    },
                    step=i + 1,
                )

            if self.rank == 0 and (i + 1) % self.config.log_interval == 0:
                logger.info(
                    f"[{i+1}/{self.config.num_iterations}] "
                    f"reward={stats['reward_mean']:.4f} ± {stats['reward_std']:.3f} | "
                    f"loss={stats['loss_total']:.4f} | "
                    f"kl={stats['kl']:.4f}"
                )

            if stats["reward_mean"] > best_reward:
                best_reward = stats["reward_mean"]
                if self.rank == 0:
                    self.policy.save_pretrained(str(output_path / "best"))

            if self.rank == 0 and (i + 1) % self.config.save_interval == 0:
                self.policy.save_pretrained(str(output_path / f"ckpt_{i+1:05d}"))

        logger.info(f"Training complete | best_reward={best_reward:.4f}")
        return {"best_reward": best_reward, "num_iterations": self.config.num_iterations, "history": history}
