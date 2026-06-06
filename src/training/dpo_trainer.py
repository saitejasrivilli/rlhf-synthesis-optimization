"""
Direct Preference Optimization (DPO) trainer for LLM synthesis policy.

DPO removes the separate reward model. Instead, it trains directly on
(chosen, rejected) preference pairs using the implicit reward defined by
the policy itself:

    r(x, y) = β * log(π_θ(y|x) / π_ref(y|x))

DPO loss (Rafailov et al., 2023):
    L = -E[ log σ( β*(log π_θ(y_w|x) - log π_ref(y_w|x))
                 - β*(log π_θ(y_l|x) - log π_ref(y_l|x)) ) ]

Reference policy uses disable_adapter() — no second model copy needed.

Advantages over PPO for this domain:
  - No separate reward model (avoids reward hacking)
  - More stable training (single supervised-style loss)
  - Directly optimizes preference ranking
"""
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


@dataclass
class DPOConfig:
    beta: float  = 0.1           # KL constraint strength
    learning_rate: float = 1e-5
    num_epochs: int = 3
    batch_size: int = 4
    max_new_tokens: int = 128
    temperature: float = 0.7
    max_grad_norm: float = 1.0
    log_interval: int = 10
    output_dir: str = "models/dpo_policy"


class DPOTrainer:
    """
    DPO trainer for LLMSynthesisPolicy.

    The reference policy is the policy with LoRA disabled (same GPU,
    no extra memory). Updates only LoRA weights.
    """

    def __init__(
        self,
        policy,           # LLMSynthesisPolicy
        config: DPOConfig,
        device: str = "cuda",
    ):
        self.policy = policy
        self.config = config
        self.device = device

        trainable = [p for p in policy.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(trainable, lr=config.learning_rate)
        self.scaler    = torch.amp.GradScaler("cuda")

        logger.info(
            f"DPOTrainer | β={config.beta} | lr={config.learning_rate} "
            f"| epochs={config.num_epochs}"
        )

    # ------------------------------------------------------------------
    # Log-probability helpers
    # ------------------------------------------------------------------

    def _get_logprobs(
        self,
        query_ids: torch.Tensor,
        response_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Return mean log-prob over response tokens. Shape: [B]"""
        lp, _ = self.policy.get_logprobs_and_values(query_ids, response_ids)
        return lp.mean(dim=-1)   # [B]

    @torch.no_grad()
    def _get_ref_logprobs(
        self,
        query_ids: torch.Tensor,
        response_ids: torch.Tensor,
    ) -> torch.Tensor:
        with self.policy.model.disable_adapter():
            lp, _ = self.policy.get_logprobs_and_values(query_ids, response_ids)
        return lp.mean(dim=-1)

    # ------------------------------------------------------------------
    # Tokenise a pair
    # ------------------------------------------------------------------

    def _prepare_pair(
        self,
        chosen_traj: Dict,
        rejected_traj: Dict,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns (query_ids_chosen, resp_ids_chosen,
                 query_ids_rejected, resp_ids_rejected).

        Both queries are identical (same molecule prompt); responses differ.
        """
        # Both share the same query (molecule prompt)
        query = self.policy.build_prompt(chosen_traj)

        def _make_response(traj: Dict) -> str:
            import json
            outcomes = traj.get("outcomes", traj)
            params   = traj.get("parameters", traj)
            cond = {
                "temperature_celsius":  params.get("temperature_celsius",  traj.get("temperature_celsius",  80.0)),
                "time_hours":           params.get("time_hours",            traj.get("time_hours",            4.0)),
                "catalyst_loading_M":   params.get("catalyst_loading_M",   traj.get("catalyst_loading_M",   0.05)),
                "solvent_ratio_ml_mmol":params.get("solvent_ratio_ml_mmol",traj.get("solvent_ratio_ml_mmol", 3.0)),
            }
            return json.dumps(cond)

        chosen_resp   = _make_response(chosen_traj)
        rejected_resp = _make_response(rejected_traj)

        tok = self.policy.tokenizer

        def _encode(q, r):
            q_ids = tok(q, return_tensors="pt", truncation=True, max_length=384).input_ids.to(self.device)
            r_ids = tok(r, return_tensors="pt", truncation=True, max_length=128,
                        add_special_tokens=False).input_ids.to(self.device)
            return q_ids, r_ids

        qc, rc  = _encode(query, chosen_resp)
        qr, rr  = _encode(query, rejected_resp)
        return qc, rc, qr, rr

    # ------------------------------------------------------------------
    # Training step
    # ------------------------------------------------------------------

    def train_step(self, pairs: List[Tuple[Dict, Dict]]) -> Dict[str, float]:
        """One DPO update step on a batch of (chosen, rejected) pairs."""
        self.policy.train()

        pol_chosen_lps,   pol_rejected_lps   = [], []
        ref_chosen_lps,   ref_rejected_lps   = [], []
        implicit_rewards_chosen, implicit_rewards_rejected = [], []

        for chosen, rejected in pairs:
            qc, rc, qr, rr = self._prepare_pair(chosen, rejected)

            # Policy log-probs (with grad)
            pc_lp = self._get_logprobs(qc, rc)
            pr_lp = self._get_logprobs(qr, rr)

            # Reference log-probs (no grad, base model)
            rc_lp = self._get_ref_logprobs(qc, rc)
            rr_lp = self._get_ref_logprobs(qr, rr)

            pol_chosen_lps.append(pc_lp)
            pol_rejected_lps.append(pr_lp)
            ref_chosen_lps.append(rc_lp)
            ref_rejected_lps.append(rr_lp)

            with torch.no_grad():
                implicit_rewards_chosen.append(
                    (self.config.beta * (pc_lp - rc_lp)).item()
                )
                implicit_rewards_rejected.append(
                    (self.config.beta * (pr_lp - rr_lp)).item()
                )

        pol_chosen   = torch.stack(pol_chosen_lps)
        pol_rejected = torch.stack(pol_rejected_lps)
        ref_chosen   = torch.stack(ref_chosen_lps)
        ref_rejected = torch.stack(ref_rejected_lps)

        # DPO loss
        beta = self.config.beta
        log_ratio = (
            beta * (pol_chosen   - ref_chosen)
          - beta * (pol_rejected - ref_rejected)
        )
        loss = -F.logsigmoid(log_ratio).mean()

        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(
            [p for p in self.policy.model.parameters() if p.requires_grad],
            self.config.max_grad_norm,
        )
        self.scaler.step(self.optimizer)
        self.scaler.update()

        reward_margin = (
            sum(implicit_rewards_chosen) / len(implicit_rewards_chosen)
          - sum(implicit_rewards_rejected) / len(implicit_rewards_rejected)
        )

        return {
            "dpo_loss":      loss.item(),
            "reward_margin": reward_margin,
            "chosen_reward": sum(implicit_rewards_chosen)   / len(implicit_rewards_chosen),
            "reject_reward": sum(implicit_rewards_rejected) / len(implicit_rewards_rejected),
        }

    # ------------------------------------------------------------------
    # Full training loop
    # ------------------------------------------------------------------

    def train(
        self,
        pairs: List[Tuple[Dict, Dict]],
        use_wandb: bool = False,
    ) -> Dict:
        output_path = Path(self.config.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        history = []
        best_margin = -999.0
        global_step = 0

        logger.info(
            f"DPO training | {self.config.num_epochs} epochs "
            f"| {len(pairs)} pairs | batch={self.config.batch_size}"
        )

        for epoch in range(self.config.num_epochs):
            random.shuffle(pairs)
            ep_stats: Dict[str, float] = {}

            for i in range(0, len(pairs), self.config.batch_size):
                batch = pairs[i : i + self.config.batch_size]
                if not batch:
                    continue
                stats = self.train_step(batch)
                stats["epoch"] = epoch + 1
                stats["step"]  = global_step
                history.append(stats)

                for k, v in stats.items():
                    if isinstance(v, float):
                        ep_stats[k] = ep_stats.get(k, 0.0) + v

                if use_wandb:
                    try:
                        import wandb
                        wandb.log(stats, step=global_step)
                    except ImportError:
                        pass

                global_step += 1

            n_batches = max(1, len(pairs) // self.config.batch_size)
            avg_loss   = ep_stats.get("dpo_loss", 0) / n_batches
            avg_margin = ep_stats.get("reward_margin", 0) / n_batches
            logger.info(
                f"Epoch {epoch+1}/{self.config.num_epochs}: "
                f"dpo_loss={avg_loss:.4f} | reward_margin={avg_margin:.4f}"
            )

            if avg_margin > best_margin:
                best_margin = avg_margin
                self.policy.save_pretrained(str(output_path / "best"))

        logger.info(f"DPO training complete | best_reward_margin={best_margin:.4f}")
        return {
            "best_reward_margin": best_margin,
            "num_epochs":   self.config.num_epochs,
            "num_pairs":    len(pairs),
            "history":      history,
        }
