"""
MultiAgentGRPOTrainer — Generator + Critic dual-agent GRPO

Motivation
----------
AgentGRPOTrainer uses a binary verifiable reward (correct=1.0 / wrong=0.0).
For hard problems this reward is extremely sparse — an entire trajectory
gets 0 reward even if it used the right tool and reached a near-correct
intermediate value.

A Critic agent fixes this by providing dense, trajectory-level feedback:
  Generator  produces G rollouts per problem (existing AgentGRPOTrainer)
  Critic     evaluates each trajectory for reasoning quality, tool use
             appropriateness, and answer coherence (separate signal)

Combined reward:
    r_combined = r_verifiable + α * r_critic

The GRPO advantage is then computed from r_combined. The critic signal
provides gradient even for failed trajectories, accelerating learning
on problems where 0/1 reward alone gives little information.

This is structurally similar to how Constitutional AI (Anthropic) and
Debate-style training (Irving et al.) augment RL with a separate
evaluator agent — but applied here to tool-use trajectories.

Usage
-----
    trainer = MultiAgentGRPOTrainer(
        policy,
        config,
        critic_alpha=0.3,    # weight for critic reward vs verifiable
        critic_mode="heuristic",   # or "llm" for a real critic model
    )
    metrics = trainer.train_step(problems)
"""

from __future__ import annotations
import logging
import re
from typing import Dict, List

import torch

from .agent_grpo_trainer import (
    AgentGRPOTrainer,
    GRPOConfig,
    build_agent_prompt,
    run_agent_rollout,
    trajectory_logprobs,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Heuristic Critic
# ---------------------------------------------------------------------------

_FINAL_ANS_RE  = re.compile(r"<final_answer>(.*?)</final_answer>", re.DOTALL | re.IGNORECASE)
_TOOL_CALL_RE  = re.compile(r"<tool_call>", re.IGNORECASE)
_REASONING_KWS = {"because", "therefore", "first", "next", "then", "since",
                  "step", "compute", "calculate", "find", "let", "we"}


def heuristic_critic_score(problem: str, output: str) -> float:
    """
    Evaluate a Generator trajectory without a second LLM call.

    Scores four dimensions (each 0.0–0.25), returns sum in [0.0, 1.0]:

    1. answer_present   — model produced a <final_answer> tag
    2. reasoning_depth  — output contains reasoning keywords (not bare answer)
    3. tool_use         — at least one <tool_call> was made
    4. coherence        — output is not repetitive or empty

    In production swap this for a real LLM critic call:
        critic_prompt = CRITIC_SYSTEM + f"Problem: {problem}\nOutput: {output}"
        critique = critic_model(critic_prompt)
        return parse_score(critique)
    """
    score = 0.0
    lower = output.lower()
    words = lower.split()

    # 1. Final answer present
    if _FINAL_ANS_RE.search(output):
        score += 0.25

    # 2. Reasoning depth — at least 3 reasoning keywords
    kw_count = sum(1 for w in words if w in _REASONING_KWS)
    if kw_count >= 3:
        score += 0.25
    elif kw_count >= 1:
        score += 0.10

    # 3. Tool use — at least one tool call attempted
    if _TOOL_CALL_RE.search(output):
        score += 0.25

    # 4. Coherence — not excessively repetitive, not trivially short
    if len(words) >= 20:
        # penalise repetition: check for repeated 4-gram
        ngrams = [" ".join(words[i:i+4]) for i in range(len(words) - 3)]
        unique_ratio = len(set(ngrams)) / max(len(ngrams), 1)
        if unique_ratio > 0.8:
            score += 0.25
        elif unique_ratio > 0.5:
            score += 0.10

    return round(min(score, 1.0), 4)


# ---------------------------------------------------------------------------
# MultiAgentGRPOTrainer
# ---------------------------------------------------------------------------

class MultiAgentGRPOTrainer(AgentGRPOTrainer):
    """
    Extends AgentGRPOTrainer with a Critic agent that provides dense
    reward signal supplementing the sparse verifiable reward.

    Parameters
    ----------
    critic_alpha : float
        Weight for the critic reward in the combined reward.
        combined_reward = verifiable_reward + critic_alpha * critic_reward
        Recommended range: 0.1–0.4. Higher values give more influence to
        the critic; lower values keep verifiable correctness dominant.
    critic_mode : str
        "heuristic" — use heuristic_critic_score (no extra GPU memory)
        "llm"       — use a separate LLM critic call (set critic_model)
    critic_model : optional
        A callable(prompt: str) -> str for LLM-based critique.
        Only used when critic_mode="llm".
    """

    def __init__(
        self,
        policy,
        config: GRPOConfig,
        critic_alpha: float = 0.3,
        critic_mode: str = "heuristic",
        critic_model=None,
        **kwargs,
    ):
        super().__init__(policy, config, **kwargs)
        self.critic_alpha = critic_alpha
        self.critic_mode  = critic_mode
        self.critic_model = critic_model

        if critic_mode == "llm" and critic_model is None:
            raise ValueError("critic_mode='llm' requires a critic_model callable.")

        logger.info(
            f"MultiAgentGRPOTrainer | α={critic_alpha} | critic={critic_mode} "
            f"| G={config.group_size}"
        )

    def _critic_score(self, problem: str, output: str) -> float:
        """Route to heuristic or LLM critic based on critic_mode."""
        if self.critic_mode == "heuristic":
            return heuristic_critic_score(problem, output)

        # LLM critic — expects a 0.0–1.0 numeric score on the last line
        prompt = (
            "You are an expert evaluator. Score this agent trajectory from 0.0 to 1.0.\n"
            "Criteria: answer present, reasoning depth, appropriate tool use, coherence.\n"
            f"Problem: {problem}\n"
            f"Agent output:\n{output}\n\n"
            "Respond with ONLY a decimal number between 0.0 and 1.0."
        )
        raw = self.critic_model(prompt).strip()
        try:
            return min(max(float(re.search(r"\d+\.?\d*", raw).group()), 0.0), 1.0)
        except (AttributeError, ValueError):
            return 0.0

    def train_step(self, problems: List[Dict]) -> Dict[str, float]:
        """
        GRPO update with critic-augmented rewards.

        Overrides AgentGRPOTrainer.train_step to inject critic scores
        into the reward signal before computing group-relative advantages.
        Everything else (log-prob computation, clipped policy gradient,
        KL penalty) is identical to the parent class.
        """
        G   = self.config.group_size
        cfg = self.config

        # ---- 1. Generator rollouts (same as parent) ----
        self.policy.eval()
        all_segments:  list = []
        all_tool_blks: list = []
        all_outputs:   list = []

        with torch.no_grad():
            for item in problems:
                prompt = build_agent_prompt(item["problem"])
                group_segs, group_blks, group_outs = [], [], []
                for _ in range(G):
                    segs, blks, out = run_agent_rollout(
                        self.policy, prompt,
                        max_new_tokens=cfg.max_new_tokens,
                        temperature=cfg.temperature,
                    )
                    group_segs.append(segs)
                    group_blks.append(blks)
                    group_outs.append(out)
                all_segments.append(group_segs)
                all_tool_blks.append(group_blks)
                all_outputs.append(group_outs)

        # ---- 2. Critic scores (new) ----
        all_critic_scores: list = []
        for prob_idx, item in enumerate(problems):
            group_critic = []
            for g in range(G):
                cs = self._critic_score(item["problem"], all_outputs[prob_idx][g])
                group_critic.append(cs)
            all_critic_scores.append(group_critic)

        # ---- 3. Verifiable rewards + combined rewards + GRPO advantages ----
        all_rewards:    list = []
        all_advantages: list = []

        for prob_idx, item in enumerate(problems):
            raw_rewards = []
            for g in range(G):
                output = all_outputs[prob_idx][g]
                fa = _FINAL_ANS_RE.search(output)
                answer = fa.group(1).strip() if fa else ""
                verifiable = 1.0 if (answer and answer == item.get("answer", "")) else 0.0
                critic     = all_critic_scores[prob_idx][g]
                combined   = verifiable + self.critic_alpha * critic
                raw_rewards.append(combined)

            rewards_t = torch.tensor(raw_rewards, dtype=torch.float32)
            mean = rewards_t.mean()
            std  = rewards_t.std().clamp(min=1e-8)
            advantages = ((rewards_t - mean) / std).tolist()

            all_rewards.append(rewards_t)
            all_advantages.append(advantages)

        # ---- 4. Policy gradient update (identical to parent) ----
        self.policy.train()
        total_loss  = 0.0
        total_kl    = 0.0
        total_steps = 0

        with torch.amp.autocast("cuda"):
            for prob_idx, item in enumerate(problems):
                prompt = build_agent_prompt(item["problem"])
                for g in range(G):
                    adv = all_advantages[prob_idx][g]
                    if abs(adv) < 1e-6:
                        continue

                    segs = all_segments[prob_idx][g]
                    blks = all_tool_blks[prob_idx][g]
                    lp, ref_lp = trajectory_logprobs(
                        self.policy, prompt, segs, blks,
                    )
                    if lp is None:
                        continue

                    ratio = torch.exp(lp - lp.detach())
                    adv_t = torch.tensor(adv, device=lp.device)
                    pg    = -torch.min(
                        ratio * adv_t,
                        torch.clamp(ratio, 1 - cfg.clip_ratio, 1 + cfg.clip_ratio) * adv_t,
                    ).mean()
                    kl   = (lp - ref_lp).mean().clamp(min=0)
                    loss = pg + cfg.beta * kl

                    total_loss  += loss.item()
                    total_kl    += kl.item()
                    total_steps += 1

                    self.scaler.scale(loss).backward()

        if total_steps:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.policy.model.parameters(), 1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        self.optimizer.zero_grad()

        # ---- 5. Metrics ----
        flat_rewards  = [r.item() for rt in all_rewards for r in rt]
        flat_critic   = [c for grp in all_critic_scores for c in grp]
        verifiable_r  = [
            1.0 if (_FINAL_ANS_RE.search(all_outputs[pi][g])
                    and _FINAL_ANS_RE.search(all_outputs[pi][g]).group(1).strip()
                    == problems[pi].get("answer", "")) else 0.0
            for pi in range(len(problems)) for g in range(G)
        ]

        metrics = {
            "loss":              round(total_loss / max(total_steps, 1), 4),
            "kl":                round(total_kl   / max(total_steps, 1), 4),
            "mean_reward":       round(sum(flat_rewards) / len(flat_rewards), 4),
            "mean_verifiable":   round(sum(verifiable_r) / len(verifiable_r), 4),
            "mean_critic":       round(sum(flat_critic)  / len(flat_critic),  4),
            "critic_alpha":      self.critic_alpha,
        }

        if self._wandb:
            self._wandb.log(metrics)

        logger.info(
            f"multi_agent_grpo | loss={metrics['loss']} "
            f"| r_verif={metrics['mean_verifiable']} "
            f"| r_critic={metrics['mean_critic']} "
            f"| r_combined={metrics['mean_reward']}"
        )
        return metrics
