"""
RLAIF (Reinforcement Learning from AI Feedback) trainer.

Instead of human preference labels, uses the same LLM as a judge to score
responses and generate preference pairs for DPO training. This enables
scalable alignment without human annotators.

Algorithm:
  For each problem in dataset:
    1. Generate K candidate responses with the current policy
    2. Judge: score each response 0-10 on correctness + reasoning quality
    3. Form pairs (chosen, rejected) where score gap >= threshold
    4. Accumulate pairs into a preference dataset
  Run one DPO epoch on the accumulated preference pairs.
  Iterate for R rounds.

Reference: Constitutional AI (Anthropic 2022), RLAIF (Lee et al. 2023).
"""
import json
import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

_SCORE_RE = re.compile(r"\b(\d+(?:\.\d+)?)\s*/\s*10\b|\bscore[:\s]+(\d+(?:\.\d+)?)\b", re.IGNORECASE)


# ── Config ────────────────────────────────────────────────────────────────────

@dataclass
class RAIFConfig:
    data_file:        str   = "data/gsm8k_train_tool_dataset.jsonl"
    output_dir:       str   = "models/rlaif"
    num_rounds:       int   = 3
    candidates_per_q: int   = 4       # K responses generated per problem
    pair_gap:         float = 2.0     # min score gap to form a preference pair
    dpo_epochs:       int   = 1
    dpo_beta:         float = 0.1     # DPO temperature
    lr:               float = 5e-6
    max_new_tokens:   int   = 256
    judge_max_tokens: int   = 64
    num_problems:     int   = 20
    log_interval:     int   = 5


# ── Judge prompt ──────────────────────────────────────────────────────────────

_JUDGE_TMPL = """You are evaluating an AI assistant's answer to a math problem.

Problem: {problem}

Response: {response}

Rate this response on a scale of 0-10:
- 10: Correct answer with clear step-by-step reasoning
- 7-9: Correct answer, reasoning partially shown
- 4-6: Incorrect answer but some valid reasoning steps
- 0-3: Incorrect answer with flawed or missing reasoning

Reply with ONLY: "Score: X/10" where X is your rating."""


# ── Scorer ────────────────────────────────────────────────────────────────────

def judge_response(policy, problem: str, response: str, max_tokens: int = 64) -> float:
    """
    Use the policy as a judge to score a response 0-10.
    Returns a normalized score in [0, 1].
    """
    prompt = _JUDGE_TMPL.format(problem=problem, response=response[:800])
    enc = policy.tokenizer(prompt, return_tensors="pt")
    device = next(policy.model.parameters()).device
    enc = {k: v.to(device) for k, v in enc.items()}

    with torch.no_grad():
        out = policy.model.generate(
            **enc,
            max_new_tokens=max_tokens,
            do_sample=False,
            pad_token_id=policy.tokenizer.eos_token_id,
        )
    text = policy.tokenizer.decode(
        out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True
    )

    for m in _SCORE_RE.finditer(text):
        score_str = m.group(1) or m.group(2)
        try:
            return min(float(score_str), 10.0) / 10.0
        except ValueError:
            pass
    return 0.5  # neutral fallback


# ── Preference pair collection ────────────────────────────────────────────────

def collect_preference_pairs(
    policy,
    problems: List[Dict],
    cfg: RAIFConfig,
) -> List[Dict]:
    """
    Generate K responses per problem, score them, form (chosen, rejected) pairs.
    Returns list of {prompt, chosen, rejected, score_gap}.
    """
    from ..tools.tool_parser import parse_output

    pairs = []
    for item in problems:
        problem = item["problem"]
        prompt  = (
            "<|im_start|>system\nSolve the math problem step by step. "
            "Show all calculations. Give your final answer as a number.<|im_end|>\n"
            f"<|im_start|>user\n{problem}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )

        # Generate K candidates
        device = next(policy.model.parameters()).device
        enc = policy.tokenizer(prompt, return_tensors="pt").to(device)
        responses = []
        with torch.no_grad():
            for _ in range(cfg.candidates_per_q):
                out = policy.model.generate(
                    **enc,
                    max_new_tokens=cfg.max_new_tokens,
                    do_sample=True,
                    temperature=0.9,
                    top_p=0.95,
                    pad_token_id=policy.tokenizer.eos_token_id,
                )
                resp = policy.tokenizer.decode(
                    out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True
                )
                responses.append(resp)

        # Score all candidates
        scores = [judge_response(policy, problem, r, cfg.judge_max_tokens)
                  for r in responses]

        # Form pairs: all (i, j) where score gap >= threshold
        for i in range(len(responses)):
            for j in range(len(responses)):
                if i == j:
                    continue
                gap = scores[i] - scores[j]
                if gap >= cfg.pair_gap / 10.0:   # normalize gap threshold
                    pairs.append({
                        "prompt":    prompt,
                        "chosen":    responses[i],
                        "rejected":  responses[j],
                        "score_gap": round(gap, 3),
                    })

    logger.info("Collected %d preference pairs from %d problems", len(pairs), len(problems))
    return pairs


# ── DPO loss ──────────────────────────────────────────────────────────────────

def _seq_logprob(policy, prompt: str, response: str) -> torch.Tensor:
    """Compute sum log p(response | prompt) under policy."""
    device = next(policy.model.parameters()).device
    enc  = policy.tokenizer(prompt, return_tensors="pt").to(device)
    resp = policy.tokenizer(response, return_tensors="pt",
                            add_special_tokens=False).input_ids.to(device)
    if resp.shape[1] == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)
    lp = policy.get_logprobs(enc["input_ids"], resp)   # [1, T]
    return lp.sum()


def dpo_loss(
    policy,
    pairs: List[Dict],
    beta: float = 0.1,
    ref_policy=None,
) -> torch.Tensor:
    """
    DPO loss over a batch of preference pairs.

    L_DPO = -E[ log σ( β*(log π_θ(y_w|x) - log π_θ(y_l|x)
                          - log π_ref(y_w|x) + log π_ref(y_l|x)) ) ]

    When ref_policy is None, uses the initial log-probs (implicit reference).
    For simplicity here we use no explicit reference (offline DPO / SFT-style).
    """
    total_loss = torch.tensor(0.0, device=next(policy.model.parameters()).device)
    valid = 0

    for pair in pairs:
        try:
            lp_chosen   = _seq_logprob(policy, pair["prompt"], pair["chosen"])
            lp_rejected = _seq_logprob(policy, pair["prompt"], pair["rejected"])
            # DPO objective (no reference — equivalent to supervised contrastive)
            loss = -F.logsigmoid(beta * (lp_chosen - lp_rejected))
            total_loss = total_loss + loss
            valid += 1
        except Exception:
            continue

    return total_loss / max(valid, 1)


# ── Main trainer ──────────────────────────────────────────────────────────────

class RAIFTrainer:
    def __init__(self, policy, cfg: RAIFConfig):
        self.policy = policy
        self.cfg    = cfg
        self.out    = Path(cfg.output_dir)
        self.out.mkdir(parents=True, exist_ok=True)

        self._load_data()
        self.optimizer = torch.optim.AdamW(
            [p for p in policy.model.parameters() if p.requires_grad],
            lr=cfg.lr,
        )
        self.history: List[Dict] = []

    def _load_data(self):
        rows = []
        with open(self.cfg.data_file) as f:
            for line in f:
                rows.append(json.loads(line))
        self.problems = rows[: self.cfg.num_problems]
        logger.info("RLAIF | %d problems | %d rounds", len(self.problems), self.cfg.num_rounds)

    def run(self) -> Dict:
        best_loss = float("inf")

        for rnd in range(1, self.cfg.num_rounds + 1):
            t0 = time.time()
            logger.info("--- RLAIF Round %d/%d ---", rnd, self.cfg.num_rounds)

            # Step 1: collect AI preference pairs
            pairs = collect_preference_pairs(self.policy, self.problems, self.cfg)
            if not pairs:
                logger.warning("Round %d: no pairs collected — skipping DPO", rnd)
                continue

            # Step 2: DPO epochs
            epoch_losses = []
            for epoch in range(self.cfg.dpo_epochs):
                loss = dpo_loss(self.policy, pairs, beta=self.cfg.dpo_beta)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [p for p in self.policy.model.parameters() if p.requires_grad], 1.0
                )
                self.optimizer.step()
                self.optimizer.zero_grad()
                epoch_losses.append(loss.item())

            avg_loss    = sum(epoch_losses) / len(epoch_losses)
            avg_gap     = sum(p["score_gap"] for p in pairs) / len(pairs)
            elapsed     = time.time() - t0

            if avg_loss < best_loss:
                best_loss = avg_loss
                self.policy.save_pretrained(str(self.out / "best"))

            record = {
                "round":    rnd,
                "pairs":    len(pairs),
                "avg_gap":  round(avg_gap, 3),
                "dpo_loss": round(avg_loss, 6),
                "elapsed_s": round(elapsed),
            }
            self.history.append(record)
            logger.info(
                "Round %d (%.0fs): pairs=%d | avg_gap=%.2f | dpo_loss=%.6f",
                rnd, elapsed, len(pairs), avg_gap, avg_loss,
            )

        results = {
            "method":      "RLAIF",
            "num_rounds":  self.cfg.num_rounds,
            "best_loss":   best_loss,
            "history":     self.history,
        }
        out_path = self.out / "rlaif_results.json"
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        logger.info("RLAIF complete | best_dpo_loss=%.6f | saved to %s", best_loss, self.out)
        return results
