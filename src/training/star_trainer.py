"""
STaR (Self-Taught Reasoner) trainer.

Iterative self-improvement loop:
  1. Generate chain-of-thought rationales + answers for training problems
  2. Keep rationales where the final answer is correct (rejection sampling)
  3. SFT fine-tune on the correct (problem, rationale, answer) triples
  4. Repeat with the updated model — each round the model learns from its
     own improved reasoning traces

The key insight: a model too weak to solve a problem cold can still
occasionally produce a correct rationale via sampling → training on
those rare successes lifts the baseline for the next round.

Reference: STaR — Bootstrapping Reasoning With Reasoning (Zelikman et al. 2022).
"""
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ── Config ────────────────────────────────────────────────────────────────────

@dataclass
class STaRConfig:
    data_file:       str   = "data/gsm8k_train_tool_dataset.jsonl"
    output_dir:      str   = "models/star"
    num_rounds:      int   = 3
    samples_per_q:   int   = 8        # rollouts per problem per round
    sft_epochs:      int   = 1        # SFT passes per round
    lr:              float = 5e-6
    max_new_tokens:  int   = 256
    num_problems:    int   = 20
    temperature:     float = 0.9
    log_interval:    int   = 1


# ── Rationale prompt ──────────────────────────────────────────────────────────

def _star_prompt(problem: str) -> str:
    return (
        "<|im_start|>system\n"
        "Solve the math problem step by step. Show every calculation.\n"
        "End with: <answer>NUMBER</answer><|im_end|>\n"
        f"<|im_start|>user\n{problem}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


# ── Answer extraction ─────────────────────────────────────────────────────────

import re as _re
_ANS_RE = _re.compile(r"<answer>\s*([\d,.\-]+)\s*</answer>", _re.IGNORECASE)
_NUM_RE = _re.compile(r"-?\d+(?:,\d{3})*(?:\.\d+)?")

def _extract_answer(text: str) -> Optional[str]:
    m = _ANS_RE.search(text)
    if m:
        return m.group(1).strip().replace(",", "")
    nums = _NUM_RE.findall(text)
    return nums[-1].replace(",", "") if nums else None

def _answers_match(pred: Optional[str], gt: str) -> bool:
    if pred is None:
        return False
    try:
        return abs(float(pred) - float(gt.replace(",", ""))) < 1e-4
    except ValueError:
        return pred.strip().lower() == gt.strip().lower()


# ── Generate rationales ───────────────────────────────────────────────────────

def generate_rationales(
    policy,
    problems: List[Dict],
    samples_per_q: int,
    max_new_tokens: int,
    temperature: float,
) -> List[Dict]:
    """
    For each problem, sample `samples_per_q` chain-of-thought responses.
    Return list of correct (problem, rationale, answer) triples.
    """
    device  = next(policy.model.parameters()).device
    correct = []

    for item in problems:
        problem = item["problem"]
        gt      = item["answer"]
        prompt  = _star_prompt(problem)
        enc     = policy.tokenizer(prompt, return_tensors="pt").to(device)

        for _ in range(samples_per_q):
            with torch.no_grad():
                out = policy.model.generate(
                    **enc,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=temperature,
                    top_p=0.95,
                    pad_token_id=policy.tokenizer.eos_token_id,
                )
            rationale = policy.tokenizer.decode(
                out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True
            )
            pred = _extract_answer(rationale)
            if _answers_match(pred, gt):
                correct.append({
                    "problem":   problem,
                    "rationale": rationale,
                    "answer":    gt,
                    "prompt":    prompt,
                })

    return correct


# ── SFT on correct rationales ─────────────────────────────────────────────────

def sft_on_rationales(
    policy,
    examples: List[Dict],
    epochs: int,
    lr: float,
    optimizer: torch.optim.Optimizer,
) -> float:
    """
    Supervised fine-tuning on correct (prompt, rationale) pairs.
    Uses teacher-forcing cross-entropy on the rationale tokens only.
    """
    device = next(policy.model.parameters()).device
    total_loss = 0.0
    steps = 0

    for epoch in range(epochs):
        for ex in examples:
            resp_ids = policy.tokenizer(
                ex["rationale"], return_tensors="pt", add_special_tokens=False
            ).input_ids.to(device)
            prompt_ids = policy.tokenizer(
                ex["prompt"], return_tensors="pt"
            ).input_ids.to(device)

            if resp_ids.shape[1] == 0:
                continue

            try:
                lp = policy.get_logprobs(prompt_ids, resp_ids)  # [1, T]
                # SFT loss = -mean log p(rationale | prompt)
                loss = -lp.mean()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [p for p in policy.model.parameters() if p.requires_grad], 1.0
                )
                optimizer.step()
                optimizer.zero_grad()
                total_loss += loss.item()
                steps += 1
            except Exception:
                optimizer.zero_grad()
                continue

    return total_loss / max(steps, 1)


# ── Evaluate ──────────────────────────────────────────────────────────────────

def evaluate(policy, eval_set: List[Dict], max_new_tokens: int) -> float:
    """Greedy-decode accuracy on held-out problems."""
    device  = next(policy.model.parameters()).device
    correct = 0

    for item in eval_set:
        prompt  = _star_prompt(item["problem"])
        enc     = policy.tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out = policy.model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=policy.tokenizer.eos_token_id,
            )
        pred_text = policy.tokenizer.decode(
            out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True
        )
        pred = _extract_answer(pred_text)
        if _answers_match(pred, item["answer"]):
            correct += 1

    return correct / max(len(eval_set), 1)


# ── Main loop ─────────────────────────────────────────────────────────────────

class STaRTrainer:
    def __init__(self, policy, cfg: STaRConfig):
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
        n_eval = max(3, len(rows) // 10)
        self.eval_set  = rows[-n_eval:]
        self.train_set = rows[:self.cfg.num_problems]
        logger.info("STaR | %d train | %d eval", len(self.train_set), len(self.eval_set))

    def run(self) -> Dict:
        # Baseline accuracy
        baseline_acc = evaluate(self.policy, self.eval_set, self.cfg.max_new_tokens)
        logger.info("Baseline accuracy: %.3f", baseline_acc)
        best_acc = baseline_acc
        self.history.append({"round": 0, "accuracy": round(baseline_acc, 4),
                             "correct_rationales": 0, "sft_loss": None})

        for rnd in range(1, self.cfg.num_rounds + 1):
            t0 = time.time()
            logger.info("--- STaR Round %d/%d ---", rnd, self.cfg.num_rounds)

            # Step 1: generate rationales, keep correct ones
            correct = generate_rationales(
                self.policy, self.train_set,
                self.cfg.samples_per_q, self.cfg.max_new_tokens, self.cfg.temperature,
            )
            n_correct = len(correct)
            yield_rate = n_correct / max(len(self.train_set) * self.cfg.samples_per_q, 1)
            logger.info("Round %d: %d/%d correct rationales (yield=%.1f%%)",
                        rnd, n_correct, len(self.train_set) * self.cfg.samples_per_q,
                        yield_rate * 100)

            if n_correct == 0:
                logger.warning("No correct rationales — skipping SFT this round")
                sft_loss = None
            else:
                # Step 2: SFT on correct rationales
                sft_loss = sft_on_rationales(
                    self.policy, correct,
                    self.cfg.sft_epochs, self.cfg.lr, self.optimizer,
                )
                logger.info("Round %d SFT loss: %.6f", rnd, sft_loss)

            # Step 3: evaluate
            acc = evaluate(self.policy, self.eval_set, self.cfg.max_new_tokens)
            elapsed = time.time() - t0

            if acc > best_acc:
                best_acc = acc
                self.policy.save_pretrained(str(self.out / "best"))
                logger.info("New best accuracy: %.3f → saved", best_acc)

            record = {
                "round":               rnd,
                "accuracy":            round(acc, 4),
                "correct_rationales":  n_correct,
                "yield_rate":          round(yield_rate, 4),
                "sft_loss":            round(sft_loss, 6) if sft_loss else None,
                "elapsed_s":           round(elapsed),
            }
            self.history.append(record)
            logger.info(
                "Round %d (%.0fs): acc=%.3f (+%.3f vs baseline) | correct=%d",
                rnd, elapsed, acc, acc - baseline_acc, n_correct,
            )

        results = {
            "method":       "STaR",
            "num_rounds":   self.cfg.num_rounds,
            "baseline_acc": round(baseline_acc, 4),
            "best_acc":     round(best_acc, 4),
            "improvement":  round(best_acc - baseline_acc, 4),
            "history":      self.history,
        }
        path = self.out / "star_results.json"
        with open(path, "w") as f:
            json.dump(results, f, indent=2)
        logger.info("STaR complete | %.3f → %.3f (+%.3f) | %s",
                    baseline_acc, best_acc, best_acc - baseline_acc, self.out)
        return results
