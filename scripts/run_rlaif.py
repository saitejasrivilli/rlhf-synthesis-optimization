#!/usr/bin/env python3
"""
RLAIF (Reinforcement Learning from AI Feedback) — pairwise judge demo.

Algorithm:
  1. For each prompt, generate K candidate responses
  2. LLM judge compares pairs (A vs B) and picks the better response
  3. Collect (prompt, chosen, rejected) preference pairs
  4. Write DPO-ready JSONL — feed into any DPO trainer

Demo mode uses rule-based generator + judge (no GPU required).
--mode hf loads any HuggingFace causal LM.

Usage:
    python scripts/run_rlaif.py --mode demo --n_prompts 10 --candidates 4
    python scripts/run_rlaif.py --mode hf --model Qwen/Qwen2.5-7B-Instruct
"""

import argparse
import json
import logging
import random
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

logger = logging.getLogger(__name__)


# ── Prompts ───────────────────────────────────────────────────────────────────

SAMPLE_PROMPTS = [
    "Explain the difference between supervised and unsupervised learning.",
    "What are the risks of using AI systems in medical diagnosis?",
    "How does gradient descent optimize a neural network?",
    "What is the difference between precision and recall?",
    "Describe three ways to reduce overfitting in deep learning.",
    "What is attention mechanism in transformers?",
    "Explain the bias-variance tradeoff.",
    "How does reinforcement learning differ from supervised learning?",
    "What are the ethical implications of large language models?",
    "Describe how RLHF works at a high level.",
]


# ── Pairwise judge prompt ─────────────────────────────────────────────────────

_JUDGE_TMPL = """\
You are an impartial judge evaluating AI assistant responses.

Prompt: {prompt}

Response A:
{response_a}

Response B:
{response_b}

Which response is better? Consider: accuracy, completeness, clarity, safety.
Respond with EXACTLY one line: "Winner: A" or "Winner: B" followed by a \
one-sentence reason.
"""


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class PreferencePair:
    prompt: str
    chosen: str
    rejected: str
    judge_reason: str
    score_gap: float = 0.0


@dataclass
class RLAIFStats:
    n_prompts: int = 0
    n_candidates: int = 0
    n_comparisons: int = 0
    n_pairs: int = 0
    avg_chosen_len: float = 0.0
    avg_rejected_len: float = 0.0
    pairs: list = field(default_factory=list)


# ── Rule-based demo components ────────────────────────────────────────────────

_GOOD_QUALIFIERS = [
    "specifically", "for example", "in detail", "importantly", "however",
    "first", "second", "therefore", "because", "which means",
]


def _rule_generator(prompt: str, variant: int) -> str:
    """Produce variant responses of varying quality."""
    if variant == 0:
        return (
            f"This is an important question about {prompt.lower()[:40]}. "
            "The answer involves several key considerations. "
            "First, it is important to understand the fundamentals. "
            "Second, practical applications matter significantly. "
            "Third, there are tradeoffs that must be carefully evaluated. "
            "In conclusion, a thorough understanding requires examining multiple perspectives."
        )
    elif variant == 1:
        return (
            f"To answer this, we need to break it down. "
            f"{prompt.split()[0]} relates to core ML concepts. "
            "There are several dimensions: theoretical foundations, practical implementation, "
            "and real-world limitations. For example, gradient-based methods rely on "
            "differentiability. The key tradeoff is between computational cost and accuracy."
        )
    elif variant == 2:
        return f"Yes. {prompt.split('?')[0]} is correct."
    else:
        return (
            f"Great question! {prompt[:30]}... involves understanding that "
            "machine learning systems work by learning patterns from data. "
            "This is relevant to many applications. Overall, it depends on the context "
            "and specific use case you are working with."
        )


def _rule_judge(prompt: str, response_a: str, response_b: str) -> tuple[str, str]:
    """Heuristic judge: prefer more specific, longer, structured responses."""
    def score(r: str) -> float:
        s = len(r) * 0.01
        s += sum(q in r.lower() for q in _GOOD_QUALIFIERS) * 0.5
        s += r.count(".") * 0.2
        s += r.count(",") * 0.1
        s -= r.lower().count("great question") * 2.0
        s -= r.lower().count("it depends") * 1.0
        return s

    sa, sb = score(response_a), score(response_b)
    if abs(sa - sb) < 0.3:
        return "A", "responses are similarly structured"
    if sa > sb:
        return "A", "response A is more specific and better structured"
    return "B", "response B is more specific and better structured"


# ── Core RLAIF loop ───────────────────────────────────────────────────────────

def run_rlaif(
    prompts: list[str],
    generator_fn,       # callable(prompt, variant_idx) -> str
    judge_fn,           # callable(prompt, response_a, response_b) -> (winner, reason)
    n_candidates: int = 4,
    min_gap_frac: float = 0.0,    # accept all pairs in demo
    seed: int = 42,
) -> RLAIFStats:
    rng = random.Random(seed)
    stats = RLAIFStats(n_prompts=len(prompts))

    for prompt in prompts:
        # Step 1: generate K candidates
        variants = list(range(n_candidates))
        rng.shuffle(variants)
        candidates = [generator_fn(prompt, v) for v in variants[:n_candidates]]
        stats.n_candidates += len(candidates)

        # Step 2: pairwise comparisons (all pairs)
        for i in range(len(candidates)):
            for j in range(i + 1, len(candidates)):
                stats.n_comparisons += 1
                winner, reason = judge_fn(prompt, candidates[i], candidates[j])
                chosen   = candidates[i] if winner == "A" else candidates[j]
                rejected = candidates[j] if winner == "A" else candidates[i]
                if chosen != rejected:
                    stats.pairs.append(PreferencePair(
                        prompt=prompt,
                        chosen=chosen,
                        rejected=rejected,
                        judge_reason=reason,
                    ))
                    stats.n_pairs += 1

    if stats.pairs:
        stats.avg_chosen_len   = sum(len(p.chosen) for p in stats.pairs) / len(stats.pairs)
        stats.avg_rejected_len = sum(len(p.rejected) for p in stats.pairs) / len(stats.pairs)

    return stats


def save_rlaif_output(stats: RLAIFStats, output_dir: str) -> None:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    dpo_path = f"{output_dir}/rlaif_dpo.jsonl"
    with open(dpo_path, "w") as f:
        for p in stats.pairs:
            f.write(json.dumps({
                "prompt":       p.prompt,
                "chosen":       p.chosen,
                "rejected":     p.rejected,
                "judge_reason": p.judge_reason,
                "source":       "rlaif",
            }) + "\n")
    summary = {
        "n_prompts":        stats.n_prompts,
        "n_candidates":     stats.n_candidates,
        "n_comparisons":    stats.n_comparisons,
        "n_pairs":          stats.n_pairs,
        "pairs_per_prompt": round(stats.n_pairs / max(stats.n_prompts, 1), 2),
        "avg_chosen_len":   round(stats.avg_chosen_len),
        "avg_rejected_len": round(stats.avg_rejected_len),
        "len_gap":          round(stats.avg_chosen_len - stats.avg_rejected_len),
    }
    with open(f"{output_dir}/rlaif_stats.json", "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("saved %d DPO pairs → %s", stats.n_pairs, dpo_path)


# ── HF model wrappers ─────────────────────────────────────────────────────────

def _make_hf_generator(model, tok, max_new_tokens: int = 200):
    import torch
    def generate(prompt: str, variant: int) -> str:
        temperature = 0.7 + variant * 0.15
        enc = tok(
            f"Answer clearly and specifically: {prompt}",
            return_tensors="pt", truncation=True, max_length=256,
        ).to(model.device)
        with torch.no_grad():
            out = model.generate(
                **enc, max_new_tokens=max_new_tokens,
                do_sample=(variant > 0), temperature=temperature, top_p=0.9,
                pad_token_id=tok.eos_token_id,
            )
        return tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
    return generate


def _make_hf_judge(model, tok, max_new_tokens: int = 64):
    import torch
    _winner_re = re.compile(r"winner\s*:\s*([AB])", re.IGNORECASE)
    def judge(prompt: str, response_a: str, response_b: str) -> tuple[str, str]:
        judge_prompt = _JUDGE_TMPL.format(
            prompt=prompt, response_a=response_a[:400], response_b=response_b[:400],
        )
        enc = tok(judge_prompt[-1800:], return_tensors="pt",
                  truncation=True, max_length=512).to(model.device)
        with torch.no_grad():
            out = model.generate(
                **enc, max_new_tokens=max_new_tokens, do_sample=False,
                pad_token_id=tok.eos_token_id,
            )
        text = tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
        m = _winner_re.search(text)
        winner = m.group(1).upper() if m else ("A" if hash(text) % 2 == 0 else "B")
        reason = text.split("\n")[0][:120] if text else "no reason given"
        return winner, reason
    return judge


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["demo", "hf"], default="demo")
    p.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--n_prompts", type=int, default=10)
    p.add_argument("--candidates", type=int, default=4)
    p.add_argument("--output_dir", default="output/rlaif")
    p.add_argument("--custom_prompts", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

    prompts = SAMPLE_PROMPTS
    if args.custom_prompts:
        prompts = [json.loads(l)["prompt"] for l in open(args.custom_prompts)]
    prompts = (prompts * 10)[:args.n_prompts]

    if args.mode == "demo":
        generator_fn = _rule_generator
        judge_fn     = _rule_judge
        logging.info("using rule-based generator + judge (no GPU)")
    else:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        logging.info("loading %s ...", args.model)
        tok = AutoTokenizer.from_pretrained(args.model)
        mdl = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.float16, device_map="auto"
        )
        generator_fn = _make_hf_generator(mdl, tok)
        judge_fn     = _make_hf_judge(mdl, tok)

    logging.info("running RLAIF on %d prompts, %d candidates each ...",
                 len(prompts), args.candidates)
    stats = run_rlaif(prompts, generator_fn, judge_fn, n_candidates=args.candidates)
    save_rlaif_output(stats, args.output_dir)

    pairs_per = stats.n_pairs / max(stats.n_prompts, 1)
    print(f"\n=== RLAIF Results ===")
    print(f"  Prompts:           {stats.n_prompts}")
    print(f"  Candidates/prompt: {args.candidates}")
    print(f"  Comparisons:       {stats.n_comparisons}")
    print(f"  DPO pairs:         {stats.n_pairs}  ({pairs_per:.1f}/prompt)")
    print(f"  Chosen avg len:    {stats.avg_chosen_len:.0f} chars")
    print(f"  Rejected avg len:  {stats.avg_rejected_len:.0f} chars")
    print(f"  Length gap:        +{stats.avg_chosen_len - stats.avg_rejected_len:.0f} chars")
    print(f"\n  Output: {args.output_dir}/")
    print(f"    rlaif_dpo.jsonl  — (prompt, chosen, rejected, judge_reason) for DPO")
    print(f"    rlaif_stats.json — summary statistics")

    if stats.pairs:
        ex = stats.pairs[0]
        print(f"\n--- Example pair ---")
        print(f"  Prompt:   {ex.prompt[:70]}")
        print(f"  Chosen:   {ex.chosen[:100]}")
        print(f"  Rejected: {ex.rejected[:100]}")
        print(f"  Reason:   {ex.judge_reason}")


if __name__ == "__main__":
    main()
