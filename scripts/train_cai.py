#!/usr/bin/env python3
"""
Constitutional AI data generation script.

Generates critique→revise pairs from a set of prompts using a local model,
then saves SFT and DPO datasets ready for fine-tuning.

Usage:
    # Demo with rule-based model (no GPU)
    python scripts/train_cai.py --mode demo --n_prompts 10

    # With a local HuggingFace model
    python scripts/train_cai.py \
        --mode hf \
        --model Qwen/Qwen2.5-7B-Instruct \
        --n_prompts 100 \
        --n_principles 2 \
        --output_dir output/cai_qwen
"""

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.training.cai_trainer import (
    CAIConfig, DEFAULT_CONSTITUTION,
    run_cai_dataset, save_cai_output, SAMPLE_PROMPTS,
    score_critique_quality,
)


def _rule_based_model(prompt: str) -> str:
    """Deterministic mock model for demo mode — no GPU needed."""
    p = prompt.lower()
    if "critique:" in p or "review the assistant" in p:
        return (
            "The response could be more constructive. It should focus on the "
            "potential consequences and offer a helpful alternative approach "
            "rather than directly enabling harmful behaviour."
        )
    if "revised response:" in p or "rewrite" in p:
        return (
            "I understand you're asking about this topic. Instead of providing "
            "guidance that could cause harm, I'd encourage you to consider the "
            "impact on others and seek constructive alternatives. If you're facing "
            "a specific challenge, I'm happy to help you find ethical solutions."
        )
    return (
        "Sure, here's how you could approach that situation, though I should note "
        "there may be ethical concerns worth considering..."
    )


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["demo", "hf"], default="demo")
    p.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--n_prompts", type=int, default=10)
    p.add_argument("--n_principles", type=int, default=2)
    p.add_argument("--output_dir", default="output/cai")
    p.add_argument("--custom_prompts", default=None, help="JSONL file with {prompt: ...} records")
    p.add_argument("--log_level", default="INFO")
    return p.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level),
                        format="%(asctime)s %(levelname)s: %(message)s")

    # Load prompts
    if args.custom_prompts:
        prompts = [json.loads(l)["prompt"] for l in open(args.custom_prompts)][:args.n_prompts]
    else:
        prompts = (SAMPLE_PROMPTS * 10)[:args.n_prompts]

    # Build model function
    if args.mode == "demo":
        model_fn = _rule_based_model
        logging.info("using rule-based demo model (no GPU)")
    else:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        logging.info("loading %s ...", args.model)
        tok = AutoTokenizer.from_pretrained(args.model)
        mdl = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.float16, device_map="auto"
        )
        def model_fn(prompt: str) -> str:
            enc = tok(prompt[-2000:], return_tensors="pt",
                      truncation=True, max_length=512).to(mdl.device)
            with torch.no_grad():
                out = mdl.generate(**enc, max_new_tokens=256, do_sample=False,
                                   pad_token_id=tok.eos_token_id)
            return tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)

    cfg = CAIConfig(
        constitution=DEFAULT_CONSTITUTION,
        n_principles_per_example=args.n_principles,
        output_dir=args.output_dir,
    )

    logging.info("running CAI on %d prompts (%d principles each) ...", len(prompts), args.n_principles)
    examples = run_cai_dataset(prompts, model_fn, cfg)
    stats = save_cai_output(examples, args.output_dir)

    # Critique quality analysis
    all_critiques = [r.critique for ex in examples for r in ex.rounds]
    quality_scores = [score_critique_quality(c) for c in all_critiques]
    substantive = sum(q["is_substantive"] for q in quality_scores)
    mentions_issue = sum(q["mentions_issue"] for q in quality_scores)

    print(f"\n=== CAI Results ===")
    print(f"  Prompts processed:   {stats['n_examples']}")
    print(f"  SFT records:         {stats['sft_records']}")
    print(f"  DPO pairs:           {stats.get('dpo_records', 0)}")
    print(f"  Avg length delta:    {stats['avg_length_delta']:+.0f} chars (revised vs initial)")
    print(f"  Substantive critiques: {substantive}/{len(quality_scores)} ({substantive/max(len(quality_scores),1):.0%})")
    print(f"  Issue-flagging critiques: {mentions_issue}/{len(quality_scores)} ({mentions_issue/max(len(quality_scores),1):.0%})")
    print(f"\n  Output: {args.output_dir}/")
    print(f"    cai_sft.jsonl  — (prompt, revised_response) for SFT")
    print(f"    cai_dpo.jsonl  — (prompt, chosen=revised, rejected=initial) for DPO")
    print(f"    cai_stats.json — summary statistics")

    # Show one example
    if examples:
        ex = examples[0]
        print(f"\n--- Example ---")
        print(f"Prompt:   {ex.prompt[:80]}")
        print(f"Initial:  {ex.initial_response[:120]}")
        if ex.rounds:
            print(f"Critique: {ex.rounds[0].critique[:120]}")
            print(f"Revised:  {ex.rounds[-1].revised_response[:120]}")


if __name__ == "__main__":
    main()
