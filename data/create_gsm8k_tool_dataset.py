#!/usr/bin/env python3
"""
Download GSM8K and format for agent RL training.

GSM8K: 7,473 grade-school math problems with step-by-step solutions.
Answers are integers — easy to verify, ideal for RLVR.

Output format (JSONL):
    {"problem": str, "answer": str, "available_tools": ["python_executor"]}

Usage:
    python data/create_gsm8k_tool_dataset.py
    python data/create_gsm8k_tool_dataset.py --split test --out data/gsm8k_test.jsonl
"""
import argparse
import json
import re
from pathlib import Path


def extract_answer(solution: str) -> str:
    """GSM8K stores the answer after '####'."""
    for line in reversed(solution.strip().split("\n")):
        if "####" in line:
            raw = line.split("####")[-1].strip()
            return re.sub(r"[,\s]", "", raw)   # strip commas/whitespace
    return ""


def load_gsm8k(split: str = "train") -> list:
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("pip install datasets")

    ds = load_dataset("gsm8k", "main", split=split)
    examples = []
    for item in ds:
        ans = extract_answer(item["answer"])
        if not ans:
            continue
        examples.append({
            "problem":         item["question"],
            "answer":          ans,
            "available_tools": ["python_executor"],
        })
    return examples


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--split", default="train", choices=["train", "test"])
    p.add_argument("--out",   default=None)
    p.add_argument("--limit", type=int, default=None, help="cap dataset size")
    args = p.parse_args()

    out_path = Path(args.out or f"data/gsm8k_{args.split}_tool_dataset.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Downloading GSM8K ({args.split})...")
    examples = load_gsm8k(args.split)
    if args.limit:
        examples = examples[: args.limit]

    with open(out_path, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")

    print(f"Wrote {len(examples)} examples → {out_path}")
    if examples:
        print(f"Sample: {examples[0]['problem'][:80]}...")
        print(f"Answer: {examples[0]['answer']}")


if __name__ == "__main__":
    main()
