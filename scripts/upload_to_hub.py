#!/usr/bin/env python3
"""
Upload trained LoRA adapters to Hugging Face Hub.

Creates a model card and pushes the adapter weights (~40MB) so
the checkpoint is publicly reproducible.

Usage:
    python scripts/upload_to_hub.py \
        --ckpt_path models/dpo_policy/best \
        --repo_id saitejasrivilli/rlhf-synthesis-dpo \
        --method DPO

    # DPO + GRPO adapters
    python scripts/upload_to_hub.py \
        --ckpt_path models/grpo_policy/best \
        --repo_id saitejasrivilli/rlhf-synthesis-grpo \
        --method GRPO

Requirements:
    huggingface_hub (pip install huggingface_hub)
    HF_TOKEN env var or `huggingface-cli login`
"""
import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

MODEL_CARD_TEMPLATE = """\
---
language: en
license: apache-2.0
base_model: Qwen/Qwen2.5-7B-Instruct
tags:
  - rlhf
  - pharmaceutical-synthesis
  - lora
  - peft
  - chemistry
datasets:
  - open-reaction-database
---

# RLHF Synthesis Optimization — {method} Adapter

LoRA adapter (r=8, α=16) fine-tuned with **{method}** on pharmaceutical synthesis data.

The policy proposes reaction conditions (temperature, time, catalyst loading, solvent ratio)
that maximize yield, selectivity, and safety for 5 drug molecules.

## Results

| Method  | Data | Avg Reward | >0.75 (%) | vs Baseline |
|---------|------|-----------|-----------|-------------|
| {method} | ORD (real) | {avg_reward} | {pct_above} | {improvement} |

## Usage

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

base = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-7B-Instruct",
    dtype=torch.float16,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
model = PeftModel.from_pretrained(base, "{repo_id}")

prompt = \"\"\"### Optimize pharmaceutical synthesis conditions

Molecule: Aspirin
CAS: PHARMA_Aspirin
Objective: maximize yield and selectivity, minimize hazard and steps

### Optimal conditions (JSON):
\"\"\"

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
out = model.generate(**inputs, max_new_tokens=128, temperature=0.7, do_sample=True)
print(tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True))
```

## Training

- Base model: `Qwen/Qwen2.5-7B-Instruct`
- LoRA: r=8, α=16, targets q/k/v/o_proj (5.05M trainable params, 0.066%)
- Data: 500 literature-based pharmaceutical synthesis trajectories (ORD format)
- Pipeline: SFT warmup (3 epochs) → {method} fine-tuning

Repo: [saitejasrivilli/rlhf-synthesis-optimization](https://github.com/saitejasrivilli/rlhf-synthesis-optimization)
"""


def read_results(results_path: str, method: str) -> dict:
    """Try to read evaluation results for the model card."""
    if not results_path or not Path(results_path).exists():
        return {"avg_reward": "N/A", "pct_above": "N/A", "improvement": "N/A"}
    try:
        with open(results_path) as f:
            data = json.load(f)
        ev = data.get("evaluation", {})
        return {
            "avg_reward":  f"{ev.get('average_reward', 0):.4f}",
            "pct_above":   f"{ev.get('pct_above_threshold', 0):.0f}%",
            "improvement": f"{ev.get('improvement', 0):+.1f}%",
        }
    except Exception:
        return {"avg_reward": "N/A", "pct_above": "N/A", "improvement": "N/A"}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt_path",    required=True,
                   help="Path to saved checkpoint (e.g. models/dpo_policy/best)")
    p.add_argument("--repo_id",      required=True,
                   help="HF Hub repo, e.g. saitejasrivilli/rlhf-synthesis-dpo")
    p.add_argument("--method",       default="DPO",
                   choices=["DPO", "GRPO", "PPO", "SFT"],
                   help="Training method used")
    p.add_argument("--results_file", default=None,
                   help="Path to JSON results file for model card stats")
    p.add_argument("--private",      action="store_true",
                   help="Create private repository")
    p.add_argument("--token",        default=None,
                   help="HF API token (or set HF_TOKEN env var)")
    return p.parse_args()


def main():
    args = parse_args()

    try:
        from huggingface_hub import HfApi, create_repo
    except ImportError:
        logger.error("pip install huggingface_hub")
        sys.exit(1)

    token = args.token or os.environ.get("HF_TOKEN")
    if not token:
        logger.error("Provide --token or set HF_TOKEN env var. "
                     "Get a token from https://huggingface.co/settings/tokens")
        sys.exit(1)

    ckpt_path = Path(args.ckpt_path)
    if not ckpt_path.exists():
        logger.error(f"Checkpoint not found: {ckpt_path}")
        sys.exit(1)

    api = HfApi(token=token)

    # Create repo if it doesn't exist
    logger.info(f"Creating/updating repo: {args.repo_id}")
    create_repo(args.repo_id, token=token, private=args.private, exist_ok=True)

    # Write model card
    stats = read_results(args.results_file, args.method)
    card_text = MODEL_CARD_TEMPLATE.format(
        method=args.method,
        repo_id=args.repo_id,
        **stats,
    )
    card_path = ckpt_path / "README.md"
    card_path.write_text(card_text)
    logger.info("Model card written to checkpoint directory")

    # Upload all files in the checkpoint directory
    logger.info(f"Uploading {ckpt_path} → {args.repo_id} ...")
    api.upload_folder(
        folder_path=str(ckpt_path),
        repo_id=args.repo_id,
        repo_type="model",
        commit_message=f"Upload {args.method} LoRA adapter — {datetime.now().strftime('%Y-%m-%d')}",
        token=token,
        ignore_patterns=["*.pt.bak", "__pycache__/**"],
    )

    logger.info(f"Uploaded to https://huggingface.co/{args.repo_id}")
    logger.info("Share link: "
                f"https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
