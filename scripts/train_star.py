#!/usr/bin/env python3
"""
STaR: Self-Taught Reasoner — iterative SFT on self-generated correct rationales.

Usage:
  CUDA_VISIBLE_DEVICES=0 python scripts/train_star.py \
    --num_rounds 3 --samples_per_q 8 --num_problems 20
"""
import argparse, logging, os, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
os.environ.setdefault("TRANSFORMERS_CACHE", "/storage/gxg8313/saiteja/hf_cache")
os.environ.setdefault("HF_HOME",            "/storage/gxg8313/saiteja/hf_cache")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s",
                    datefmt="%H:%M:%S")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model",         default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--data_file",     default="data/gsm8k_train_tool_dataset.jsonl")
    p.add_argument("--num_rounds",    type=int, default=3)
    p.add_argument("--samples_per_q", type=int, default=8)
    p.add_argument("--sft_epochs",    type=int, default=1)
    p.add_argument("--lr",            type=float, default=5e-6)
    p.add_argument("--num_problems",  type=int, default=20)
    p.add_argument("--temperature",   type=float, default=0.9)
    p.add_argument("--output_dir",    default="models/star")
    p.add_argument("--lora_r",        type=int, default=8)
    args = p.parse_args()

    from src.policy.llm_synthesis_policy import LLMSynthesisPolicy
    from src.training.star_trainer import STaRConfig, STaRTrainer

    policy = LLMSynthesisPolicy(args.model, lora_r=args.lora_r)
    cfg = STaRConfig(
        data_file      = args.data_file,
        output_dir     = args.output_dir,
        num_rounds     = args.num_rounds,
        samples_per_q  = args.samples_per_q,
        sft_epochs     = args.sft_epochs,
        lr             = args.lr,
        num_problems   = args.num_problems,
        temperature    = args.temperature,
    )
    trainer = STaRTrainer(policy, cfg)
    results = trainer.run()

    print("\n=== STaR Results ===")
    print(f"Baseline accuracy: {results['baseline_acc']:.3f}")
    print(f"Best accuracy:     {results['best_acc']:.3f}  (+{results['improvement']:.3f})")
    for r in results["history"]:
        suffix = f" | SFT loss={r['sft_loss']:.4f}" if r['sft_loss'] else ""
        print(f"  Round {r['round']}: acc={r['accuracy']:.3f} | "
              f"correct_rationales={r['correct_rationales']}{suffix}")

if __name__ == "__main__":
    main()
