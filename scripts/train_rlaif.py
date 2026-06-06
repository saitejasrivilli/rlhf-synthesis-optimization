#!/usr/bin/env python3
"""
RLAIF: train with AI-generated preference pairs via DPO.

Usage:
  CUDA_VISIBLE_DEVICES=0 python scripts/train_rlaif.py \
    --num_rounds 3 --candidates 4 --num_problems 20
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
    p.add_argument("--candidates",    type=int, default=4)
    p.add_argument("--pair_gap",      type=float, default=2.0)
    p.add_argument("--dpo_beta",      type=float, default=0.1)
    p.add_argument("--lr",            type=float, default=5e-6)
    p.add_argument("--num_problems",  type=int, default=20)
    p.add_argument("--output_dir",    default="models/rlaif")
    p.add_argument("--lora_r",        type=int, default=8)
    args = p.parse_args()

    from src.policy.llm_synthesis_policy import LLMSynthesisPolicy
    from src.training.rlaif_trainer import RAIFConfig, RAIFTrainer

    policy = LLMSynthesisPolicy(args.model, lora_r=args.lora_r)
    cfg = RAIFConfig(
        data_file        = args.data_file,
        output_dir       = args.output_dir,
        num_rounds       = args.num_rounds,
        candidates_per_q = args.candidates,
        pair_gap         = args.pair_gap,
        dpo_beta         = args.dpo_beta,
        lr               = args.lr,
        num_problems     = args.num_problems,
    )
    trainer = RAIFTrainer(policy, cfg)
    results = trainer.run()

    print("\n=== RLAIF Results ===")
    print(f"Rounds:    {results['num_rounds']}")
    print(f"Best loss: {results['best_loss']:.6f}")
    for r in results["history"]:
        print(f"  Round {r['round']}: pairs={r['pairs']} | gap={r['avg_gap']:.2f} | loss={r['dpo_loss']:.6f}")

if __name__ == "__main__":
    main()
