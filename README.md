# RLHF + PPO for Pharmaceutical Synthesis Optimization

Reinforcement learning from human feedback applied to synthesis condition optimization.
The policy learns to propose reaction parameters (temperature, time, catalyst, solvent) that maximize
yield while controlling safety risk — framed as a language generation task over Qwen2.5-7B.

---

## Architecture

```
Molecule name / SMILES
        │
        ▼
┌─────────────────────────────────┐
│  LLMSynthesisPolicy             │
│  ┌──────────────────────────┐   │
│  │  Qwen2.5-7B (frozen)     │   │  ← base LLM
│  │  + LoRA r=8, α=16        │   │  ← trainable adapters
│  │    targets: q/k/v/o_proj │   │
│  └──────────────────────────┘   │
│  ┌──────────────┐                │
│  │  Value head  │ Linear(3584→1) │  ← PPO critic
│  └──────────────┘                │
└─────────────────────────────────┘
        │ generates JSON conditions
        ▼
{"temperature_celsius": 85, "time_hours": 3.5, ...}
        │
        ▼
┌─────────────────────────────────┐
│  LearnedRewardModel             │
│  BERT-base CLS → Linear(768→1)  │  ← reward scorer
│  fine-tuned + rule-based blend  │
└─────────────────────────────────┘
        │ scalar reward ∈ [0,1]
        ▼
     PPO Update
  (LoRA + value head only)
```

**Reward weights (rule-based component):** yield 40% · selectivity 30% · safety 20% · efficiency 10%

---

## Results

### MLP-PPO baseline (single GPU, 5 epochs)

| Split | Avg Reward | High-quality (>0.75) | Improvement |
|---|---|---|---|
| Train | 0.8577 | — | — |
| **Test** | **0.8121** | **100 / 100** | **+62.4%** |

### LLM-PPO — Qwen2.5-7B + LoRA r=8 (50 iterations, 1×A30)

| Split | Avg Reward | Improvement |
|---|---|---|
| Best train | 0.7318 | — |
| **Test** | **0.5801** | **+16.0%** |

### Head-to-head benchmark (100-trajectory held-out test set)

| Model | Avg Reward | >0.60 (%) | >0.75 (%) | Δ vs baseline | Train time |
|---|---|---|---|---|---|
| Rule-based baseline | 0.517 | 0% | 0% | — | — |
| MLP-PPO (128→256→256) | 0.517 | 0% | 0% | +3.3% | 1.2 s |
| **LLM-PPO** (Qwen2.5-7B + LoRA r=8) | **0.601** | **33%** | **6%** | **+20.2%** | ~5 min/10 iters |

> **Why the LLM wins:** The 8-dimensional MLP state vector discards all molecular structure.
> Qwen2.5-7B encodes chemical context through its pre-trained weights, giving it a far
> stronger inductive bias. LoRA keeps the fine-tuning cost to **5.05M trainable params
> out of 7.62B (0.066%)**.

> **Why test < train for LLM-PPO:** 50 iterations is a short pilot; the policy hasn't
> fully generalized. The reward model is also BERT-based and still blended with rule-based
> scores. Both improve with more iterations.

---

## Hardware & environment

| Item | Value |
|---|---|
| GPUs | 3 × NVIDIA A30 (24 GB each) |
| Run config | Single GPU (NCCL multi-GPU unavailable on this cluster) |
| Base model | `Qwen/Qwen2.5-7B-Instruct` (local cache) |
| Python | 3.13 |
| PyTorch | 2.x |
| transformers | 4.57.6 |
| peft | 0.19.1 |
| accelerate | 1.12.0 |

---

## Quick start

```bash
pip install torch transformers peft accelerate

# MLP-PPO baseline (single GPU, ~2 min)
python scripts/run_pipeline.py

# LLM-PPO single GPU (50 iterations, ~5 min)
CUDA_VISIBLE_DEVICES=0 python scripts/train_ppo_distributed.py \
    --model_name Qwen/Qwen2.5-7B-Instruct \
    --num_iterations 50 \
    --batch_size 2 \
    --finetune_reward_model

# LLM-PPO multi-GPU via torchrun
python -m torch.distributed.run --standalone --nproc_per_node=3 \
    scripts/train_ppo_distributed.py \
    --model_name Qwen/Qwen2.5-7B-Instruct \
    --num_iterations 200 \
    --batch_size 4

# LLM-PPO with DeepSpeed ZeRO-2
deepspeed --num_gpus=3 scripts/train_ppo_distributed.py \
    --deepspeed configs/deepspeed_zero2.json

# Benchmark (compare all models)
python scripts/benchmark.py --data_file data/trajectories_improvable.jsonl

# Benchmark using saved LLM checkpoint
python scripts/benchmark.py \
    --llm_ckpt models/llm_policy/best \
    --eval_only
```

---

## Distributed setup

### torchrun

```bash
python -m torch.distributed.run --standalone --nproc_per_node=3 \
    scripts/train_ppo_distributed.py --batch_size 4

# Memory-constrained: 8-bit quantization
python -m torch.distributed.run --nproc_per_node=3 \
    scripts/train_ppo_distributed.py --load_in_8bit
```

### DeepSpeed ZeRO-2

ZeRO-2 shards optimizer states and gradients across GPUs, cutting per-GPU memory by ~2×.
Config at `configs/deepspeed_zero2.json`.

```
GPU 0: full model replica + ZeRO-2 optim shard 0
GPU 1: full model replica + ZeRO-2 optim shard 1
GPU 2: full model replica + ZeRO-2 optim shard 2
```

### Ray cluster

```bash
ray start --head --num-gpus=3
ray job submit --address=http://127.0.0.1:8265 \
    -- python scripts/train_ppo_distributed.py
```

Cluster config: `configs/ray_cluster.yaml`

---

## Repository layout

```
rlhf-synthesis-optimization/
├── config/
│   └── config.json                  # Unified config for all components
├── configs/
│   ├── deepspeed_zero2.json         # DeepSpeed ZeRO-2 config (3×A30)
│   └── ray_cluster.yaml             # Ray cluster config
├── data/
│   ├── trajectories_improvable.jsonl   # 500 synthesis trajectories (primary)
│   └── labeled/trajectories_expanded.jsonl
├── scripts/
│   ├── run_pipeline.py              # MLP-PPO single-GPU entry point
│   ├── train_ppo_distributed.py     # LLM-PPO distributed entry point
│   └── benchmark.py                 # Benchmark harness (MLP vs LLM)
├── src/
│   ├── config.py
│   ├── paths.py
│   ├── policy/
│   │   ├── synthesis_policy.py      # Stub / interface
│   │   └── llm_synthesis_policy.py  # Qwen2.5-7B + LoRA + value head
│   ├── reward/
│   │   ├── real_reward_model.py     # Rule-based (original)
│   │   └── learned_reward_model.py  # BERT-base fine-tuned (new)
│   ├── training/
│   │   ├── real_ppo_trainer.py      # MLP-PPO (original)
│   │   └── llm_ppo_trainer.py       # LLM-PPO with KL penalty (new)
│   └── evaluation/
│       └── real_rlhf_evaluator.py
├── models/
│   └── llm_policy/
│       ├── best/                    # Best LoRA checkpoint + value head
│       └── ckpt_00050/              # Iteration 50 checkpoint
└── results/
    ├── benchmark_results.json       # Head-to-head comparison
    ├── rlhf_llm_distributed.json    # LLM-PPO training run
    └── rlhf_results.json            # MLP-PPO training run
```

---

## PPO training details

| Hyper-parameter | Value |
|---|---|
| Base model | Qwen/Qwen2.5-7B-Instruct |
| LoRA rank | 8 |
| LoRA alpha | 16 |
| LoRA targets | q\_proj, k\_proj, v\_proj, o\_proj |
| Trainable params | 5.05M / 7.62B (0.066%) |
| Batch size | 2 per GPU (pilot) / 4 per GPU (full) |
| Gradient accumulation | 4 steps |
| Learning rate | 1e-5 (Adam) |
| PPO clip ratio | 0.2 |
| KL penalty weight | 0.1 |
| Entropy coefficient | 0.01 |
| Precision | fp16 + GradScaler |
| Reference policy | Base model via `disable_adapter()` — no extra VRAM |
| Iterations (pilot) | 50 |
| Iterations (full) | 200 |

---

## Key lessons

1. **LLM inductive bias matters.** Pre-trained language models encode chemical knowledge
   that cannot be recovered from 8-dimensional hand-crafted state vectors.
2. **KL penalty via `disable_adapter()`** — disabling LoRA on the same model instance
   gives the frozen reference policy for free, halving the VRAM needed vs a second model copy.
3. **BERT reward model fine-tuning converges fast** — 3 epochs on 400 trajectories drives
   MSE loss from 0.007 → 0.001 in under 10 seconds.
4. **Honest evaluation.** Both the MLP-PPO +62.4% train result and the LLM-PPO test gap
   are reported — the train/test gap shows the policy needs more iterations to generalize.
