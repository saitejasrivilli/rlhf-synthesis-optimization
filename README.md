# RLHF + Agent RL for LLM Post-Training

Full post-training pipeline: SFT warmup → DPO → PPO / GRPO / **Agent GRPO with tool use** / **PRM step-level rewards**.

Two domains:
- **Pharmaceutical synthesis** — policy proposes reaction conditions (yield, selectivity, safety)
- **Agent RL on GSM8K** — policy calls a Python executor tool, reward = verifiable answer match (RLVR)

---

## Process Reward Model (PRM): Step-Level Rewards

Standard GRPO uses a binary terminal reward (1.0 if correct, 0 otherwise). PRM adds
**intermediate rewards after each reasoning step**, providing denser gradient signal
and penalising shortcut solutions that guess the final answer without showing work.

### Reward structure

```
Response with 4 steps → PRM scores each step individually:

  Step 1: "Sarah has 3 bags × 8 apples = 24"       → 0.15  (verified: 3×8=24 ✓)
  Step 2: "Half of 24 oranges = 12"                  → 0.10  (expr parsed, not grounded)
  Step 3: "Total = 24 + 12 = 36"                     → 0.15  (verified: 24+12=36 ✓)
  Final:  <final_answer>36</final_answer> vs GT=36   → 1.00  (verifiable reward)

  Discounted total (γ=0.9):  Σ γ^(T-1-t) * r_t  +  r_final
                           =  0.15×0.81 + 0.10×0.90 + 0.15×1.0  +  1.0
                           =  1.36

  Wrong answer (GT=40):      same step rewards + 0.0 final = 0.36
                             (step signal preserved — model still learns to show work)
```

### Verification approach (rule-based, no second model)

1. **Fenced code blocks** — execute with subprocess, reward 0.15 if valid output
2. **Assignment statements** — `x = 3 * 24 = 72`: eval LHS, check vs stated RHS
3. **Bare arithmetic** — any `+/-/×/÷` expression that evaluates: reward 0.10

### Usage

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/train_prm_grpo.py \
    --num_iters 60 \
    --group_size 4 \
    --gamma 0.9 \
    --step_weight 0.5 \
    --output_dir models/prm_grpo
```

**Measured results** (30 iters, 15 problems, γ=0.9, step_weight=0.5, G=4):

| Iter | Reward | Step Frac | Notes |
|------|--------|-----------|-------|
| 5 | ~0.40 | 0.75 | High step signal early — model reasoning without final answer |
| 10 | **1.0659** | 0.14 | Peak — correct answers + step verification aligned |
| 20 | 0.657 | 0.07 | Variance increases as model overfits prompt format |
| 30 | 0.782 | 0.32 | Recovers; best checkpoint saved at iter 10 |

**Best reward: 1.0659** (correct answer + step rewards, γ-discounted). Step frac=0.15 means ~15% of reward signal came from intermediate step verification — the PRM signal keeps gradients alive on partially-correct responses.

---

---

## Agent RL with Verifiable Rewards (new)

The core new addition: training an LLM agent with GRPO where reward comes from
**verifiable outcomes** (ground truth match), not a learned reward model.

### Architecture

```
Problem: "Janet earns $25/hr. She works 52 hrs/week. How much does she earn in 4 weeks?"

Model turn 1 (generate):
  <think>I'll compute 25 * 52 * 4 in Python</think>
  <tool_call>{"name": "python_executor", "args": {"code": "25 * 52 * 4"}}</tool_call>

Tool execution (injected, not part of policy gradient):
  <tool_result>5200</tool_result>

Model turn 2 (generate, conditioned on full context):
  <final_answer>5200</final_answer>

Reward: compute_reward(output, "5200") → 1.0  ← verifiable, no learned RM
```

### GRPO update

For each problem, generate G=4 trajectories → compute group-relative advantages:
```
advantages_i = (reward_i - mean(rewards)) / (std(rewards) + 1e-8)
```
Log-probs computed per-segment, each conditioned on the full context including
injected tool results — not on a naive concatenation that would ignore tool outputs.

### Usage

```bash
# Download dataset (7,473 GSM8K math problems)
python data/create_gsm8k_tool_dataset.py

# Train
CUDA_VISIBLE_DEVICES=0 python scripts/train_agent_grpo.py \
    --data_file data/gsm8k_train_tool_dataset.jsonl \
    --num_iterations 200 \
    --group_size 4

# With W&B
python scripts/train_agent_grpo.py --use_wandb --wandb_project rlhf-synthesis
```

---

## Pipeline

```
Real ORD Data (500 trajectories, 5 molecules)
        │
        ▼
  ① SFT Warmup
  Top-quartile trajectories (yield > 0.88)
  Fine-tune Qwen2.5-7B + LoRA on (prompt → JSON conditions)
        │
        ├──────────────────────────────────┐
        ▼                                  ▼
  ② PPO Branch                      ③ DPO Branch
  BERT reward model                  Preference pairs
  (BT-trained or rule-based)         (chosen / rejected)
  + KL via disable_adapter()         Direct BT loss, no RM
        │                                  │
        └──────────────┬───────────────────┘
                       ▼
              Evaluate on held-out test set
```

**Architecture:**
- Policy: Qwen2.5-7B-Instruct + LoRA (r=8, α=16, q/k/v/o_proj) — **5.05M / 7.62B trainable (0.066%)**
- Reward: BERT-base with Bradley-Terry ranking loss on preference pairs
- Reference policy: base model via `disable_adapter()` — zero extra VRAM

---

## Results

### All experiments — held-out test set (100 trajectories)

| Method | Data | Avg Reward | >0.75 (%) | Improvement | Notes |
|---|---|---|---|---|---|
| Rule-based baseline | improvable | 0.517 | 0% | — | Hand-crafted weights |
| **MLP-PPO** (128→256→256, fixed) | improvable | 0.812 | 100% | +62.4% | Real state + Morgan fingerprints, 5 epochs |
| LLM-PPO pilot | improvable | 0.580 | 1% | +16.0% | 50 iters, batch=2 — under-trained |
| **LLM-PPO** (Qwen2.5-7B + LoRA) | ORD (real) | **0.808** | **100%** | **+61.6%** | 200 iters, batch=4, best train=0.9007 |
| **SFT warmup** | ORD (real) | — | — | loss 1.66→1.01 | 139 high-yield pairs, 3 epochs |
| **DPO** (SFT → DPO) | ORD (real) | **0.808** | **100%** | **+61.6%** | 80 preference pairs, β=0.1 |
| **GRPO** (G=4) | ORD (real) | — | — | — | `scripts/train_grpo.py` — run to compare |
| **Agent GRPO** (GSM8K, tool use) | GSM8K train | **0.5575** | — | — | 200 iters, G=4, RLVR, best at iter 180 |
| **RLAIF** (AI judge → DPO) | GSM8K | — | — | — | `scripts/train_rlaif.py` — AI preference pairs |
| **STaR** (self-taught reasoner) | GSM8K | — | — | — | `scripts/train_star.py` — iterative SFT |
| **PRM-GRPO** (step rewards, γ=0.9) | GSM8K | — | — | — | `scripts/train_prm_grpo.py` — dense rewards |

### Agent GRPO — GSM8K training history (200 iterations, G=4)

Agent calls a Python executor tool; reward = verifiable ground-truth match (RLVR).

| Iter | Reward | Notes |
|------|--------|-------|
| 1 | 0.2900 | Cold start — mostly wrong tool calls |
| 16 | 0.4100 | First major jump — tool use improving |
| 50 | 0.2800 | Transient dip — exploring harder problems |
| 106 | 0.4125 | Stabilises above 0.40 |
| 180 | **0.5575** | **Peak reward — best checkpoint** |
| 200 | 0.2425 | Late-stage variance (best checkpoint saved at 180) |

> Training reward peak **0.5575** at iter 180 over 200 total iterations.
> GSM8K tool-use GRPO: the policy learns to write `<tool_call>{"name":"python_executor","args":{"code":"..."}}` and read back `<tool_result>` before generating `<final_answer>`.

### Hill Climbing (reject-then-SFT loop)

Iterative self-improvement: generate N rollouts per problem → keep trajectories above reward threshold → SFT → repeat.

**Final results (3 rounds complete):**

| Round | Dataset | Accuracy | Tool Use | Avg Reward | Notes |
|-------|---------|----------|----------|-----------|-------|
| 0 (seed) | 32 | **0.125** | 0.000 | 0.138 | Base model — 4/32 problems solved |
| 1 | 44 | 0.000 | 0.000 | 0.000 | Reward threshold filtered all trajectories |
| 2 | 32 | 0.000 | 0.000 | 0.000 | Same — threshold too strict for 7B model |
| 3 | 32 | 0.000 | 0.000 | 0.000 | Checkpoints saved; SFT ran but didn't recover |

> Checkpoints saved at `models/hill_climb_fast/round_{1,2,3}/best/`.
> The zero eval accuracy reflects the strict rejection threshold (reward ≥ 0.5) discarding all rollouts before SFT — the 7B base model rarely exceeds this threshold on tool-use GSM8K cold. Lowering the threshold or using a curriculum of easier problems is the correct fix.

### LLM-PPO training history (200 iterations, batch=4)

| Iters | Best reward | Avg reward | KL | Status |
|---|---|---|---|---|
| 1–10  | 0.848 | 0.848 | ~0 | Strong start on ORD high-quality data |
| 10–50 | **0.879** | 0.862 | ~0 | Peak region |
| 50–100 | 0.879 | 0.851 | ~0 | Stable plateau |
| 100–200 | **0.9007** | 0.847 | ~0 | New best at iter 131+ |
| Final | — | 0.808 (test) | ~0 | 100/100 above 0.75 |

> With 200 iters and batch=4, LLM-PPO matches DPO (0.808 test) and reaches best training reward 0.9007.
> The 50-iter pilot was simply under-trained — more iterations is the key lever.

### Molecule generalization (held-out test)

Train on 4 molecules, evaluate on 1 unseen — tests whether the policy learned
transferable chemistry or just molecule-specific reward hacking.

| Held-out | Train reward | Test reward | Transfer |
|---|---|---|---|
| Aspirin | 0.825 | 0.897 | ✓ |
| Ibuprofen | 0.836 | 0.852 | ✓ |
| Naproxen | 0.859 | 0.761 | ✓ |
| Paracetamol | 0.829 | 0.878 | ✓ |
| Ketoprofen | 0.847 | 0.808 | ✓ |

All 5 held-out molecules score above 0.75 with zero adaptation — the policy generalizes.

### SFT + DPO pipeline (recommended path)

```
SFT:  139 high-yield ORD trajectories  →  loss 1.663 → 1.008  (3 epochs)
DPO:  80 preference pairs (chosen/rejected by yield rank)
      reward_margin: -0.0006 → 0.0003 (2 epochs)
Test: avg_reward=0.808, 100/100 above 0.75 threshold
```

---

## Training curves

Generated by `python scripts/plot_results.py`:

| Plot | File | Contents |
|---|---|---|
| LLM-PPO curves | `results/training_curves.png` | Reward + loss + KL over 50 iterations |
| MLP-PPO curves | `results/mlp_curves.png` | Epoch reward + policy/value loss |
| Method comparison | `results/comparison.png` | Bar chart: all methods side by side |

---

## Quick start

```bash
pip install torch transformers peft accelerate wandb

# ① SFT warmup on real ORD data (recommended first step)
python scripts/sft_warmup.py \
    --data_file data/trajectories_real_ord.jsonl \
    --yield_threshold 0.88 \
    --num_epochs 3 \
    --output_dir models/sft_policy

# ② DPO from SFT checkpoint
python scripts/train_dpo.py \
    --sft_ckpt models/sft_policy/best \
    --data_file data/trajectories_real_ord.jsonl \
    --num_epochs 3 \
    --output_dir models/dpo_policy

# ③ PPO from SFT checkpoint (alternative to DPO)
CUDA_VISIBLE_DEVICES=0 python scripts/train_ppo_distributed.py \
    --model_name models/sft_policy/best \
    --data_file data/trajectories_real_ord.jsonl \
    --num_iterations 200 --batch_size 4 \
    --output_dir models/llm_policy

# ④ GRPO (G=4 completions per prompt, no value head)
CUDA_VISIBLE_DEVICES=0 python scripts/train_grpo.py \
    --sft_ckpt models/sft_policy/best \
    --data_file data/trajectories_real_ord.jsonl \
    --group_size 4 --num_iterations 200

# MLP-PPO baseline
python scripts/run_pipeline.py

# Benchmark (all methods)
python scripts/benchmark.py --data_file data/trajectories_real_ord.jsonl --eval_only

# Training curves
python scripts/plot_results.py

# W&B logging
python scripts/train_dpo.py --use_wandb --wandb_project rlhf-synthesis
```

---

## Multi-GPU (torchrun)

```bash
# 3x A30, single node
python -m torch.distributed.run --standalone --nproc_per_node=3 \
    scripts/train_ppo_distributed.py \
    --data_file data/trajectories_real_ord.jsonl \
    --use_preference_reward \
    --batch_size 4 --num_iterations 200

# DeepSpeed ZeRO-2
deepspeed --num_gpus=3 scripts/train_ppo_distributed.py \
    --deepspeed configs/deepspeed_zero2.json
```

---

## Repository layout

```
rlhf-synthesis-optimization/
├── config/config.json                   # Unified config
├── configs/
│   ├── deepspeed_zero2.json             # ZeRO-2 config (3×A30)
│   └── ray_cluster.yaml                 # Ray cluster config
├── data/
│   ├── trajectories_improvable.jsonl    # 500 synthetic trajectories
│   ├── trajectories_real_ord.jsonl      # 500 real ORD trajectories ← recommended
│   └── labeled/trajectories_expanded.jsonl
├── scripts/
│   ├── sft_warmup.py                    # ① SFT on high-yield data
│   ├── train_dpo.py                     # ② DPO from SFT checkpoint
│   ├── train_ppo_distributed.py         # ② PPO (torchrun / DeepSpeed)
│   ├── run_pipeline.py                  # MLP-PPO baseline
│   ├── benchmark.py                     # Head-to-head comparison
│   └── plot_results.py                  # Training curve plots
├── src/
│   ├── policy/
│   │   └── llm_synthesis_policy.py      # Qwen2.5-7B + LoRA + value head
│   ├── reward/
│   │   ├── real_reward_model.py         # Rule-based (handles flat + nested)
│   │   ├── learned_reward_model.py      # BERT fine-tuned (MSE pseudo-labels)
│   │   └── preference_reward_model.py   # BERT Bradley-Terry (preference pairs)
│   └── training/
│       ├── real_ppo_trainer.py          # MLP-PPO (real state vectors)
│       ├── llm_ppo_trainer.py           # LLM-PPO (W&B, disable_adapter KL)
│       └── dpo_trainer.py               # DPO (no reward model needed)
├── models/
│   ├── sft_policy/best/                 # SFT checkpoint
│   ├── dpo_policy/best/                 # DPO checkpoint
│   └── llm_policy/best/                 # PPO checkpoint
└── results/
    ├── training_curves.png              # LLM-PPO reward/loss/KL curves
    ├── mlp_curves.png                   # MLP-PPO epoch curves
    ├── comparison.png                   # Method comparison bar chart
    ├── benchmark_results.json
    ├── dpo_results.json
    ├── sft_results.json
    ├── rlhf_llm_distributed.json
    └── rlhf_results.json
```

---

## Training details

| Hyper-parameter | SFT | PPO | DPO |
|---|---|---|---|
| Base model | Qwen2.5-7B-Instruct | Qwen2.5-7B-Instruct | Qwen2.5-7B-Instruct |
| LoRA rank | 8 | 8 | 8 |
| Trainable params | 5.05M (0.066%) | 5.05M (0.066%) | 5.05M (0.066%) |
| Learning rate | 2e-5 | 1e-5 | 1e-5 |
| Epochs / iters | 3 epochs | 200 iters (batch=4) | 3 epochs |
| Batch size | 2 | 2–4 | 2–4 |
| Reference policy | — | `disable_adapter()` | `disable_adapter()` |
| Data | ORD (real) | ORD / improvable | ORD (real) |
| Precision | fp16 + GradScaler | fp16 + GradScaler | fp16 + GradScaler |

---

## Key engineering decisions

1. **`disable_adapter()` for reference policy** — instead of loading a second 7B model copy (doubles VRAM), calling `policy.model.disable_adapter()` gives the frozen base model for KL/DPO reference computation at zero extra cost.

2. **Real state vectors in MLP-PPO** — original code used `torch.randn()` as input (random noise). Fixed to encode 8 chemistry features (temp, time, catalyst, solvent, yield, selectivity, safety, steps) → 128-dim vector with zero-padding for future molecular fingerprint extension.

3. **Real ORD data** — `data/trajectories_real_ord.jsonl` contains 500 literature-based pharmaceutical synthesis trajectories (Aspirin, Ibuprofen, Naproxen, Paracetamol, Ketoprofen) with yield 0.73–0.93. All reward model fixes required handling nested `outcomes` dicts.

4. **SFT → DPO** is the most stable path. PPO with batch=2 and 50 iterations shows high reward variance (±0.13); DPO with 80 preference pairs converges in 2 epochs.

5. **BT preference reward model** uses actual (chosen, rejected) pairs sorted by yield, not hand-crafted weights. The 40/60 blend (rule + neural) prevents the model from discarding chemistry constraints entirely.
