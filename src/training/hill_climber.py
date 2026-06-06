"""
Hill Climber — closed-loop dataset refinement via Agent GRPO.

Algorithm per round:
  1. Generate G rollouts per problem with the current policy
  2. Score each rollout with verifiable_reward
  3. Filter: keep rollouts with reward >= threshold (rejection sampling)
  4. Append filtered (problem, trajectory) pairs to the training dataset
  5. Run agent_grpo_trainer on the augmented dataset for `grpo_iters` steps
  6. Evaluate on held-out test set; checkpoint if accuracy improves

After R rounds the dataset contains only high-reward trajectories — the
policy iteratively distills its own best solutions back into training data.

This directly implements "automatically hill climb datasets" from the
Scale Enterprise GenAI JD and matches the rejection-sampling + RL loop
described in DeepSeek-R1 and related work.
"""
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

from .agent_grpo_trainer import (
    AgentGRPOTrainer,
    build_agent_prompt,
    run_agent_rollout,
)
from .grpo_trainer import GRPOConfig
from ..reward.verifiable_reward import compute_reward, extract_final_answer

logger = logging.getLogger(__name__)


@dataclass
class HillClimbConfig:
    # Dataset
    data_file: str = "data/gsm8k_train_tool_dataset.jsonl"
    output_dir: str = "models/hill_climb"

    # Loop
    num_rounds: int = 5
    rollouts_per_problem: int = 4   # G — trajectories generated per problem per round
    reward_threshold: float = 0.5   # min reward to include in augmented dataset
    top_k_per_problem: int = 2      # keep at most this many per problem

    # GRPOConfig fields used per round
    grpo_iters_per_round: int = 50
    grpo_group_size: int = 4
    grpo_lr: float = 5e-6
    grpo_clip: float = 0.2          # maps to GRPOConfig.clip_ratio
    grpo_kl_coeff: float = 0.04     # maps to GRPOConfig.beta
    grpo_batch_size: int = 2
    max_new_tokens: int = 200

    # Eval
    eval_problems: int = 20

    # Misc
    seed: int = 42


class HillClimber:
    def __init__(self, policy, config: HillClimbConfig):
        self.policy  = policy
        self.cfg     = config
        self.out_dir = Path(config.output_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

        self._load_data()
        self.results: List[Dict] = []
        self.best_accuracy = 0.0

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------

    def _load_data(self):
        problems = []
        with open(self.cfg.data_file) as f:
            for line in f:
                problems.append(json.loads(line))

        n_eval = min(self.cfg.eval_problems, len(problems) // 5)
        self.eval_set  = problems[-n_eval:]
        self.train_set = problems[:-n_eval]
        logger.info("HillClimber | %d train | %d eval", len(self.train_set), len(self.eval_set))

    def _write_augmented_dataset(self, extra_rows: List[Dict]) -> str:
        path = self.out_dir / "augmented_train.jsonl"
        with open(path, "w") as f:
            for row in self.train_set:
                f.write(json.dumps(row) + "\n")
            for row in extra_rows:
                f.write(json.dumps(row) + "\n")
        return str(path)

    # ------------------------------------------------------------------
    # Rollout & filter
    # ------------------------------------------------------------------

    def _collect_rollouts(self, problems: List[Dict]) -> List[Dict]:
        """Generate rollouts; keep those with reward >= threshold (rejection sampling)."""
        kept: List[Dict] = []
        total = rewarded = 0

        for item in problems:
            problem = item["problem"]
            answer  = item["answer"]
            prompt  = build_agent_prompt(problem)

            candidates = []
            for _ in range(self.cfg.rollouts_per_problem):
                _, _, output = run_agent_rollout(
                    self.policy, prompt,
                    max_new_tokens=self.cfg.max_new_tokens,
                    temperature=0.9,
                )
                reward = compute_reward(output, answer)
                total += 1
                if reward >= self.cfg.reward_threshold:
                    rewarded += 1
                    candidates.append({"trajectory": output, "reward": reward})

            # Keep top-k by reward
            candidates.sort(key=lambda x: x["reward"], reverse=True)
            for c in candidates[: self.cfg.top_k_per_problem]:
                kept.append({
                    "problem":         problem,
                    "answer":          answer,
                    "trajectory":      c["trajectory"],
                    "reward":          c["reward"],
                    "available_tools": item.get("available_tools", ["python_executor"]),
                })

        logger.info(
            "Rollout collection: %d/%d passed threshold=%.2f → %d kept",
            rewarded, total, self.cfg.reward_threshold, len(kept),
        )
        return kept

    # ------------------------------------------------------------------
    # Eval
    # ------------------------------------------------------------------

    def _evaluate(self) -> Dict:
        from ..tools.tool_parser import parse_output
        correct = tool_uses = 0
        rewards = []

        for item in self.eval_set:
            prompt = build_agent_prompt(item["problem"])
            _, _, output = run_agent_rollout(
                self.policy, prompt,
                max_new_tokens=self.cfg.max_new_tokens,
                temperature=0.0,
            )
            rewards.append(compute_reward(output, item["answer"]))
            pred = extract_final_answer(output)
            if pred is not None and pred.strip() == item["answer"].strip():
                correct += 1
            if parse_output(output).tool_call is not None:
                tool_uses += 1

        n = len(self.eval_set)
        return {
            "accuracy":   correct / n,
            "tool_use":   tool_uses / n,
            "avg_reward": sum(rewards) / n,
        }

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self) -> List[Dict]:
        logger.info("=== Hill Climb | %d rounds | G=%d | threshold=%.2f ===",
                    self.cfg.num_rounds, self.cfg.rollouts_per_problem, self.cfg.reward_threshold)

        # Baseline
        baseline = self._evaluate()
        logger.info("Baseline: acc=%.3f | tool_use=%.3f | reward=%.3f",
                    baseline["accuracy"], baseline["tool_use"], baseline["avg_reward"])
        self.best_accuracy = baseline["accuracy"]
        self.results.append({"round": 0, **baseline, "dataset_size": len(self.train_set)})

        for rnd in range(1, self.cfg.num_rounds + 1):
            t0 = time.time()
            logger.info("--- Round %d/%d ---", rnd, self.cfg.num_rounds)

            # Step 1: rejection sampling — collect high-reward rollouts
            filtered = self._collect_rollouts(self.train_set)
            logger.info("Round %d: %d trajectories added to dataset", rnd, len(filtered))

            # Step 2: augment dataset and run GRPO
            aug_path = self._write_augmented_dataset(filtered)
            self._run_grpo_round(aug_path, rnd)

            # Step 3: eval
            metrics = self._evaluate()
            elapsed = time.time() - t0
            logger.info(
                "Round %d (%.0fs): acc=%.3f | tool_use=%.3f | reward=%.3f | dataset=%d",
                rnd, elapsed,
                metrics["accuracy"], metrics["tool_use"], metrics["avg_reward"],
                len(self.train_set) + len(filtered),
            )

            if metrics["accuracy"] > self.best_accuracy:
                self.best_accuracy = metrics["accuracy"]
                ckpt = self.out_dir / f"best_round_{rnd}"
                self.policy.save_pretrained(str(ckpt))
                logger.info("New best acc=%.3f → saved to %s", self.best_accuracy, ckpt)

            self.results.append({
                "round":        rnd,
                "dataset_size": len(self.train_set) + len(filtered),
                **metrics,
            })

        self._save_results()
        return self.results

    def _run_grpo_round(self, data_file: str, rnd: int):
        import random
        import torch
        torch.cuda.empty_cache()
        rows = []
        with open(data_file) as f:
            for line in f:
                rows.append(json.loads(line))
        random.shuffle(rows)

        grpo_cfg = GRPOConfig(
            num_iterations  = self.cfg.grpo_iters_per_round,
            group_size      = self.cfg.grpo_group_size,
            learning_rate   = self.cfg.grpo_lr,
            clip_ratio      = self.cfg.grpo_clip,
            beta            = self.cfg.grpo_kl_coeff,
            batch_size      = self.cfg.grpo_batch_size,
            max_new_tokens  = self.cfg.max_new_tokens,
            output_dir      = str(self.out_dir / f"round_{rnd}"),
            log_interval    = 10,
        )
        trainer = AgentGRPOTrainer(self.policy, grpo_cfg)
        trainer.train(rows)

    def _save_results(self):
        path = self.out_dir / "hill_climb_results.json"
        with open(path, "w") as f:
            json.dump(self.results, f, indent=2)
        logger.info("Results saved to %s", path)

        if self.results:
            baseline = next((r for r in self.results if r["round"] == 0), None)
            final    = self.results[-1]
            if baseline:
                delta = final["accuracy"] - baseline["accuracy"]
                logger.info(
                    "=== Summary: acc %.3f → %.3f (%+.3f) over %d rounds | dataset %d → %d ===",
                    baseline["accuracy"], final["accuracy"], delta, self.cfg.num_rounds,
                    baseline["dataset_size"], final["dataset_size"],
                )
