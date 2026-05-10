#!/usr/bin/env python3
"""Project 3: RLHF Fine-tuning - REAL PPO"""

import sys
import logging
import json
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.paths import initialize_directories, get_results_dir
from src.config import get_config
from src.reward.real_reward_model import RealRewardModel
from src.policy.synthesis_policy import SynthesisPolicy
from src.training.real_ppo import RealPPOTrainer
from src.evaluation.real_rlhf_evaluator import evaluate_rlhf_policy, save_rlhf_results

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_trajectories(file_path: Path):
    """Load synthesis trajectories"""
    trajectories = []
    with open(file_path) as f:
        for line in f:
            trajectories.append(json.loads(line))
    return trajectories

def main():
    initialize_directories()
    config = get_config()
    
    logger.info("=" * 80)
    logger.info("Project 3: RLHF Fine-tuning with Real PPO")
    logger.info("=" * 80)
    
    # 1. Load trajectories
    logger.info("[1/6] Loading synthesis trajectories...")
    trajectories = load_trajectories(Path("data/labeled/trajectories.jsonl"))
    split_idx = int(len(trajectories) * config.data.train_val_split)
    train_traj = trajectories[:split_idx]
    test_traj = trajectories[split_idx:]
    logger.info(f"✓ Loaded {len(train_traj)} train, {len(test_traj)} test trajectories")
    
    # 2. Initialize reward model
    logger.info("[2/6] Initializing reward model...")
    reward_model = RealRewardModel()
    
    # 3. Initialize policy
    logger.info("[3/6] Initializing synthesis policy...")
    policy = SynthesisPolicy()
    
    # 4. Create PPO trainer
    logger.info("[4/6] Setting up PPO trainer...")
    trainer = RealPPOTrainer(policy, reward_model, config.rlhf_training)
    
    # 5. Train policy
    logger.info("[5/6] Training policy with PPO...")
    training_metrics = trainer.train(train_traj, epochs=config.rlhf_training.ppo_epochs)
    
    # 6. Evaluate
    logger.info("[6/6] Evaluating policy...")
    eval_results = evaluate_rlhf_policy(test_traj, reward_model)
    
    # Save results
    output_file = get_results_dir() / "rlhf_results.json"
    final_results = {
        "timestamp": datetime.now().isoformat(),
        "training_metrics": training_metrics,
        "evaluation": eval_results,
        "policy_summary": {
            "base_model": config.policy.base_model,
            "training_method": "PPO",
            "trajectories_trained": len(train_traj),
            "trajectories_tested": len(test_traj)
        }
    }
    
    save_rlhf_results(final_results, output_file)
    
    logger.info("=" * 80)
    logger.info("✓ RLHF TRAINING COMPLETE")
    logger.info(f"  Avg Reward: {eval_results['average_reward']:.4f}")
    logger.info(f"  Improvement: {eval_results['improvement']:+.1f}%")
    logger.info(f"  High Quality: {eval_results['high_yield_trajectories']}/{len(test_traj)}")
    logger.info("=" * 80)

if __name__ == "__main__":
    main()
