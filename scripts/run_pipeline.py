#!/usr/bin/env python3
import sys
import logging
import json
from pathlib import Path
from datetime import datetime
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.paths import initialize_directories, get_results_dir
from src.config import get_config
from src.reward.real_reward_model import RealRewardModel
from src.training.real_ppo_trainer import RealPPOPolicy, RealPPOTrainer
from src.evaluation.real_rlhf_evaluator import evaluate_rlhf_policy, save_rlhf_results

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_trajectories(file_path):
    trajectories = []
    with open(file_path) as f:
        for line in f:
            trajectories.append(json.loads(line))
    return trajectories

def main():
    initialize_directories()
    config = get_config()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    logger.info("=" * 80)
    logger.info("Project 3: RLHF with PPO (EXPANDED DATASET)")
    logger.info("=" * 80)
    
    logger.info("[1/5] Loading 500 synthesis trajectories...")
    trajectories = load_trajectories(Path("data/labeled/trajectories_expanded.jsonl"))
    split_idx = int(len(trajectories) * 0.8)
    train_traj = trajectories[:split_idx]
    test_traj = trajectories[split_idx:]
    logger.info(f"✓ Loaded {len(train_traj)} train, {len(test_traj)} test")
    
    logger.info("[2/5] Initializing reward model...")
    reward_model = RealRewardModel()
    
    logger.info("[3/5] Initializing PPO policy...")
    policy = RealPPOPolicy(state_dim=128, action_dim=32, hidden_dim=256).to(device)
    
    logger.info("[4/5] Setting up trainer...")
    trainer = RealPPOTrainer(policy, reward_model, learning_rate=1e-4, device=device)
    
    logger.info("[5/5] Training PPO on expanded dataset...")
    training_metrics = trainer.train(train_traj, epochs=5)
    
    eval_results = evaluate_rlhf_policy(test_traj, reward_model)
    
    output_file = get_results_dir() / "rlhf_results.json"
    final_results = {
        "timestamp": datetime.now().isoformat(),
        "dataset": {
            "total_trajectories": len(trajectories),
            "train_size": len(train_traj),
            "test_size": len(test_traj),
            "expansion": "500 trajectories (5x original)"
        },
        "training_metrics": training_metrics,
        "evaluation": eval_results
    }
    
    save_rlhf_results(final_results, output_file)
    
    logger.info("=" * 80)
    logger.info("✓ PPO TRAINING COMPLETE (EXPANDED)")
    logger.info(f"  Trajectories: {len(trajectories)}")
    logger.info(f"  Final Reward: {eval_results['average_reward']:.4f}")
    logger.info(f"  Improvement: {eval_results['improvement']:+.1f}%")
    logger.info("=" * 80)

if __name__ == "__main__":
    main()
