import logging
import json
from pathlib import Path
from typing import List, Dict

logger = logging.getLogger(__name__)

def evaluate_rlhf_policy(test_trajectories: List[Dict], reward_model) -> Dict:
    """Evaluate RLHF-trained policy"""
    
    rewards = reward_model.score_batch(test_trajectories)
    
    results = {
        "num_trajectories": len(test_trajectories),
        "average_reward": sum(rewards) / len(rewards) if rewards else 0.0,
        "max_reward": max(rewards) if rewards else 0.0,
        "min_reward": min(rewards) if rewards else 0.0,
        "reward_std": (sum((r - (sum(rewards)/len(rewards)))**2 for r in rewards) / len(rewards)) ** 0.5 if rewards else 0.0,
        "successful_trajectories": sum(1 for r in rewards if r > 0.6),
        "high_yield_trajectories": sum(1 for r in rewards if r > 0.75),
        "baseline_reward": 0.5,
        "improvement": (sum(rewards) / len(rewards) - 0.5) / 0.5 * 100 if rewards else 0.0
    }
    
    logger.info(f"Evaluation Results:")
    logger.info(f"  Avg Reward: {results['average_reward']:.4f}")
    logger.info(f"  Improvement: {results['improvement']:+.1f}%")
    logger.info(f"  High Quality: {results['high_yield_trajectories']}/{len(test_trajectories)}")
    
    return results

def save_rlhf_results(results: Dict, output_file: Path):
    """Save results"""
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"✓ Results saved to {output_file}")
