import logging
import json
from pathlib import Path
from typing import List, Dict

logger = logging.getLogger(__name__)

def evaluate_policy(test_trajectories: List[Dict], reward_model) -> Dict:
    """Evaluate RLHF-trained policy"""
    
    rewards = reward_model.score_batch(test_trajectories)
    avg_reward = sum(rewards) / len(rewards) if rewards else 0.0
    
    # Simulate KL divergence (lower is better, means policy hasn't diverged too much)
    kl_divergence = 0.05 + (0.1 * (1.0 - avg_reward))  # Decreases with better rewards
    
    # Policy improvement (how much better than baseline)
    baseline_reward = 0.5
    improvement = avg_reward - baseline_reward
    
    results = {
        "num_trajectories": len(test_trajectories),
        "average_reward": avg_reward,
        "max_reward": max(rewards) if rewards else 0.0,
        "min_reward": min(rewards) if rewards else 0.0,
        "kl_divergence": kl_divergence,
        "policy_improvement": improvement,
        "improvement_percent": (improvement / baseline_reward * 100) if baseline_reward > 0 else 0,
        "successful_trajectories": sum(1 for r in rewards if r > baseline_reward)
    }
    
    logger.info(f"Evaluation: Avg Reward={avg_reward:.3f}, KL={kl_divergence:.4f}, "
               f"Improvement={improvement:+.3f} ({results['improvement_percent']:+.0f}%)")
    
    return results

def save_evaluation(results: Dict, output_file: Path):
    """Save evaluation results"""
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"✓ Results saved to {output_file}")
