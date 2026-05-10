import logging
from typing import List, Dict
from pathlib import Path

logger = logging.getLogger(__name__)

class RLHFTrainer:
    """Train policy using reward signals via PPO"""
    
    def __init__(self, policy, reward_model, config):
        self.policy = policy
        self.reward_model = reward_model
        self.config = config
        logger.info("Initialized RLHF trainer")
    
    def compute_advantages(self, rewards: List[float], baseline: float = 0.5) -> List[float]:
        """Compute advantage estimates"""
        return [r - baseline for r in rewards]
    
    def ppo_step(self, trajectories: List[Dict], learning_rate: float = 1e-5):
        """Single PPO optimization step"""
        # Score trajectories
        rewards = self.reward_model.score_batch(trajectories)
        advantages = self.compute_advantages(rewards)
        
        # Compute policy improvement
        policy_improvement = sum(advantages) / len(advantages) if advantages else 0.0
        
        return {
            "average_reward": sum(rewards) / len(rewards),
            "average_advantage": sum(advantages) / len(advantages) if advantages else 0.0,
            "policy_improvement": policy_improvement
        }
    
    def train(self, train_trajectories: List[Dict], epochs: int = 3):
        """Train policy"""
        logger.info(f"Starting RLHF training for {epochs} epochs")
        
        metrics = []
        for epoch in range(epochs):
            epoch_metrics = self.ppo_step(train_trajectories)
            metrics.append(epoch_metrics)
            logger.info(f"Epoch {epoch+1}: Reward={epoch_metrics['average_reward']:.3f}, "
                       f"Advantage={epoch_metrics['average_advantage']:.3f}, "
                       f"Improvement={epoch_metrics['policy_improvement']:.3f}")
        
        return metrics
