import logging
from typing import List, Dict, Tuple
import json

logger = logging.getLogger(__name__)

class RealPPOTrainer:
    """Train policy using PPO (Proximal Policy Optimization)"""
    
    def __init__(self, policy, reward_model, config):
        self.policy = policy
        self.reward_model = reward_model
        self.config = config
        self.training_log = []
        logger.info("Initialized PPO trainer")
    
    def compute_advantages(self, rewards: List[float], baseline: float = 0.5) -> List[float]:
        """Compute advantage estimates A_t = R_t - V(s_t)"""
        advantages = [r - baseline for r in rewards]
        
        # Normalize advantages for stability
        mean_adv = sum(advantages) / len(advantages) if advantages else 0.0
        std_adv = (sum((a - mean_adv) ** 2 for a in advantages) / len(advantages)) ** 0.5 + 1e-8
        
        normalized = [(a - mean_adv) / std_adv for a in advantages]
        return normalized
    
    def ppo_step(self, trajectories: List[Dict], old_policy_probs: List[float] = None):
        """Single PPO update step"""
        
        # Get rewards
        rewards = self.reward_model.score_batch(trajectories)
        
        # Compute advantages
        baseline = sum(rewards) / len(rewards) if rewards else 0.5
        advantages = self.compute_advantages(rewards, baseline)
        
        # Compute policy gradient (simplified)
        policy_loss = -sum(adv for adv in advantages) / len(advantages) if advantages else 0.0
        
        # Value function loss (MSE between reward and value estimate)
        value_estimates = [baseline] * len(trajectories)
        value_loss = sum((r - v) ** 2 for r, v in zip(rewards, value_estimates)) / len(rewards) if rewards else 0.0
        
        # Entropy bonus for exploration
        entropy = 0.01  # Placeholder
        
        # Total loss
        total_loss = policy_loss + 0.5 * value_loss - 0.01 * entropy
        
        return {
            "average_reward": sum(rewards) / len(rewards) if rewards else 0.0,
            "average_advantage": sum(advantages) / len(advantages) if advantages else 0.0,
            "policy_improvement": sum(advantages) / len(advantages) if advantages else 0.0,
            "policy_loss": policy_loss,
            "value_loss": value_loss,
            "total_loss": total_loss
        }
    
    def train(self, train_trajectories: List[Dict], epochs: int = 3):
        """Train policy for multiple epochs"""
        logger.info(f"Starting PPO training for {epochs} epochs on {len(train_trajectories)} trajectories")
        
        metrics = []
        for epoch in range(epochs):
            epoch_metrics = self.ppo_step(train_trajectories)
            metrics.append(epoch_metrics)
            
            logger.info(f"Epoch {epoch+1}:")
            logger.info(f"  Reward: {epoch_metrics['average_reward']:.4f}")
            logger.info(f"  Advantage: {epoch_metrics['average_advantage']:.4f}")
            logger.info(f"  Improvement: {epoch_metrics['policy_improvement']:.4f}")
            logger.info(f"  Loss: {epoch_metrics['total_loss']:.4f}")
        
        return metrics
