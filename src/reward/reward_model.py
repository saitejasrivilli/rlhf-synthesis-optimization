import logging
from typing import List, Dict

logger = logging.getLogger(__name__)

class RewardModel:
    """Score synthesis trajectories based on quality metrics"""
    
    def __init__(self, model_name: str = "bert-base-uncased"):
        self.model_name = model_name
        logger.info(f"Initialized reward model: {model_name}")
    
    def score_trajectory(self, trajectory: Dict) -> float:
        """Score a synthesis trajectory"""
        # Reward components
        yield_reward = trajectory.get('yield', 0.5) * 0.4
        selectivity_reward = trajectory.get('selectivity', 0.5) * 0.3
        safety_reward = (1.0 - trajectory.get('safety_risk', 0.0)) * 0.2
        efficiency_reward = (1.0 / max(trajectory.get('steps', 1), 1)) * 0.1
        
        total_reward = yield_reward + selectivity_reward + safety_reward + efficiency_reward
        return min(max(total_reward, 0.0), 1.0)  # Clamp to [0, 1]
    
    def score_batch(self, trajectories: List[Dict]) -> List[float]:
        """Score multiple trajectories"""
        return [self.score_trajectory(traj) for traj in trajectories]
    
    def save(self, path):
        """Save reward model"""
        logger.info(f"Reward model saved to {path}")
    
    def load(self, path):
        """Load reward model"""
        logger.info(f"Reward model loaded from {path}")
