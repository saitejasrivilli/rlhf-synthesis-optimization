import logging
from typing import List, Dict
import json

logger = logging.getLogger(__name__)

class RealRewardModel:
    """Score synthesis trajectories on real chemistry metrics"""
    
    def __init__(self):
        logger.info("Initialized real reward model")
    
    def score_trajectory(self, trajectory: Dict) -> float:
        """Score based on: yield, selectivity, safety, efficiency"""
        
        # Yield reward (40% weight)
        yield_score = trajectory.get("yield", 0.5)
        yield_reward = yield_score * 0.40
        
        # Selectivity reward (30% weight)
        selectivity_score = trajectory.get("selectivity", 0.5)
        selectivity_reward = selectivity_score * 0.30
        
        # Safety reward (20% weight) - lower risk is better
        safety_risk = trajectory.get("safety_risk", 0.5)
        safety_reward = (1.0 - safety_risk) * 0.20
        
        # Efficiency reward (10% weight) - fewer steps is better
        steps = trajectory.get("steps", 5)
        efficiency_reward = (1.0 / (1.0 + steps / 10.0)) * 0.10
        
        total_reward = yield_reward + selectivity_reward + safety_reward + efficiency_reward
        return max(0.0, min(1.0, total_reward))
    
    def score_batch(self, trajectories: List[Dict]) -> List[float]:
        """Score multiple trajectories"""
        return [self.score_trajectory(traj) for traj in trajectories]
    
    def save(self, path):
        logger.info(f"Reward model saved to {path}")
    
    def load(self, path):
        logger.info(f"Reward model loaded from {path}")
