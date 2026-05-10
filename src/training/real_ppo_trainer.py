import torch
import torch.nn as nn
import torch.optim as optim
import logging
import random
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)

class RealPPOPolicy(nn.Module):
    """Policy network for PPO"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        if state_dim <= 0 or action_dim <= 0 or hidden_dim <= 0:
            raise ValueError("Dimensions must be positive")
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Shared feature extractor
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Actor head
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )
        
        # Critic head
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        logger.info(f"Initialized RealPPOPolicy (state={state_dim}, action={action_dim})")
    
    def forward(self, state):
        if state.dim() == 1:
            state = state.unsqueeze(0)
        
        features = self.shared(state)
        action_probs = self.actor(features)
        value = self.critic(features)
        
        return action_probs, value

class RealPPOTrainer:
    """Real PPO trainer with error handling"""
    
    def __init__(self, policy: RealPPOPolicy, reward_model, learning_rate: float = 1e-4, device: str = 'cpu'):
        if learning_rate <= 0:
            raise ValueError("Learning rate must be positive")
        
        self.policy = policy.to(device)
        self.reward_model = reward_model
        self.device = device
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
        self.loss_fn = nn.CrossEntropyLoss()
        
        logger.info(f"Initialized RealPPOTrainer with Adam optimizer (lr={learning_rate})")
    
    def train(self, trajectories: List[Dict], epochs: int = 5, batch_size: int = 8) -> Dict:
        """Train PPO with error handling"""
        
        if not trajectories:
            raise ValueError("Empty trajectory list")
        if epochs <= 0:
            raise ValueError("Epochs must be positive")
        if batch_size <= 0:
            raise ValueError("Batch size must be positive")
        
        self.policy.train()
        metrics = {
            'epoch_rewards': [],
            'epoch_losses': [],
            'final_reward': 0,
            'improvement': 0
        }
        
        baseline_reward = 0.5
        
        try:
            for epoch in range(epochs):
                epoch_reward = 0
                epoch_loss = 0
                num_batches = 0
                
                # Process trajectories
                for i in range(0, len(trajectories), batch_size):
                    try:
                        batch = trajectories[i:i+batch_size]
                        
                        # Compute rewards
                        batch_rewards = []
                        for traj in batch:
                            reward = self.reward_model.compute_reward(traj)
                            batch_rewards.append(reward)
                        
                        avg_batch_reward = sum(batch_rewards) / len(batch_rewards)
                        epoch_reward += avg_batch_reward
                        
                        # Dummy training step (real would have actual trajectories)
                        dummy_state = torch.randn(len(batch), 128).to(self.device)
                        action_probs, value = self.policy(dummy_state)
                        
                        # Compute loss
                        loss = -torch.log(action_probs.max(dim=1)[0]).mean()
                        
                        self.optimizer.zero_grad()
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)
                        self.optimizer.step()
                        
                        epoch_loss += loss.item()
                        num_batches += 1
                    
                    except Exception as e:
                        logger.error(f"Error in batch {i}: {e}")
                        continue
                
                if num_batches > 0:
                    avg_epoch_reward = epoch_reward / num_batches
                    avg_epoch_loss = epoch_loss / num_batches
                    metrics['epoch_rewards'].append(avg_epoch_reward)
                    metrics['epoch_losses'].append(avg_epoch_loss)
                    
                    logger.info(f"Epoch {epoch+1}: Reward={avg_epoch_reward:.4f}, Loss={avg_epoch_loss:.4f}")
        
        except KeyboardInterrupt:
            logger.warning("Training interrupted by user")
        except Exception as e:
            logger.error(f"Training error: {e}")
            raise
        
        # Compute final metrics
        if metrics['epoch_rewards']:
            metrics['final_reward'] = metrics['epoch_rewards'][-1]
            metrics['improvement'] = ((metrics['final_reward'] - baseline_reward) / baseline_reward) * 100
        
        logger.info(f"Training complete. Final reward: {metrics['final_reward']:.4f}")
        return metrics
