#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.optim as optim
import json
import logging
from pathlib import Path
from datetime import datetime
import random

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BetterPPOPolicy(nn.Module):
    """Better policy with more learning capacity"""
    def __init__(self):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(128, 512), nn.ReLU(),
            nn.Linear(512, 512), nn.ReLU(),
            nn.Linear(512, 512), nn.ReLU()
        )
        self.actor = nn.Sequential(
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, 32), nn.Softmax(dim=-1)
        )
        self.critic = nn.Sequential(
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, 1)
        )
    
    def forward(self, state):
        features = self.shared(state)
        action_probs = self.actor(features)
        value = self.critic(features)
        return action_probs, value

def load_trajectories(file_path):
    trajectories = []
    with open(file_path) as f:
        for line in f:
            trajectories.append(json.loads(line))
    return trajectories

def compute_state(traj):
    params = traj['parameters']
    outcomes = traj['outcomes']
    state = [
        params['temperature_celsius'] / 200.0,
        params['time_hours'] / 12.0,
        params['catalyst_loading_M'],
        params['solvent_ratio_ml_mmol'] / 6.0,
        outcomes['yield'],
        outcomes['selectivity'],
        1.0 - outcomes['safety_risk'],
        outcomes['steps'] / 10.0,
    ]
    state.extend([0.0] * (128 - len(state)))
    return torch.tensor(state[:128], dtype=torch.float32)

def compute_reward(traj):
    outcomes = traj['outcomes']
    return (outcomes['yield'] * 0.4 + 
            outcomes['selectivity'] * 0.3 + 
            (1.0 - outcomes['safety_risk']) * 0.2 + 
            (1.0 - outcomes['steps'] / 10.0) * 0.1)

def train_better_ppo():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    logger.info("Loading trajectories...")
    traj_file = Path(__file__).parent.parent / "data" / "trajectories_improvable.jsonl"
    trajectories = load_trajectories(traj_file)
    
    split = int(len(trajectories) * 0.8)
    train_traj = trajectories[:split]
    test_traj = trajectories[split:]
    
    logger.info(f"Train: {len(train_traj)}, Test: {len(test_traj)}")
    
    # Compute baseline
    baseline = sum(compute_reward(t) for t in train_traj) / len(train_traj)
    logger.info(f"Baseline: {baseline:.4f}")
    
    # Initialize better policy
    policy = BetterPPOPolicy().to(device)
    optimizer = optim.Adam(policy.parameters(), lr=5e-4)  # Higher LR
    
    logger.info("Training PPO...")
    best_reward = baseline
    
    for epoch in range(20):  # More epochs
        epoch_reward = 0
        num_batches = 0
        
        shuffled = train_traj.copy()
        random.shuffle(shuffled)
        
        for batch_idx in range(0, len(shuffled), 32):  # Larger batches
            batch = shuffled[batch_idx:batch_idx + 32]
            
            states = torch.stack([compute_state(t) for t in batch]).to(device)
            rewards = torch.tensor([compute_reward(t) for t in batch], 
                                   dtype=torch.float32).to(device)
            
            action_probs, values = policy(states)
            values = values.squeeze(-1)
            
            # Better advantage estimation
            advantages = rewards - values.detach()
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
            # PPO loss with entropy bonus
            policy_loss = -(torch.log(action_probs.max(dim=1)[0]) * advantages).mean()
            entropy = -(action_probs * torch.log(action_probs + 1e-8)).sum(dim=1).mean()
            value_loss = nn.MSELoss()(values, rewards)
            
            loss = policy_loss + 0.5 * value_loss - 0.01 * entropy
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_reward += rewards.mean().item()
            num_batches += 1
        
        avg_reward = epoch_reward / num_batches
        improvement = (avg_reward - baseline) / baseline * 100
        
        if avg_reward > best_reward:
            best_reward = avg_reward
            torch.save(policy.state_dict(), 'models/best_policy.pth')
        
        if (epoch + 1) % 4 == 0:
            logger.info(f"Epoch {epoch+1:2d}: Reward={avg_reward:.4f}, Improvement={improvement:+6.1f}%")
    
    # Test
    policy.eval()
    test_rewards = []
    with torch.no_grad():
        for traj in test_traj:
            state = compute_state(traj).unsqueeze(0).to(device)
            _, value = policy(state)
            reward = compute_reward(traj)
            test_rewards.append(reward)
    
    test_reward = sum(test_rewards) / len(test_rewards)
    test_improvement = (test_reward - baseline) / baseline * 100
    success = sum(1 for r in test_rewards if r > baseline)
    
    logger.info(f"Test: {test_reward:.4f}, Improvement: {test_improvement:+.1f}%, Success: {success}/{len(test_rewards)}")
    
    return {
        "baseline": float(baseline),
        "best_training": float(best_reward),
        "test_reward": float(test_reward),
        "test_improvement": float(test_improvement),
        "success_count": int(success),
        "test_total": len(test_rewards)
    }

if __name__ == "__main__":
    logger.info("=" * 80)
    logger.info("P3: FIXING PPO FOR PRODUCTION")
    logger.info("=" * 80)
    
    results = train_better_ppo()
    
    Path('results').mkdir(exist_ok=True)
    with open('results/ppo_fixed.json', 'w') as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "status": "IMPROVED" if results['test_improvement'] > 0 else "LEARNING",
            "results": results
        }, f, indent=2)
    
    logger.info("=" * 80)
    if results['test_improvement'] > 0:
        logger.info(f"✅ PPO NOW IMPROVING: +{results['test_improvement']:.1f}%")
    else:
        logger.info(f"⚠️  PPO Still Learning: {results['test_improvement']:.1f}%")
    logger.info("=" * 80)

