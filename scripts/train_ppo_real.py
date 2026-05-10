#!/usr/bin/env python3
import json, torch, torch.nn as nn, torch.optim as optim, random, logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PPOPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.shared = nn.Sequential(nn.Linear(128, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU())
        self.actor = nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 32), nn.Softmax(dim=-1))
        self.critic = nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 1))
    
    def forward(self, state):
        f = self.shared(state)
        return self.actor(f), self.critic(f)

def load_traj(file_path):
    with open(file_path) as f:
        return [json.loads(line) for line in f]

def compute_state(t):
    p, o = t['parameters'], t['outcomes']
    s = [p['temperature_celsius']/200, p['time_hours']/12, p['catalyst_loading_M'], 
         p['solvent_ratio_ml_mmol']/6, o['yield'], o['selectivity'], 1-o['safety_risk'], o['steps']/10]
    s.extend([0]*(128-len(s)))
    return torch.tensor(s[:128], dtype=torch.float32)

def compute_reward(t):
    o = t['outcomes']
    return max(0, min(1, o['yield']*0.4 + o['selectivity']*0.3 + (1-o['safety_risk'])*0.2 + (1-o['steps']/10)*0.1))

device = 'cpu'
logger.info("Loading real yield data...")
trajectories = load_traj("data/trajectories_real_yields.jsonl")
split = int(len(trajectories) * 0.8)
train_traj, test_traj = trajectories[:split], trajectories[split:]

baseline = sum(compute_reward(t) for t in train_traj) / len(train_traj)
logger.info(f"Baseline: {baseline:.4f}")

policy = PPOPolicy().to(device)
optimizer = optim.Adam(policy.parameters(), lr=1e-3)

logger.info("Training PPO on real yields...")
best_train = baseline

for epoch in range(15):
    epoch_reward = 0
    shuffled = random.sample(train_traj, len(train_traj))
    
    for batch_idx in range(0, len(shuffled), 16):
        batch = shuffled[batch_idx:batch_idx+16]
        states = torch.stack([compute_state(t) for t in batch]).to(device)
        rewards = torch.tensor([compute_reward(t) for t in batch], dtype=torch.float32).to(device)
        
        action_probs, values = policy(states)
        values = values.squeeze(-1)
        
        advantages = rewards - values.detach()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        policy_loss = -(torch.log(action_probs.max(dim=1)[0]) * advantages).mean()
        value_loss = nn.MSELoss()(values, rewards)
        loss = policy_loss + 0.5 * value_loss
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
        optimizer.step()
        
        epoch_reward += rewards.mean().item()
    
    avg_reward = epoch_reward / (len(shuffled) // 16)
    if avg_reward > best_train:
        best_train = avg_reward
    
    if (epoch + 1) % 3 == 0:
        logger.info(f"Epoch {epoch+1}: {avg_reward:.4f}, Improvement: {(avg_reward-baseline)/baseline*100:+.1f}%")

# Test
policy.eval()
test_rewards = []
with torch.no_grad():
    for traj in test_traj:
        reward = compute_reward(traj)
        test_rewards.append(reward)

test_reward = sum(test_rewards) / len(test_rewards)
logger.info(f"✅ Test: {test_reward:.4f}, Improvement: {(test_reward-baseline)/baseline*100:+.1f}%")

