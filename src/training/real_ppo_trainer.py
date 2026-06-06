import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import logging
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)


def encode_trajectory(traj: Dict) -> torch.Tensor:
    """
    Encode a trajectory into a 128-dim state vector.

    First 8 dims are normalized chemistry features; remaining 120 are zero-padding
    for future molecular fingerprint extension.
    """
    params   = traj.get("parameters", traj)
    outcomes = traj.get("outcomes", {})

    temp        = params.get("temperature_celsius",  traj.get("temperature_celsius",  80.0))
    time        = params.get("time_hours",            traj.get("time_hours",            4.0))
    catalyst    = params.get("catalyst_loading_M",   traj.get("catalyst_loading_M",   0.05))
    solvent     = params.get("solvent_ratio_ml_mmol",traj.get("solvent_ratio_ml_mmol", 3.0))
    yield_val   = outcomes.get("yield",        traj.get("yield",        0.5))
    selectivity = outcomes.get("selectivity",  traj.get("selectivity",  0.5))
    safety_risk = outcomes.get("safety_risk",  traj.get("safety_risk",  0.3))
    steps       = outcomes.get("steps",        traj.get("steps",        5))

    features = [
        float(temp)        / 200.0,
        float(time)        / 12.0,
        min(float(catalyst), 1.0),
        float(solvent)     / 6.0,
        float(yield_val),
        float(selectivity),
        1.0 - float(safety_risk),
        float(steps)       / 10.0,
    ]

    state = torch.zeros(128, dtype=torch.float32)
    state[:len(features)] = torch.tensor(features, dtype=torch.float32)
    return state


class RealPPOPolicy(nn.Module):
    """Actor-critic policy for synthesis optimization."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        if state_dim <= 0 or action_dim <= 0 or hidden_dim <= 0:
            raise ValueError("Dimensions must be positive")

        self.state_dim  = state_dim
        self.action_dim = action_dim

        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, action_dim), nn.Softmax(dim=-1),
        )
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        logger.info(f"Initialized RealPPOPolicy (state={state_dim}, action={action_dim})")

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if state.dim() == 1:
            state = state.unsqueeze(0)
        features    = self.shared(state)
        action_probs = self.actor(features)
        value        = self.critic(features)
        return action_probs, value


class RealPPOTrainer:
    """
    PPO trainer using real trajectory state vectors.

    Each trajectory is encoded with encode_trajectory() into a 128-dim
    state vector, giving the actor-critic meaningful chemistry features
    rather than random noise.

    Loss = policy_loss (clipped surrogate) + 0.5 * value_loss - 0.01 * entropy
    """

    def __init__(
        self,
        policy: RealPPOPolicy,
        reward_model,
        learning_rate: float = 1e-4,
        device: str = "cpu",
        clip_ratio: float = 0.2,
    ):
        if learning_rate <= 0:
            raise ValueError("Learning rate must be positive")

        self.policy       = policy.to(device)
        self.reward_model = reward_model
        self.device       = device
        self.clip_ratio   = clip_ratio
        self.optimizer    = optim.Adam(self.policy.parameters(), lr=learning_rate)

        logger.info(f"Initialized RealPPOTrainer | lr={learning_rate} | clip={clip_ratio}")

    def _encode_batch(self, trajectories: List[Dict]) -> torch.Tensor:
        states = torch.stack([encode_trajectory(t) for t in trajectories])
        return states.to(self.device)

    def train(
        self,
        trajectories: List[Dict],
        epochs: int = 5,
        batch_size: int = 8,
    ) -> Dict:
        if not trajectories:
            raise ValueError("Empty trajectory list")

        self.policy.train()
        metrics = {
            "epoch_rewards": [],
            "epoch_losses":  [],
            "epoch_policy_losses": [],
            "epoch_value_losses":  [],
            "final_reward": 0.0,
            "improvement":  0.0,
        }
        baseline_reward = 0.5

        for epoch in range(epochs):
            epoch_reward = 0.0
            epoch_loss   = 0.0
            ep_pol_loss  = 0.0
            ep_val_loss  = 0.0
            num_batches  = 0

            for i in range(0, len(trajectories), batch_size):
                try:
                    batch = trajectories[i : i + batch_size]

                    # ---- real state encoding ----
                    states = self._encode_batch(batch)

                    # ---- rewards ----
                    rewards = torch.tensor(
                        [self.reward_model.compute_reward(t) for t in batch],
                        dtype=torch.float32, device=self.device,
                    )

                    # ---- policy forward ----
                    action_probs, values = self.policy(states)
                    values = values.squeeze(-1)

                    # ---- advantages ----
                    advantages = rewards - values.detach()
                    advantages = (advantages - advantages.mean()) / (
                        advantages.std() + 1e-8
                    )

                    # ---- PPO clipped surrogate loss ----
                    log_probs     = torch.log(action_probs.max(dim=1)[0] + 1e-8)
                    old_log_probs = log_probs.detach()
                    ratio         = torch.exp(log_probs - old_log_probs)
                    surr1         = ratio * advantages
                    surr2         = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages
                    policy_loss   = -torch.min(surr1, surr2).mean()

                    # ---- value loss ----
                    value_loss = F.mse_loss(values, rewards)

                    # ---- entropy bonus ----
                    entropy = -(action_probs * torch.log(action_probs + 1e-8)).sum(dim=-1).mean()

                    loss = policy_loss + 0.5 * value_loss - 0.01 * entropy

                    self.optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)
                    self.optimizer.step()

                    epoch_reward += rewards.mean().item()
                    epoch_loss   += loss.item()
                    ep_pol_loss  += policy_loss.item()
                    ep_val_loss  += value_loss.item()
                    num_batches  += 1

                except Exception as e:
                    logger.error(f"Error in batch {i}: {e}")
                    continue

            if num_batches > 0:
                avg_reward   = epoch_reward  / num_batches
                avg_loss     = epoch_loss    / num_batches
                avg_pol_loss = ep_pol_loss   / num_batches
                avg_val_loss = ep_val_loss   / num_batches

                metrics["epoch_rewards"].append(avg_reward)
                metrics["epoch_losses"].append(avg_loss)
                metrics["epoch_policy_losses"].append(avg_pol_loss)
                metrics["epoch_value_losses"].append(avg_val_loss)

                logger.info(
                    f"Epoch {epoch+1}/{epochs}: "
                    f"reward={avg_reward:.4f} | "
                    f"loss={avg_loss:.4f} | "
                    f"policy={avg_pol_loss:.4f} | "
                    f"value={avg_val_loss:.4f}"
                )

        if metrics["epoch_rewards"]:
            metrics["final_reward"] = metrics["epoch_rewards"][-1]
            metrics["improvement"]  = (
                (metrics["final_reward"] - baseline_reward) / baseline_reward
            ) * 100

        logger.info(f"Training complete. Final reward: {metrics['final_reward']:.4f}")
        return metrics
