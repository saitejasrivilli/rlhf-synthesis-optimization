"""
BERT-based learned reward model for synthesis trajectory scoring.

Two modes:
  1. Rule-based (default):  weighted sum over yield/selectivity/safety/efficiency.
  2. Neural (after fine-tuning): BERT CLS head, blended 40/60 with rule score.

Both expose the same score_trajectory / score_batch / compute_reward API so they
are drop-in replacements for RealRewardModel in existing scripts.
"""
import logging
from typing import Dict, List, Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

_RULE_WEIGHTS = {"yield": 0.40, "selectivity": 0.30, "safety": 0.20, "efficiency": 0.10}


def _extract_outcomes(traj: Dict) -> Dict:
    """Normalise flat and nested trajectory formats to a flat outcomes dict."""
    out = traj.get("outcomes", {})
    params = traj.get("parameters", {})
    return {
        "yield":        out.get("yield",        traj.get("yield", 0.5)),
        "selectivity":  out.get("selectivity",  traj.get("selectivity", 0.5)),
        "safety_risk":  out.get("safety_risk",  traj.get("safety_risk", 0.3)),
        "steps":        out.get("steps",        traj.get("steps", 5)),
        "temperature":  params.get("temperature_celsius", traj.get("temperature_celsius", 80)),
        "time":         params.get("time_hours",          traj.get("time_hours", 4)),
    }


def _rule_score(traj: Dict) -> float:
    o = _extract_outcomes(traj)
    score = (
        o["yield"]                        * _RULE_WEIGHTS["yield"]
        + o["selectivity"]                * _RULE_WEIGHTS["selectivity"]
        + (1.0 - o["safety_risk"])        * _RULE_WEIGHTS["safety"]
        + (1.0 / (1.0 + o["steps"] / 10.0)) * _RULE_WEIGHTS["efficiency"]
    )
    return float(max(0.0, min(1.0, score)))


def _traj_to_text(traj: Dict) -> str:
    o = _extract_outcomes(traj)
    mol = traj.get("molecule", "compound")
    return (
        f"Synthesis of {mol}: temp {o['temperature']:.0f}°C, "
        f"time {o['time']:.1f}h. "
        f"Yield {o['yield']:.2f}, selectivity {o['selectivity']:.2f}, "
        f"safety risk {o['safety_risk']:.2f}, steps {o['steps']}."
    )


class LearnedRewardModel(nn.Module):
    """
    BERT-base reward model.

    Architecture:
      BERT CLS → Dropout → Linear(768, 384) → GELU → Dropout → Linear(384, 1) → Sigmoid
    """

    def __init__(
        self,
        backbone: str = "bert-base-uncased",
        hidden_dim: int = 768,
        dropout: float = 0.1,
    ):
        super().__init__()
        self._backbone_name = backbone
        self._is_trained = False

        try:
            from transformers import AutoModel, AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(backbone)
            self.encoder = AutoModel.from_pretrained(backbone)
            self.reward_head = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, 1),
                nn.Sigmoid(),
            )
            logger.info(f"LearnedRewardModel: {backbone} backbone loaded")
        except ImportError:
            self.tokenizer = None
            self.encoder = None
            self.reward_head = None
            logger.warning("transformers not available — using rule-based fallback only")

    # ------------------------------------------------------------------
    # nn.Module forward
    # ------------------------------------------------------------------

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0, :]
        return self.reward_head(cls).squeeze(-1)

    # ------------------------------------------------------------------
    # Public scoring API
    # ------------------------------------------------------------------

    def score_trajectory(self, traj: Dict) -> float:
        rule = _rule_score(traj)
        if not (self._is_trained and self.encoder is not None):
            return rule

        text = _traj_to_text(traj)
        enc = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
        dev = next(self.encoder.parameters()).device
        enc = {k: v.to(dev) for k, v in enc.items()}
        self.eval()
        with torch.no_grad():
            neural = self(enc["input_ids"], enc["attention_mask"]).item()
        # Blend: 40% rule + 60% neural
        return max(0.0, min(1.0, 0.4 * rule + 0.6 * neural))

    def score_batch(self, trajectories: List[Dict]) -> List[float]:
        return [self.score_trajectory(t) for t in trajectories]

    # Alias used by existing RealPPOTrainer
    def compute_reward(self, traj: Dict) -> float:
        return self.score_trajectory(traj)

    def score_text_batch(
        self, texts: List[str], device: str = "cpu"
    ) -> torch.Tensor:
        """Score a batch of text strings — used during LLM PPO training."""
        if self.tokenizer is None:
            return torch.zeros(len(texts))
        enc = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128,
        ).to(device)
        self.eval()
        with torch.no_grad():
            return self(enc["input_ids"], enc["attention_mask"])

    # ------------------------------------------------------------------
    # Fine-tuning on (trajectory, score) pairs
    # ------------------------------------------------------------------

    def finetune(
        self,
        trajectories: List[Dict],
        epochs: int = 3,
        lr: float = 1e-4,
        batch_size: int = 16,
        device: str = "cpu",
    ) -> List[float]:
        """Supervised fine-tuning on rule-based pseudo-labels."""
        if self.encoder is None:
            logger.warning("No encoder — skipping fine-tuning")
            return []

        self.to(device)
        self.train()
        opt = torch.optim.Adam(self.parameters(), lr=lr)
        losses = []

        texts = [_traj_to_text(t) for t in trajectories]
        targets = torch.tensor([_rule_score(t) for t in trajectories], dtype=torch.float32)

        for epoch in range(epochs):
            ep_loss = 0.0
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i : i + batch_size]
                batch_targets = targets[i : i + batch_size].to(device)

                enc = self.tokenizer(
                    batch_texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=128,
                ).to(device)
                preds = self(enc["input_ids"], enc["attention_mask"])
                loss = nn.functional.mse_loss(preds, batch_targets)

                opt.zero_grad()
                loss.backward()
                opt.step()
                ep_loss += loss.item()

            avg = ep_loss / max(1, len(texts) // batch_size)
            losses.append(avg)
            logger.info(f"Reward model fine-tune epoch {epoch+1}/{epochs}: loss={avg:.4f}")

        self._is_trained = True
        return losses

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str):
        torch.save(self.state_dict(), path)
        logger.info(f"Reward model saved to {path}")

    def load(self, path: str):
        self.load_state_dict(torch.load(path, map_location="cpu"))
        self._is_trained = True
        logger.info(f"Reward model loaded from {path}")
