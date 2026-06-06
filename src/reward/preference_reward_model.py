"""
Preference-based reward model using Bradley-Terry ranking loss.

Instead of fitting BERT to a rule-based scalar (circular), we train on
preference pairs (chosen, rejected) using the BT ranking objective:

    loss = -log σ(r_chosen - r_rejected)

Pairs are built from trajectory data: for each molecule, trajectories with
yield > median are "chosen" and those below are "rejected". This makes the
reward model learn what chemists prefer from the data structure itself,
without relying on hand-crafted weights.
"""
import logging
import random
from itertools import combinations
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def _traj_to_text(traj: Dict) -> str:
    params   = traj.get("parameters", traj)
    outcomes = traj.get("outcomes", {})
    mol      = traj.get("molecule", "compound")

    temp     = params.get("temperature_celsius",  traj.get("temperature_celsius",  80))
    time     = params.get("time_hours",            traj.get("time_hours",            4))
    catalyst = params.get("catalyst_loading_M",   traj.get("catalyst_loading_M",   0.05))
    solvent  = params.get("solvent_ratio_ml_mmol",traj.get("solvent_ratio_ml_mmol", 3.0))

    yield_val   = outcomes.get("yield",       traj.get("yield",       0.5))
    selectivity = outcomes.get("selectivity", traj.get("selectivity", 0.5))
    safety_risk = outcomes.get("safety_risk", traj.get("safety_risk", 0.3))
    steps       = outcomes.get("steps",       traj.get("steps",       5))

    return (
        f"Synthesis of {mol}: temp {temp:.0f}°C, time {time:.1f}h, "
        f"catalyst {catalyst:.3f}M, solvent {solvent:.1f}. "
        f"Yield {yield_val:.2f}, selectivity {selectivity:.2f}, "
        f"safety risk {safety_risk:.2f}, steps {steps}."
    )


def build_preference_pairs(
    trajectories: List[Dict],
    pairs_per_molecule: int = 20,
) -> List[Tuple[Dict, Dict]]:
    """
    Build (chosen, rejected) preference pairs.

    Strategy: group by molecule, sort by yield, pair top-half vs bottom-half.
    Each chosen trajectory has higher yield than its paired rejected one.
    """
    by_molecule: Dict[str, List[Dict]] = {}
    for traj in trajectories:
        mol = traj.get("molecule", "unknown")
        by_molecule.setdefault(mol, []).append(traj)

    pairs = []
    for mol, trajs in by_molecule.items():
        if len(trajs) < 2:
            continue
        outcomes_key = lambda t: t.get("outcomes", t).get("yield", t.get("yield", 0))
        sorted_trajs = sorted(trajs, key=outcomes_key, reverse=True)
        n     = len(sorted_trajs)
        top   = sorted_trajs[: n // 2]
        bottom = sorted_trajs[n // 2 :]

        # Sample pairs (chosen from top, rejected from bottom)
        sampled = 0
        while sampled < pairs_per_molecule and top and bottom:
            chosen   = random.choice(top)
            rejected = random.choice(bottom)
            pairs.append((chosen, rejected))
            sampled += 1

    logger.info(f"Built {len(pairs)} preference pairs from {len(trajectories)} trajectories")
    return pairs


class PreferenceRewardModel(nn.Module):
    """
    BERT-base reward model trained with Bradley-Terry ranking loss.

    Architecture: BERT CLS → Dropout → Linear(768, 384) → GELU → Linear(384, 1)
    No Sigmoid — raw logit used in BT loss for numerical stability.
    """

    def __init__(
        self,
        backbone: str = "bert-base-uncased",
        hidden_dim: int = 768,
        dropout: float = 0.1,
    ):
        super().__init__()
        self._backbone = backbone

        from transformers import AutoModel, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(backbone)
        self.encoder   = AutoModel.from_pretrained(backbone)
        self.head      = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        # Rule-based fallback (used before training)
        self._is_trained = False
        logger.info(f"PreferenceRewardModel initialized with {backbone}")

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0, :]
        return self.head(cls).squeeze(-1)   # raw logit, no sigmoid

    def _encode(self, texts: List[str], device: str) -> Dict:
        enc = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128,
        )
        return {k: v.to(device) for k, v in enc.items()}

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(
        self,
        pairs: List[Tuple[Dict, Dict]],
        epochs: int = 3,
        lr: float = 2e-5,
        batch_size: int = 16,
        device: str = "cpu",
    ) -> List[float]:
        """Fine-tune on (chosen, rejected) pairs using BT loss."""
        self.to(device)
        self.train()

        opt    = torch.optim.AdamW(self.parameters(), lr=lr)
        losses = []

        chosen_texts   = [_traj_to_text(c) for c, _ in pairs]
        rejected_texts = [_traj_to_text(r) for _, r in pairs]

        for epoch in range(epochs):
            ep_loss = 0.0
            indices = list(range(len(pairs)))
            random.shuffle(indices)

            for i in range(0, len(indices), batch_size):
                idx = indices[i : i + batch_size]
                c_texts = [chosen_texts[j]   for j in idx]
                r_texts = [rejected_texts[j] for j in idx]

                c_enc = self._encode(c_texts, device)
                r_enc = self._encode(r_texts, device)

                c_scores = self(c_enc["input_ids"], c_enc["attention_mask"])
                r_scores = self(r_enc["input_ids"], r_enc["attention_mask"])

                # Bradley-Terry loss
                loss = -F.logsigmoid(c_scores - r_scores).mean()

                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
                opt.step()
                ep_loss += loss.item()

            avg = ep_loss / max(1, len(pairs) // batch_size)
            losses.append(avg)
            logger.info(f"BT reward model epoch {epoch+1}/{epochs}: loss={avg:.4f}")

        self._is_trained = True
        return losses

    # ------------------------------------------------------------------
    # Scoring API (compatible with RealRewardModel)
    # ------------------------------------------------------------------

    def _rule_score(self, traj: Dict) -> float:
        out = traj.get("outcomes", traj)
        y   = out.get("yield",       traj.get("yield",       0.5))
        s   = out.get("selectivity", traj.get("selectivity", 0.5))
        sr  = out.get("safety_risk", traj.get("safety_risk", 0.3))
        st  = out.get("steps",       traj.get("steps",       5))
        return float(max(0, min(1,
            y*0.40 + s*0.30 + (1-sr)*0.20 + (1/(1+st/10))*0.10
        )))

    def score_trajectory(self, traj: Dict) -> float:
        rule = self._rule_score(traj)
        if not self._is_trained:
            return rule

        dev  = next(self.encoder.parameters()).device
        text = _traj_to_text(traj)
        enc  = self._encode([text], str(dev))
        self.eval()
        with torch.no_grad():
            logit  = self(enc["input_ids"], enc["attention_mask"]).item()
            neural = torch.sigmoid(torch.tensor(logit)).item()
        return max(0.0, min(1.0, 0.3 * rule + 0.7 * neural))

    def compute_reward(self, traj: Dict) -> float:
        return self.score_trajectory(traj)

    def score_batch(self, trajectories: List[Dict]) -> List[float]:
        return [self.score_trajectory(t) for t in trajectories]

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str):
        torch.save(self.state_dict(), path)
        logger.info(f"Preference reward model saved to {path}")

    def load(self, path: str):
        self.load_state_dict(torch.load(path, map_location="cpu"))
        self._is_trained = True
        logger.info(f"Preference reward model loaded from {path}")
