"""Tests for preference pair building and Bradley-Terry reward model."""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.reward.preference_reward_model import build_preference_pairs, PreferenceRewardModel


# -----------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------

def make_traj(molecule, yield_val, selectivity=0.8):
    return {
        "molecule": molecule,
        "cas_number": f"TEST_{molecule}",
        "parameters": {"temperature_celsius": 80.0, "time_hours": 4.0,
                       "catalyst_loading_M": 0.05, "solvent_ratio_ml_mmol": 3.0},
        "outcomes": {"yield": yield_val, "selectivity": selectivity,
                     "safety_risk": 0.15, "steps": 4},
    }


# 10 Aspirin trajectories with varying yields
ASPIRIN_TRAJS = [make_traj("Aspirin", 0.50 + i * 0.04) for i in range(10)]
# 10 Ibuprofen trajectories
IBU_TRAJS     = [make_traj("Ibuprofen", 0.45 + i * 0.05) for i in range(10)]

ALL_TRAJS = ASPIRIN_TRAJS + IBU_TRAJS


# -----------------------------------------------------------------------
# build_preference_pairs
# -----------------------------------------------------------------------

def test_pairs_produced():
    pairs = build_preference_pairs(ALL_TRAJS, pairs_per_molecule=5)
    assert len(pairs) > 0, "Should produce at least one pair"

def test_pair_structure():
    pairs = build_preference_pairs(ALL_TRAJS, pairs_per_molecule=5)
    p = pairs[0]
    # pairs are (chosen, rejected) tuples
    assert isinstance(p, (tuple, list)) and len(p) == 2

def test_chosen_higher_yield():
    pairs = build_preference_pairs(ALL_TRAJS, pairs_per_molecule=10)
    for chosen, rejected in pairs:
        chosen_yield   = chosen.get("outcomes",   {}).get("yield",
                         chosen.get("yield", 0))
        rejected_yield = rejected.get("outcomes", {}).get("yield",
                         rejected.get("yield", 0))
        assert chosen_yield >= rejected_yield, (
            f"Chosen yield {chosen_yield} should be >= rejected {rejected_yield}"
        )

def test_pairs_capped_per_molecule():
    pairs = build_preference_pairs(ALL_TRAJS, pairs_per_molecule=3)
    # 2 molecules × 3 pairs = 6 pairs
    assert len(pairs) <= 6 + 2, "pairs_per_molecule cap should be respected"

def test_single_molecule():
    pairs = build_preference_pairs(ASPIRIN_TRAJS, pairs_per_molecule=5)
    assert len(pairs) > 0

def test_empty_trajectories():
    pairs = build_preference_pairs([], pairs_per_molecule=5)
    assert pairs == []


# -----------------------------------------------------------------------
# PreferenceRewardModel.score_trajectory
# -----------------------------------------------------------------------

def test_score_range():
    rm   = PreferenceRewardModel()
    traj = make_traj("Aspirin", 0.85)
    score = rm.score_trajectory(traj)
    assert 0.0 <= score <= 1.0, f"Score out of range: {score}"

def test_untrained_score():
    """Before training, uses 30% rule-based blend — should still return a float."""
    rm    = PreferenceRewardModel()
    high  = make_traj("Aspirin", 0.95)
    low   = make_traj("Aspirin", 0.20)
    score_high = rm.score_trajectory(high)
    score_low  = rm.score_trajectory(low)
    assert isinstance(score_high, float)
    assert isinstance(score_low,  float)

def test_score_batch_length():
    rm     = PreferenceRewardModel()
    trajs  = ASPIRIN_TRAJS[:5]
    scores = [rm.score_trajectory(t) for t in trajs]
    assert len(scores) == 5
    assert all(0.0 <= s <= 1.0 for s in scores)
