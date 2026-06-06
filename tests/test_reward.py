"""Tests for RealRewardModel — both flat and nested trajectory formats."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.reward.real_reward_model import RealRewardModel


@pytest.fixture
def rm():
    return RealRewardModel()


# ---- flat format (improvable trajectories) ----

FLAT_HIGH = {"yield": 0.95, "selectivity": 0.90, "safety_risk": 0.05, "steps": 3}
FLAT_LOW  = {"yield": 0.20, "selectivity": 0.30, "safety_risk": 0.90, "steps": 15}

def test_flat_high_yield(rm):
    score = rm.score_trajectory(FLAT_HIGH)
    assert score > 0.80, f"Expected > 0.80, got {score}"

def test_flat_low_yield(rm):
    score = rm.score_trajectory(FLAT_LOW)
    assert score < 0.40, f"Expected < 0.40, got {score}"

def test_flat_range(rm):
    score = rm.score_trajectory(FLAT_HIGH)
    assert 0.0 <= score <= 1.0


# ---- nested format (ORD trajectories) ----

ORD_HIGH = {
    "molecule": "Aspirin",
    "parameters": {"temperature_celsius": 85.0, "time_hours": 2.0},
    "outcomes":   {"yield": 0.94, "selectivity": 0.88, "safety_risk": 0.10, "steps": 3},
}
ORD_LOW = {
    "molecule": "Aspirin",
    "parameters": {"temperature_celsius": 60.0, "time_hours": 8.0},
    "outcomes":   {"yield": 0.30, "selectivity": 0.40, "safety_risk": 0.70, "steps": 10},
}

def test_nested_high_yield(rm):
    score = rm.score_trajectory(ORD_HIGH)
    assert score > 0.75

def test_nested_low_yield(rm):
    score = rm.score_trajectory(ORD_LOW)
    assert score < 0.50

def test_nested_vs_flat_ordering(rm):
    """Nested ORD high-yield should score higher than nested low-yield."""
    assert rm.score_trajectory(ORD_HIGH) > rm.score_trajectory(ORD_LOW)


# ---- compute_reward alias ----

def test_compute_reward_alias(rm):
    s1 = rm.score_trajectory(FLAT_HIGH)
    s2 = rm.compute_reward(FLAT_HIGH)
    assert abs(s1 - s2) < 1e-9


# ---- batch scoring ----

def test_score_batch(rm):
    batch  = [FLAT_HIGH, FLAT_LOW, ORD_HIGH]
    scores = rm.score_batch(batch)
    assert len(scores) == 3
    assert scores[0] > scores[1], "High-yield trajectory should score higher"


# ---- edge cases ----

def test_empty_trajectory(rm):
    score = rm.score_trajectory({})
    assert 0.0 <= score <= 1.0, "Should handle missing keys gracefully"

def test_zero_yield(rm):
    score = rm.score_trajectory({"yield": 0.0, "selectivity": 0.0, "safety_risk": 1.0, "steps": 20})
    assert score >= 0.0

def test_perfect_trajectory(rm):
    score = rm.score_trajectory({"yield": 1.0, "selectivity": 1.0, "safety_risk": 0.0, "steps": 1})
    assert score > 0.90
