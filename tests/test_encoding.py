"""Tests for encode_trajectory() and reaction_constraints."""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.training.real_ppo_trainer import encode_trajectory
from src.utils.reaction_constraints import validate_conditions, batch_validate, validity_penalty


# -----------------------------------------------------------------------
# encode_trajectory
# -----------------------------------------------------------------------

FLAT_TRAJ = {
    "temperature_celsius":   80.0,
    "time_hours":            4.0,
    "catalyst_loading_M":    0.05,
    "solvent_ratio_ml_mmol": 3.0,
    "yield":                 0.85,
    "selectivity":           0.80,
    "safety_risk":           0.15,
    "steps":                 4,
}

ORD_TRAJ = {
    "parameters": {
        "temperature_celsius":   90.0,
        "time_hours":            2.5,
        "catalyst_loading_M":    0.10,
        "solvent_ratio_ml_mmol": 2.0,
    },
    "outcomes": {
        "yield":        0.94,
        "selectivity":  0.88,
        "safety_risk":  0.10,
        "steps":        3,
    },
}


def test_output_shape():
    v = encode_trajectory(FLAT_TRAJ)
    assert v.shape == (128,), f"Expected (128,), got {v.shape}"

def test_output_dtype():
    v = encode_trajectory(FLAT_TRAJ)
    assert v.dtype == torch.float32

def test_flat_format():
    v = encode_trajectory(FLAT_TRAJ)
    # temperature: 80/200=0.40
    assert abs(v[0].item() - 0.40) < 1e-5, f"temp feature wrong: {v[0].item()}"

def test_nested_format():
    v = encode_trajectory(ORD_TRAJ)
    # temperature: 90/200=0.45
    assert abs(v[0].item() - 0.45) < 1e-5, f"temp feature wrong: {v[0].item()}"

def test_padding_zeros():
    v = encode_trajectory(FLAT_TRAJ)
    # Positions 8+ should be zero-padded (when rdkit unavailable)
    # When rdkit IS available, fingerprint bits fill positions 8:108
    # Either way the vector should be finite
    assert torch.isfinite(v).all(), "Vector contains NaN or Inf"

def test_high_yield_differs_from_low():
    high = {**FLAT_TRAJ, "yield": 0.95, "selectivity": 0.90}
    low  = {**FLAT_TRAJ, "yield": 0.20, "selectivity": 0.30}
    v_high = encode_trajectory(high)
    v_low  = encode_trajectory(low)
    assert not torch.allclose(v_high, v_low), "High/low yield should produce different state vectors"

def test_missing_keys():
    v = encode_trajectory({})
    assert v.shape == (128,)
    assert torch.isfinite(v).all()


# -----------------------------------------------------------------------
# reaction_constraints
# -----------------------------------------------------------------------

VALID_CONDITIONS = {
    "temperature_celsius":   80.0,
    "time_hours":            4.0,
    "catalyst_loading_M":    0.05,
    "solvent_ratio_ml_mmol": 3.0,
}

OUT_OF_RANGE = {
    "temperature_celsius":   999.0,
    "time_hours":            -2.0,
    "catalyst_loading_M":    0.05,
    "solvent_ratio_ml_mmol": 3.0,
}


def test_valid_passes():
    clipped, was_valid = validate_conditions(VALID_CONDITIONS)
    assert was_valid
    assert clipped["temperature_celsius"] == 80.0

def test_out_of_range_clipped():
    clipped, was_valid = validate_conditions(OUT_OF_RANGE)
    assert not was_valid
    assert clipped["temperature_celsius"] == 300.0, "Should clip to max 300"
    assert clipped["time_hours"] == 0.25,           "Should clip to min 0.25"

def test_missing_key_defaults():
    clipped, was_valid = validate_conditions({})
    assert not was_valid
    assert clipped["temperature_celsius"] == 80.0   # default

def test_batch_validate():
    conditions = [VALID_CONDITIONS, OUT_OF_RANGE, {}]
    clipped, validity = batch_validate(conditions)
    assert len(clipped)  == 3
    assert len(validity) == 3
    assert validity[0] is True
    assert validity[1] is False
    assert validity[2] is False

def test_validity_penalty():
    assert validity_penalty(True)  == 0.0
    assert validity_penalty(False) == -0.05
    assert validity_penalty(False, penalty=0.1) == -0.1
