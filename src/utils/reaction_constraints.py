"""
Hard constraints for pharmaceutical synthesis conditions.

The LLM can generate out-of-range values (e.g. temperature=999°C, negative time).
This module clips and validates generated conditions before reward computation.
"""
from typing import Dict, Tuple

CONSTRAINTS = {
    "temperature_celsius":   (0.0,   300.0),
    "time_hours":            (0.25,  48.0),
    "catalyst_loading_M":    (0.001, 1.0),
    "solvent_ratio_ml_mmol": (0.5,   20.0),
}

DEFAULTS = {
    "temperature_celsius":   80.0,
    "time_hours":            4.0,
    "catalyst_loading_M":    0.05,
    "solvent_ratio_ml_mmol": 3.0,
}


def validate_conditions(conditions: Dict) -> Tuple[Dict, bool]:
    """
    Clip synthesis conditions to chemically feasible ranges.

    Returns (clipped_conditions, was_valid) where was_valid=False means
    at least one value was outside bounds and had to be clipped.
    """
    clipped = dict(conditions)
    valid   = True

    for key, (lo, hi) in CONSTRAINTS.items():
        raw = conditions.get(key)
        if raw is None:
            clipped[key] = DEFAULTS[key]
            valid = False
            continue
        try:
            val = float(raw)
        except (TypeError, ValueError):
            clipped[key] = DEFAULTS[key]
            valid = False
            continue

        if val < lo or val > hi:
            clipped[key] = max(lo, min(hi, val))
            valid = False

    return clipped, valid


def batch_validate(conditions_list):
    """Validate a list of condition dicts, return (clipped_list, validity_flags)."""
    results  = [validate_conditions(c) for c in conditions_list]
    clipped  = [r[0] for r in results]
    validity = [r[1] for r in results]
    return clipped, validity


def validity_penalty(was_valid: bool, penalty: float = 0.05) -> float:
    """Small reward penalty if conditions required clipping."""
    return 0.0 if was_valid else -penalty
