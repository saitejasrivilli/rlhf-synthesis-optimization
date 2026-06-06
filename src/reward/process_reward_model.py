"""
Process Reward Model (PRM): step-level rewards for chain-of-thought reasoning.

Instead of a single terminal signal (1.0 if correct), PRM gives intermediate
feedback after each reasoning step:
  +0.15  step contains a verifiable numeric computation
  +0.05  step contains executable Python code (but no numeric grounding)
  +1.0   correct final answer (from verifiable_reward)

Denser gradient signal → faster convergence, less reward hacking on
shortcuts that skip intermediate work.

Algorithm (Math-Shepherd style):
  total = Σ_{t=1}^{T} γ^(T-1-t) * step_reward_t  +  final_reward
  γ = 0.9 by default — earlier steps weighted higher.

Reference: Math-Shepherd (Wang et al. 2023), Lightman et al. 2023 (PRM800K).
"""
import re
import sys
import subprocess
from typing import List, Optional, Tuple


_STEP_SEP = re.compile(
    r"\n\s*\n"
    r"|(?:^|\n)(?:Step\s+\d+[:\.]|#+\s)"
    r"|(?:^|\n)(?:First|Second|Third|Fourth|Next|Then|Finally)[,:\s]",
    re.MULTILINE | re.IGNORECASE,
)
_CODE_BLOCK = re.compile(r"```(?:python)?\s*(.*?)```", re.DOTALL)
_ASSIGN_STMT = re.compile(r"([\w_]+)\s*=\s*(.+)")
_MATH_OPS = re.compile(r"[-\d\s()+\-*/%^.]+")
_NUMERIC = re.compile(r"-?\d+(?:,\d{3})*(?:\.\d+)?")


def _safe_eval(expr: str) -> Optional[float]:
    """Evaluate a pure-arithmetic expression; returns None on any error."""
    expr = re.sub(r",", "", expr)      # 1,000 → 1000
    expr = re.sub(r"\^", "**", expr)   # ^ → **
    try:
        result = eval(expr, {"__builtins__": {}})
        if isinstance(result, (int, float)) and not (
            result != result or abs(result) == float("inf")
        ):
            return float(result)
    except Exception:
        pass
    return None


def _exec_code(code: str, timeout: int = 3) -> Optional[str]:
    """Run a Python snippet; return stdout or None on failure."""
    try:
        r = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True, text=True, timeout=timeout,
        )
        return r.stdout.strip() if r.returncode == 0 and r.stdout.strip() else None
    except Exception:
        return None


def parse_steps(response: str) -> List[str]:
    """
    Split a model response into individual reasoning steps.

    Handles:
    - Blank-line separated paragraphs
    - Numbered steps ("Step 1: ...")
    - Transition words ("First, ...", "Then, ...")
    """
    raw = _STEP_SEP.split(response)
    steps = []
    for s in raw:
        s = s.strip()
        if not s:
            continue
        # Merge very short fragments into previous step
        if steps and len(s) < 25:
            steps[-1] += " " + s
        else:
            steps.append(s)
    return steps if steps else [response]


def score_step(step: str) -> float:
    """
    Score a single reasoning step: 0.0, 0.05, 0.10, or 0.15.

    Scoring hierarchy:
    1. Fenced code block → execute it. If valid output: 0.15. If error: 0.05.
    2. Assignment with verifiable numeric RHS: 0.15.
    3. Bare arithmetic expression that evaluates to a number: 0.10.
    4. No computable content: 0.0.
    """
    # 1. Code blocks
    for code in _CODE_BLOCK.findall(step):
        code = code.strip()
        if not code:
            continue
        out = _exec_code(code)
        return 0.15 if out is not None else 0.05

    # 2. Assignment: "x = 3 * 24 = 72" or "cost = 4 * 5"
    for m in _ASSIGN_STMT.finditer(step):
        rhs = m.group(2).strip()
        # Strip trailing "= <stated_result>"
        parts = re.split(r"\s*=\s*(?=\d)", rhs, maxsplit=1)
        expr = parts[0].strip()
        result = _safe_eval(expr)
        if result is None:
            continue
        if len(parts) > 1:
            stated_nums = _NUMERIC.findall(parts[1])
            if stated_nums:
                try:
                    stated = float(stated_nums[0].replace(",", ""))
                    if abs(result - stated) < max(1e-4, abs(stated) * 1e-4):
                        return 0.15
                    continue
                except ValueError:
                    pass
        return 0.10   # expression parsed but no stated result to verify

    # 3. Bare arithmetic in the step text
    for m in _MATH_OPS.finditer(step):
        expr = m.group(0).strip()
        if len(expr) < 4:
            continue
        if not any(op in expr for op in ["+", "-", "*", "/"]):
            continue
        result = _safe_eval(expr)
        if result is not None:
            return 0.10

    return 0.0


def compute_prm_reward(
    response: str,
    ground_truth: str,
    gamma: float = 0.9,
    step_weight: float = 1.0,
) -> Tuple[float, List[float]]:
    """
    Compute process reward for a full chain-of-thought response.

    Returns:
        (total_reward, step_rewards)  where step_rewards[-1] is the
        final-answer component.

    Formula:
        total = step_weight * Σ_t γ^(T-1-t) * r_t  +  final_answer_reward

    γ < 1 weights earlier steps higher, incentivising front-loaded reasoning.
    """
    from .verifiable_reward import compute_reward

    steps = parse_steps(response)
    step_scores = [score_step(s) for s in steps]

    T = len(step_scores)
    discounted = step_weight * sum(
        (gamma ** (T - 1 - t)) * r for t, r in enumerate(step_scores)
    )

    final = compute_reward(response, ground_truth)
    total = discounted + final

    return total, step_scores + [final]


def prm_reward_batch(
    responses: List[str],
    ground_truths: List[str],
    gamma: float = 0.9,
    step_weight: float = 1.0,
) -> List[float]:
    return [
        compute_prm_reward(r, gt, gamma=gamma, step_weight=step_weight)[0]
        for r, gt in zip(responses, ground_truths)
    ]
