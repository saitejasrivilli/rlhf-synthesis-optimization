"""
Verifiable outcome reward for agent RL — no learned reward model required.

Reward schedule:
  1.0  — <final_answer> matches ground truth (exact or numeric)
  0.1  — model used a tool correctly (format ok) but got wrong answer
  0.0  — wrong answer, no valid tool use

Plus an auxiliary format reward (max 0.06) that encourages using
<think>, <tool_call>, and <final_answer> tags even when the answer is wrong.
This prevents reward hacking via empty outputs during early training.

Reference: DeepSeek-R1, RLVR (Reinforcement Learning with Verifiable Rewards).
"""
import re
from typing import Optional


_FINAL_ANS_RE = re.compile(r"<final_answer>(.*?)</final_answer>", re.DOTALL)
_TOOL_CALL_RE = re.compile(r"<tool_call>")
_TOOL_RES_RE  = re.compile(r"<tool_result>")


def normalize_answer(s: str) -> str:
    """Strip formatting and normalise to a canonical numeric string where possible."""
    s = s.strip().lower()
    s = re.sub(r"[$,%]", "", s)
    s = re.sub(r"\s+", " ", s)
    try:
        return str(float(s.replace(",", "")))
    except ValueError:
        return s


def extract_final_answer(text: str) -> Optional[str]:
    m = _FINAL_ANS_RE.search(text)
    if m:
        return m.group(1).strip()
    # Fallback: last standalone number in the text
    nums = re.findall(r"(?<!\w)-?\d+(?:,\d{3})*(?:\.\d+)?(?!\w)", text)
    return nums[-1] if nums else None


def compute_reward(response: str, ground_truth: str) -> float:
    """Binary verifiable reward (+ partial for correct tool-use format)."""
    answer = extract_final_answer(response)

    if answer is not None:
        if normalize_answer(answer) == normalize_answer(ground_truth):
            return 1.0

    # Partial: model attempted tool use with correct syntax
    if _TOOL_CALL_RE.search(response) and _TOOL_RES_RE.search(response):
        return 0.1

    return 0.0


def format_reward(response: str) -> float:
    """
    Small auxiliary reward for well-structured responses.
    Keeps gradients alive during early training when the agent rarely
    reaches a correct final answer.
    """
    score = 0.0
    if "<think>" in response:     score += 0.02
    if "<tool_call>" in response: score += 0.02
    if "<final_answer>" in response: score += 0.02
    return score
