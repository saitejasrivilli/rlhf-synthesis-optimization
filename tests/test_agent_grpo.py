"""Tests for verifiable reward and AgentGRPO components (no GPU required)."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.reward.verifiable_reward import (
    compute_reward,
    extract_final_answer,
    format_reward,
    normalize_answer,
)
from src.training.agent_grpo_trainer import build_agent_prompt


# ── normalize_answer ──────────────────────────────────────────────────────

def test_normalize_strips_dollar():
    assert normalize_answer("$1,234") == normalize_answer("1234")


def test_normalize_strips_percent():
    assert normalize_answer("50%") == normalize_answer("50")


def test_normalize_float_int_equal():
    assert normalize_answer("42.0") == normalize_answer("42")


def test_normalize_text_passthrough():
    assert normalize_answer("apple") == "apple"


# ── extract_final_answer ──────────────────────────────────────────────────

def test_extract_tagged_answer():
    text = "reasoning\n<final_answer>42</final_answer>"
    assert extract_final_answer(text) == "42"


def test_extract_fallback_last_number():
    text = "I count 3 steps, then 7 more, so the total is 17."
    assert extract_final_answer(text) == "17"


def test_extract_no_answer_returns_none():
    # No numbers, no tag
    assert extract_final_answer("no numbers here at all") is None


# ── compute_reward ────────────────────────────────────────────────────────

def test_reward_correct_tagged():
    r = compute_reward("<final_answer>42</final_answer>", "42")
    assert r == 1.0


def test_reward_correct_dollar_amount():
    r = compute_reward("<final_answer>$1,200</final_answer>", "1200")
    assert r == 1.0


def test_reward_partial_tool_use():
    response = (
        '<tool_call>{"name": "python_executor", "args": {"code": "1+1"}}</tool_call>\n'
        "<tool_result>2</tool_result>"
    )
    r = compute_reward(response, "999")
    assert r == pytest.approx(0.1)


def test_reward_wrong_no_tool():
    r = compute_reward("I think the answer is 5", "42")
    assert r == 0.0


def test_reward_wrong_answer_tagged():
    r = compute_reward("<final_answer>99</final_answer>", "42")
    assert r == 0.0


# ── format_reward ─────────────────────────────────────────────────────────

def test_format_reward_all_tags():
    text = "<think>x</think> <tool_call>y</tool_call> <final_answer>z</final_answer>"
    assert format_reward(text) == pytest.approx(0.06)


def test_format_reward_no_tags():
    assert format_reward("plain text") == 0.0


def test_format_reward_partial():
    assert format_reward("<think>reasoning</think>") == pytest.approx(0.02)


# ── combined reward (compute + format) ───────────────────────────────────

def test_correct_answer_with_tags_is_above_1():
    response = "<think>step</think><tool_call>c</tool_call><final_answer>42</final_answer>"
    total = compute_reward(response, "42") + format_reward(response)
    assert total == pytest.approx(1.06)


# ── build_agent_prompt ────────────────────────────────────────────────────

def test_prompt_contains_problem():
    prompt = build_agent_prompt("What is 2 + 2?")
    assert "What is 2 + 2?" in prompt
    assert "python_executor" in prompt
    assert "<final_answer>" in prompt
