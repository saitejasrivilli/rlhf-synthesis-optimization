"""Tests for tool infrastructure: python_executor + tool_parser."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.tools.python_executor import execute
from src.tools.tool_parser import parse_output


# ── python_executor ────────────────────────────────────────────────────────

def test_executor_basic_arithmetic():
    r = execute("print(2 + 2)")
    assert r.stdout == "4"
    assert not r.error
    assert not r.timed_out


def test_executor_multiline():
    code = "x = 10\ny = 32\nprint(x + y)"
    r = execute(code)
    assert r.stdout == "42"
    assert not r.error


def test_executor_import_math():
    r = execute("import math; print(int(math.factorial(5)))")
    assert r.stdout == "120"


def test_executor_syntax_error():
    r = execute("def (:")
    assert r.error


def test_executor_runtime_error():
    r = execute("print(1 / 0)")
    assert r.error
    assert "ZeroDivisionError" in r.stderr or "ERROR" in r.output


def test_executor_timeout():
    r = execute("while True: pass", timeout=1.0)
    assert r.timed_out
    assert "timed out" in r.output


def test_executor_output_truncation():
    r = execute("print('x' * 1000)")
    assert len(r.output) <= 510   # MAX_OUTPUT_CHARS + small slack


# ── tool_parser ───────────────────────────────────────────────────────────

def test_parser_tool_call():
    text = '<tool_call>{"name": "python_executor", "args": {"code": "2+2"}}</tool_call>'
    p = parse_output(text)
    assert p.tool_call is not None
    assert p.tool_call.name == "python_executor"
    assert p.tool_call.args["code"] == "2+2"
    assert p.final_answer is None


def test_parser_final_answer():
    text = "reasoning here\n<final_answer>42</final_answer>"
    p = parse_output(text)
    assert p.final_answer == "42"
    assert p.tool_call is None


def test_parser_think_tag():
    text = "<think>I should compute this</think>\n<tool_call>...</tool_call>"
    p = parse_output(text)
    assert p.thinking == "I should compute this"


def test_parser_full_trajectory():
    text = (
        "<think>Let me use Python</think>\n"
        '<tool_call>{"name": "python_executor", "args": {"code": "15*23+7"}}</tool_call>\n'
        "<tool_result>352</tool_result>\n"
        "<final_answer>352</final_answer>"
    )
    p = parse_output(text)
    assert p.tool_call.name == "python_executor"
    assert p.tool_results == ["352"]
    assert p.final_answer == "352"


def test_parser_invalid_json_tool_call():
    text = "<tool_call>NOT_JSON</tool_call>"
    p = parse_output(text)
    assert p.tool_call is None   # graceful fallback


def test_parser_no_tags():
    text = "The answer is 7."
    p = parse_output(text)
    assert p.tool_call is None
    assert p.final_answer is None
    assert p.thinking is None
    assert p.tool_results == []
