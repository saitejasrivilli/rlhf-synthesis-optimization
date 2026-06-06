"""
Parse <tool_call> / <tool_result> / <think> / <final_answer> tags from model output.
Tags follow a minimal XML convention — no schema validation, just regex extraction.
"""
import json
import re
from dataclasses import dataclass
from typing import Optional

_TOOL_CALL_RE    = re.compile(r"<tool_call>(.*?)</tool_call>",       re.DOTALL)
_FINAL_ANS_RE    = re.compile(r"<final_answer>(.*?)</final_answer>", re.DOTALL)
_THINK_RE        = re.compile(r"<think>(.*?)</think>",               re.DOTALL)
_TOOL_RESULT_RE  = re.compile(r"<tool_result>(.*?)</tool_result>",   re.DOTALL)


@dataclass
class ToolCall:
    name: str
    args: dict


@dataclass
class ParsedOutput:
    tool_call:    Optional[ToolCall]
    final_answer: Optional[str]
    thinking:     Optional[str]
    tool_results: list   # list[str], injected results already in the text
    raw:          str


def parse_output(text: str) -> ParsedOutput:
    tool_call = None
    m = _TOOL_CALL_RE.search(text)
    if m:
        try:
            data = json.loads(m.group(1).strip())
            tool_call = ToolCall(name=data["name"], args=data.get("args", {}))
        except (json.JSONDecodeError, KeyError):
            pass

    final_answer = None
    m = _FINAL_ANS_RE.search(text)
    if m:
        final_answer = m.group(1).strip()

    thinking = None
    m = _THINK_RE.search(text)
    if m:
        thinking = m.group(1).strip()

    tool_results = [m.group(1).strip() for m in _TOOL_RESULT_RE.finditer(text)]

    return ParsedOutput(
        tool_call=tool_call,
        final_answer=final_answer,
        thinking=thinking,
        tool_results=tool_results,
        raw=text,
    )
