"""
Safe Python code executor for agent RL rollouts.
Runs model-generated code in a subprocess with a hard timeout.
No imports are blocked — the model learns which tools are useful
through RL signal, not through a blocklist.
"""
import subprocess
import sys
import textwrap
from dataclasses import dataclass

TIMEOUT_SEC = 5
MAX_OUTPUT_CHARS = 500


@dataclass
class ExecutionResult:
    stdout: str
    stderr: str
    timed_out: bool
    error: bool

    @property
    def output(self) -> str:
        if self.timed_out:
            return "ERROR: execution timed out (5s limit)"
        if self.error and not self.stdout:
            return f"ERROR: {self.stderr[:200]}"
        out = self.stdout[:MAX_OUTPUT_CHARS]
        if self.stderr and not self.error:
            out = out.rstrip() + f"\n# stderr: {self.stderr[:100]}"
        return out


def execute(code: str, timeout: float = TIMEOUT_SEC) -> ExecutionResult:
    """Execute Python code in an isolated subprocess. Returns stdout/stderr."""
    try:
        result = subprocess.run(
            [sys.executable, "-c", textwrap.dedent(code)],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return ExecutionResult(
            stdout=result.stdout.strip(),
            stderr=result.stderr.strip(),
            timed_out=False,
            error=result.returncode != 0,
        )
    except subprocess.TimeoutExpired:
        return ExecutionResult("", "", timed_out=True, error=True)
    except Exception as e:
        return ExecutionResult("", str(e), timed_out=False, error=True)
