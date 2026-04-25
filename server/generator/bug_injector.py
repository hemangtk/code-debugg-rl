"""Inject a single bug into a correct function's source code."""
from __future__ import annotations

from typing import Dict


def inject_bug(correct_code: str, bug: Dict) -> str:
    """Replace bug['bug_line'] of correct_code with bug['buggy_code'].

    Raises ValueError if the line at bug_line does not match
    bug['original_code'] (catches drift between templates and code).
    """
    lines = correct_code.split("\n")
    idx = bug["bug_line"] - 1
    if idx < 0 or idx >= len(lines):
        raise ValueError(
            f"bug_line {bug['bug_line']} out of range for code with {len(lines)} lines"
        )
    if lines[idx] != bug["original_code"]:
        raise ValueError(
            f"original_code mismatch at line {bug['bug_line']}\n"
            f"  expected: {bug['original_code']!r}\n"
            f"  actual:   {lines[idx]!r}"
        )
    lines[idx] = bug["buggy_code"]
    return "\n".join(lines)
