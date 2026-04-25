"""Anti-cheating penalties for shortcut behavior."""
from __future__ import annotations

from typing import Dict, List, Tuple


def anti_cheat(
    tool_calls: List[Dict],
    submitted: bool,
    max_turns_reached: bool,
) -> Tuple[float, dict]:
    penalty = 0.0
    info: Dict = {"flags": []}

    used_read = any(c["tool_name"] == "read_code" for c in tool_calls)
    used_tests = any(c["tool_name"] == "run_tests" for c in tool_calls)

    if not used_read and not used_tests:
        penalty -= 0.2
        info["flags"].append("no_read_or_tests_before_submit")

    seen = set()
    repeats = 0
    for c in tool_calls:
        sig = (c["tool_name"], repr(c.get("tool_args")))
        if sig in seen:
            repeats += 1
        else:
            seen.add(sig)
    penalty -= 0.05 * repeats
    if repeats:
        info["flags"].append(f"repeated_tool_calls:{repeats}")

    if max_turns_reached and not submitted:
        penalty -= 0.3
        info["flags"].append("exceeded_max_turns")

    # apply_fix without ever running tests beforehand.
    fixed_before_testing = False
    saw_tests = False
    for c in tool_calls:
        if c["tool_name"] == "run_tests":
            saw_tests = True
        if c["tool_name"] == "apply_fix" and not saw_tests:
            fixed_before_testing = True
            break
    if fixed_before_testing:
        penalty -= 0.1
        info["flags"].append("apply_fix_before_running_tests")

    info["penalty"] = penalty
    return penalty, info
