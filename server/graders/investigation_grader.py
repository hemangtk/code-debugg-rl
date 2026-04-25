"""Score the agent's investigation quality based on tool usage."""
from __future__ import annotations

from typing import Dict, List, Tuple

from server.models import Scenario


def grade_investigation(
    scenario: Scenario, tool_calls: List[Dict]
) -> Tuple[float, dict]:
    score = 0.0
    info: Dict = {"breakdown": {}}

    used_run_tests = any(c["tool_name"] == "run_tests" for c in tool_calls)
    used_read_code = any(c["tool_name"] == "read_code" for c in tool_calls)

    if used_run_tests:
        score += 0.05
        info["breakdown"]["run_tests"] = 0.05
    if used_read_code:
        score += 0.03
        info["breakdown"]["read_code"] = 0.03

    failing = set(scenario.grading.failing_inputs)
    run_code_failing = 0
    run_code_other = 0
    seen_failing_inputs: set = set()
    seen_other_inputs: set = set()
    for c in tool_calls:
        if c["tool_name"] == "run_code":
            arg = (c.get("tool_args") or {}).get("input_value", "")
            if not arg:
                continue
            if arg in failing:
                # Only credit each failing input once to prevent farming
                # by spamming run_code with the same args.
                if arg in seen_failing_inputs:
                    continue
                seen_failing_inputs.add(arg)
                run_code_failing += 1
                score += 0.05
            else:
                if arg in seen_other_inputs:
                    continue
                seen_other_inputs.add(arg)
                run_code_other += 1
                score += 0.02

    info["breakdown"]["run_code_failing"] = run_code_failing * 0.05
    info["breakdown"]["run_code_other"] = run_code_other * 0.02

    # Bonus: investigated near the actual bug line — but only credit it if
    # the agent's *final* fix attempt landed on the bug line and they did
    # NOT also touch the wrong line. This blocks the hack of touring the
    # bug line for the bonus, then caving by fixing the wrong line.
    bug_line = scenario.grading.bug_line
    wrong_line = scenario.grading.wrong_line
    fix_lines: List[int] = []
    for c in tool_calls:
        if c["tool_name"] == "apply_fix":
            ln = (c.get("tool_args") or {}).get("line_number")
            if isinstance(ln, int):
                fix_lines.append(ln)
    last_fix = fix_lines[-1] if fix_lines else None
    near_bug = (
        last_fix is not None
        and abs(last_fix - bug_line) <= 1
        and wrong_line not in fix_lines
    )
    if near_bug:
        score += 0.10
        info["breakdown"]["near_bug_bonus"] = 0.10

    # Penalty: redundant identical tool calls.
    seen = set()
    redundant = 0
    for c in tool_calls:
        sig = (c["tool_name"], repr(c.get("tool_args")))
        if sig in seen:
            redundant += 1
        else:
            seen.add(sig)
    score -= 0.03 * redundant
    info["breakdown"]["redundant_penalty"] = -0.03 * redundant
    info["redundant_calls"] = redundant

    score = max(-0.2, min(0.5, score))
    info["score"] = score
    return score, info
