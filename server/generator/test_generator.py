"""Determine which test cases pass/fail under correct vs buggy code.

Used by the scenario generator to compute the failing-input set so the
graders and the user-misconception engine can reference real failures.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

from server.models import TestCase
from server.tools.code_runner import CodeRunner


def evaluate_tests(
    function_name: str, code: str, test_cases: List[Dict]
) -> Tuple[List[str], List[str]]:
    """Run all test cases under `code` and return (failing_inputs, passing_inputs)."""
    tcs = [TestCase(**tc) for tc in test_cases]
    runner = CodeRunner(function_name, code, tcs, timeout_s=0.15)
    _, results = runner.run_tests()
    failing = [r.input for r in results if not r.passed]
    passing = [r.input for r in results if r.passed]
    return failing, passing


def first_failing_input(
    function_name: str, code: str, test_cases: List[Dict]
) -> str:
    """Return the first failing input under buggy code, or empty string."""
    failing, _ = evaluate_tests(function_name, code, test_cases)
    return failing[0] if failing else ""


def expected_for_input(test_cases: List[Dict], input_str: str) -> str:
    for tc in test_cases:
        if tc["input"] == input_str:
            return tc["expected"]
    return ""


def actual_for_input(function_name: str, buggy_code: str, input_str: str) -> str:
    runner = CodeRunner(function_name, buggy_code, [], timeout_s=0.15)
    return runner.run_code(input_str)
