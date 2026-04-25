"""Combine templates, bugs, and misconceptions into a Scenario."""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, List, Optional

from server.misconception_engine import (
    get_wrong_reason,
    render_followups,
    render_initial,
)
from server.models import (
    BugDefinition,
    DifficultyConfig,
    GradingCriteria,
    Scenario,
    TestCase,
)
from server.generator.bug_injector import inject_bug
from server.generator.test_generator import (
    actual_for_input,
    evaluate_tests,
    expected_for_input,
)

_TEMPLATE_DIR = Path(__file__).parent.parent / "templates"


def _load_functions() -> Dict:
    with open(_TEMPLATE_DIR / "functions.json") as f:
        return json.load(f)


_FUNCTIONS = _load_functions()


class CodeDebuggerGenerator:
    def __init__(self, seed: Optional[int] = None) -> None:
        self.rng = random.Random(seed)

    def generate(self, difficulty: DifficultyConfig) -> Scenario:
        # 1. Pick a function template, biased by code complexity.
        names = list(_FUNCTIONS.keys())
        if difficulty.code_complexity > 0.6:
            # Favor higher-line-count templates.
            names.sort(key=lambda n: -_FUNCTIONS[n]["lines"])
            names = names[: max(5, len(names) // 2)]
        template = _FUNCTIONS[self.rng.choice(names)]

        # 2. Pick a compatible bug.
        bug_dict = self.rng.choice(template["compatible_bugs"])
        buggy_code = inject_bug(template["correct_code"], bug_dict)

        # 3. Resolve failing inputs under the buggy code.
        failing_inputs, _ = evaluate_tests(
            template["name"], buggy_code, template["test_cases"]
        )
        if not failing_inputs:
            # Defensive: every bug should fail at least one test, but if a
            # template/bug combo slipped through, retry with a different bug.
            return self.generate(difficulty)

        failing_input = self.rng.choice(failing_inputs)
        expected_output = expected_for_input(template["test_cases"], failing_input)
        wrong_output = actual_for_input(
            template["name"], buggy_code, failing_input
        )

        # 4. Decide if the user is right or wrong.
        user_is_correct = self.rng.random() < difficulty.correct_user_ratio
        if user_is_correct:
            wrong_line = bug_dict["bug_line"]
        else:
            candidates = [
                ln
                for ln in template["wrong_lines"]
                if ln != bug_dict["bug_line"]
            ]
            if not candidates:
                # Fall back to any line that isn't the bug line.
                candidates = [
                    ln
                    for ln in range(1, template["lines"] + 1)
                    if ln != bug_dict["bug_line"]
                ]
            wrong_line = self.rng.choice(candidates)

        wrong_reason = get_wrong_reason(template["name"], wrong_line)

        # 5. Render messages.
        initial_message = render_initial(
            function_name=template["name"],
            wrong_line=wrong_line,
            wrong_reason=wrong_reason,
            failing_input=failing_input,
            wrong_output=wrong_output,
            expected_output=expected_output,
            convincingness=difficulty.misconception_convincingness,
            rng=self.rng,
            user_is_correct=user_is_correct,
        )
        followups = render_followups(
            function_name=template["name"],
            wrong_line=wrong_line,
            bug_line=bug_dict["bug_line"],
            wrong_reason=wrong_reason,
            followup_count=difficulty.followup_count,
            rng=self.rng,
        )

        # 6. Triggers describe when each follow-up is delivered.
        triggers = ["after_run_tests", "after_investigate_bug_area", "emotional"]
        followup_triggers = triggers[: len(followups)]

        # 7. Wrap it up.
        bug_def = BugDefinition(
            bug_type=bug_dict["bug_type"],
            bug_line=bug_dict["bug_line"],
            original_code=bug_dict["original_code"],
            buggy_code=bug_dict["buggy_code"],
            explanation=bug_dict["explanation"],
        )
        grading = GradingCriteria(
            bug_line=bug_dict["bug_line"],
            wrong_line=wrong_line,
            correct_code=template["correct_code"],
            user_was_correct=user_is_correct,
            failing_inputs=failing_inputs,
        )
        return Scenario(
            function_name=template["name"],
            correct_code=template["correct_code"],
            buggy_code=buggy_code,
            test_cases=[TestCase(**tc) for tc in template["test_cases"]],
            bug=bug_def,
            wrong_line=wrong_line,
            wrong_reason=wrong_reason,
            initial_message=initial_message,
            followup_messages=followups,
            followup_triggers=followup_triggers,
            grading=grading,
            difficulty=difficulty,
            user_was_correct=user_is_correct,
        )
