"""Generate natural-sounding user misconception messages for a scenario."""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, List

_TEMPLATE_DIR = Path(__file__).parent / "templates"


def _load_json(name: str) -> Dict:
    with open(_TEMPLATE_DIR / name) as f:
        return json.load(f)


_MISCONCEPTIONS = _load_json("misconceptions.json")
_WRONG_REASONS = _load_json("wrong_reasons.json")


def get_wrong_reason(function_name: str, line: int) -> str:
    """Look up the canned wrong reason for a function/line, or fabricate."""
    fn_reasons = _WRONG_REASONS.get(function_name, {})
    reason = fn_reasons.get(str(line))
    if reason:
        return reason
    return (
        f"the logic on line {line} looks suspicious to me — it doesn't "
        f"match what I'd expect for this kind of operation."
    )


def render_initial(
    function_name: str,
    wrong_line: int,
    wrong_reason: str,
    failing_input: str,
    wrong_output: str,
    expected_output: str,
    convincingness: float,
    rng: random.Random,
    user_is_correct: bool = False,
) -> str:
    if user_is_correct:
        templates = _MISCONCEPTIONS["correct_user_templates"]
    else:
        templates = _MISCONCEPTIONS["initial_templates"]
    # At higher convincingness, prefer longer/more detailed templates.
    if convincingness > 0.5 and not user_is_correct:
        templates = templates[:6]  # Longer ones tend to be the first six.
    template = rng.choice(templates)
    return template.format(
        func_name=function_name,
        wrong_line=wrong_line,
        wrong_reason=wrong_reason,
        failing_input=failing_input,
        wrong_output=wrong_output,
        expected_output=expected_output,
        bug_line=wrong_line,
    )


def render_followups(
    function_name: str,
    wrong_line: int,
    bug_line: int,
    wrong_reason: str,
    followup_count: int,
    rng: random.Random,
) -> List[str]:
    if followup_count <= 0:
        return []
    pushbacks = list(_MISCONCEPTIONS["pushback_templates"])
    rng.shuffle(pushbacks)
    chosen = pushbacks[: max(0, followup_count - 1)]
    messages = [
        t.format(
            func_name=function_name,
            wrong_line=wrong_line,
            bug_line=bug_line,
            wrong_reason=wrong_reason,
        )
        for t in chosen
    ]
    if followup_count >= 3:
        emo = rng.choice(_MISCONCEPTIONS["emotional_templates"])
        messages.append(
            emo.format(wrong_line=wrong_line, func_name=function_name)
        )
    return messages
