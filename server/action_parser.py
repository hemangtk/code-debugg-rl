"""Robust LLM-output to Action parser.

Real LLM completions don't always match a rigid format. This parser
accepts the canonical:

    ACTION: tool_call
    TOOL: run_code
    ARGS: {"input_value": "f(1, 2)"}
    REASONING: ...

but also tolerates:

* Markdown code-block fences (``` or ```json) around ARGS
* Pretty-printed multi-line JSON with nested objects/strings
* Lower or upper case keywords, extra whitespace
* JSON-only completions (no ACTION header — we infer tool_call)
* "submit" with various phrasings (submit, submit_fix, finish, done)
* The full message in tag form: <action>tool_call</action> etc.

The parser is conservative: when in doubt it returns a "noop" tool call
(``read_code()``) rather than fabricating arguments. The caller can
detect malformed completions by checking ``Action.reasoning`` against a
``parse_warnings`` list returned alongside.
"""
from __future__ import annotations

import json
import re
from typing import Dict, List, Tuple

from server.models import Action


_VALID_TOOLS = {"read_code", "run_tests", "run_code", "apply_fix", "submit_fix"}


_CODE_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def _strip_fences(text: str) -> str:
    m = _CODE_FENCE_RE.search(text)
    return m.group(1) if m else text


def _find_balanced_json(text: str, start: int) -> str:
    """Return the JSON object substring starting at index ``start``.

    Walks brace depth so nested ``{...}`` inside string values don't end
    the match early. ``text[start]`` must be ``{``. Returns ``""`` if
    the braces don't balance.
    """
    if start >= len(text) or text[start] != "{":
        return ""
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(text)):
        ch = text[i]
        if esc:
            esc = False
            continue
        if ch == "\\" and in_str:
            esc = True
            continue
        if ch == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return ""


def _extract_args(text: str) -> Tuple[Dict, List[str]]:
    """Find an ``ARGS:`` line (or a bare JSON object) and parse it."""
    warnings: List[str] = []
    m = re.search(r"ARGS\s*:\s*", text, re.IGNORECASE)
    if m is not None:
        rest = text[m.end():].lstrip()
        rest = _strip_fences(rest)
        rest = rest.lstrip()
        if rest.startswith("{"):
            blob = _find_balanced_json(rest, 0)
            if blob:
                try:
                    return json.loads(blob), warnings
                except json.JSONDecodeError as e:
                    warnings.append(f"args_json_decode_error: {e}")

    # Fallback: any balanced JSON object anywhere in the text.
    for i, ch in enumerate(text):
        if ch == "{":
            blob = _find_balanced_json(text, i)
            if blob:
                try:
                    return json.loads(blob), warnings
                except json.JSONDecodeError:
                    continue
    return {}, warnings


_TOOL_NAME_PATTERNS = [
    re.compile(r"TOOL\s*:\s*([\w]+)", re.IGNORECASE),
    re.compile(r"<tool>\s*([\w]+)\s*</tool>", re.IGNORECASE),
    re.compile(r'"tool"\s*:\s*"([\w]+)"', re.IGNORECASE),
    re.compile(r'"tool_name"\s*:\s*"([\w]+)"', re.IGNORECASE),
]

_ACTION_TYPE_PATTERNS = [
    re.compile(r"ACTION\s*:\s*([\w_]+)", re.IGNORECASE),
    re.compile(r"<action>\s*([\w_]+)\s*</action>", re.IGNORECASE),
    re.compile(r'"action_type"\s*:\s*"([\w_]+)"', re.IGNORECASE),
]

_SUBMIT_KEYWORDS = ("submit", "finish", "done", "final answer", "submit_fix")


def parse_action(text: str) -> Tuple[Action, List[str]]:
    """Parse an LLM completion. Returns (Action, warnings)."""
    warnings: List[str] = []
    if not isinstance(text, str):
        warnings.append("non_string_input")
        return Action(action_type="tool_call", tool_name="read_code", tool_args={},
                      reasoning=""), warnings

    # 1. Detect tool name.
    tool_name = None
    for pat in _TOOL_NAME_PATTERNS:
        m = pat.search(text)
        if m:
            cand = m.group(1).lower()
            if cand in _VALID_TOOLS:
                tool_name = cand
                break
            else:
                warnings.append(f"invalid_tool_in_text: {cand!r}")

    # 2. Detect action_type.
    action_type = None
    for pat in _ACTION_TYPE_PATTERNS:
        m = pat.search(text)
        if m:
            cand = m.group(1).lower()
            if cand in ("tool_call", "submit"):
                action_type = cand
                break
            warnings.append(f"invalid_action_in_text: {cand!r}")

    # 3. Heuristic fallback — keyword scan for submit-like intent.
    if action_type is None:
        head = text.lower()[:300]
        if any(kw in head for kw in _SUBMIT_KEYWORDS) and tool_name != "submit_fix":
            # Be careful not to misread "submit_fix is the next thing I want to call" as a submit.
            for kw in _SUBMIT_KEYWORDS:
                if re.search(rf"\b{re.escape(kw)}\b", head):
                    action_type = "submit"
                    warnings.append(f"inferred_submit_from_keyword:{kw}")
                    break

    # 4. Tool fallback when we have a clear non-submit shape but no tool name.
    if action_type is None and tool_name is None:
        # Look for bare function calls like "run_code(...)".
        m = re.search(r"\b(run_tests|read_code|run_code|apply_fix|submit_fix)\s*\(", text)
        if m:
            tool_name = m.group(1)
            action_type = "submit" if tool_name == "submit_fix" else "tool_call"
        else:
            warnings.append("no_action_detected")
            action_type = "tool_call"
            tool_name = "read_code"

    # 5. Resolve combined cases.
    if tool_name == "submit_fix":
        action_type = "submit"
    if action_type == "submit":
        return Action(action_type="submit", reasoning=text[:500]), warnings
    if action_type is None:
        action_type = "tool_call"
    if tool_name is None:
        tool_name = "read_code"
        warnings.append("defaulted_to_read_code")

    # 6. Args (only meaningful for run_code / apply_fix).
    args, arg_warnings = _extract_args(text)
    warnings.extend(arg_warnings)

    # 7. Validate args per tool.
    if tool_name == "run_code" and "input_value" not in args:
        warnings.append("run_code_missing_input_value")
    if tool_name == "apply_fix":
        if "line_number" not in args:
            warnings.append("apply_fix_missing_line_number")
        elif not isinstance(args["line_number"], int):
            try:
                args["line_number"] = int(args["line_number"])
            except (TypeError, ValueError):
                warnings.append("apply_fix_line_number_not_int")
        if "new_code" not in args:
            warnings.append("apply_fix_missing_new_code")
            args["new_code"] = ""

    return Action(
        action_type="tool_call",
        tool_name=tool_name,
        tool_args=args,
        reasoning=text[:500],
    ), warnings
