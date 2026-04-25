"""Robust action parser tests."""
from __future__ import annotations

from server.action_parser import parse_action


def test_canonical_format():
    text = 'ACTION: tool_call\nTOOL: run_tests\nARGS: {}\nREASONING: x'
    a, w = parse_action(text)
    assert a.action_type == "tool_call"
    assert a.tool_name == "run_tests"
    assert a.tool_args == {}


def test_apply_fix_with_nested_braces_in_new_code():
    text = 'ACTION: tool_call\nTOOL: apply_fix\nARGS: {"line_number": 5, "new_code": "x = {1: 2}"}\nREASONING: try'
    a, w = parse_action(text)
    assert a.tool_name == "apply_fix"
    assert a.tool_args["line_number"] == 5
    assert a.tool_args["new_code"] == "x = {1: 2}"


def test_args_in_code_fence():
    text = 'ACTION: tool_call\nTOOL: run_code\nARGS:\n```json\n{"input_value": "f(1, 2)"}\n```\nREASONING: x'
    a, w = parse_action(text)
    assert a.tool_args == {"input_value": "f(1, 2)"}


def test_submit_keyword_inferred():
    text = "I think I have it. Let me submit."
    a, w = parse_action(text)
    assert a.action_type == "submit"
    assert any("inferred_submit" in str(x) for x in w)


def test_json_only_completion():
    text = '{"action_type": "tool_call", "tool_name": "read_code", "tool_args": {}}'
    a, w = parse_action(text)
    assert a.action_type == "tool_call"
    assert a.tool_name == "read_code"


def test_invalid_tool_name_falls_back():
    text = 'ACTION: tool_call\nTOOL: hack_me\nARGS: {}\nREASONING: x'
    a, w = parse_action(text)
    assert a.tool_name == "read_code"
    assert any("invalid_tool" in str(x) for x in w)


def test_garbage_falls_back_safely():
    text = "lol idk"
    a, w = parse_action(text)
    assert a.action_type == "tool_call"
    assert a.tool_name == "read_code"


def test_multiline_args_preserved():
    text = 'ACTION: tool_call\nTOOL: apply_fix\nARGS: {"line_number": 4, "new_code": "for i in range(\\n    10\\n):\\n    print(i)"}'
    a, w = parse_action(text)
    assert a.tool_args["line_number"] == 4
    assert "range" in a.tool_args["new_code"]


def test_submit_fix_routes_to_submit_action_type():
    text = 'ACTION: tool_call\nTOOL: submit_fix\nARGS: {}\nREASONING: done'
    a, w = parse_action(text)
    assert a.action_type == "submit"


def test_run_code_missing_input_warns():
    text = 'ACTION: tool_call\nTOOL: run_code\nARGS: {}\nREASONING: x'
    a, w = parse_action(text)
    assert any("run_code_missing_input_value" in str(x) for x in w)
