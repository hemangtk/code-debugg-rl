"""Sandbox safety tests. Run with `pytest tests/`."""
from __future__ import annotations

from server.models import TestCase
from server.tools.code_runner import CodeRunner


def _run(code: str, expr: str, expected: str = "0", timeout: float = 1.0):
    runner = CodeRunner("f", code, [TestCase(input=expr, expected=expected)], timeout_s=timeout)
    return runner.run_tests()[1][0]


def test_normal_function_passes():
    code = "def f(x):\n    return x + 1"
    r = _run(code, "f(2)", expected="3")
    assert r.passed
    assert r.actual == "3"


def test_import_blocked():
    code = "def f():\n    import os\n    return os.listdir('/')"
    r = _run(code, "f()", expected="[]")
    assert not r.passed
    assert "blocked" in (r.error or "").lower()


def test_open_blocked():
    code = "def f():\n    return open('/etc/passwd').read()"
    r = _run(code, "f()", expected="")
    assert not r.passed
    assert "blocked" in (r.error or "").lower()


def test_class_escape_blocked():
    code = "def f():\n    return ().__class__.__base__.__subclasses__()"
    r = _run(code, "f()", expected="[]")
    assert not r.passed, "subclasses escape leaked"


def test_infinite_loop_killed():
    code = "def f():\n    while True:\n        pass"
    r = _run(code, "f()", expected="None", timeout=0.4)
    assert not r.passed
    assert r.actual == "TIMEOUT"


def test_recursion_does_not_blow_host():
    code = "def f(n):\n    return f(n + 1)"
    r = _run(code, "f(0)", expected="None", timeout=2.0)
    assert not r.passed
    # Either RecursionError or runs into the worker's setrecursionlimit cap.


def test_float_tolerance_pass():
    code = "def f():\n    return 0.1 + 0.2"
    r = _run(code, "f()", expected="0.3")
    assert r.passed


def test_float_tolerance_fail():
    code = "def f():\n    return 1.0"
    r = _run(code, "f()", expected="2.0")
    assert not r.passed


def test_apply_fix_modifies_code():
    code = "def f():\n    return 1"
    runner = CodeRunner("f", code, [TestCase(input="f()", expected="2")])
    runner.apply_fix(2, "    return 2")
    _, results = runner.run_tests()
    assert results[0].passed
