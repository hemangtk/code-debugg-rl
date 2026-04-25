"""Sandboxed Python code execution engine.

Runs the buggy function code in a fresh subprocess with restricted
globals. Two reasons we use a subprocess rather than threading:

1. Hard kill — daemon threads cannot be terminated, so an infinite-loop
   bug pinned to a thread keeps running after `t.join(timeout)` returns.
   In long-running servers this leaks threads; on container exit the
   process can hang. ``subprocess.run(..., timeout=...)`` raises
   ``TimeoutExpired`` and the parent kills the child cleanly.
2. Sandbox isolation — the worker process cannot read host secrets
   (`env={}`), cannot reach the project directory (`cwd=tempdir`), and
   the ``getattr``/builtins shim blocks ``__class__.__subclasses__``
   chains that the in-thread sandbox couldn't.

If subprocess startup latency becomes a bottleneck, set
``ARG_SANDBOX_THREADED=1`` to fall back to the legacy in-thread runner —
faster but unsafe in shared deployments. The threaded path remains for
local CI smoke tests and is *not* what we ship to HF Spaces.
"""
from __future__ import annotations

import builtins
import json
import os
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from server.models import TestCase, TestResult


_WORKER_PATH = Path(__file__).parent / "_sandbox_worker.py"


def _use_threaded_sandbox() -> bool:
    return os.environ.get("ARG_SANDBOX_THREADED", "0") == "1"


# ---------------------------------------------------------------------------
# Subprocess sandbox (default).
# ---------------------------------------------------------------------------


def _run_subprocess(code: str, expressions: List[str], timeout_s: float
                    ) -> Tuple[Optional[str], List[Dict[str, Any]]]:
    """Run code + zero or more expressions in a worker subprocess.

    Returns (compile_error, results). On wall-clock timeout returns
    ('TIMEOUT', [{'ok': False, 'error': 'TIMEOUT'}, ...]).
    """
    payload = json.dumps({"code": code, "expressions": expressions})
    try:
        proc = subprocess.run(
            [sys.executable, str(_WORKER_PATH)],
            input=payload,
            capture_output=True,
            text=True,
            timeout=max(0.1, timeout_s),
            env={
                "PATH": os.environ.get("PATH", ""),
                "ARG_SANDBOX_RECURSION_LIMIT": "200",
                "ARG_SANDBOX_MEM_MB": os.environ.get("ARG_SANDBOX_MEM_MB", "512"),
                "ARG_SANDBOX_CPU_S": os.environ.get("ARG_SANDBOX_CPU_S", "10"),
            },
            check=False,
        )
    except subprocess.TimeoutExpired:
        return ("TIMEOUT after %.2fs" % timeout_s,
                [{"input": e, "ok": False, "value": None, "error": "TIMEOUT"} for e in expressions])
    except FileNotFoundError as e:
        return (f"sandbox launch failed: {e}",
                [{"input": e, "ok": False, "value": None, "error": "LAUNCH_FAILED"} for e in expressions])

    if proc.returncode != 0 and not proc.stdout:
        return (f"sandbox crashed (rc={proc.returncode}): {proc.stderr.strip()[:200]}",
                [{"input": e, "ok": False, "value": None, "error": "WORKER_CRASH"} for e in expressions])

    try:
        data = json.loads(proc.stdout)
    except json.JSONDecodeError:
        return (f"invalid worker output: {proc.stdout[:200]}",
                [{"input": e, "ok": False, "value": None, "error": "WORKER_BAD_JSON"} for e in expressions])

    return data.get("compile_error"), data.get("results", [])


# ---------------------------------------------------------------------------
# In-thread sandbox (legacy, opt-in via env var). Faster but cannot kill
# infinite loops and is vulnerable to __class__.__subclasses__ escape.
# ---------------------------------------------------------------------------


_SAFE_BUILTIN_NAMES = {
    "abs", "all", "any", "bool", "bytes", "chr", "dict", "divmod",
    "enumerate", "filter", "float", "frozenset", "hash", "hex", "int",
    "isinstance", "issubclass", "iter", "len", "list", "map", "max", "min",
    "next", "object", "oct", "ord", "pow", "print", "range", "repr",
    "reversed", "round", "set", "slice", "sorted", "str", "sum", "tuple",
    "type", "zip",
    "True", "False", "None",
    "Exception", "ValueError", "TypeError", "IndexError", "KeyError",
    "ZeroDivisionError", "StopIteration", "ArithmeticError", "RuntimeError",
}


def _build_safe_thread_builtins() -> Dict[str, Any]:
    safe: Dict[str, Any] = {}
    for name in _SAFE_BUILTIN_NAMES:
        if hasattr(builtins, name):
            safe[name] = getattr(builtins, name)
    def _blocked(*_a, **_kw):
        raise RuntimeError("operation blocked in sandbox")
    safe["__import__"] = _blocked
    safe["open"] = _blocked
    return safe


def _run_threaded(code: str, expressions: List[str], timeout_s: float
                  ) -> Tuple[Optional[str], List[Dict[str, Any]]]:
    holder: Dict[str, Any] = {"results": [], "compile_error": None}

    def worker() -> None:
        ns = {"__builtins__": _build_safe_thread_builtins()}
        try:
            exec(code, ns)
        except Exception as e:
            holder["compile_error"] = f"{type(e).__name__}: {e}"
            for expr in expressions:
                holder["results"].append(
                    {"input": expr, "ok": False, "value": None,
                     "error": holder["compile_error"]}
                )
            return
        for expr in expressions:
            try:
                val = eval(expr, ns)
                value = val if isinstance(val, str) else repr(val)
                holder["results"].append(
                    {"input": expr, "ok": True, "value": value, "error": None}
                )
            except Exception as e:
                holder["results"].append(
                    {"input": expr, "ok": False, "value": None,
                     "error": f"{type(e).__name__}: {e}"}
                )

    t = threading.Thread(target=worker, daemon=True)
    t.start()
    t.join(timeout_s)
    if t.is_alive():
        # Pad missing entries with TIMEOUT — the thread keeps running but
        # will be reaped on process exit.
        already = {r["input"] for r in holder["results"]}
        for expr in expressions:
            if expr not in already:
                holder["results"].append(
                    {"input": expr, "ok": False, "value": None, "error": "TIMEOUT"}
                )
        return ("TIMEOUT after %.2fs" % timeout_s, holder["results"])
    return holder["compile_error"], holder["results"]


def _run_sandbox(code: str, expressions: List[str], timeout_s: float
                 ) -> Tuple[Optional[str], List[Dict[str, Any]]]:
    if _use_threaded_sandbox():
        return _run_threaded(code, expressions, timeout_s)
    return _run_subprocess(code, expressions, timeout_s)


# ---------------------------------------------------------------------------
# Test result comparison — float-tolerant.
# ---------------------------------------------------------------------------


def _values_equal(actual: str, expected: str, tol: float = 1e-6) -> bool:
    """Compare actual (string from sandbox) to expected (string in template).

    Exact string match wins. For floats and float-containing structures we
    also try numeric eval and tolerance-compare so that 1.0000000000001 vs
    1.0 doesn't flake across platforms.
    """
    if actual == expected:
        return True
    try:
        a = _safe_literal(actual)
        e = _safe_literal(expected)
    except Exception:
        return False
    return _numeric_close(a, e, tol)


def _safe_literal(text: str) -> Any:
    import ast
    return ast.literal_eval(text)


def _numeric_close(a: Any, b: Any, tol: float) -> bool:
    if isinstance(a, float) or isinstance(b, float):
        try:
            return abs(float(a) - float(b)) <= tol
        except (TypeError, ValueError):
            return False
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            return False
        return all(_numeric_close(x, y, tol) for x, y in zip(a, b))
    if isinstance(a, dict) and isinstance(b, dict):
        if a.keys() != b.keys():
            return False
        return all(_numeric_close(a[k], b[k], tol) for k in a)
    return a == b


class CodeRunner:
    """Executes a single user-supplied Python function in a sandbox."""

    def __init__(
        self,
        function_name: str,
        code: str,
        test_cases: List[TestCase],
        timeout_s: float = 5.0,
    ) -> None:
        self.function_name = function_name
        self.code = code
        self.test_cases = test_cases
        self.timeout_s = timeout_s

    def read_code(self) -> str:
        lines = self.code.split("\n")
        width = len(str(len(lines)))
        return "\n".join(
            f"{str(i + 1).rjust(width)}  {line}" for i, line in enumerate(lines)
        )

    def run_code(self, expression: str) -> str:
        compile_err, results = _run_sandbox(self.code, [expression], self.timeout_s)
        if compile_err and not results:
            return f"ERROR: {compile_err}"
        r = results[0]
        if r["ok"]:
            return r["value"]
        if r["error"] == "TIMEOUT":
            return f"TIMEOUT after {self.timeout_s}s"
        return f"ERROR: {r['error']}"

    def run_tests(self) -> Tuple[str, List[TestResult]]:
        budget = max(self.timeout_s, self.timeout_s * len(self.test_cases))
        expressions = [tc.input for tc in self.test_cases]
        compile_err, raw = _run_sandbox(self.code, expressions, budget)

        # Map raw results back to TestResult, preserving order.
        results: List[TestResult] = []
        by_input = {r["input"]: r for r in raw}
        for tc in self.test_cases:
            r = by_input.get(tc.input)
            if r is None:
                results.append(
                    TestResult(input=tc.input, expected=tc.expected,
                               actual="", passed=False,
                               error="missing result from sandbox")
                )
                continue
            if not r["ok"]:
                err = r["error"] or "unknown error"
                actual_marker = "TIMEOUT" if err == "TIMEOUT" else "ERROR"
                results.append(
                    TestResult(input=tc.input, expected=tc.expected,
                               actual=actual_marker, passed=False, error=err)
                )
                continue
            value = r["value"] or ""
            passed = _values_equal(value, tc.expected) or value == tc.expected
            results.append(
                TestResult(input=tc.input, expected=tc.expected,
                           actual=value, passed=passed)
            )
        return self._format_results(results), results

    @staticmethod
    def _format_results(results: List[TestResult]) -> str:
        lines = []
        for i, r in enumerate(results, 1):
            mark = "PASS" if r.passed else "FAIL"
            actual = r.actual if r.actual else "<no output>"
            if r.error:
                lines.append(
                    f"Test {i}: {r.input} -> expected: {r.expected}, error: {r.error} [{mark}]"
                )
            else:
                lines.append(
                    f"Test {i}: {r.input} -> expected: {r.expected}, actual: {actual} [{mark}]"
                )
        passed = sum(1 for r in results if r.passed)
        total = len(results)
        lines.append(f"\n{passed}/{total} tests passed.")
        return "\n".join(lines)

    def apply_fix(self, line_number: int, new_code: str) -> str:
        lines = self.code.split("\n")
        if line_number < 1 or line_number > len(lines):
            return f"ERROR: line {line_number} is out of range (1..{len(lines)})"
        lines[line_number - 1] = new_code
        self.code = "\n".join(lines)
        return f"Line {line_number} updated. Use run_tests() to verify your fix."

    def submit_fix(self) -> Tuple[str, List[TestResult], bool]:
        formatted, results = self.run_tests()
        all_pass = bool(results) and all(r.passed for r in results)
        if all_pass:
            msg = f"All {len(results)} tests passed! Fix verified."
        else:
            passed = sum(1 for r in results if r.passed)
            msg = f"{passed}/{len(results)} tests passed. Fix incomplete.\n\n" + formatted
        return msg, results, all_pass
