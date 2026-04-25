"""Sandbox worker — runs in a fresh subprocess with restricted globals.

Communication protocol: parent writes a JSON request to stdin, worker
writes a single JSON response to stdout, then exits. The parent enforces
a wall-clock timeout via subprocess.run; if we run past it the parent
kills us. Inside the worker we additionally cap recursion depth.

Request: {
    "code": str,                 # function source
    "expressions": [str, ...]    # zero or more `func(...)` calls to eval
}

Response: {
    "compile_error": str | null,
    "results": [                 # one per expression in input order
        {"input": str, "ok": bool, "value": str | null, "error": str | null},
        ...
    ]
}

Worker also reads optional env var ARG_SANDBOX_RECURSION_LIMIT (default 200)
to cap stack depth so a malicious recursive function doesn't blow up the
host's default recursion limit.
"""
from __future__ import annotations

import ast
import builtins
import json
import os
import sys


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


def _blocked(*_a, **_kw):
    raise RuntimeError("operation blocked in sandbox")


def _build_safe_globals() -> dict:
    safe = {}
    for name in _SAFE_BUILTIN_NAMES:
        if hasattr(builtins, name):
            safe[name] = getattr(builtins, name)
    safe["__import__"] = _blocked
    safe["open"] = _blocked
    safe["compile"] = _blocked
    safe["eval"] = _blocked
    safe["exec"] = _blocked
    safe["globals"] = _blocked
    safe["locals"] = _blocked
    safe["vars"] = _blocked
    safe["getattr"] = _safe_getattr
    return safe


_BLOCKED_ATTRS = {
    # Common escape vectors via __class__.__base__.__subclasses__() etc.
    "__class__", "__bases__", "__subclasses__", "__mro__",
    "__globals__", "__code__", "__builtins__", "__import__",
    "__getattribute__", "__reduce__", "__reduce_ex__",
    "func_globals", "gi_frame", "f_globals", "f_locals",
    "__init_subclass__", "__subclasshook__",
}

_BLOCKED_NAMES = {
    "open", "exec", "eval", "compile", "__import__",
    "globals", "locals", "vars", "memoryview",
}


def _safe_getattr(obj, name, *default):
    if name in _BLOCKED_ATTRS:
        raise AttributeError(f"access to {name!r} is blocked in the sandbox")
    if default:
        return getattr(obj, name, default[0])
    return getattr(obj, name)


def _ast_check(source: str) -> str | None:
    """Walk the AST and reject dangerous attribute / name patterns.

    Returns an error string if the code is rejected, else None. We do
    this BEFORE compile() so that even if the runtime ``getattr`` shim
    is bypassed (e.g. via dot-notation, which Python resolves
    internally), the dangerous code never runs.
    """
    try:
        tree = ast.parse(source, mode="exec")
    except SyntaxError:
        # Let the real compile() report the syntax error to the parent.
        return None
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            if node.attr in _BLOCKED_ATTRS:
                return f"blocked attribute access: {node.attr!r}"
        if isinstance(node, ast.Name):
            if node.id in _BLOCKED_NAMES:
                return f"blocked name reference: {node.id!r}"
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            return "import statements are blocked in the sandbox"
    return None


def _value_to_str(val):
    if isinstance(val, str):
        return val
    return repr(val)


def _apply_resource_limits() -> None:
    """Cap address space, CPU time, and file size so adversarial code
    can't OOM the host or write large files. POSIX-only; on Windows
    these limits silently no-op (sandbox falls back to subprocess wall
    timeout for protection)."""
    try:
        import resource
    except ImportError:
        return
    try:
        mem_mb = int(os.environ.get("ARG_SANDBOX_MEM_MB", "512"))
        # RLIMIT_AS is virtual address space — covers process-wide allocation.
        resource.setrlimit(resource.RLIMIT_AS, (mem_mb * 1024 * 1024, mem_mb * 1024 * 1024))
    except (ValueError, OSError):
        pass
    try:
        cpu_s = int(os.environ.get("ARG_SANDBOX_CPU_S", "10"))
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_s, cpu_s))
    except (ValueError, OSError):
        pass
    try:
        # No filesystem writes from the worker.
        resource.setrlimit(resource.RLIMIT_FSIZE, (0, 0))
    except (ValueError, OSError):
        pass


def main() -> int:
    _apply_resource_limits()

    try:
        payload = json.loads(sys.stdin.read())
    except json.JSONDecodeError as e:
        sys.stdout.write(json.dumps({"compile_error": f"invalid request: {e}", "results": []}))
        return 1

    code = payload.get("code", "")
    expressions = payload.get("expressions", [])

    sys.setrecursionlimit(int(os.environ.get("ARG_SANDBOX_RECURSION_LIMIT", "200")))

    safe_globals = {"__builtins__": _build_safe_globals()}

    response = {"compile_error": None, "results": []}

    ast_error = _ast_check(code)
    if ast_error is not None:
        response["compile_error"] = ast_error
        for expr in expressions:
            response["results"].append(
                {"input": expr, "ok": False, "value": None, "error": ast_error}
            )
        sys.stdout.write(json.dumps(response))
        return 0

    try:
        compiled = compile(code, "<sandbox>", "exec")
        exec(compiled, safe_globals)
    except Exception as e:
        response["compile_error"] = f"{type(e).__name__}: {e}"
        # Still emit one error result per expression so the parent can
        # surface them uniformly.
        for expr in expressions:
            response["results"].append(
                {"input": expr, "ok": False, "value": None,
                 "error": response["compile_error"]}
            )
        sys.stdout.write(json.dumps(response))
        return 0

    for expr in expressions:
        expr_err = _ast_check(expr)
        if expr_err is not None:
            response["results"].append(
                {"input": expr, "ok": False, "value": None, "error": expr_err}
            )
            continue
        try:
            val = eval(expr, safe_globals)
            response["results"].append(
                {"input": expr, "ok": True, "value": _value_to_str(val), "error": None}
            )
        except Exception as e:
            response["results"].append(
                {"input": expr, "ok": False, "value": None,
                 "error": f"{type(e).__name__}: {e}"}
            )

    sys.stdout.write(json.dumps(response))
    return 0


if __name__ == "__main__":
    sys.exit(main())
