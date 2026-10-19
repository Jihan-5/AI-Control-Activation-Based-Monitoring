"""Restricted, deterministic execution of *our own generated* candidate code.

Candidate implementations in the program environment are produced by this
repository's templates, never by the actor, so this sandbox is a correctness
harness, not a security boundary for untrusted code. It still refuses anything
outside a small, explicit subset of Python so a template bug cannot reach the
filesystem, imports, or dunder machinery:

* AST whitelist: one top-level ``def``; simple statements, loops, comparisons,
  comprehensions, slicing. No imports, ``global``, ``lambda``, ``with``, ``try``,
  ``class``, ``yield``, f-strings, or deletion.
* Names: no dunder names; no ``eval``/``exec``/``open``/``getattr``-style
  builtins. Attributes: only a whitelist of list/str/set methods.
* Builtins: an explicit minimal table (``len``, ``range``, ``sorted`` ...).
* Step budget: a ``sys.settrace`` line counter aborts runaway loops
  (``SandboxTimeout``), deterministically and without signals or threads.

Arguments are deep-copied per call so a mutating candidate cannot change the
test inputs seen by another candidate.
"""

from __future__ import annotations

import ast
import copy
import sys
from dataclasses import dataclass
from typing import Any

FILENAME = "<candidate>"

ALLOWED_NODES: tuple[type, ...] = (
    ast.Module, ast.FunctionDef, ast.arguments, ast.arg, ast.Return, ast.Assign,
    ast.AugAssign, ast.For, ast.While, ast.If, ast.Expr, ast.Call, ast.Name, ast.Load,
    ast.Store, ast.Constant, ast.BinOp, ast.UnaryOp, ast.BoolOp, ast.Compare, ast.List,
    ast.Tuple, ast.Dict, ast.Set, ast.Subscript, ast.Slice, ast.Attribute, ast.ListComp,
    ast.comprehension, ast.IfExp, ast.Pass, ast.Break, ast.Continue, ast.keyword,
    # operators
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.USub, ast.UAdd,
    ast.Not, ast.And, ast.Or, ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE,
    ast.In, ast.NotIn, ast.Is, ast.IsNot,
)

ALLOWED_ATTRIBUTES: frozenset[str] = frozenset({
    "append", "add", "reverse", "lower", "upper", "split", "join", "extend", "pop",
    "insert", "count", "strip", "isalpha", "isdigit", "startswith", "endswith",
})

BANNED_NAMES: frozenset[str] = frozenset({
    "eval", "exec", "open", "compile", "globals", "locals", "vars", "getattr", "setattr",
    "delattr", "input", "breakpoint", "help", "dir", "type", "object", "super", "memoryview",
    "bytearray", "bytes", "classmethod", "staticmethod", "property", "exit", "quit",
})

SAFE_BUILTINS: dict[str, Any] = {
    "len": len, "range": range, "reversed": reversed, "sorted": sorted, "list": list,
    "set": set, "dict": dict, "tuple": tuple, "str": str, "int": int, "sum": sum,
    "max": max, "min": min, "abs": abs, "enumerate": enumerate, "zip": zip, "bool": bool,
}

DEFAULT_MAX_STEPS = 20_000


class SandboxViolation(ValueError):
    """Source uses a construct outside the whitelist."""


class SandboxTimeout(RuntimeError):
    """Execution exceeded the line-step budget."""


@dataclass(frozen=True)
class TestOutcome:
    passed: bool
    value: Any
    error: str | None = None


def check_source(source: str) -> tuple[ast.Module, str]:
    """Parse and validate; return (tree, function name)."""
    try:
        tree = ast.parse(source, filename=FILENAME, mode="exec")
    except SyntaxError as exc:
        raise SandboxViolation(f"syntax error: {exc}") from exc
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef):
        raise SandboxViolation("source must contain exactly one top-level function definition")
    fn = tree.body[0]
    if fn.decorator_list or fn.args.vararg or fn.args.kwarg or fn.returns is not None:
        raise SandboxViolation("decorators, *args/**kwargs and annotations are not allowed")
    for node in ast.walk(tree):
        if not isinstance(node, ALLOWED_NODES):
            raise SandboxViolation(f"disallowed syntax: {type(node).__name__}")
        if isinstance(node, ast.Name):
            if node.id.startswith("__") or node.id in BANNED_NAMES:
                raise SandboxViolation(f"disallowed name: {node.id}")
        if isinstance(node, ast.arg) and node.arg.startswith("__"):
            raise SandboxViolation(f"disallowed argument name: {node.arg}")
        if isinstance(node, ast.FunctionDef) and node is not fn:
            raise SandboxViolation("nested function definitions are not allowed")
        if isinstance(node, ast.Attribute):
            if node.attr.startswith("_") or node.attr not in ALLOWED_ATTRIBUTES:
                raise SandboxViolation(f"disallowed attribute: .{node.attr}")
        if isinstance(node, ast.Constant) and isinstance(node.value, (bytes, complex)):
            raise SandboxViolation("bytes/complex constants are not allowed")
    return tree, fn.name


def load_function(source: str):
    """Validate, compile and return the candidate function object."""
    tree, name = check_source(source)
    namespace: dict[str, Any] = {}
    code = compile(tree, FILENAME, "exec")
    exec(code, {"__builtins__": dict(SAFE_BUILTINS)}, namespace)  # noqa: S102 - whitelisted AST only
    return namespace[name]


def call_with_budget(fn, args: tuple, max_steps: int = DEFAULT_MAX_STEPS) -> Any:
    """Call ``fn(*deepcopy(args))`` aborting after ``max_steps`` traced lines."""
    steps = 0

    def local_trace(frame, event, arg):
        nonlocal steps
        if event == "line":
            steps += 1
            if steps > max_steps:
                raise SandboxTimeout(f"exceeded {max_steps} steps")
        return local_trace

    def global_trace(frame, event, arg):
        if frame.f_code.co_filename == FILENAME:
            return local_trace
        return None

    previous = sys.gettrace()
    sys.settrace(global_trace)
    try:
        return fn(*copy.deepcopy(args))
    finally:
        sys.settrace(previous)


def run_tests(
    source: str,
    cases: list[tuple[tuple, Any]],
    max_steps: int = DEFAULT_MAX_STEPS,
) -> list[TestOutcome]:
    """Run ``source`` on ``(args, expected)`` cases. Runtime errors and timeouts
    are failures; AST violations raise ``SandboxViolation``."""
    fn = load_function(source)
    out = []
    for args, expected in cases:
        try:
            value = call_with_budget(fn, args, max_steps)
        except SandboxTimeout as exc:
            out.append(TestOutcome(False, None, f"timeout: {exc}"))
            continue
        except Exception as exc:  # noqa: BLE001 - a failing candidate is data, not a crash
            out.append(TestOutcome(False, None, f"{type(exc).__name__}: {exc}"))
            continue
        out.append(TestOutcome(_same(value, expected), value))
    return out


def _same(a: Any, b: Any) -> bool:
    """Type-sensitive equality (``None`` != ``[]`` != ``0`` != ``''``)."""
    if type(a) is not type(b):
        return False
    return a == b
