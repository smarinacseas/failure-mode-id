"""Precision-pool verifiers (CAUSE_B archetype: exact execution — counts,
arithmetic, ordering, exact repetition). Importing this module registers them.

These are the mechanically-checkable constraints RLVR can reward directly; the
count/arithmetic/repetition checks are deliberately strict (off-by-one fails).
"""
from __future__ import annotations

import ast
import operator
import re

from base import CheckResult, register

# --- shared helpers ----------------------------------------------------------

_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
        ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv,
        ast.Mod: operator.mod, ast.Pow: operator.pow,
        ast.USub: operator.neg, ast.UAdd: operator.pos}


def _safe_eval(expr: str):
    """Evaluate a pure-arithmetic expression.

    SECURITY: this does NOT call Python's eval(). It parses to an AST and walks
    it with an explicit whitelist — only numeric literals and the arithmetic
    BinOp/UnaryOp nodes in _OPS are permitted; any Name, Call, Attribute,
    subscript, etc. raises. So a datagen-supplied `expression` string cannot
    execute arbitrary code. Inputs are our own generated specs, not user input,
    but this keeps that guarantee regardless.
    """
    def ev(node):
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](ev(node.left), ev(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](ev(node.operand))
        raise ValueError(f"unsupported arithmetic expression: {expr!r}")
    return ev(ast.parse(expr, mode="eval"))


def _compare(actual: int, spec: dict) -> CheckResult:
    """Shared op comparison for the *_count verifiers."""
    op = spec["op"]
    if op == "exact":
        ok, want = actual == spec["n"], f"exactly {spec['n']}"
    elif op == "min":
        ok, want = actual >= spec["n"], f">= {spec['n']}"
    elif op == "max":
        ok, want = actual <= spec["n"], f"<= {spec['n']}"
    elif op == "range":
        ok, want = spec["min"] <= actual <= spec["max"], f"in [{spec['min']}, {spec['max']}]"
    else:
        raise ValueError(f"unknown count op: {op!r}")
    return CheckResult(ok, f"count={actual}, wanted {want}")


# --- counts ------------------------------------------------------------------

@register("word_count")
def word_count(response: str, spec: dict) -> CheckResult:
    return _compare(len(response.split()), spec)


_SENT_DECIMAL = re.compile(r"(\d)\.(\d)")
_SENT_SPLIT = re.compile(r"[.!?]+(?=\s|$)")


@register("sentence_count")
def sentence_count(response: str, spec: dict) -> CheckResult:
    # neutralize decimal points (3.14) so they aren't sentence boundaries,
    # then count non-empty segments between terminal punctuation.
    neutral = _SENT_DECIMAL.sub("\\1․\\2", response)
    parts = [p for p in _SENT_SPLIT.split(neutral) if p.strip()]
    return _compare(len(parts), spec)


_ITEM_LINE = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+\S")


@register("item_count")
def item_count(response: str, spec: dict) -> CheckResult:
    n = sum(1 for line in response.splitlines() if _ITEM_LINE.match(line))
    return _compare(n, spec)


# --- arithmetic --------------------------------------------------------------

@register("arithmetic_result")
def arithmetic_result(response: str, spec: dict) -> CheckResult:
    val = spec.get("expected")
    if val is None:
        val = _safe_eval(spec["expression"])
    if isinstance(val, float) and val.is_integer():
        val = int(val)
    # standalone number: not glued to another digit (so 42 != the 42 in 4200)
    if re.search(r"(?<!\d)" + re.escape(str(val)) + r"(?!\d)", response):
        return CheckResult(True, f"result {val} present")
    return CheckResult(False, f"result {val} not present as a standalone number")


# --- ordering ----------------------------------------------------------------

@register("ordering")
def ordering(response: str, spec: dict) -> CheckResult:
    hay = response.lower()
    idxs = []
    for item in spec["items"]:
        i = hay.find(item.lower())
        if i == -1:
            return CheckResult(False, f"item not found: {item!r}")
        idxs.append(i)
    if idxs != sorted(idxs):
        return CheckResult(False, f"items out of order: {spec['items']}")
    return CheckResult(True, "items in required order")


# --- exact repetition --------------------------------------------------------

@register("exact_repetition")
def exact_repetition(response: str, spec: dict) -> CheckResult:
    n = response.lower().count(spec["phrase"].lower())
    want = spec["count"]
    if n != want:
        return CheckResult(False, f"phrase count={n}, wanted exactly {want}")
    return CheckResult(True, f"phrase appears exactly {want} times")
