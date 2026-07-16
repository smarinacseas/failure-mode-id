"""Verifier core (T1.1). A verifier is `fn(response: str, spec: dict) ->
CheckResult`; specs are JSON-serializable so datagen can emit them and the
GRPO reward + eval can consume the same objects. `spec["type"]` selects the
verifier from the registry."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass
class CheckResult:
    passed: bool
    detail: str = ""


Verifier = Callable[[str, dict], CheckResult]
VERIFIERS: dict[str, Verifier] = {}


def register(type_name: str) -> Callable[[Verifier], Verifier]:
    """Decorator: register a verifier under spec['type'] == type_name."""
    def deco(fn: Verifier) -> Verifier:
        if type_name in VERIFIERS:
            raise ValueError(f"verifier type {type_name!r} already registered")
        VERIFIERS[type_name] = fn
        return fn
    return deco


def check(response: str, spec: dict) -> CheckResult:
    """Dispatch to the verifier named by spec['type']. Raises KeyError for an
    unknown type — datagen must only emit registered types (guarded in T1.2)."""
    return VERIFIERS[spec["type"]](response, spec)
