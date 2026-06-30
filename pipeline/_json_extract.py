"""Defensive extraction of a single JSON array from a noisy LLM response."""

from __future__ import annotations

import json
import re


_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)


def extract_json_array(text: str) -> list:
    """Return the first JSON array found in `text`. Raises ValueError if none parse.

    Strategy:
      1. Strip ``` fences if present.
      2. Try parsing the trimmed body directly.
      3. Fall back to scanning for the first balanced [...] block.
    """
    candidates: list[str] = []

    fenced = _FENCE_RE.findall(text)
    candidates.extend(s.strip() for s in fenced)

    candidates.append(text.strip())

    for c in candidates:
        try:
            parsed = json.loads(c)
            if isinstance(parsed, list):
                return parsed
        except json.JSONDecodeError:
            pass

    # Last resort: find a balanced [...] block.
    start = text.find("[")
    while start != -1:
        depth = 0
        in_str = False
        esc = False
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
            else:
                if ch == '"':
                    in_str = True
                elif ch == "[":
                    depth += 1
                elif ch == "]":
                    depth -= 1
                    if depth == 0:
                        block = text[start : i + 1]
                        try:
                            parsed = json.loads(block)
                            if isinstance(parsed, list):
                                return parsed
                        except json.JSONDecodeError:
                            break
        start = text.find("[", start + 1)

    raise ValueError("no parseable JSON array found")
