"""Pre-flight connectivity check. One tiny call to each candidate + the judge.

Fails fast on the first error so the user can fix model IDs against
openrouter.ai/models or anthropic docs before paying for a batch run.
"""

from __future__ import annotations

from config import CANDIDATES, JUDGE, anthropic, router


def _ping_candidate(key: str, model_id: str) -> None:
    print(f"  → candidate {key} ({model_id}) ...", end=" ", flush=True)
    resp = router.chat.completions.create(
        model=model_id,
        temperature=0,
        max_tokens=4,
        messages=[{"role": "user", "content": "Say 'ok'."}],
    )
    txt = (resp.choices[0].message.content or "").strip()
    print(f"ok ({txt[:40]!r})")


def _ping_judge() -> None:
    print(f"  → judge ({JUDGE}) ...", end=" ", flush=True)
    resp = anthropic.messages.create(
        model=JUDGE,
        max_tokens=4,
        messages=[{"role": "user", "content": "Say 'ok'."}],
    )
    txt = "".join(b.text for b in resp.content if hasattr(b, "text")).strip()
    print(f"ok ({txt[:40]!r})")


def run() -> None:
    print("connectivity check:")
    failures: list[tuple[str, str, str]] = []
    for key, model_id in CANDIDATES.items():
        try:
            _ping_candidate(key, model_id)
        except Exception as e:  # noqa: BLE001
            print(f"FAIL ({type(e).__name__}: {e})")
            failures.append((key, model_id, f"{type(e).__name__}: {e}"))
    try:
        _ping_judge()
    except Exception as e:  # noqa: BLE001
        print(f"FAIL ({type(e).__name__}: {e})")
        failures.append(("judge", JUDGE, f"{type(e).__name__}: {e}"))

    if failures:
        print("\nconnectivity check FAILED. Fix these IDs and retry:")
        for key, model_id, err in failures:
            print(f"  - {key}: {model_id}  →  {err}")
        raise SystemExit(2)
    print("connectivity check passed.")


if __name__ == "__main__":
    run()
