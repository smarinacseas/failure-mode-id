"""One-shot: turn a Claude Design bundled `.dc.html` into deployable static files.

The bundled design ships as a single self-extracting HTML: base64-gzipped
assets in a manifest, a template with UUID placeholders, and a bootstrap
script that decodes assets to blob URLs on every load. That's convenient
for the design tool but wasteful for a shipping dashboard — every visit
does the base64/gzip dance client-side.

This script unpacks the bundle once:
  · Assets → real files under `dashboard/` (fonts, runtime JS)
  · Template UUIDs → relative paths to those files
  · JSON assets bundled by the design (sample runs.json / results.json) are
    DROPPED so the dashboard fetches live data written by dashboard_sync.py
  · The `<script>window.__resources = {…}</script>` bootstrap is not emitted,
    so `__assetUrl()` inside the template falls through to the real URLs.

Run whenever you drop a new `ConstraintLens Dashboard.html` from the design
tool. Output is committed to git so GitHub Pages deploys with no build step.

Usage: uv run python scripts/unpack_design.py
"""

from __future__ import annotations

import argparse
import base64
import gzip
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BUNDLE = REPO_ROOT / "design" / "ConstraintLens Dashboard.html"
DEFAULT_OUT = REPO_ROOT / "dashboard"

# ext_resources entries with these `id`s carry sample data the design shipped
# with — we don't want them in the deployed dashboard because the pipeline
# writes real versions at the same relative paths.
LIVE_DATA_IDS = frozenset({"runsJson", "resultsJson"})

# Tag names the HTML parser applies special "table parsing" rules to. When
# they contain anything that isn't a table child (e.g. our <sc-for>), the
# parser hoists it out. Renaming them so the parser sees "unknown elements"
# keeps our structure intact; dc-runtime unwraps them at render time.
TABLE_TAGS = ("table", "tbody", "thead", "tfoot", "tr", "td", "th", "caption")


def _encode_table_tags(html: str) -> str:
    for tag in TABLE_TAGS:
        html = re.sub(
            rf"(<\/?){tag}(?=[\s>])", rf"\1sc-raw-{tag}", html, flags=re.IGNORECASE
        )
    return html


_EXTRA_RUN_DETAIL_ROWS = (
    "{label:'Reasoning mode', value:"
    " meta.reasoning_enabled === false ? 'Disabled'"
    " : (meta.reasoning_enabled === true ? 'Enabled'"
    " : (meta.reasoning_enabled==null ? 'Not recorded' : String(meta.reasoning_enabled)))},"
    "{label:'Judge validation', value:"
    " (meta.validation && meta.validation.status === 'scored' && meta.validation.agreement_pct != null)"
    " ? (fmtPct(meta.validation.agreement_pct)+' agreement (n='+(meta.validation.n_scored||0)+')')"
    " : ((meta.validation && meta.validation.status === 'sampled')"
    "     ? ('sample drawn (n='+(meta.validation.n_sampled||0)+', ungraded)')"
    "     : 'not run')},"
    "{label:'Git commit', value:"
    " (meta.git && meta.git.commit)"
    " ? (meta.git.commit + (meta.git.dirty?' (dirty)':''))"
    " : '—'},"
)


def _augment_run_details(html: str) -> str:
    """Insert three extra rows into the design's runDetails literal.

    The design's Logic hardcodes `runDetails:[…]` inside a `<script
    type="text/x-dc">` block. We locate the shipped `Coverage` row (a
    stable landmark that closes the array) and append the extra items
    directly after it. Idempotent — bails out if the marker Reasoning
    mode row is already present.
    """
    if "'Reasoning mode'" in html:
        return html
    coverage_re = re.compile(
        r"(\{label:'Coverage',\s*value:\s*useCases\.length\+' use cases · '\+promptStyles\.length\+' prompt styles'\},)",
    )
    m = coverage_re.search(html)
    if not m:
        return html  # design's runDetails changed shape — patch would be brittle
    return html[: m.end()] + _EXTRA_RUN_DETAIL_ROWS + html[m.end():]


def _patch_scfor_around_tr(html: str, list_expr: str, tr_var: str) -> str:
    """Wrap the `<tr onclick="{{ <tr_var>.onClick }}">…</tr>` inside its
    surrounding `<tbody>` with a `<sc-for list="{{ <list_expr> }}" as="<tr_var>">`.

    Fixes a design-export bug where the tool emitted the sc-for tag with an
    empty body ahead of the table instead of wrapping the row template. The
    tr's inner bindings still reference `<tr_var>` so the intent is clear
    from the shape — we just have to re-attach the loop.
    """
    empty_sc = re.compile(
        r"<sc-for\s+list=\"\{\{\s*"
        + re.escape(list_expr)
        + r"\s*\}\}\"[^>]*>\s*</sc-for>",
        re.IGNORECASE,
    )
    m = empty_sc.search(html)
    if not m:
        return html
    scfor_open = m.group(0).replace("</sc-for>", "").rstrip()
    html = html[: m.start()] + html[m.end():]

    tr_pattern = re.compile(
        r"(<tr\b[^>]*onclick=\"\{\{\s*"
        + re.escape(tr_var)
        + r"\.[^\"]+\}\}\"[\s\S]*?</tr>)",
        re.IGNORECASE,
    )
    m2 = tr_pattern.search(html)
    if not m2:
        return html  # bail rather than emit half-fixed output
    return html[: m2.start()] + scfor_open + m2.group(1) + "</sc-for>" + html[m2.end():]


def _read_json_script(html: str, script_type: str):
    match = re.search(
        rf'<script type="{re.escape(script_type)}">([\s\S]*?)</script>',
        html,
    )
    if not match:
        raise RuntimeError(f"missing <script type='{script_type}'> in bundle")
    return json.loads(match.group(1))


def _decode_asset(entry: dict) -> bytes:
    raw = base64.b64decode(entry["data"])
    return gzip.decompress(raw) if entry.get("compressed") else raw


def _filename_for(uuid: str, mime: str, seen_js: list[bool]) -> str | None:
    """Map (uuid, mime) → relative path under `dashboard/`.

    Returns None when the asset should NOT be extracted (e.g. bundled sample
    JSONs the pipeline overwrites at runtime — those we drop instead of
    burning them into the repo).
    """
    if mime == "text/javascript":
        # Every design bundle carries exactly one text/javascript blob — the
        # dc-runtime (`support.js`). Overwriting the user's copy is safe
        # because the source is the same design tool.
        if seen_js[0]:
            return f"assets/{uuid}.js"
        seen_js[0] = True
        return "support.js"
    if mime == "font/woff2":
        return f"fonts/{uuid}.woff2"
    if mime == "font/woff":
        return f"fonts/{uuid}.woff"
    if mime == "application/json":
        return None  # dashboard_sync.py provides these at runtime
    if mime.startswith("image/"):
        ext = mime.split("/")[1].split("+")[0]
        return f"assets/{uuid}.{ext}"
    if mime == "text/css":
        return f"assets/{uuid}.css"
    return f"assets/{uuid}.bin"


def unpack(bundle_path: Path, out_dir: Path) -> dict:
    html = bundle_path.read_text(encoding="utf-8")

    template = _read_json_script(html, "__bundler/template")
    manifest = _read_json_script(html, "__bundler/manifest")
    ext_resources = _read_json_script(html, "__bundler/ext_resources")

    # Track which UUIDs are the live-data JSONs so we can wipe any lingering
    # references to them from the template.
    live_data_uuids = {
        e["uuid"] for e in ext_resources if e["id"] in LIVE_DATA_IDS
    }

    seen_js = [False]
    uuid_to_path: dict[str, str] = {}
    written: list[tuple[str, int]] = []
    dropped: list[str] = []

    for uuid, entry in manifest.items():
        mime = entry.get("mime", "application/octet-stream")
        if uuid in live_data_uuids:
            dropped.append(f"{uuid} ({mime}) — live data via sync script")
            continue
        rel_path = _filename_for(uuid, mime, seen_js)
        if rel_path is None:
            dropped.append(f"{uuid} ({mime}) — no export rule")
            continue
        data = _decode_asset(entry)
        abs_path = out_dir / rel_path
        abs_path.parent.mkdir(parents=True, exist_ok=True)
        abs_path.write_bytes(data)
        uuid_to_path[uuid] = rel_path
        written.append((rel_path, len(data)))

    # Substitute UUIDs. The bundler emits UUIDs as raw hex-with-dashes with no
    # protocol — they appear inside src=/url() strings and are unique enough
    # to swap as plain string replacements.
    unpacked = template
    for uuid, rel_path in uuid_to_path.items():
        unpacked = unpacked.replace(uuid, rel_path)

    # Design export bugs we patch here. Each one is a targeted regex, gated on
    # the specific `{{ ... }}` binding so we can't silently re-apply after the
    # design is updated to fix them upstream.
    unpacked = _patch_scfor_around_tr(
        unpacked, list_expr="drillRows", tr_var="r"
    )

    # Design extension: the shipped runDetails array covers models / judge /
    # counts / date / token / benchmark / coverage. Reasoning-mode,
    # judge-validation status, and git commit are equally load-bearing for
    # interpreting a run — append them here so the pipeline's config is
    # visible on the dashboard without touching the design source.
    unpacked = _augment_run_details(unpacked)

    # HTML parser applies table-parsing rules to <table>/<tbody>/<tr>/<td>/etc.
    # and will hoist any non-table children (like <sc-for>) OUT of the table.
    # Renaming them to `sc-raw-*` avoids that; the dc-runtime unwraps them
    # back to real tags at render time via its RAW_UNWRAP map. Idempotent —
    # only the exact bare tag names are matched, `sc-raw-table` is safe.
    unpacked = _encode_table_tags(unpacked)

    # Any references to live-data UUIDs are stale (design's sample data). The
    # template usually references them only via ext_resources → __resources
    # (which we don't emit), but leave a comment breadcrumb if any survive.
    stale = [u for u in live_data_uuids if u in unpacked]
    if stale:
        for u in stale:
            unpacked = unpacked.replace(u, f"unbound-{u}")

    index_path = out_dir / "index.html"
    index_path.write_text(unpacked, encoding="utf-8")
    written.append(("index.html", len(unpacked.encode("utf-8"))))

    return {
        "index_path": str(index_path),
        "assets_written": [(p, n) for p, n in written],
        "dropped": dropped,
        "stale_refs": stale,
    }


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="unpack_design.py",
        description="Unpack Claude Design bundled HTML → clean static files under dashboard/.",
    )
    p.add_argument("--bundle", default=str(DEFAULT_BUNDLE), help="Path to the bundled .html file.")
    p.add_argument("--out", default=str(DEFAULT_OUT), help="Destination directory.")
    p.add_argument("--quiet", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    bundle = Path(args.bundle)
    out = Path(args.out)
    if not bundle.exists():
        print(f"error: bundle not found: {bundle}", file=sys.stderr)
        return 2
    result = unpack(bundle, out)
    if args.quiet:
        return 0
    total = sum(n for _, n in result["assets_written"])
    print(f"unpack_design: wrote {len(result['assets_written'])} files ({total:,} bytes) → {out}")
    for path, size in result["assets_written"]:
        print(f"  · {path}  ({size:,} bytes)")
    if result["dropped"]:
        print(f"unpack_design: dropped {len(result['dropped'])} bundled asset(s):")
        for note in result["dropped"]:
            print(f"  · {note}")
    if result["stale_refs"]:
        print(
            "unpack_design: WARN — template still references live-data UUIDs after substitution: "
            + ", ".join(result["stale_refs"])
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
