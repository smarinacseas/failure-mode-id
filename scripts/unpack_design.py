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


# Responsive shim: the design ships every layout as inline styles, so
# attribute selectors + !important is the only way to override without
# forking the source. Rules only fire at their breakpoints — desktop
# stays pixel-identical.
#
# The browser normalizes inline styles when parsing (adds a space after
# every colon, after every comma inside function values). Substring
# selectors must match the NORMALIZED form the DOM presents, not the
# unnormalized form in the source HTML. Every selector below uses the
# normalized form.
_RESPONSIVE_CSS = """<!-- responsive-overrides v1 -->
<style id="responsive-overrides-v1">
@media (min-width: 1600px) {
  div[style*="max-width: 1320px"] {
    max-width: min(1600px, 96vw) !important;
  }
}
@media (max-width: 1024px) {
  div[style*="max-width: 1320px"] {
    padding-left: 20px !important;
    padding-right: 20px !important;
  }
  div[style*="grid-template-columns: repeat(4, 1fr)"],
  div[style*="grid-template-columns: repeat(5, 1fr)"] {
    grid-template-columns: repeat(2, minmax(0, 1fr)) !important;
  }
  /* `1fr 1fr` inner grids (comparison cards, filter rows) hold content
     that won't shrink below its natural min-width. Substituting
     minmax(0, 1fr) lets them compress into the tablet's 2-col chrome
     without breaking. Below 640px they collapse to 1-col via the
     mobile rule instead. */
  div[style*="grid-template-columns: 1fr 1fr"] {
    grid-template-columns: minmax(0, 1fr) minmax(0, 1fr) !important;
  }
}
@media (max-width: 640px) {
  div[style*="max-width: 1320px"] {
    padding-left: 12px !important;
    padding-right: 12px !important;
  }
  div[style*="grid-template-columns: repeat(4, 1fr)"],
  div[style*="grid-template-columns: repeat(5, 1fr)"],
  div[style*="grid-template-columns: 1fr 1fr"] {
    grid-template-columns: minmax(0, 1fr) !important;
  }
  /* Header topbar and footer both need to wrap on narrow viewports. */
  div[style*="max-width: 1320px"][style*="padding: 13px 30px"],
  div[style*="max-width: 1320px"][style*="padding: 0px 30px 30px"] {
    flex-wrap: wrap !important;
    row-gap: 10px !important;
  }
  /* Popovers must stay inside the viewport. Without this the run
     selector's 262px-min-width dropdown blows out a 320px screen. */
  div[style*="position: absolute"][style*="top: calc(100% + 8px)"],
  div[style*="position: absolute"][style*="top: calc(100% + 5px)"] {
    max-width: calc(100vw - 24px) !important;
    min-width: 0 !important;
  }
  /* Truncate the mono date/count caption inside the run selector so
     the header topbar doesn't overflow. font-size:10.5px is unique to
     that one span in the header, hence the specificity. */
  div[style*="max-width: 1320px"][style*="padding: 13px 30px"] span[style*="font-size: 10.5px"] {
    max-width: 120px !important;
    overflow: hidden !important;
    text-overflow: ellipsis !important;
  }
  /* Tables become horizontally scrollable rather than overflowing. */
  sc-raw-table, table {
    display: block !important;
    max-width: 100% !important;
    overflow-x: auto !important;
    -webkit-overflow-scrolling: touch !important;
  }
  h1 { font-size: 20px !important; }
  /* Compact panel padding to reclaim horizontal space. */
  div[style*="padding: 16px 20px 18px"] { padding: 14px 14px 16px !important; }
}
</style>
"""


def _fix_selection_color(html: str) -> str:
    """Replace the design's washed-out selection highlight in-place.

    The shipped rule is `background:#e3d4bf` with no `color` set, so the
    original text color survives on top of a pale cream. On the design's
    muted-gray metadata (`var(--ink-4,#a8a294)` etc.) the contrast drops
    to near-invisible when the user drags-selects. Swapping in the
    design's warm accent (matches `--pill-imp`) with an explicit light
    text color guarantees legibility across every text tone. Idempotent
    on the source string — a design update that changes the color just
    skips this patch.
    """
    source = "::selection{background:#e3d4bf}"
    if source not in html:
        return html
    replacement = (
        "::selection{background:#8a5736;color:#faf8f2}"
        "::-moz-selection{background:#8a5736;color:#faf8f2}"
    )
    return html.replace(source, replacement, 1)


def _inject_responsive_css(html: str) -> str:
    """Add media-query rules to the design's <head>.

    The design's layout is inline-styled top-to-bottom; !important on an
    attribute selector is the only way to override without editing the
    source. Idempotent gate on the marker id.
    """
    if "responsive-overrides-v1" in html:
        return html
    marker = "</head>"
    idx = html.find(marker)
    if idx < 0:
        return html
    return html[:idx] + _RESPONSIVE_CSS + html[idx:]


# Descriptions match README §Glossary. Kept terse for native title="…"
# tooltips (browsers truncate long titles and don't honor newlines).
_GLOSSARY_JS = r"""
const GLOSSARY = {
  use_case: {
    'Data Processing, Formatting & Math': 'Structured output under numerical + formatting constraints. Spreadsheets, reports, JSON schemas with arithmetic that must reconcile.',
    'Logistics, Scheduling & Event Planning': 'Multi-entity plans under real-world constraints. Rotas with union/age/shift rules, routing with prohibitions, event flows.'
  },
  instruction_type: {
    'Negative': "Must-not constraints (\"avoid X\", \"no Y\"). Canonical failure: introducing the forbidden element anyway.",
    'Multistep': 'Sequenced operations where step N depends on step N−1. Canonical failure: intermediate-step errors compounding.'
  },
  prompt_style: {
    'Direct prompting': 'Clear, numbered/bulleted instructions.',
    'Context prompting': 'Instructions embedded in a narrative scenario (a stakeholder message, a briefing).',
    'Rambling/Stream-of-Consciousness': 'Instructions buried in conversational, redundant, or noisy prose.'
  }
};
function describe(field, value){ return (GLOSSARY[field] && GLOSSARY[field][value]) || ''; }
"""


def _add_tag_glossary(html: str) -> str:
    """Wire native title="…" tooltips onto tag pills and category cells.

    Four coordinated edits (all idempotent):
      1. Inject GLOSSARY + describe() at the top of the Logic block.
      2. Append useTitle/instrTitle/styleTitle to the drill-row builder.
      3. Append dUseTitle/dInstrTitle/dStyleTitle to the detail row.
      4. Add title="…" to the six template spans that render tags.

    Follows the design's own pattern — `title="{{ c.gameTitle }}"` on the
    gameable pill already relies on native browser tooltips, so this
    keeps the vocabulary consistent.
    """
    if "const GLOSSARY" in html:
        return html

    # (1) Insert GLOSSARY + describe just before the design's first
    # top-level `const PILL_BASE` declaration, which sits at the top of
    # the Logic block right after the <script type="text/x-dc"> tag.
    anchor_1 = "const PILL_BASE="
    if anchor_1 not in html:
        return html
    html = html.replace(anchor_1, _GLOSSARY_JS.strip() + "\n" + anchor_1, 1)

    # (2) Extend the drill-row return literal — insert titles right after
    # the styleLabel field so the object stays legible.
    anchor_2 = "styleLabel:p.prompt_style,"
    replacement_2 = (
        "styleLabel:p.prompt_style, "
        "useTitle:describe('use_case',p.use_case), "
        "instrTitle:describe('instruction_type',p.instruction_type), "
        "styleTitle:describe('prompt_style',p.prompt_style),"
    )
    if anchor_2 in html and replacement_2 not in html:
        html = html.replace(anchor_2, replacement_2, 1)

    # (3) Extend the detail-row (dPrompt) literal similarly.
    anchor_3 = "dUseLabel:dPrompt.use_case||'', dStyleLabel:dPrompt.prompt_style||'',"
    replacement_3 = (
        "dUseLabel:dPrompt.use_case||'', dStyleLabel:dPrompt.prompt_style||'', "
        "dUseTitle:describe('use_case',dPrompt.use_case), "
        "dStyleTitle:describe('prompt_style',dPrompt.prompt_style), "
        "dInstrTitle:describe('instruction_type',dPrompt.instruction_type),"
    )
    if anchor_3 in html and replacement_3 not in html:
        html = html.replace(anchor_3, replacement_3, 1)

    # (4a) Drill table — use_case cell (plain text) gets a title span.
    html = html.replace(
        '<sc-raw-td style="padding:11px 12px;color:var(--ink-1,#2a2f3a)">{{ r.useLabel }}</sc-raw-td>',
        '<sc-raw-td style="padding:11px 12px;color:var(--ink-1,#2a2f3a)"><span title="{{ r.useTitle }}">{{ r.useLabel }}</span></sc-raw-td>',
        1,
    )

    # (4b) Drill table — instruction pill.
    html = html.replace(
        '<span style="{{ r.instrCss }}">{{ r.instrLabel }}</span>',
        '<span title="{{ r.instrTitle }}" style="{{ r.instrCss }}">{{ r.instrLabel }}</span>',
        1,
    )

    # (4c) Drill table — prompt style cell.
    html = html.replace(
        '<sc-raw-td style="padding:11px 12px;color:var(--ink-2,#5b616d)">{{ r.styleLabel }}</sc-raw-td>',
        '<sc-raw-td style="padding:11px 12px;color:var(--ink-2,#5b616d)"><span title="{{ r.styleTitle }}">{{ r.styleLabel }}</span></sc-raw-td>',
        1,
    )

    # (4d) Detail modal — use case pill.
    html = html.replace(
        '<span style="font-size:11.5px;background:var(--line,#eef0f3);color:var(--ink-1,#3a4150);padding:3px 9px;border-radius:6px">{{ dUseLabel }}</span>',
        '<span title="{{ dUseTitle }}" style="font-size:11.5px;background:var(--line,#eef0f3);color:var(--ink-1,#3a4150);padding:3px 9px;border-radius:6px">{{ dUseLabel }}</span>',
        1,
    )

    # (4e) Detail modal — instruction pill.
    html = html.replace(
        '<span style="{{ dInstrCss }}">{{ dInstrLabel }}</span>',
        '<span title="{{ dInstrTitle }}" style="{{ dInstrCss }}">{{ dInstrLabel }}</span>',
        1,
    )

    # (4f) Detail modal — prompt style pill.
    html = html.replace(
        '<span style="font-size:11.5px;background:var(--line,#eef0f3);color:var(--ink-1,#3a4150);padding:3px 9px;border-radius:6px">{{ dStyleLabel }}</span>',
        '<span title="{{ dStyleTitle }}" style="font-size:11.5px;background:var(--line,#eef0f3);color:var(--ink-1,#3a4150);padding:3px 9px;border-radius:6px">{{ dStyleLabel }}</span>',
        1,
    )

    return html


_GLOSSARY_LINK_HTML = (
    '<span style="flex:1"></span>'
    '<a href="https://github.com/smarinacseas/failure-mode-id#glossary" '
    'target="_blank" rel="noopener" '
    'title="Definitions for every tag pill on this dashboard" '
    'style="color:var(--ink-4,#aab0bb);text-decoration:none;'
    'border-bottom:1px dotted currentColor">Glossary ↗</a>'
)


def _inject_glossary_link(html: str) -> str:
    """Add a discreet 'Glossary ↗' link at the far right of the footer.

    Chosen over the topbar because the footer is where reference-type
    metadata already lives (run/judge/counts) — the link matches the
    footer's IBM Plex Mono chrome and doesn't crowd the model tabs.
    Idempotent on the anchor href.
    """
    if "#glossary" in html and "Glossary" in html:
        return html
    anchor = '<span>{{ footModels }} models</span>'
    if anchor not in html:
        return html
    return html.replace(anchor, anchor + _GLOSSARY_LINK_HTML, 1)


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

    # Tooltips on tag pills. Must run AFTER _encode_table_tags because four
    # of the six template patches target `<sc-raw-td>`, which only exists
    # after encoding. Also patches the Logic block to add GLOSSARY + a
    # describe() helper + tag-title fields on drill and detail row builders.
    unpacked = _add_tag_glossary(unpacked)

    # Responsive shim: single stylesheet in <head>, media-query-gated. The
    # design ships every layout as inline styles, so this uses attribute
    # selectors + !important to override without touching the source.
    unpacked = _inject_responsive_css(unpacked)

    # Legibility fix on ::selection — the shipped rule is background-only
    # and drops muted-gray text to near-invisible when highlighted.
    unpacked = _fix_selection_color(unpacked)

    # Discreet link out to the repo's Glossary section from the footer.
    unpacked = _inject_glossary_link(unpacked)

    # Any references to live-data UUIDs are stale (design's sample data). The
    # template usually references them only via ext_resources → __resources
    # (which we don't emit), but leave a comment breadcrumb if any survive.
    stale = [u for u in live_data_uuids if u in unpacked]
    if stale:
        for u in stale:
            unpacked = unpacked.replace(u, f"unbound-{u}")

    # The unpacked design is the eval app; index.html is the hand-written landing page.
    index_path = out_dir / "eval.html"
    index_path.write_text(unpacked, encoding="utf-8")
    written.append(("eval.html", len(unpacked.encode("utf-8"))))

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
