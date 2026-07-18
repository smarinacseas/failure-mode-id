#!/usr/bin/env python3
"""T1.3 live status reporter — READ-ONLY.

Regenerates `experiments/T01/results/LIVE_STATUS.md` from the training artifacts
already written by the runners/callbacks (results/logs/*.jsonl, *_summary.json,
*_lengths.json, results/adapters/, RA_gate_step50.json). It does NOT import
torch/trl, does NOT touch any training process, and does NOT poll — run it once
per meaningful checkpoint (per-arm completion, gate fire, cap hit).

Usage:  python experiments/T01/results/live_status.py
"""
from __future__ import annotations

import json
import math
import os
import re
import statistics as st
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
LOGS = REPO / "results" / "logs"
ADAPTERS = REPO / "results" / "adapters"
FROZEN = REPO / "experiments" / "T01" / "config" / "t1_3_frozen.md"
OUT = REPO / "experiments" / "T01" / "results" / "LIVE_STATUS.md"

# Arm registry — tag maps to the runner's CSVLogger basename.
ARMS = [
    {"arm": "SA", "method": "SFT",  "cause": "coverage",  "tag": "sft_SA_coverage",  "total_steps": 16,  "n": 123, "budget_s": None},
    {"arm": "SB", "method": "SFT",  "cause": "precision", "tag": "sft_SB_precision", "total_steps": 16,  "n": 123, "budget_s": None},
    {"arm": "RA", "method": "GRPO", "cause": "coverage",  "tag": "grpo_RA_coverage", "total_steps": 300, "n": 300, "budget_s": 10800},
    {"arm": "RB", "method": "GRPO", "cause": "precision", "tag": "grpo_RB_precision","total_steps": 300, "n": 300, "budget_s": 10800},
]

# Probe-derived expected ranges (hardened coverage / precision) for anomaly flags.
EXPECT = {"reward_lo": 0.40, "reward_hi": 0.60, "cap_hit_hi": 0.15, "kl_hi": 0.5,
          "len_drift_hi": 1536, "format_ok_min": 0.95}


# ---------- small io helpers ----------
def read_jsonl(p: Path) -> list[dict]:
    if not p.exists():
        return []
    out = []
    for line in p.read_text().splitlines():
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out


def read_json(p: Path) -> dict | None:
    try:
        return json.loads(p.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def hms(sec: float | None) -> str:
    if sec is None or sec < 0 or math.isinf(sec):
        return "—"
    sec = int(sec)
    h, r = divmod(sec, 3600)
    m, s = divmod(r, 60)
    return f"{h}h{m:02d}m" if h else f"{m}m{s:02d}s"


def pid_alive(tag_arm: str) -> bool | None:
    """Best-effort liveness from results/logs/{ARM}_PID.txt (line 'X_PID=NNN')."""
    pf = LOGS / f"{tag_arm}_PID.txt"
    if not pf.exists():
        return None
    m = re.search(r"=(\d+)", pf.read_text())
    if not m:
        return None
    try:
        os.kill(int(m.group(1)), 0)
        return True
    except (OSError, ProcessLookupError, PermissionError) as e:
        return isinstance(e, PermissionError)  # perm error => exists (alive)


def frozen_summary() -> dict:
    """Pull the headline frozen knobs from t1_3_frozen.md (regex, with fallbacks
    to the canonical values so a formatting change never crashes the report)."""
    txt = FROZEN.read_text() if FROZEN.exists() else ""

    def grab(pattern, fallback):
        m = re.search(pattern, txt)
        return m.group(1) if m else fallback

    return {
        "sft_lr": grab(r"LR=(1e-4|[0-9.eE+-]+)\s*·\s*cosine", "1e-4"),
        "grpo_lr": grab(r"LR\s*=\s*(7\.5e-6|[0-9.eE+-]+)\*\*", "7.5e-6"),
        "k": grab(r"num_generations k=(\d+)", "6"),
        "temp": grab(r"rollout temp=(0?\.\d+)", "0.9"),
        "beta": grab(r"β=(0?\.\d+)", "0.04"),
        "seed": grab(r"seed=(\d+)", "20260715"),
        "epochs": "2",
        "found": bool(txt),
    }


# ---------- per-arm status ----------
def classify(a: dict) -> dict:
    tag, arm = a["tag"], a["arm"]
    jl = LOGS / f"{tag}.jsonl"
    rows = read_jsonl(jl)
    metric_rows = [r for r in rows if ("loss" in r or "reward" in r)]
    steps = [r["step"] for r in metric_rows if "step" in r]
    cur_step = max(steps) if steps else 0
    elapsed = max((r.get("elapsed_s", 0) for r in metric_rows), default=0)

    override = LOGS / f"{arm}.status"
    summary = read_json(LOGS / f"{tag}_summary.json")
    sft_done = any("train_runtime" in r for r in rows)
    grpo_done = summary is not None
    adapter = (ADAPTERS / f"T01-{arm}" / "adapter_model.safetensors").exists()

    if override.exists():
        status = override.read_text().strip().split(":", 1)[0].upper() or "PAUSED"
        note = override.read_text().strip()
    elif (a["method"] == "SFT" and sft_done) or (a["method"] == "GRPO" and grpo_done):
        status, note = "DONE", ""
    elif not jl.exists():
        status, note = "QUEUED", ""
    else:
        alive = pid_alive(arm)
        stale = jl.exists() and (time.time() - jl.stat().st_mtime) > 300
        if alive is True:
            status, note = "RUNNING", ""
        elif alive is False and not (sft_done or grpo_done):
            status, note = "FAILED?", "PID not alive and no completion marker — check stdout"
        elif stale:
            status, note = "RUNNING?", "log stale >5m — possible stall/crash"
        else:
            status, note = "RUNNING", ""

    # s/step from inter-step elapsed deltas (skip warmup step 1). GRPO step time is
    # length-driven and bimodal (periodic long-completion batches ≈3-5x the median),
    # so the MEAN is the honest estimator for "finishes in budget" (total wall ≈ n·mean);
    # the median alone under-projects the cap. Report both, project on the mean.
    deltas = []
    ss = sorted((r["step"], r.get("elapsed_s")) for r in metric_rows if "step" in r and r.get("elapsed_s") is not None)
    for (s0, e0), (s1, e1) in zip(ss, ss[1:]):
        if s1 > s0 and s1 > 1:
            deltas.append((e1 - e0) / (s1 - s0))
    s_med = st.median(deltas) if deltas else None
    s_mean = st.mean(deltas) if deltas else None
    s_proj = s_mean  # conservative

    total = a["total_steps"]
    steps_per_epoch = max(total // 2, 1)
    remaining = max(total - cur_step, 0)
    eta = remaining * s_proj if s_proj else None
    cap_note = ""
    trunc = None
    if a["budget_s"] and s_proj and status.startswith("RUNNING"):
        budget_left = a["budget_s"] - elapsed
        proj_stop = cur_step + int(budget_left / s_proj)
        if proj_stop < total:
            eta = max(budget_left, 0)
            over_min = round((total - proj_stop) * s_proj / 60)
            trunc = {"proj_stop": proj_stop, "proj_epochs": round(proj_stop / steps_per_epoch, 2),
                     "s_mean": round(s_proj, 1), "s_med": round(s_med, 1), "over_min": over_min}
            cap_note = f"⚠️ 3h cap first — proj stop ≈step {proj_stop} (~{trunc['proj_epochs']}/2.0 ep)"

    last = max(metric_rows, key=lambda r: r.get("step", -1)) if metric_rows else {}
    last_val = None
    last_kind = "loss" if a["method"] == "SFT" else "reward(last10)"
    if a["method"] == "SFT":
        if status == "DONE" and summary is None:
            tr = [r for r in rows if "train_runtime" in r]
            last_val = tr[-1].get("train_loss") if tr else last.get("loss")
        else:
            last_val = last.get("loss")
    else:
        # Windowed last10 mean — NOT the single noisy last step (avoids the
        # single-endpoint-noise pitfall flagged for the LR probe).
        fm = (summary or {}).get("final_metrics", {})
        if fm.get("reward") is not None:
            last_val, last_kind = fm["reward"], "reward(final)"
        else:
            w = grpo_windows(metric_rows)
            last_val = w.get("last10")

    return {**a, "rows": rows, "metric_rows": metric_rows, "cur_step": cur_step,
            "elapsed": elapsed, "eta": eta, "cap_note": cap_note,
            "status": status, "note": note, "summary": summary, "last_val": last_val,
            "last_kind": last_kind, "adapter": adapter, "trunc": trunc,
            "s_med": s_med, "s_mean": s_mean}


# ---------- diagnostics rendering ----------
def sft_diag(a: dict) -> list[str]:
    L = [f"### {a['arm']} — SFT · {a['cause']}", ""]
    lengths = read_json(LOGS / f"{a['tag']}_lengths.json")
    rows = [r for r in a["metric_rows"] if "loss" in r and "train_runtime" not in r]
    if not rows:
        L += ["_no metric rows yet._", ""]
        return L
    spe = a["total_steps"] // 2 or 8  # steps per epoch
    L += ["| epoch | steps | first loss | last loss | mean loss | Δ | tok-acc (first→last) |",
          "|---|---|---|---|---|---|---|"]
    for ep in (1, 2):
        er = [r for r in rows if math.ceil(r["step"] / spe) == ep]
        if not er:
            continue
        f, l = er[0]["loss"], er[-1]["loss"]
        acc = f"{er[0].get('mean_token_accuracy', float('nan')):.3f}→{er[-1].get('mean_token_accuracy', float('nan')):.3f}"
        L.append(f"| {ep} | {er[0]['step']}–{er[-1]['step']} | {f:.4f} | {l:.4f} | "
                 f"{st.mean(x['loss'] for x in er):.4f} | {l-f:+.4f} | {acc} |")
    if len(rows) >= 2:
        f3 = st.mean(r["loss"] for r in rows[:3])
        l3 = st.mean(r["loss"] for r in rows[-3:])
        L += ["", f"**Overall loss slope:** mean(first3) {f3:.4f} → mean(last3) {l3:.4f} "
              f"(Δ **{l3-f3:+.4f}**, {'decreasing ✅' if l3 < f3 else 'NOT decreasing ⚠️'})."]
    if lengths:
        tr = lengths["truncation_rate"]
        L += [f"**Truncation @{lengths['max_length']}:** {tr*100:.2f}% "
              f"({'ok' if tr <= 0.01 else 'FLAG >1% → bump to 8192'}) "
              f"| lengths n={lengths['n']} med {lengths['median']:.0f} p95 {lengths['p95']} max {lengths['max']}."]
    masked = "Building labels" in (LOGS / f"{a['tag']}.stdout").read_text() if (LOGS / f"{a['tag']}.stdout").exists() else None
    L += [f"**Masking sanity:** completion_only_loss=True (user prompt → -100; loss on scaffold+answer)"
          f"{'; label-build step present in run log ✅' if masked else ''}.", ""]
    return L


def grpo_windows(rows: list[dict]) -> dict:
    rw = sorted([r for r in rows if "reward" in r], key=lambda r: r["step"])
    if not rw:
        return {}
    rewards = [r["reward"] for r in rw]
    w_first = [r["reward"] for r in rw if r["step"] <= 10]
    w_last = rewards[-10:]
    out = {"n": len(rw), "max_step": rw[-1]["step"],
           "first10": st.mean(w_first), "last10": st.mean(w_last),
           "delta": st.mean(w_last) - st.mean(w_first)}
    g1 = [r["reward"] for r in rw if 1 <= r["step"] <= 10]
    g2 = [r["reward"] for r in rw if 41 <= r["step"] <= 50]
    if len(g2) >= 1:
        out["gate_1_10"] = st.mean(g1)
        out["gate_41_50"] = st.mean(g2)
        out["gate_delta"] = st.mean(g2) - st.mean(g1)
    return out


def grpo_diag(a: dict) -> list[str]:
    L = [f"### {a['arm']} — GRPO · {a['cause']}", ""]
    rows = a["metric_rows"]
    rw = sorted([r for r in rows if "reward" in r], key=lambda r: r["step"])
    if not rw:
        L += ["_no reward rows yet._", ""]
        return L
    w = grpo_windows(rows)
    L += ["**Windowed reward (first10 vs last10 — probe methodology, not single-endpoint):**", "",
          "| window | steps | mean reward |", "|---|---|---|",
          f"| first10 | 1–10 | {w['first10']:.4f} |",
          f"| last10 | {max(1,w['max_step']-9)}–{w['max_step']} | {w['last10']:.4f} |",
          f"| **Δ** |  | **{w['delta']:+.4f}** ({'rising' if w['delta']>0 else 'flat/declining'}) |", ""]
    last = rw[-1]
    summ = (a["summary"] or {}).get("final_metrics", {})
    reward = summ.get("reward", last.get("reward"))
    rstd = summ.get("reward_std", last.get("reward_std"))
    kl = summ.get("kl", last.get("kl"))
    fmt = last.get("format_ok")
    len0 = rw[0].get("completions/mean_length")
    lenN = last.get("completions/mean_length")
    cap = summ.get("completions/clipped_ratio", last.get("completions/clipped_ratio"))
    caps = [r.get("completions/clipped_ratio") for r in rw if r.get("completions/clipped_ratio") is not None]
    cap_mean = st.mean(caps) if caps else None
    tok_s = (a["summary"] or {}).get("throughput", {}).get("tok_per_s")
    L += ["**Health metrics (last step / summary):**", "",
          "| metric | value |", "|---|---|",
          f"| reward (last step, noisy — see windowed table above) | {reward:.4f} |" if reward is not None else "| reward | — |",
          f"| reward_std | {rstd:.4f} |" if rstd is not None else "| reward_std | — |",
          f"| kl | {kl:.4g} |" if kl is not None else "| kl | — |",
          f"| format_ok | {fmt:.3f} |" if fmt is not None else "| format_ok | — |",
          f"| length drift (mean_len first→last) | {len0:.0f}→{lenN:.0f} (cap 1536) |" if len0 and lenN else "| length drift | — |",
          f"| cap-hit % (last / mean) | {cap*100:.1f}% / {cap_mean*100:.1f}% |" if cap is not None and cap_mean is not None else "| cap-hit % | — |",
          f"| rollout tok/s | {tok_s} |" if tok_s else "| rollout tok/s | n/a until completion |",
          f"| step_time (last) | {last.get('step_time', float('nan')):.1f}s |" if last.get("step_time") else "| step_time | — |",
          ""]
    if a.get("budget_s"):
        t = a.get("trunc")
        L += ["**3h hardcap behavior + partial-run note.**"]
        if t:
            L += [f"> 🟠 **Truncation likely.** Honest projection (mean {t['s_mean']}s/step; median {t['s_med']}s — "
                  f"GRPO step time is length-driven and bimodal, so mean is the correct estimator, not median): "
                  f"stops ≈**step {t['proj_stop']}/{a['total_steps']} (~{t['proj_epochs']} of 2.0 epochs)** at the "
                  f"3h cap; 2 full epochs would need ≈{t['over_min']} min past the cap.",
                  ">",
                  "> **On truncation:** the `TimeBudget` callback sets `should_training_stop` at the first step "
                  "past 10800s; the run then exits gracefully and `grpo.py` calls `save_model()` → a final adapter "
                  "is written at the stop step (plus `save_steps=50` checkpoints at 50/100/150/200/250). The summary "
                  "records `time_budget_hit=true` and the actual step count.",
                  ">",
                  "> **What \"RA complete\" then means:** an adapter trained ~1.7 (not 2.0) epochs — usable for eval, "
                  "but **under-trained vs the frozen 2-epoch spec**. If RB (precision, shorter rollouts) finishes 2 "
                  "full epochs, the GRPO 'dose' differs across causes → a confound on the method×cause interaction "
                  "(the analog of the SFT teacher-yield asymmetry already equalized by down-sampling). **Decision "
                  "point for the operator:** (a) let RA run to 2 full epochs (relax cap ~{over} min), (b) cap both "
                  "GRPO arms at a common step count for symmetry, or (c) accept partial + log the deviation and "
                  "caveat the coverage arm.".replace("{over}", str(t["over_min"])), ""]
        else:
            L += ["> On current pace RA is projected to reach the full step count within the 3h cap. If pace "
                  "slows (length drift up), truncation risk returns — this note will flip to 🟠. At any cap-stop the "
                  "adapter still saves (`save_model` + `save_steps=50`).", ""]
    if a["arm"] == "RA":
        L += ra_gate_block(w)
    return L


def ra_gate_block(w: dict) -> list[str]:
    gate = read_json(LOGS / "RA_gate_step50.json")
    L = ["#### 🔺 RA step-50 windowed-trend gate (pre-committed decision point)", ""]
    if gate:
        v = gate["suggested_verdict"]
        d = gate["delta"]
        if v.startswith("POSITIVE"):
            L += [f"> **✅ GATE RESULT — PROCEED.** mean(reward[1:10])={gate['mean_reward_1_10']} → "
                  f"mean(reward[41:50])={gate['mean_reward_41_50']}, **Δ={d:+.4f}** (clear positive slope). "
                  f"end-window std {gate['end_window_std']}.", ""]
        else:
            L += [f"> **🛑 GATE RESULT — PAUSE (flat-or-declining).** mean(reward[1:10])={gate['mean_reward_1_10']} → "
                  f"mean(reward[41:50])={gate['mean_reward_41_50']}, **Δ={d:+.4f}**. end-window std "
                  f"{gate['end_window_std']}. Per the pre-committed rule, RA is halted for operator review.", ""]
    elif w.get("gate_delta") is not None:
        d = w["gate_delta"]
        L += [f"> **Gate window forming (step {w['max_step']}/50):** interim "
              f"mean[1:10]={w['gate_1_10']:.4f} → mean[41:50]={w['gate_41_50']:.4f}, Δ={d:+.4f}. "
              f"Final verdict written at step 50.", ""]
    else:
        L += [f"> **Gate pending** — needs step 50 (currently {w.get('max_step', 0)}/50). "
              f"Probe reference on this hardened pool was Δ=−0.032 (declining), so a PAUSE is plausible.", ""]
    return L


# ---------- flags ----------
def health_flags(arms: list[dict]) -> list[str]:
    flags = []
    for a in arms:
        arm = a["arm"]
        # OOM / traceback scan
        sp = LOGS / f"{a['tag']}.stdout"
        if sp.exists():
            t = sp.read_text()
            if re.search(r"out of memory|OutOfMemoryError|CUDA error", t):
                flags.append(f"🛑 **{arm}: CUDA OOM / CUDA error** in stdout — investigate.")
            if "Traceback (most recent call last)" in t and a["status"].startswith(("FAIL", "RUNNING?")):
                flags.append(f"🛑 **{arm}: traceback in stdout** while not marked DONE.")
        if a["status"] in ("FAILED?", "RUNNING?"):
            flags.append(f"⚠️ **{arm}: {a['status']}** — {a['note']}.")
        if a.get("trunc"):
            t = a["trunc"]
            flags.append(
                f"🟠 **{arm}: PARTIAL-RUN RISK — the 3h hardcap will likely truncate before 2 full epochs.** "
                f"At the honest mean pace {t['s_mean']}s/step (median {t['s_med']}s — steps are length-driven & "
                f"bimodal, ~100s on long-completion batches vs ~30s on short), RA projects to stop ≈step "
                f"{t['proj_stop']}/{a['total_steps']} (**~{t['proj_epochs']} of 2.0 epochs**); reaching 300 would "
                f"need ≈{t['over_min']} min past the cap. The adapter still saves at the cap (`save_model` + "
                f"`save_steps=50` checkpoints) so it's usable but under-trained vs the 2-epoch spec. **Analysis "
                f"impact: if RB (precision, shorter rollouts) completes 2 full epochs, RA↔RB training dose is "
                f"unequal → confounds the GRPO×cause interaction.** Operator decision needed — see RA diagnostics note.")
        elif a["cap_note"]:
            flags.append(f"⏱️ **{arm}: {a['cap_note']}** (approaching 3h hardcap).")
        if a["method"] == "GRPO":
            w = grpo_windows(a["metric_rows"])
            last = ([r for r in a["metric_rows"] if "reward" in r] or [{}])[-1]
            if w:
                if not (EXPECT["reward_lo"] <= w["last10"] <= EXPECT["reward_hi"]):
                    flags.append(f"⚠️ **{arm}: last10 reward {w['last10']:.3f}** outside probe band "
                                 f"[{EXPECT['reward_lo']},{EXPECT['reward_hi']}].")
            fmt = last.get("format_ok")
            if fmt is not None and fmt < EXPECT["format_ok_min"]:
                flags.append(f"⚠️ **{arm}: format_ok {fmt:.2f}** < {EXPECT['format_ok_min']} (malformed rollouts rising).")
            kl = last.get("kl")
            if kl is not None and kl > EXPECT["kl_hi"]:
                flags.append(f"⚠️ **{arm}: KL {kl:.3f}** > {EXPECT['kl_hi']} (KL blow-up).")
            cap = last.get("completions/clipped_ratio")
            if cap is not None and cap > EXPECT["cap_hit_hi"]:
                flags.append(f"⚠️ **{arm}: cap-hit {cap*100:.0f}%** > {EXPECT['cap_hit_hi']*100:.0f}% (length runaway).")
            lenN = last.get("completions/mean_length")
            if lenN is not None and lenN > 0.85 * EXPECT["len_drift_hi"]:
                flags.append(f"⚠️ **{arm}: mean completion {lenN:.0f}** approaching cap {EXPECT['len_drift_hi']} (drift).")
    gate = read_json(LOGS / "RA_gate_step50.json")
    if gate and gate["suggested_verdict"].startswith("FLAT"):
        flags.insert(0, f"🛑 **RA step-50 gate → PAUSE** (Δ={gate['delta']:+.4f}). Decision point — operator review required.")
    elif gate:
        flags.insert(0, f"✅ **RA step-50 gate → PROCEED** (Δ={gate['delta']:+.4f}).")
    return flags or ["_None. All arms within expected ranges._"]


# ---------- assemble ----------
def main():
    fz = frozen_summary()
    arms = [classify(a) for a in ARMS]
    now = time.strftime("%Y-%m-%d %H:%M:%S %Z")

    # batch start proxy = earliest lengths.json / jsonl mtime
    starts = [p.stat().st_mtime for p in LOGS.glob("*_lengths.json")] + \
             [p.stat().st_mtime for p in LOGS.glob("*.jsonl")]
    start_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(min(starts))) if starts else "—"

    order = {"RUNNING": 0, "RUNNING?": 0, "PAUSED": 1, "FAILED?": 1, "DONE": 2, "QUEUED": 3}
    running = [a["arm"] for a in arms if a["status"].startswith("RUNNING")]
    queued = [a["arm"] for a in arms if a["status"] == "QUEUED"]
    done = [a["arm"] for a in arms if a["status"] == "DONE"]

    md = []
    md += [f"# T1.3 — LIVE STATUS (auto-generated, read-only)", "",
           f"_Generated: **{now}** · batch start ≈ {start_str} · source: `results/logs/` (volume) + "
           f"`config/t1_3_frozen.md`. Regenerate: `python experiments/T01/results/live_status.py`._", "",
           "> This file is regenerated at checkpoints (arm completion / gate fire), not by continuous "
           "polling — it reads training output only and never touches the training processes.", "",
           "---", "", "## 1 · Run header", "",
           f"- **Frozen config** (`config/t1_3_frozen.md`): SFT LR **{fz['sft_lr']}** (cosine, 2 ep) · "
           f"GRPO LR **{fz['grpo_lr']}** · k=**{fz['k']}** (frozen) · rollout temp {fz['temp']} · β {fz['beta']} · "
           f"seed **{fz['seed']}** · 2 epochs · identical LoRA (r16/α32) all arms.",
           f"- **Running:** {', '.join(running) or '—'}  ·  **Queued:** {', '.join(queued) or '—'}  ·  "
           f"**Done:** {', '.join(done) or '—'}",
           f"- **Estimand:** method (SFT vs GRPO) × cause (coverage vs precision). SA/RA = coverage, "
           f"SB/RB = precision.", "", "---", "", "## 2 · Per-arm status", "",
           "| arm | method · cause | status | step / total | elapsed | ETA | last loss/reward |",
           "|---|---|---|---|---|---|---|"]
    for a in sorted(arms, key=lambda x: order.get(x["status"], 9)):
        lv = a["last_val"]
        lv_s = f"{lv:.4f}" if isinstance(lv, (int, float)) else "—"
        kind = a["last_kind"]
        eta_s = hms(a["eta"]) + (f" · {a['cap_note']}" if a["cap_note"] else "") if a["status"].startswith("RUNNING") else ("done" if a["status"] == "DONE" else "—")
        step_s = f"{a['cur_step']} / {a['total_steps']}" + (f" (~{a['total_steps']}→cap)" if a["budget_s"] and a["status"].startswith("RUNNING") else "")
        md.append(f"| **{a['arm']}** | {a['method']} · {a['cause']} | {status_badge(a['status'])} | "
                  f"{step_s} | {hms(a['elapsed'])} | {eta_s} | {lv_s} ({kind}) |")
    md += ["", "---", "", "## 3 · Per-arm diagnostics", ""]
    for a in arms:
        md += (sft_diag(a) if a["method"] == "SFT" else grpo_diag(a))
    md += ["---", "", "## 4 · Flags / health", ""]
    md += [f"- {f}" for f in health_flags(arms)]
    md += ["", "---", "",
           "## 5 · Artifacts (durable, on /workspace volume — gitignored)", "",
           "| arm | adapter | live log | summary |", "|---|---|---|---|"]
    for a in arms:
        ad = "✅" if a["adapter"] else "—"
        jl = f"`results/logs/{a['tag']}.jsonl`" if (LOGS / f"{a['tag']}.jsonl").exists() else "—"
        sm = f"`results/logs/{a['tag']}_summary.json`" if a["summary"] else ("`RUN_" + a["arm"] + ".md` (SFT)" if a["adapter"] and a["method"] == "SFT" else "—")
        md.append(f"| {a['arm']} | {ad} `results/adapters/T01-{a['arm']}/` | {jl} | {sm} |")
    md += ["", f"_Gate JSON: `results/logs/RA_gate_step50.json`"
           f"{' (present)' if (LOGS/'RA_gate_step50.json').exists() else ' (pending)'}._", ""]

    OUT.write_text("\n".join(md))
    print(f"[live_status] wrote {OUT} ({len(md)} lines) at {now}")


def status_badge(s: str) -> str:
    return {"RUNNING": "🟢 running", "RUNNING?": "🟡 running?", "DONE": "✅ done",
            "QUEUED": "⚪ queued", "PAUSED": "⏸️ PAUSED", "FAILED?": "🔴 FAILED?"}.get(s, s)


if __name__ == "__main__":
    main()
