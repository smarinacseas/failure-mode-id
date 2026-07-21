/* FMID: shared helpers for the FailureModeID pages. No build step. */
(function () {
  "use strict";

  var FMID = {};

  /* ---------- theme ---------- */
  FMID.currentTheme = function () {
    try { return localStorage.getItem("fmid-theme") === "light" ? "light" : "dark"; }
    catch (e) { return "dark"; }
  };
  FMID.initTheme = function () {
    document.documentElement.dataset.theme = FMID.currentTheme();
  };
  FMID.toggleTheme = function () {
    var next = FMID.currentTheme() === "dark" ? "light" : "dark";
    try { localStorage.setItem("fmid-theme", next); } catch (e) {}
    document.documentElement.dataset.theme = next;
    return next;
  };

  /* ---------- nav ---------- */
  FMID.navHTML = function (active) {
    function link(href, label, key) {
      var cur = active === key ? ' aria-current="page"' : "";
      return '<a href="' + href + '"' + cur + ">" + label + "</a>";
    }
    return (
      '<div class="fmid-nav-inner">' +
      '<a class="fmid-nav-brand" href="./index.html">' +
      '<span style="width:21px;height:21px;border-radius:6px;background:var(--ink);display:inline-flex;align-items:center;justify-content:center">' +
      '<span style="width:8px;height:8px;border:2px solid var(--surface);border-radius:50%"></span></span>' +
      "FailureModeID</a>" +
      '<nav class="fmid-nav-links">' +
      link("./index.html", "Findings", "home") +
      link("./eval.html", "E-series Eval", "eval") +
      link("./t01.html", "T01 Training", "t01") +
      link("./eval.html#tab=reference", "Reference", "reference") +
      "</nav>" +
      '<button class="fmid-theme-btn" type="button" data-fmid-theme-btn></button>' +
      "</div>"
    );
  };
  FMID.mountNav = function (el, active) {
    el.className = "fmid-nav";
    el.innerHTML = FMID.navHTML(active);
    var btn = el.querySelector("[data-fmid-theme-btn]");
    function label() { btn.textContent = FMID.currentTheme() === "dark" ? "Light mode" : "Dark mode"; }
    btn.addEventListener("click", function () { FMID.toggleTheme(); label(); });
    label();
  };

  /* ---------- tooltips ---------- */
  FMID.TIPS = {
    "judge-agreement": { title: "Judge agreement",
      body: "How often the panel's judges cast the same pass/fail vote on the same criterion. Fleiss' kappa corrects raw agreement for chance; 0 means chance-level, 1 means perfect agreement." },
    "verifiability": { title: "Verifiability",
      body: "Auto-verifiable criteria can be checked deterministically (counts, exact phrases, formatting). Judge-required criteria need a model judge's reading. Pass rates are split so judge noise can be isolated." },
    "panel-consensus": { title: "Panel consensus",
      body: "A synthetic judge whose verdict on each criterion is the majority vote of the judge panel. It reduces single-judge bias; per-judge views remain available in the Judge switcher." },
    "root-cause": { title: "Root-cause taxonomy",
      body: "Every failed criterion is diagnosed by a blinded classifier into causes such as: never engaged the constraint (coverage), engaged but executed wrong (precision), misread the constraint or input, degenerate output, or judge-suspect." },
    "coverage": { title: "Coverage failure",
      body: "The model never engages a stated requirement at all (root cause constraint_unaddressed). In the E08 census this was 28.8% of diagnosed failures." },
    "precision": { title: "Precision failure",
      body: "The model engages the requirement but executes it incorrectly (root cause execution_slip). In the E08 census this was 36.4% of diagnosed failures." },
    "grpo": { title: "GRPO",
      body: "Group Relative Policy Optimization: on-policy reinforcement learning where the model samples several answers per prompt and is updated toward the ones a verifier scores higher, with a KL penalty keeping it near the base model." },
    "sft": { title: "SFT",
      body: "Supervised fine-tuning: the model imitates teacher-written target answers with a standard next-token loss. Here the teacher data was distilled from a stronger model using a scaffolded prompt." },
    "lora": { title: "LoRA",
      body: "Low-Rank Adaptation: instead of updating all weights, small low-rank matrices are trained and added to the frozen base model. Cheaper, but lower capacity than full fine-tuning." },
    "interaction": { title: "Interaction effect",
      body: "The difference of differences: (GRPO minus SFT on precision) minus (GRPO minus SFT on coverage). It isolates whether the training method's advantage depends on the failure type, and cancels shared confounds such as extra compute." },
    "recovery": { title: "Recovery",
      body: "Of the criteria the base model failed, the fraction the trained model now passes (majority vote over 3 decodes). The primary estimand of T01." },
    "breakage": { title: "Breakage",
      body: "Of the criteria the base model passed, the fraction the trained model now fails. The complement of recovery: an arm can look good on recovery while quietly breaking what worked." },
    "cluster-bootstrap": { title: "Cluster bootstrap",
      body: "Confidence intervals computed by resampling whole prompts (clusters), not individual criteria, because criteria within one prompt are correlated. 10,000 resamples, fixed seed." },
    "estimand": { title: "Estimand",
      body: "The precise quantity an experiment commits to estimating before running. T01's estimand is the recovery interaction, fixed in the pre-registration." },
    "prereg": { title: "Pre-registered",
      body: "Hypotheses, decision rules, and analyses fixed and committed to git before training ran. Pre-registered claims are confirmatory; everything else is labeled post-hoc and is exploratory." },
    "posthoc": { title: "Post-hoc",
      body: "Analysis chosen after seeing results. Useful for understanding mechanisms, but carries multiple-comparisons and hindsight risk, so it is labeled and never treated as confirmatory." }
  };

  var pop = null;
  function closeTip() { if (pop) { pop.remove(); pop = null; } }
  function openTip(el) {
    closeTip();
    var def = FMID.TIPS[el.getAttribute("data-tip")] ||
      { title: "", body: el.getAttribute("data-tip-text") || "" };
    pop = document.createElement("div");
    pop.className = "fmid-tip-pop";
    pop.innerHTML = (def.title ? "<b></b>" : "");
    if (def.title) pop.querySelector("b").textContent = def.title;
    pop.appendChild(document.createTextNode(def.body));
    document.body.appendChild(pop);
    var r = el.getBoundingClientRect();
    var top = r.bottom + window.scrollY + 7;
    var left = Math.min(r.left + window.scrollX - 8,
      window.scrollX + document.documentElement.clientWidth - pop.offsetWidth - 12);
    pop.style.top = top + "px";
    pop.style.left = Math.max(12, left) + "px";
  }
  FMID.initTips = function () {
    if (FMID._tipsWired) return;
    FMID._tipsWired = true;
    document.addEventListener("mouseover", function (e) {
      var t = e.target.closest && e.target.closest(".fmid-tip");
      if (t) openTip(t);
    });
    document.addEventListener("mouseout", function (e) {
      if (e.target.closest && e.target.closest(".fmid-tip")) closeTip();
    });
    // Tap/keyboard: click toggles, Escape and outside-click close.
    document.addEventListener("click", function (e) {
      var t = e.target.closest && e.target.closest(".fmid-tip");
      if (t) { pop ? closeTip() : openTip(t); e.preventDefault(); }
      else if (pop && !e.target.closest(".fmid-tip-pop")) closeTip();
    });
    document.addEventListener("keydown", function (e) {
      if (e.key === "Escape") closeTip();
    });
  };

  /* ---------- curated run metadata (safe from dashboard_sync) ---------- */
  FMID.RUN_META = {
    "E01-smoke-3p":        { tier: "dev", takeaway: "First 3-prompt smoke: pipeline plumbing works end to end." },
    "E02-smoke-fable-3p":  { tier: "dev", takeaway: "Judge-swap smoke (Fable): grading transport and schema hold up." },
    "E03-judge-compare-3p":{ tier: "dev", takeaway: "Two judges on identical responses: verdict deltas are real, judge choice matters." },
    "E04-reasoning-smoke-3p": { tier: "dev", takeaway: "Reasoning-mode smoke: thinking budgets and decode health wiring validated." },
    "E05-reasoning-rand20p": { tier: "primary", takeaway: "Reasoning on vs off, 20-prompt stratified sample: reasoning helps, motivating the full-benchmark E07." },
    "E06-temp06-3p":       { tier: "dev", takeaway: "Temperature 0.6 spot-check: no verdict-changing effect at smoke scale." },
    "E07-reasoning-full75":{ tier: "primary", takeaway: "Full 75-prompt replication: reasoning-on gains hold at population scale across the Qwen ladder." },
    "E08-llama3-2-3b-cc75":{ tier: "primary", takeaway: "Failure census of Llama-3.2-3B: 43.4% criterion pass, 0% full-prompt pass; execution slips (36.4%) and unaddressed constraints (28.8%) dominate. Basis for T01." },
    "E80-drysmoke-3p":     { tier: "dev", takeaway: "Dry-run smoke: no-API-cost path for pipeline changes." },
    "E91-panel-smoke":     { tier: "dev", takeaway: "Judge-panel smoke: multi-judge consensus rollup validated before full runs." }
  };
  FMID.runMeta = function (id) { return FMID.RUN_META[id] || null; };

  /* ---------- misc ---------- */
  FMID.fmtPct = function (x) {
    var n = (x == null || isNaN(x)) ? 0 : Number(x);
    return Math.round(n * 10) / 10 + "%";
  };

  window.FMID = FMID;
})();
