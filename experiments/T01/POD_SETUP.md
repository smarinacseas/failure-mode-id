# T01 pod setup — RunPod A100 → Gate GT0

Step-by-step to take a freshly-deployed RunPod A100 from zero to a validated
training env (**Gate GT0**: env reproduces + smoke test passes). Copy-paste in
order. Total ~20–30 min, most of it the weights download.

**Roles:** this local (Mac) Claude Code session is the planner/scaffolder; a
Claude Code session **on the pod** drives the GPU work from Phase T1.1 on.
**Cost:** the A100 bills ~$1.20–1.80/hr — so we clear GT0, then **Stop the pod**
(step 8) until data-gen/training.

Prereqs you already have: pod deployed, HF token with the `meta-llama/Llama-3.2-3B`
license accepted, and working Claude Code auth on your Mac (this session).

---

## 1 · SSH into the pod (~2 min)

One-time — register your Mac's public key with RunPod:

```bash
# on your MAC — make a key if you don't have one:
ls ~/.ssh/id_ed25519.pub || ssh-keygen -t ed25519 -C "runpod"
cat ~/.ssh/id_ed25519.pub          # copy this
```

Paste it into **RunPod → Settings → SSH Public Keys**. Then open your pod →
**Connect** and copy the SSH command RunPod shows (either the proxy form
`ssh <id>@ssh.runpod.io -i ~/.ssh/id_ed25519` or a direct
`ssh root@<ip> -p <port> -i ~/.ssh/id_ed25519`). Run it from your Mac terminal.
*(Fallback: the console's browser "Web Terminal" needs no SSH.)*

Start a `tmux` so a dropped SSH doesn't kill a long run:

```bash
# on the POD:
apt-get update -qq && apt-get install -y -qq tmux git
tmux new -s t01                    # reattach later with: tmux attach -t t01
```

## 2 · Install Claude Code on the pod (~5 min)

```bash
# on the POD — native installer (no Node needed):
curl -fsSL https://claude.ai/install.sh | bash
exec bash                          # reload PATH
claude --version
```

**Authenticate (headless — pick ONE).** OpenRouter keys do **not** work for
Claude Code; it needs Anthropic auth. Since you already run Claude Code on your
Mac, the cleanest path reuses that:

```bash
# RECOMMENDED — reuses your Claude subscription, no extra API billing.
# Run ONCE on your MAC (it has a browser):
claude setup-token                 # prints a long-lived OAuth token — copy it
# then on the POD:
export CLAUDE_CODE_OAUTH_TOKEN="paste-the-token"
```

Alternative if you have an Anthropic API key (console.anthropic.com):
`export ANTHROPIC_API_KEY="sk-ant-..."` on the pod.

Persist Claude's config on the volume so a pod restart keeps your auth:

```bash
export CLAUDE_CONFIG_DIR=/workspace/.claude && mkdir -p $CLAUDE_CONFIG_DIR
```

## 3 · Clone the repo (~1 min)

`/workspace` is RunPod's persistent volume — put everything there so a Stop/Start
keeps it.

```bash
cd /workspace
git clone https://github.com/smarinacseas/failure-mode-id.git
cd failure-mode-id && git checkout e08-t01
```

## 4 · Secrets — recreate `.env` on the pod (~1 min)

`.env` is gitignored, so it's not in the clone. T01 needs two keys (OpenRouter
for the SFT teacher + Tier-2 judging; HF for the gated weights):

```bash
# on the POD, in /workspace/failure-mode-id:
cat > .env <<'EOF'
OPENROUTER_API_KEY=sk-or-v1-...your-key...
HF_TOKEN=hf_...your-token...
EOF
```

Export the HF token + cache weights on the volume (avoids re-download after a
restart):

```bash
export HF_TOKEN=$(grep HF_TOKEN .env | cut -d= -f2)
export HF_HOME=/workspace/.hf
```

## 5 · Python env — TRL stack (~5–10 min)

The RunPod PyTorch template already ships a CUDA-matched `torch`. Use the pod's
base `python`/`pip` (the repo's `uv run` convention is a local-Mac thing —
`python main.py …` works anywhere the deps are installed).

**No vLLM.** Gate GT0 proved vLLM forces an incompatible torch/CUDA downgrade (a
4-round spiral); training runs on stock `model.generate()`
(`GRPOConfig(use_vllm=False)` — PREREG amendment 2026-07-16). Install the frozen
GT0 lock directly — do **not** `pip install vllm`:

```bash
# on the POD, in /workspace/failure-mode-id:
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
pip install -q -r requirements-t01.txt    # frozen GT0 lock — torch 2.5.1+cu121, trl 1.8.0, NO vllm
```

The frozen lock *is* the resolution of the earlier torch/CUDA/vLLM conflict — it
reproduces the exact GT0-validated env; the smoke test (next step) revalidates it.

## 6 · Download weights + smoke test → **Gate GT0** (~5–10 min)

The smoke test downloads Llama-3.2-3B (~6 GB, cached to `$HF_HOME`), runs 5
generations, takes one LoRA step, and imports trl (vLLM intentionally absent):

```bash
# on the POD, in /workspace/failure-mode-id:
python experiments/T01/smoke_test.py
```

**Gate GT0 clears** when it prints `GT0 SMOKE: PASS`. If it fails on a version
conflict, tighten the offending pin in `experiments/T01/requirements.txt`,
reinstall, rerun — and log the change (PREREG §9: no silent config drift).

## 7 · Hand off to the pod's Claude Code

```bash
# on the POD, in /workspace/failure-mode-id:
claude                              # interactive; or: claude -p "…" --allowedTools "Bash,Read,Edit"
```

From here the pod session drives T1.1 (verifiers) → T1.2 (data-gen) → T1.3
(training) → T1.4–T1.5 (eval), following `PREREG.md`. Long runs go in `tmux`.

## 8 · Stop the pod (billing hygiene)

Once GT0 is green and you're pausing before data-gen/training: **RunPod console →
Stop** (not Terminate). Compute billing halts; `/workspace` (repo + cached
weights + Claude auth) persists, so Start resumes in seconds. Storage cost while
stopped is small.

---

### Quick reference — the whole thing, on the pod

```bash
apt-get update -qq && apt-get install -y -qq tmux git && tmux new -s t01
curl -fsSL https://claude.ai/install.sh | bash && exec bash
export CLAUDE_CODE_OAUTH_TOKEN="…"     # from `claude setup-token` on your Mac
export CLAUDE_CONFIG_DIR=/workspace/.claude && mkdir -p $CLAUDE_CONFIG_DIR
cd /workspace && git clone https://github.com/smarinacseas/failure-mode-id.git
cd failure-mode-id && git checkout e08-t01
cat > .env <<'EOF'
OPENROUTER_API_KEY=…
HF_TOKEN=…
EOF
export HF_TOKEN=$(grep HF_TOKEN .env | cut -d= -f2) HF_HOME=/workspace/.hf
pip install -q -r requirements-t01.txt     # frozen GT0 lock — NO vllm (see §5)
python experiments/T01/smoke_test.py     # → GT0 SMOKE: PASS
```
