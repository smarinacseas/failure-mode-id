# Judge-validation labeling — setup

Human PASS/FAIL labeling for `runs/<slug>/judge_validation.json`, done in the
browser on the GitHub Pages dashboard and committed back to the repo via the
GitHub API. One-time setup below takes ~10 minutes.

## How it fits together

```
labeling/sync.py ──► dashboard/labeling/<slug>.json   (60 rows + full responses; static, served by Pages)
                                   │
dashboard/label.html  ◄────────────┘   blind labeling UI (judge verdict hidden until you vote)
        │  GitHub OAuth (token via Cloudflare Worker proxy — Pages can't hold the secret)
        ▼
commits runs/<slug>/judge_validation.json  ──►  `labels` branch   (runs/ is gitignored on main,
                                   │             so labels get their own branch; commits there
labeling/pull.py  ◄────────────────┘             also don't trigger a Pages redeploy)
        ▼
runs/<slug>/judge_validation.json locally  ──►  main.py validate --mode score
```

The committed file is byte-for-byte the `validate --mode sample` schema with
`human` filled in — `--mode score` reads it as-is.

## 1. Create the GitHub OAuth app (~2 min)

1. GitHub → Settings → Developer settings → **OAuth Apps** → *New OAuth App*.
2. Fill in:
   - **Application name**: `ConstraintLens labeling` (anything)
   - **Homepage URL**: `https://smarinacseas.github.io/failure-mode-id/`
   - **Authorization callback URL**: `https://smarinacseas.github.io/failure-mode-id/label.html`
3. Register, then note the **Client ID** and generate a **Client secret**
   (shown once — copy it).

The app only needs the `public_repo` scope (requested by label.html). If the
repo ever goes private, change `scope` in `label.html` `signIn()` to `repo`.

## 2. Deploy the OAuth proxy (Cloudflare Worker, free tier) (~5 min)

The repo has no existing serverless setup, so this uses a Cloudflare Worker.
You need a (free) Cloudflare account and Node installed.

```bash
cd proxy/oauth-worker
# edit wrangler.toml: set GITHUB_CLIENT_ID to your client id
npx wrangler login          # first time only
npx wrangler deploy         # prints the worker URL
npx wrangler secret put GITHUB_CLIENT_SECRET   # paste the client secret
```

`wrangler.toml` vars to review:

- `GITHUB_CLIENT_ID` — from step 1.
- `ALLOWED_USERS` — comma-separated GitHub logins allowed to label
  (default `smarinacseas`). Enforced server-side; anyone else gets 403.
- `ALLOWED_ORIGINS` — origins allowed to call the worker. Defaults to the
  Pages origin plus `http://localhost:8000` for local testing; drop the
  localhost entry once you're done testing.

Note the deployed URL, e.g. `https://constraintlens-oauth.<account>.workers.dev`.

## 3. Configure and publish the labeling page (~2 min)

1. Edit the `CONFIG` block at the top of `dashboard/label.html`:
   - `oauthClientId` — from step 1.
   - `proxyUrl` — from step 2.
   - (`owner`/`repo`/`allowlist`/`labelsBranch` are pre-set for this repo.)
2. The data bundle `dashboard/labeling/E08-llama3-2-3b-cc75.json` is already
   generated. After any future `validate --mode sample`, regenerate with
   `uv run python labeling/sync.py --experiment <slug>`.
3. Commit `dashboard/label.html`, `dashboard/labeling/`, `labeling/`,
   `proxy/` and push to **main** — the existing
   `.github/workflows/dashboard-deploy.yml` publishes `dashboard/` to Pages
   automatically (Pages is already enabled for this repo; if it ever isn't:
   repo → Settings → Pages → Source: *GitHub Actions*).

## 4. Label, then score

1. Open `https://smarinacseas.github.io/failure-mode-id/label.html`
   (other runs: `label.html?run=<slug>`).
2. **Sign in with GitHub** → authorize.
3. Vote PASS/FAIL per row (`p` / `f` / `x` clear / `←` `→` navigate). The
   judge's identity, verdict and reason stay hidden until you vote, then a
   reveal panel shows agreement. Full response text is shown, not the
   800-char excerpt. Labels are staged in the browser (localStorage) until
   you press **Commit** — one commit per batch, to the `labels` branch
   (created automatically on first commit).
4. Back on your machine:

   ```bash
   uv run python labeling/pull.py --experiment E08-llama3-2-3b-cc75
   uv run python main.py validate --experiment E08-llama3-2-3b-cc75 --mode score
   ```

## Local testing (optional)

```bash
python -m http.server 8000 --directory dashboard
# add http://localhost:8000/label.html as a second OAuth app callback, or just
# test the UI unauthenticated (labels stage locally without signing in)
```

## Security notes

- The client secret lives only in the Worker (`wrangler secret`); Pages ships
  no secrets. The user token is kept in `sessionStorage` (cleared when the
  tab closes) and is only sent to `api.github.com`.
- Two write barriers: the Worker's `ALLOWED_USERS` allowlist (won't mint
  tokens for other logins), and GitHub itself (a token can only push to repos
  its user can write to). The `allowlist` in `label.html` is UX-only — keep
  it in sync with the Worker.
- Blinding is procedural: the verdict data exists in the public bundle and
  repo; the UI just never renders it pre-vote. Don't peek.
