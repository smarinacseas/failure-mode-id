/**
 * ConstraintLens OAuth proxy: Cloudflare Worker.
 *
 * GitHub Pages is static, so it can't hold the OAuth app's client secret.
 * This worker does exactly one privileged thing: exchange the authorization
 * `code` (POST /exchange) for a user access token, verify the user is on the
 * labeling allowlist, and hand the token back to dashboard/label.html.
 * The token never touches this worker again; all repo reads/writes happen
 * browser → api.github.com directly.
 *
 * Bindings (see wrangler.toml / SETUP.md):
 *   GITHUB_CLIENT_ID      var     OAuth app client id
 *   GITHUB_CLIENT_SECRET  secret  `wrangler secret put GITHUB_CLIENT_SECRET`
 *   ALLOWED_USERS         var     comma-separated GitHub logins allowed to label
 *   ALLOWED_ORIGINS       var     comma-separated origins allowed to call this
 *                                 worker (the Pages origin; add
 *                                 http://localhost:8000 while testing)
 */

const JSON_HEADERS = { "Content-Type": "application/json" };

function corsHeaders(request, env) {
  const origin = request.headers.get("Origin") || "";
  const allowed = (env.ALLOWED_ORIGINS || "")
    .split(",").map((s) => s.trim()).filter(Boolean);
  if (!allowed.includes(origin)) return null;
  return {
    "Access-Control-Allow-Origin": origin,
    "Access-Control-Allow-Methods": "POST, OPTIONS",
    "Access-Control-Allow-Headers": "Content-Type",
    "Vary": "Origin",
  };
}

function json(body, status, extraHeaders = {}) {
  return new Response(JSON.stringify(body), {
    status,
    headers: { ...JSON_HEADERS, ...extraHeaders },
  });
}

export default {
  async fetch(request, env) {
    const url = new URL(request.url);

    if (url.pathname === "/" && request.method === "GET") {
      return new Response("constraintlens oauth proxy: POST /exchange {code}", { status: 200 });
    }

    if (url.pathname !== "/exchange") {
      return json({ error: "not_found" }, 404);
    }

    const cors = corsHeaders(request, env);
    if (request.method === "OPTIONS") {
      return cors
        ? new Response(null, { status: 204, headers: cors })
        : json({ error: "origin_not_allowed" }, 403);
    }
    if (!cors) return json({ error: "origin_not_allowed" }, 403);
    if (request.method !== "POST") {
      return json({ error: "method_not_allowed" }, 405, cors);
    }

    let code;
    try {
      ({ code } = await request.json());
    } catch {
      return json({ error: "bad_request", error_description: "JSON body with `code` required" }, 400, cors);
    }
    if (!code) {
      return json({ error: "bad_request", error_description: "`code` missing" }, 400, cors);
    }

    // 1. code -> token (this is the step that needs the client secret).
    const tokenRes = await fetch("https://github.com/login/oauth/access_token", {
      method: "POST",
      headers: { ...JSON_HEADERS, "Accept": "application/json" },
      body: JSON.stringify({
        client_id: env.GITHUB_CLIENT_ID,
        client_secret: env.GITHUB_CLIENT_SECRET,
        code,
      }),
    });
    const token = await tokenRes.json().catch(() => ({}));
    if (!tokenRes.ok || token.error || !token.access_token) {
      return json({
        error: token.error || "exchange_failed",
        error_description: token.error_description || `GitHub returned ${tokenRes.status}`,
      }, 400, cors);
    }

    // 2. Who is this? (GitHub requires a User-Agent on API calls.)
    const userRes = await fetch("https://api.github.com/user", {
      headers: {
        "Authorization": `Bearer ${token.access_token}`,
        "Accept": "application/vnd.github+json",
        "User-Agent": "constraintlens-oauth-proxy",
      },
    });
    if (!userRes.ok) {
      return json({ error: "user_lookup_failed", error_description: `GitHub returned ${userRes.status}` }, 502, cors);
    }
    const login = (await userRes.json()).login || "";

    // 3. Allowlist gate: don't hand tokens to logins we don't recognize.
    //    (GitHub repo permissions are the real write barrier; this just keeps
    //    the labeling flow closed by default.)
    const allowedUsers = (env.ALLOWED_USERS || "")
      .split(",").map((s) => s.trim().toLowerCase()).filter(Boolean);
    if (allowedUsers.length && !allowedUsers.includes(login.toLowerCase())) {
      return json({ error: "not_allowed", login }, 403, cors);
    }

    return json({ access_token: token.access_token, login, scope: token.scope || "" }, 200, cors);
  },
};
