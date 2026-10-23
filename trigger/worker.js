// Cloudflare Worker: 手機頁「補跑這天」代理。POST /replay {pin, date, workflow} → GitHub workflow_dispatch; GET /runs?workflow=signals 看最近執行。
// 變數 (Workers → Settings → Variables and Secrets): GH_TOKEN (GitHub fine-grained token, 只給 quant_trading 的 Actions: Read and write, 設為 Secret), API_PIN (自訂, Secret)。
const OWNER = "raywu918daytrade", REPO = "quant_trading";
const WORKFLOWS = { signals: "daily_signals.yml", paper: "paper_trade.yml" };
const ALLOW = ["https://quant.just1stock.com", "https://raywu918daytrade.github.io", "http://localhost:8790"];
let lastFire = {};   // 節流 (每個 isolate 各自計, 盡力而為)

function cors(req) {
  const o = req.headers.get("Origin") || "";
  const h = { "Access-Control-Allow-Methods": "GET,POST,OPTIONS", "Access-Control-Allow-Headers": "Content-Type", "Access-Control-Max-Age": "86400" };
  if (ALLOW.includes(o)) h["Access-Control-Allow-Origin"] = o;
  return h;
}
const json = (obj, status, req) => new Response(JSON.stringify(obj), { status, headers: { "Content-Type": "application/json; charset=utf-8", ...cors(req) } });

export default {
  async fetch(req, env) {
    const url = new URL(req.url); const h = cors(req);
    if (req.method === "OPTIONS") return new Response(null, { status: 204, headers: h });
    if (url.pathname === "/" ) return json({ ok: true, repo: `${OWNER}/${REPO}` }, 200, req);
    if (!env.GH_TOKEN) return json({ detail: "GH_TOKEN not set" }, 500, req);
    const gh = { "Authorization": `Bearer ${env.GH_TOKEN}`, "Accept": "application/vnd.github+json", "User-Agent": "quant-trigger" };
    if (url.pathname === "/runs" && req.method === "GET") {
      const wf = WORKFLOWS[url.searchParams.get("workflow") || "signals"] || WORKFLOWS.signals;
      const r = await fetch(`https://api.github.com/repos/${OWNER}/${REPO}/actions/workflows/${wf}/runs?per_page=${url.searchParams.get("n") || 5}`, { headers: gh });
      if (r.status !== 200) return json({ detail: `GitHub ${r.status}` }, 502, req);
      const j = await r.json();
      return json((j.workflow_runs || []).map(x => ({ created_at: x.created_at, status: x.status, conclusion: x.conclusion, event: x.event, url: x.html_url })), 200, req);
    }
    if (url.pathname === "/replay" && req.method === "POST") {
      let b; try { b = await req.json(); } catch { return json({ detail: "bad json" }, 400, req); }
      if (!env.API_PIN || b.pin !== env.API_PIN) return json({ detail: "PIN 錯誤" }, 401, req);
      const date = (b.date || "").trim();
      if (date && !/^(\d{4}-\d{2}(-\d{2})?|\d{4}-\d{2}-\d{2}\.\.\d{4}-\d{2}-\d{2})$/.test(date)) return json({ detail: "date 格式要 YYYY-MM-DD / YYYY-MM / A..B" }, 400, req);
      const wf = WORKFLOWS[b.workflow || "signals"]; if (!wf) return json({ detail: "workflow 只能是 signals / paper" }, 400, req);
      if (Date.now() - (lastFire[wf] || 0) < 60000) return json({ detail: "60 秒內已觸發過，請稍候" }, 429, req);
      // 伺服器端防重複: GitHub 上同一 workflow 有排隊/執行中 → 不再觸發 (isolate 的 lastFire 不可靠)
      const rr = await fetch(`https://api.github.com/repos/${OWNER}/${REPO}/actions/workflows/${wf}/runs?per_page=5`, { headers: gh });
      if (rr.status === 200) {
        const active = ((await rr.json()).workflow_runs || []).find(x => x.status === "queued" || x.status === "in_progress" || x.status === "waiting");
        if (active) return json({ detail: "GitHub 已有執行中的補跑，請等它完成", runs_url: active.html_url }, 429, req);
      }
      const r = await fetch(`https://api.github.com/repos/${OWNER}/${REPO}/actions/workflows/${wf}/dispatches`, { method: "POST", headers: { ...gh, "Content-Type": "application/json" }, body: JSON.stringify({ ref: "main", inputs: { date } }) });
      if (r.status !== 204) return json({ detail: `GitHub ${r.status}: ${(await r.text()).slice(0, 200)}` }, 502, req);
      lastFire[wf] = Date.now();
      return json({ ok: true, workflow: wf, date: date || "today", runs_url: `https://github.com/${OWNER}/${REPO}/actions/workflows/${wf}` }, 200, req);
    }
    return json({ detail: "not found" }, 404, req);
  }
};
