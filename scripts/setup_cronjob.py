"""用 cron-job.org API 建立/更新「平日 09:01 觸發 GitHub daily_signals workflow」的定時工作。

需要兩個環境變數 (值絕不印出):
  CRONJOB_API_KEY     cron-job.org 的 API key (console.cron-job.org → 設定 → API)
  GH_DISPATCH_TOKEN   GitHub fine-grained token, 只需 quant_trading repo 的 Actions: Read and write

用法:
  CRONJOB_API_KEY=... GH_DISPATCH_TOKEN=... python3 scripts/setup_cronjob.py          # 建立或更新 job
  CRONJOB_API_KEY=... GH_DISPATCH_TOKEN=... python3 scripts/setup_cronjob.py --test   # 另外直接打一次 GitHub dispatch 驗證 token
  python3 scripts/setup_cronjob.py --env-file /path/to/.env --test   # 直接從 .env 讀 CRONJOB_API_KEY_DAY_TRADE / CORN_JOB_TOKEN_DAY_TRADE
"""
import os, sys, json, requests

OWNER, REPO, WF = "raywu918daytrade", "quant_trading", "daily_signals.yml"
TITLE = "quant_trading daily signals 09:01"
DISPATCH_URL = f"https://api.github.com/repos/{OWNER}/{REPO}/actions/workflows/{WF}/dispatches"
API = "https://api.cron-job.org"

def read_env_file(path):
    """寬鬆解析 .env: 忽略註解/空行, 容許 'KEY = value'、引號、export 前綴. 回傳 dict (值不印出)."""
    out = {}
    for line in open(path, encoding="utf-8", errors="ignore"):
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line: continue
        if line.startswith("export "): line = line[7:]
        k, v = line.split("=", 1); k = k.strip(); v = v.strip()
        if len(v) >= 2 and v[0] == v[-1] and v[0] in "\"'": v = v[1:-1]
        out[k] = v
    return out

def main():
    key = os.environ.get("CRONJOB_API_KEY"); gh = os.environ.get("GH_DISPATCH_TOKEN")
    if "--env-file" in sys.argv:   # 直接讀 .env (可多個), 不靠 shell source
        env = {}
        for i, a in enumerate(sys.argv):
            if a == "--env-file": env.update(read_env_file(sys.argv[i + 1]))
        key = key or env.get("CRONJOB_API_KEY_DAY_TRADE") or env.get("CORN_JOB_TOKEN_DAY_TRADE") or env.get("CRONJOB_API_KEY")
        gh = gh or env.get("GITHUB_DISPATCH_TOKEN_DAY_TRADE") or env.get("GITHUB_DAY_TRADE") or env.get("GH_DISPATCH_TOKEN") or env.get("GITHUB_RAYWU918PYTHON")
        print("在 .env 找到的候選變數:", [k for k in env if "CRON" in k.upper() or "CORN" in k.upper() or "DISPATCH" in k.upper() or "GITHUB" in k.upper()])
    if bool(key) != bool(gh):   # 只有一把: 判斷它是哪一邊的 (只看狀態碼, 不印值)
        tok = key or gh
        is_cron = requests.get(f"{API}/jobs", headers={"Authorization": f"Bearer {tok}"}, timeout=15).status_code == 200
        is_gh = requests.get("https://api.github.com/user", headers={"Authorization": f"Bearer {tok}"}, timeout=15).status_code == 200
        kind = "cron-job.org API key" if is_cron else "GitHub token" if is_gh else "兩邊都認不得 (可能貼錯或已失效)"
        print(f"目前這把 token 是: {kind}")
        if is_cron and not is_gh: key, gh = tok, None
        elif is_gh and not is_cron: key, gh = None, tok
    missing = [n for n, v in (("CRONJOB_API_KEY_DAY_TRADE (cron-job.org 設定→API 產生的 key)", key), ("GITHUB_DISPATCH_TOKEN_DAY_TRADE (GitHub fine-grained token, Actions 寫入權)", gh)) if not v]
    if missing: sys.exit("錯誤: 還缺 " + "；".join(missing) + "，請加進 .env 後再跑一次")
    h = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}

    if "--test" in sys.argv:
        r = requests.post(DISPATCH_URL, headers={"Authorization": f"Bearer {gh}", "Accept": "application/vnd.github+json"},
                          json={"ref": "main"}, timeout=20)
        print("GitHub dispatch 測試:", r.status_code, "(204 = 成功觸發)" if r.status_code == 204 else r.text[:200])
        if r.status_code != 204:
            sys.exit("這把 GitHub token 無法觸發 raywu918daytrade/quant_trading 的 workflow (404/403 = 沒權限)。"
                     "請用 raywu918daytrade 帳號到 https://github.com/settings/personal-access-tokens/new 建 fine-grained token"
                     " (只選 quant_trading repo, Actions: Read and write)，加進 .env 叫 GITHUB_DISPATCH_TOKEN_DAY_TRADE 再跑。")

    job = {"title": TITLE, "url": DISPATCH_URL, "enabled": True, "saveResponses": True,
           "requestMethod": 1,  # POST
           "extendedData": {"headers": {"Authorization": f"Bearer {gh}", "Accept": "application/vnd.github+json",
                                        "Content-Type": "application/json"}, "body": '{"ref":"main"}'},
           "schedule": {"timezone": "Asia/Taipei", "expiresAt": 0, "hours": [9], "minutes": [1], "mdays": [-1], "months": [-1], "wdays": [1, 2, 3, 4, 5]}}

    r = requests.get(f"{API}/jobs", headers=h, timeout=20); r.raise_for_status()
    existing = [j for j in r.json().get("jobs", []) if j.get("title") == TITLE]
    if existing:
        jid = existing[0]["jobId"]
        r = requests.patch(f"{API}/jobs/{jid}", headers=h, json={"job": job}, timeout=20); r.raise_for_status()
        print(f"已更新 cron-job.org job {jid}: {TITLE}")
    else:
        r = requests.put(f"{API}/jobs", headers=h, json={"job": job}, timeout=20); r.raise_for_status()
        jid = r.json().get("jobId"); print(f"已建立 cron-job.org job {jid}: {TITLE}")
    r = requests.get(f"{API}/jobs/{jid}", headers=h, timeout=20); r.raise_for_status()
    jd = r.json().get("jobDetails", {})
    print("schedule:", json.dumps(jd.get("schedule"), ensure_ascii=False), "| method:", jd.get("requestMethod"), "| enabled:", jd.get("enabled"))
    print("url:", jd.get("url"))

if __name__ == "__main__":
    main()
