"""LINE 推播 (Messaging API push)。給 paper_trade.py 送開倉 / 平倉 / 每日結算通知。

LINE Notify 已於 2025-03-31 停止服務, 這裡走 LINE 官方帳號的 Messaging API:
  1. https://developers.line.biz/console/ → 建 Provider → 建 Messaging API channel (會一併建立一個 LINE 官方帳號)
  2. Messaging API 分頁 → Channel access token (long-lived) → Issue  → 環境變數 LINE_CHANNEL_ACCESS_TOKEN
  3. Basic settings 分頁最下方「Your user ID」(U 開頭 33 碼)              → 環境變數 LINE_USER_ID
  4. 用手機掃該官方帳號的 QR code 加好友 (沒加好友收不到 push)
  免費方案每月約 200 則 push; 一個交易日約 3~4 則 (開倉、強制回補、收盤回補、結算)。

兩個值只從環境變數 (或 --env-file) 讀, 絕不印出。沒設定 → push() 回 False 並略過, 不會讓呼叫端掛掉。

用法:
  python3 scripts/notify_line.py --env-file ~/.../.env --check          # 驗證 token: 印出官方帳號名稱
  python3 scripts/notify_line.py --env-file ~/.../.env "測試訊息"        # 推一則測試
"""
import os, sys
import requests

API = "https://api.line.me/v2/bot"
MAX_LEN = 5000   # LINE text message 上限

def load_env_file(path):
    for line in open(path, encoding="utf-8", errors="ignore"):
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line: continue
        if line.startswith("export "): line = line[7:]
        k, v = line.split("=", 1); k = k.strip(); v = v.strip()
        if len(v) >= 2 and v[0] == v[-1] and v[0] in "\"'": v = v[1:-1]
        os.environ.setdefault(k, v)

def creds():
    return os.environ.get("LINE_CHANNEL_ACCESS_TOKEN", "").strip(), os.environ.get("LINE_USER_ID", "").strip()

def enabled() -> bool:
    tok, uid = creds(); return bool(tok and uid)

def push(text: str, quiet: bool = False) -> bool:
    """推一則文字訊息給 LINE_USER_ID。成功回 True; 未設定或失敗回 False (只印一行, 不丟例外)。"""
    tok, uid = creds()
    if not tok or not uid:
        if not quiet: print("LINE 未設定 (LINE_CHANNEL_ACCESS_TOKEN / LINE_USER_ID), 略過推播", flush=True)
        return False
    text = (text or "").strip()
    if not text: return False
    if len(text) > MAX_LEN: text = text[:MAX_LEN - 20] + "\n…(截斷)"
    try:
        r = requests.post(f"{API}/message/push", headers={"Authorization": f"Bearer {tok}", "Content-Type": "application/json"},
                          json={"to": uid, "messages": [{"type": "text", "text": text}]}, timeout=10)
        if r.status_code == 200: return True
        print(f"LINE 推播失敗 HTTP {r.status_code}: {r.text[:200]}", flush=True)   # 回應不含 token
        return False
    except Exception as e:
        print("LINE 推播失敗:", e, flush=True); return False

def check() -> bool:
    """驗證 token 可用: GET /info 印出官方帳號名稱與 basicId; 另查本月 push 額度 (quota) 與已用量。"""
    tok, uid = creds()
    if not tok: print("缺 LINE_CHANNEL_ACCESS_TOKEN"); return False
    h = {"Authorization": f"Bearer {tok}"}
    r = requests.get(f"{API}/info", headers=h, timeout=10)
    if r.status_code != 200: print(f"token 無效 HTTP {r.status_code}: {r.text[:200]}"); return False
    j = r.json(); print(f"官方帳號: {j.get('displayName')} ({j.get('basicId')})")
    try:
        q = requests.get(f"{API}/message/quota", headers=h, timeout=10).json()
        c = requests.get(f"{API}/message/quota/consumption", headers=h, timeout=10).json()
        print(f"本月額度: {q.get('value', '無上限' if q.get('type') == 'none' else '?')} 則, 已用 {c.get('totalUsage', '?')} 則")
    except Exception as e: print("查額度失敗:", e)
    print("LINE_USER_ID:", "已設定" if uid else "未設定 (Basic settings 頁最下方 Your user ID)")
    return bool(uid)

def main():
    args = sys.argv[1:]
    while "--env-file" in args:
        i = args.index("--env-file"); load_env_file(args[i + 1]); del args[i:i + 2]
    if "--check" in args: sys.exit(0 if check() else 1)
    msg = " ".join(args).strip() or "quant_trading LINE 推播測試"
    sys.exit(0 if push(msg) else 1)

if __name__ == "__main__":
    main()
