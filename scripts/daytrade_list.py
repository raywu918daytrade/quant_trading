"""證交所 + 櫃買「當日可現股當沖 (先賣後買)」名單. 公開 API, 不需登入.

回傳 {stock_id: True/False}: True = 可現股當沖且可先賣後買 (能做空);
False = 在名單但被註記「暫停現股賣出後現款買進」(只能先買後賣). 不在名單的股票 = 不可當沖, 字典裡沒有這個 key.
證交所每天盤前就公布當天名單 (TWTB4U); 櫃買同理. 兩邊都支援查歷史日期, 供回放.
快取在 data/raw/meta/daytrade_<YYYYMMDD>.json.
"""
import json, sys, time
from pathlib import Path
import requests

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "data/raw/meta"
UA = {"User-Agent": "Mozilla/5.0"}

def _get_json(url, tries=3):
    for i in range(tries):
        try:
            r = requests.get(url, headers=UA, timeout=20); r.raise_for_status(); return r.json()
        except Exception as e:
            if i == tries - 1: raise
            time.sleep(2)

def fetch_twse(ymd: str) -> dict:
    """ymd: 'YYYYMMDD'. 今天用不帶 date 的查詢 (盤前就有); 歷史日帶 date."""
    today = time.strftime("%Y%m%d")
    url = "https://www.twse.com.tw/exchangeReport/TWTB4U?response=json" + ("" if ymd == today else f"&date={ymd}")
    d = _get_json(url)
    if d.get("stat") != "OK": raise RuntimeError(f"TWSE TWTB4U stat={d.get('stat')}")
    out = {}
    for t in d.get("tables", []):
        f = t.get("fields") or []
        if len(f) >= 3 and f[0] == "證券代號":
            for row in t.get("data", []):
                sid, mark = str(row[0]).strip(), str(row[2]).strip()
                out[sid] = (mark == "")
    if not out: raise RuntimeError("TWSE TWTB4U: 找不到名單表格")
    return out

def fetch_tpex(ymd: str) -> dict:
    url = f"https://www.tpex.org.tw/www/zh-tw/intraday/list?date={ymd[:4]}/{ymd[4:6]}/{ymd[6:]}&response=json"
    d = _get_json(url)
    out = {}
    for t in d.get("tables", []):
        for row in t.get("data", []):
            sid, mark = str(row[0]).strip(), str(row[2]).strip() if len(row) > 2 else ""
            out[sid] = (mark == "")
    if not out: raise RuntimeError("TPEx intraday/list: 名單為空")
    return out

def fetch_daytrade_eligible(date) -> dict:
    """date: 'YYYY-MM-DD' 或 pandas Timestamp. 回傳 {stock_id: can_short}. 失敗會 raise."""
    ymd = str(date)[:10].replace("-", "")
    CACHE.mkdir(parents=True, exist_ok=True)
    p = CACHE / f"daytrade_{ymd}.json"
    if p.exists():
        return json.load(open(p))
    tw = fetch_twse(ymd); tp = fetch_tpex(ymd)
    out = {**tw, **tp}
    json.dump(out, open(p, "w"), ensure_ascii=False)
    return out

if __name__ == "__main__":
    date = sys.argv[1] if len(sys.argv) > 1 else time.strftime("%Y-%m-%d")
    m = fetch_daytrade_eligible(date)
    n_short = sum(1 for v in m.values() if v)
    print(f"{date}: 可當沖 {len(m)} 檔, 其中可先賣後買 {n_short} 檔, 暫停先賣 {len(m)-n_short} 檔")
