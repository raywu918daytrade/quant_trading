"""證交所 + 櫃買「處置有價證券」公告. 公開 API, 不需登入.

fetch_disposal(start, end) -> DataFrame[stock_id, name, market, announce, start, end, level, content]
  start/end: 處置期間 (西元 date); level: 1 = 第一次處置 (5 分鐘撮合), 2 = 第二次 (20 分鐘撮合, 全面預收款券), 0 = 不明.
disposed_on(date) -> {stock_id: level}: 該日處於處置期間的股票.
快取在 data/raw/meta/disposal_<YYYYMM>.json (原始列, 以公布月份分檔; 當月不快取).
python3 scripts/disposal_list.py 2020-01 2026-10   # 抓區間, 印出統計並存 data/raw/meta/disposal_all.csv
"""
import json, re, sys, time, datetime as dt
from pathlib import Path
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "data/raw/meta"
UA = {"User-Agent": "Mozilla/5.0"}

def _get_json(url, params=None, tries=3, method="get"):
    for i in range(tries):
        try:
            r = (requests.post if method == "post" else requests.get)(url, params=params if method == "get" else None, data=params if method == "post" else None, headers=UA, timeout=30)
            r.raise_for_status(); return r.json()
        except Exception:
            if i == tries - 1: raise
            time.sleep(3)

def _tables(d):
    """TWSE 舊版 {fields,data} / 新版 {tables:[{fields,data}]} 都吃."""
    if d.get("tables"): return [(t.get("fields") or [], t.get("data") or []) for t in d["tables"]]
    if d.get("fields"): return [(d["fields"], d.get("data") or [])]
    return []

def _rows(fields, data, market):
    return [{"market": market, **{str(f).strip(): (str(v).strip() if v is not None else "") for f, v in zip(fields, row)}} for row in data]

def fetch_twse_month(y, m):
    s = dt.date(y, m, 1); e = (s + dt.timedelta(days=32)).replace(day=1) - dt.timedelta(days=1)
    d = _get_json("https://www.twse.com.tw/rwd/zh/announcement/punish", {"response": "json", "startDate": s.strftime("%Y%m%d"), "endDate": e.strftime("%Y%m%d")})
    out = []
    for f, rows in _tables(d): out += _rows(f, rows, "twse")
    return out

def fetch_tpex_month(y, m):
    s = dt.date(y, m, 1); e = (s + dt.timedelta(days=32)).replace(day=1) - dt.timedelta(days=1)
    p = {"startDate": s.strftime("%Y/%m/%d"), "endDate": e.strftime("%Y/%m/%d"), "response": "json"}
    url = "https://www.tpex.org.tw/www/zh-tw/bulletin/disposal"
    try: d = _get_json(url, p, method="post")
    except Exception: d = _get_json(url, p)
    out = []
    for f, rows in _tables(d): out += _rows(f, rows, "tpex")
    return out

def fetch_month_raw(y, m):
    p = CACHE / f"disposal_{y}{m:02d}.json"
    cur = dt.date.today().strftime("%Y%m") == f"{y}{m:02d}"
    if p.exists() and not cur: return json.load(open(p))
    out = fetch_twse_month(y, m); time.sleep(1.5)
    out += fetch_tpex_month(y, m); time.sleep(1.5)
    if not cur:
        CACHE.mkdir(parents=True, exist_ok=True); json.dump(out, open(p, "w"), ensure_ascii=False)
    return out

def _roc(s):
    """'113/01/02' 或 '2024/01/02' 或 '1130102' -> date."""
    s = s.strip()
    m = re.match(r"(\d{2,4})[/\-.](\d{1,2})[/\-.](\d{1,2})", s)
    if m: y, mo, d = map(int, m.groups())
    elif re.fullmatch(r"\d{7,8}", s): y, mo, d = int(s[:-4]), int(s[-4:-2]), int(s[-2:])
    else: return None
    if y < 1911: y += 1911
    return dt.date(y, mo, d)

def _pick(r, *keys):
    for k in r:
        if any(x in k for x in keys): return r[k]
    return ""

def parse(raw) -> pd.DataFrame:
    out = []
    for r in raw:
        sid = _pick(r, "代號")
        period = _pick(r, "起迄", "起訖", "處置期間", "期間")
        dates = re.findall(r"\d{2,4}/\d{1,2}/\d{1,2}", period)
        if not sid or len(dates) < 2: continue
        content = " ".join(v for k, v in r.items() if k not in ("market",))
        level = 2 if ("第二次" in content or "再次處置" in content or "二十分鐘" in content or "20分鐘" in content) else 1 if ("第一次" in content or "五分鐘" in content or "5分鐘" in content) else 0
        out.append({"stock_id": sid.split()[0], "name": re.sub(r"\(.*", "", _pick(r, "名稱")), "market": r["market"], "announce": _roc(_pick(r, "公布日期", "公告日期")),
                    "start": _roc(dates[0]), "end": _roc(dates[1]), "level": level, "content": _pick(r, "處置內容", "處置措施")[:200]})
    df = pd.DataFrame(out)
    if len(df): df = df.drop_duplicates(["stock_id", "start", "end"]).sort_values(["start", "stock_id"]).reset_index(drop=True)
    return df

def fetch_disposal(start, end) -> pd.DataFrame:
    s = pd.Timestamp(start); e = pd.Timestamp(end); raw = []
    for p in pd.period_range(s.to_period("M") - 1, e.to_period("M"), freq="M"):   # 前一個月公布的處置會延續到本月
        raw += fetch_month_raw(p.year, p.month)
    return parse(raw)

def disposed_on(date) -> dict:
    d = pd.Timestamp(date).date()
    df = fetch_disposal(d - dt.timedelta(days=40), d)
    if not len(df): return {}
    df = df[(df.start <= d) & (df.end >= d)]
    return df.groupby("stock_id").level.max().to_dict()

if __name__ == "__main__":
    a = sys.argv[1:] or [dt.date.today().strftime("%Y-%m")]
    s, e = pd.Period(a[0], "M"), pd.Period(a[-1], "M")
    raw = []
    for p in pd.period_range(s, e, freq="M"):
        try:
            r = fetch_month_raw(p.year, p.month); raw += r
            print(f"{p}: {len(r)} 列", (sorted(r[0].keys()) if r else ""), flush=True)
        except Exception as ex: print(f"{p}: 失敗 {ex}", flush=True)
    df = parse(raw); df.to_csv(CACHE / "disposal_all.csv", index=False)
    print(f"解析 {len(df)} 筆處置; 依市場/等級:\n{df.groupby(['market','level']).size() if len(df) else ''}")
    print(df.tail(15).to_string() if len(df) else "")
