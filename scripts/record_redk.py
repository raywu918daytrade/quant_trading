"""「開盤 5 分紅 K 延到 09:05 才空」影子紀錄 (stats/red_k_delay.md: 紅K>1% 延後, 日淨 0.686→0.799%, Sharpe 4.44→5.20).
正式名單每檔: 紅K = (09:00~09:04 那根 5 分 K 收 − 開) / 昨收.
  紅K ≤ 1% → 跟現行一樣 (record_results.settle, D1 開盤×1.0003 進場)
  紅K > 1% → 09:05 那根 5 分 K 開盤 ×(1−0.1%) 才空; 09:05 前已摸 昨收×1.08 → 不進 (0 元);
             出場: 09:05 之後 high ≥ 昨收×1.08 → forced (昨收×1.08×1.003); 否則同 settle (收盤 / 漲停鎖死隔日開盤 +7%)
張數照正式名單. 來源: HF db/signals/bars/<date>.json (build_bars.py 產生) + d1.parquet. 每檔一列, 追加到 HF db/signals/history_redk.csv, 純追加不刪.
用法: HF_TOKEN=... python3 scripts/record_redk.py [--dry]   (每天在 build_bars.py 之後跑)
"""
import os, sys, io, csv, argparse
from pathlib import Path
import pandas as pd, requests
sys.path.insert(0, str(Path(__file__).resolve().parent)); import config as CFG
from record_results import settle, REPO, BASE_URL, ROOT, K, SLIP, COST, PEN
TH, HIST = 0.01, "history_redk.csv"
KEYS = ["date","stock_id","name","rank","lots","prev_close","red_pct","delayed","entry_cur","exit_cur","reason_cur","net_cur_pct","pnl_cur",
        "entry_new","exit_new","reason_new","net_new_pct","pnl_new","diff_pnl","model"]

def settle_delayed(bars, d1row, pc):
    """bars: [[t, o, h, l, c, v], ...] 5 分 K. 回傳 (進場, 出場, 原因, 淨) 或 None (不進)."""
    pre = [b for b in bars if b[0] < 5]; post = [b for b in bars if b[0] >= 5]
    if not post or any(b[2] >= pc*(1+K) for b in pre): return None
    entry = post[0][1]*(1-SLIP)
    if max(b[2] for b in post) >= pc*(1+K): exit_px, reason = pc*(1+K)*(1+SLIP*3), "forced"
    elif d1row.close >= pc*1.0975:
        exit_px = float(d1row.next_open)*(1+PEN) if pd.notna(d1row.next_open) else float(d1row.close); reason = "lock" if pd.notna(d1row.next_open) else "lock_pending"
    else: exit_px, reason = float(d1row.close), "close"
    return entry, exit_px, reason, entry/exit_px - 1 - COST

def settle_day(j, bars, d1, d):
    dt = pd.Timestamp(d); rows = []
    for p in j.get("picks", []):
        key = (p["stock_id"], dt); b = bars.get(p["stock_id"])
        if key not in d1.index or not b or b[0][0] != 0: continue
        r = d1.loc[key]; pc = float(p["prev_close"]); lots = int(p["lots"])
        e0, x0, w0, n0 = settle(r, pc); red = (b[0][4] - b[0][1])/pc
        if red > TH:
            s = settle_delayed(b, r, pc)
            e1, x1, w1, n1 = s if s else (float("nan"), float("nan"), "skip", 0.0)
        else: e1, x1, w1, n1 = e0, x0, w0, n0
        pc0 = e0*1000*lots*n0; pc1 = (e1*1000*lots*n1) if w1 != "skip" else 0.0
        rows.append({"date": d, "stock_id": p["stock_id"], "name": p.get("name",""), "rank": p.get("rank"), "lots": lots, "prev_close": pc, "red_pct": round(red*100, 2),
                     "delayed": int(red > TH), "entry_cur": round(e0, 2), "exit_cur": round(x0, 2), "reason_cur": w0, "net_cur_pct": round(n0*100, 3), "pnl_cur": int(round(pc0)),
                     "entry_new": round(e1, 2) if w1 != "skip" else "", "exit_new": round(x1, 2) if w1 != "skip" else "", "reason_new": w1, "net_new_pct": round(n1*100, 3),
                     "pnl_new": int(round(pc1)), "diff_pnl": int(round(pc1 - pc0)), "model": j.get("model","")})
    return rows

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dry", action="store_true"); a = ap.parse_args()
    token = os.environ.get("HF_TOKEN")
    if not token and not a.dry: sys.exit("缺 HF_TOKEN")
    from huggingface_hub import HfApi, CommitOperationAdd
    api = HfApi(token=token)
    d1 = pd.read_parquet(ROOT/"data/processed/d1.parquet", columns=["stock_id","date","open","high","close"]); d1["date"] = pd.to_datetime(d1.date)
    d1 = d1.sort_values(["stock_id","date"]); d1["next_open"] = d1.groupby("stock_id").open.shift(-1); d1 = d1.set_index(["stock_id","date"])
    files = set(api.list_repo_files(REPO, repo_type="dataset"))
    get = lambda name: requests.get(BASE_URL + name, params={"t": pd.Timestamp.now().timestamp()}, timeout=30)
    hist = list(csv.DictReader(io.StringIO(get(HIST).text))) if f"db/signals/{HIST}" in files else []
    done = {r["date"] for r in hist}
    sfx = CFG.suffix(CFG.MAIN_VARIANT)
    pub = sorted({f.split("/")[-1][:10] for f in files if f.startswith("db/signals/20") and f.endswith(f"{sfx}.json") and (sfx or "_" not in f.split("/")[-1][10:])})
    have_bars = {f.split("/")[-1][:10] for f in files if f.startswith("db/signals/bars/")}
    d1_dates = {x.strftime("%Y-%m-%d") for x in d1.index.get_level_values("date").unique()}
    todo = [d for d in pub if d not in done and d in have_bars and d in d1_dates]
    print(f"[redk] 名單 {len(pub)} 天, 已記錄 {len(done)} 天, 待結算 {len(todo)} 天: {todo}")
    new = []
    for d in todo:
        j = get(f"{d}{sfx}.json").json()
        if j.get("no_trade"): print(f"  {d}: 不交易"); continue
        rows = settle_day(j, get(f"bars/{d}.json").json().get("bars", {}), d1, d)
        if not rows: continue
        new += rows; nd = sum(r["delayed"] for r in rows)
        print(f"  {d}: {len(rows)} 檔, 延後 {nd} 檔, 現行 {sum(r['pnl_cur'] for r in rows):+,} 元 → 延後版 {sum(r['pnl_new'] for r in rows):+,} 元")
    if a.dry: print("dry run, 不上傳"); return
    if not new: print("沒有新的要寫"); return
    allr = sorted(hist + new, key=lambda r: (r["date"], int(r.get("rank") or 0)))
    s = io.StringIO(); w = csv.DictWriter(s, fieldnames=KEYS, extrasaction="ignore"); w.writeheader(); [w.writerow(r) for r in allr]
    api.create_commit(repo_id=REPO, repo_type="dataset", commit_message=f"{HIST}: +{len({r['date'] for r in new})} days",
                      operations=[CommitOperationAdd(f"db/signals/{HIST}", s.getvalue().encode("utf-8"))])
    print(f"已更新 {HIST} (共 {len({r['date'] for r in allr})} 天)")

if __name__ == "__main__":
    main()
