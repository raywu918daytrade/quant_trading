"""把「已發布的名單」用 D1 結算, 追加到 HF db/signals/history.csv (正式) 與 history_<variant>.csv (影子). 每天 09:01 名單 workflow 在 D1 更新後執行.

規則 (D1 近似, 與回測一致): 進場 = 開盤價 × (1+0.03%) [09:01~09:03 VWAP 校準]; 當日 high ≥ 昨收×1.08 → 以 昨收×1.08×1.003 回補 (forced);
否則 收盤價 (close); 收盤 ≥ 昨收×1.0975 且未觸發 → 隔日開盤 −7% (lock, 隔日才知道, 先記 pending). 成本 0.23%. no_trade 的日子記 0 檔.
只補「有發布名單、D1 已有該日、history 還沒有」的日子; 每天一行 + 每檔一列.\n紀錄永遠接續 (換模型也不刪, model 欄位可區分); 只有 --reset (reset_history workflow) 才把舊紀錄搬到 db/signals/archive/ 後從零開始.
用法: HF_TOKEN=... python3 scripts/record_results.py [--variant base]   (需 data/processed/d1.parquet 已是最新)
"""
import os, sys, io, json, csv, argparse
from pathlib import Path
import pandas as pd, requests
sys.path.insert(0, str(Path(__file__).resolve().parent)); import config as CFG
ROOT = Path(__file__).resolve().parent.parent; REPO = "raywu918daytrade/daytrade"
BASE_URL = f"https://huggingface.co/datasets/{REPO}/resolve/main/db/signals/"
COST, ENTRY_ADJ, K, SLIP, PEN, CAP = 0.0023, 0.0003, 0.08, 0.001, 0.07, 3_000_000

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--variant", default=CFG.MAIN_VARIANT, choices=CFG.VARIANTS); ap.add_argument("--dry", action="store_true"); ap.add_argument("--reset", action="store_true", help="強制把現有紀錄搬到 archive/ 後從零開始"); a = ap.parse_args()
    sfx = CFG.suffix(a.variant); hist_name = f"history{sfx}.csv"; daily_name = f"history_daily{sfx}.csv"
    token = os.environ.get("HF_TOKEN")
    if not token and not a.dry: sys.exit("缺 HF_TOKEN")
    from huggingface_hub import HfApi, CommitOperationAdd
    api = HfApi(token=token)
    d1 = pd.read_parquet(ROOT/"data/processed/d1.parquet", columns=["stock_id","date","open","high","close"]); d1["date"] = pd.to_datetime(d1.date)
    d1 = d1.sort_values(["stock_id","date"]); d1["next_open"] = d1.groupby("stock_id").open.shift(-1)
    d1 = d1.set_index(["stock_id","date"])
    files = set(api.list_repo_files(REPO, repo_type="dataset"))
    def read_csv(name):
        if f"db/signals/{name}" not in files: return []
        r = requests.get(BASE_URL + name, params={"t": pd.Timestamp.now().timestamp()}, timeout=30); r.raise_for_status(); return list(csv.DictReader(io.StringIO(r.text)))
    hist = read_csv(hist_name); daily = read_csv(daily_name)
    # 模型換了 (最新名單的 model != 紀錄最後一行的 model) 或 --reset → 把舊紀錄搬到 archive/, 從零開始
    latest = requests.get(BASE_URL + f"latest{sfx}.json", params={"t": 1}, timeout=30).json()
    cur_model = latest.get("model", ""); last_model = daily[-1]["model"] if daily else cur_model
    ops_extra = []
    if daily and cur_model and last_model and cur_model != last_model: print(f"[{a.variant}] 注意: 模型已從 {last_model} 換成 {cur_model}，紀錄照常接續 (model 欄位可區分)；要重來請手動跑 reset_history workflow")
    if daily and a.reset:
        tag = f"{daily[0]['date']}_{daily[-1]['date']}_{last_model.replace('.txt','')}"
        print(f"[{a.variant}] --reset: 舊紀錄 {len(daily)} 天搬到 archive/{tag}，重新開始")
        from huggingface_hub import CommitOperationDelete
        for name in (hist_name, daily_name):
            if f"db/signals/{name}" in files:
                raw = requests.get(BASE_URL + name, params={"t": 2}, timeout=30).content
                ops_extra += [CommitOperationAdd(f"db/signals/archive/{tag}_{name}", raw), CommitOperationDelete(f"db/signals/{name}")]
        hist = []; daily = []
    done = {r["date"] for r in daily}
    pub = sorted({f.split("/")[-1][:10] for f in files if f.startswith("db/signals/20") and f.endswith(f"{sfx}.json") and (sfx or "_" not in f.split("/")[-1][10:])})
    d1_dates = set(d1.index.get_level_values("date").unique())
    todo = [d for d in pub if d not in done and pd.Timestamp(d) in d1_dates]
    print(f"[{a.variant}] 已發布 {len(pub)} 天, 已記錄 {len(done)} 天, 待結算 {len(todo)} 天: {todo}")
    for d in todo:
        j = requests.get(BASE_URL + f"{d}{sfx}.json", params={"t": 1}, timeout=30).json(); dt = pd.Timestamp(d)
        rows = []; pnl = 0.0; amt = 0.0
        if not j.get("no_trade"):
            for p in j.get("picks", []):
                key = (p["stock_id"], dt)
                if key not in d1.index: continue
                row = d1.loc[key]; pc = float(p["prev_close"]); entry = float(row.open)*(1+ENTRY_ADJ); lots = int(p["lots"]); a_ = entry*1000*lots
                if row.high >= pc*(1+K) and row.open < pc*(1+K): exit_px = pc*(1+K)*(1+SLIP*3); reason = "forced"
                elif row.close >= pc*1.0975: exit_px = float(row.next_open)*(1+PEN) if pd.notna(row.next_open) else float(row.close); reason = "lock" if pd.notna(row.next_open) else "lock_pending"
                else: exit_px = float(row.close); reason = "close"
                net = entry/exit_px - 1 - COST; pnl += a_*net; amt += a_
                rows.append({"date": d, "stock_id": p["stock_id"], "name": p.get("name",""), "rank": p.get("rank"), "lots": lots, "amount": int(a_), "prev_close": pc, "entry": round(entry,2), "exit": round(exit_px,2), "high": float(row.high), "exit_reason": reason, "net_pct": round(net*100,2), "pnl": int(round(a_*net))})
        drow = {"date": d, "variant": a.variant, "no_trade": "|".join(j.get("no_trade", [])), "n": len(rows), "amount": int(amt), "wins": sum(1 for r in rows if r["pnl"]>0), "pnl": int(round(pnl)), "ret_pct": round(pnl/CAP*100, 3), "model": j.get("model","")}
        hist += rows; daily.append(drow); print(f"  {d}: {len(rows)} 檔, 損益 {drow['pnl']:+,} 元 ({drow['ret_pct']:+.2f}%)" + (f" [不交易: {drow['no_trade']}]" if drow["no_trade"] else ""))
    if a.dry: print("dry run, 不上傳"); return
    if not todo and not ops_extra: print("沒有新的要寫"); return
    def to_csv(rows, keys):
        s = io.StringIO(); w = csv.DictWriter(s, fieldnames=keys, extrasaction="ignore"); w.writeheader(); [w.writerow(r) for r in rows]; return s.getvalue().encode("utf-8")
    hk = ["date","stock_id","name","rank","lots","amount","prev_close","entry","exit","high","exit_reason","net_pct","pnl"]; dk = ["date","variant","no_trade","n","amount","wins","pnl","ret_pct","model"]
    ops = [o for o in ops_extra if type(o).__name__ == "CommitOperationAdd"]   # archive copies
    if daily: ops += [CommitOperationAdd(f"db/signals/{hist_name}", to_csv(sorted(hist, key=lambda r: (r["date"], int(r.get("rank") or 0))), hk)),
                      CommitOperationAdd(f"db/signals/{daily_name}", to_csv(sorted(daily, key=lambda r: r["date"]), dk))]
    else: ops += [o for o in ops_extra if type(o).__name__ == "CommitOperationDelete"]
    api.create_commit(repo_id=REPO, repo_type="dataset", commit_message=f"history{sfx}: +{len(todo)} days" + (" (reset, archived old)" if ops_extra else ""), operations=ops)
    print(f"已更新 {hist_name} / {daily_name} (共 {len(daily)} 天)")

if __name__ == "__main__":
    main()
