"""「每天只做一支」影子紀錄: 從正式名單當天的候選 (池內可交易的前 10 名) 用組合分數選 1 支, 用 D1 結算, 追加到 HF db/signals/history_pick1.csv.
組合分數 (stats/pick_one.md 第 3 節, 只用 2020~23 選出, 2024~ 樣本外每筆淨 0.97% vs 模型第 1 名 0.71%):
  分數 = 模型排名的當日百分位 (第 1 名最高) + 2 × 今日跳空的當日百分位   (「跳空 − 大盤跳空」同日排序與跳空相同, 所以跳空算兩次)
  平手取模型排名較前. 部位 = 60 萬 (整張). 結算規則與 record_results.py 完全相同 (共用 settle).
每天一列, 另記當天前 10 名平均與模型第 1 名的每筆淨, 用來長期驗證「選股規則有沒有比照名單第 1 名 / 隨便選好」. 純追加不刪.
用法: HF_TOKEN=... python3 scripts/record_pick1.py [--dry]   (需 data/processed/d1.parquet 已是最新; 每天在 record_results.py 之後跑)
"""
import os, sys, io, csv, argparse
from pathlib import Path
import pandas as pd, requests
sys.path.insert(0, str(Path(__file__).resolve().parent)); import config as CFG
from record_results import settle, REPO, BASE_URL, ROOT
N_CAND, POS, HIST = 10, 600_000, "history_pick1.csv"
KEYS = ["date","no_trade","stock_id","name","rank","gap_pct","score","n_cand","lots","amount","entry","exit","exit_reason","net_pct","pnl",
        "top10_mean_pct","rank1_stock","rank1_net_pct","excess_pct","model","source"]

def pick_one(cand: pd.DataFrame) -> pd.DataFrame:
    """cand: 當天候選 (需 rank, gap). 回傳依分數排好的候選 (第一列 = 選中), 附 score 欄."""
    c = cand.copy()
    c["score"] = -c["rank"].rank(pct=True, method="average") + 2*c["gap"].rank(pct=True, method="average")
    return c.sort_values(["score","rank"], ascending=[False, True])

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
    sfx = CFG.suffix(CFG.MAIN_VARIANT)   # 正式名單 (不帶後綴)
    pub = sorted({f.split("/")[-1][:10] for f in files if f.startswith("db/signals/20") and f.endswith(f"{sfx}_pool.csv") and len(f.split("/")[-1]) == len(f"2026-01-01{sfx}_pool.csv")})
    d1_dates = set(d1.index.get_level_values("date").unique())
    todo = [d for d in pub if d not in done and pd.Timestamp(d) in d1_dates]
    print(f"[pick1] 有候選池 {len(pub)} 天, 已記錄 {len(done)} 天, 待結算 {len(todo)} 天: {todo}")
    for d in todo:
        j = get(f"{d}{sfx}.json").json(); dt = pd.Timestamp(d)
        row = {"date": d, "no_trade": "|".join(j.get("no_trade", [])), "model": j.get("model", ""), "source": j.get("source", "")}
        pool = pd.read_csv(io.BytesIO(get(f"{d}{sfx}_pool.csv").content), dtype={"stock_id": str}, encoding="utf-8-sig")
        el = pool[pool.elig.astype(str).str.lower() == "true"].sort_values("rank_all")
        el = el[[(s, dt) in d1.index for s in el.stock_id]].head(N_CAND).reset_index(drop=True)
        el["rank"] = range(1, len(el)+1)
        if row["no_trade"] or el.empty:
            hist.append(row); print(f"  {d}: 不交易 ({row['no_trade'] or '沒有候選'})"); continue
        res = [settle(d1.loc[(r.stock_id, dt)], float(r.prev_close)) for r in el.itertuples()]
        el["entry"], el["exit"], el["exit_reason"], el["net"] = zip(*res)
        P = pick_one(el); p = P.iloc[0]; r1 = el.iloc[0]
        lots = int(POS // (p.entry*1000)); amt = p.entry*1000*lots
        row.update(stock_id=p.stock_id, name=p["name"], rank=int(p["rank"]), gap_pct=round(p.gap*100, 2), score=round(p.score, 3), n_cand=len(el), lots=lots, amount=int(amt),
                   entry=round(p.entry, 2), exit=round(p.exit, 2), exit_reason=p.exit_reason, net_pct=round(p.net*100, 3), pnl=int(round(amt*p.net)),
                   top10_mean_pct=round(el.net.mean()*100, 3), rank1_stock=r1.stock_id, rank1_net_pct=round(r1.net*100, 3), excess_pct=round((p.net - el.net.mean())*100, 3))
        hist.append(row)
        print(f"  {d}: 選 {p.stock_id} {p['name']} (名單第 {int(p['rank'])}, 跳空 {p.gap*100:+.2f}%) 淨 {p.net*100:+.2f}%；前 {len(el)} 名平均 {el.net.mean()*100:+.2f}%，第 1 名 {r1.net*100:+.2f}%")
    if a.dry: print("dry run, 不上傳"); return
    if not todo: print("沒有新的要寫"); return
    s = io.StringIO(); w = csv.DictWriter(s, fieldnames=KEYS, extrasaction="ignore"); w.writeheader(); [w.writerow(r) for r in sorted(hist, key=lambda r: r["date"])]
    api.create_commit(repo_id=REPO, repo_type="dataset", commit_message=f"{HIST}: +{len(todo)} days",
                      operations=[CommitOperationAdd(f"db/signals/{HIST}", s.getvalue().encode("utf-8"))])
    print(f"已更新 {HIST} (共 {len(hist)} 天)")

if __name__ == "__main__":
    main()
