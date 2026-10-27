"""驗證 daily_candidates 的盤前預算 (prepare_live + finish_live) 與原本「開盤價放進 D1 整個重算」結果相同.
用 D1 裡某個歷史日當作「今天」: 只用該日以前的 D1, 開盤價取該日 D1 的 open.
用法: python3 scripts/check_live_prep.py [--date YYYY-MM-DD]   (需先 daily_candidates.refresh_d1)"""
import argparse, sys, time
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import features as F, daily_candidates as DC

def old_path(d1, last, tdate, op):
    base_last = d1[d1.date == last][["stock_id","market_type","name","close","volume"]]
    today = base_last[["stock_id","market_type","name"]].merge(op[["stock_id","open"]], on="stock_id")
    today["date"] = tdate
    for c in ["high","low","close","volume"]: today[c] = np.nan
    d1x = pd.concat([d1, today], ignore_index=True)
    base_df = F.d1_base(d1x); f = F.build_B(base_df)
    t = f[f.date == tdate].copy().merge(d1x[d1x.date == tdate][["stock_id","name"]].drop_duplicates("stock_id"), on="stock_id", how="left")
    return t, base_df

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--date", default=None); a = ap.parse_args()
    full = pd.read_parquet(DC.ROOT/"data/processed/d1.parquet")
    tdate = pd.Timestamp(a.date) if a.date else full.date.max()
    d1 = full[full.date < tdate].copy(); last = d1.date.max()
    t0 = time.time(); P = DC.prepare_live(d1, last, tdate); t_prep = time.time() - t0
    op = full[(full.date == tdate) & full.stock_id.isin(P["need"].stock_id)][["stock_id","open"]]
    op = op[op.open.notna() & (op.open > 0)]
    t0 = time.time(); t_new, b_new = DC.finish_live(P, op, tdate); t_fin = time.time() - t0
    t0 = time.time(); t_old, b_old = old_path(d1, last, tdate, op); t_full = time.time() - t0
    print(f"日期 {tdate.date()}  開盤價 {len(op)} 檔  盤前預算 {t_prep:.1f}s  09:01 後補算 {t_fin:.2f}s  (原本整個重算 {t_full:.1f}s)")
    cols = F.D1_FEATS + ["open","prev_close","dv20","dv20_prev","atr20","name","market_type"]
    o = t_old.set_index("stock_id").sort_index(); n = t_new.set_index("stock_id").sort_index()
    bad = []
    if list(o.index) != list(n.index): bad.append(f"股票集合不同: 舊 {len(o)} 新 {len(n)}")
    else:
        for c in cols:
            x, y = o[c], n[c]
            if x.dtype.kind in "fi" and y.dtype.kind in "fi":
                ok = np.allclose(x.astype(float), y.astype(float), rtol=1e-9, atol=1e-12, equal_nan=True)
            else: ok = (x.fillna("") == y.fillna("")).all()
            if not ok: bad.append(c)
    pool = t_old[F.universe_mask(t_old) & (t_old.dv20_prev >= DC.POOL_DV) & (t_old.atr20 >= DC.POOL_ATR)].stock_id.unique()
    so = F.build_sr(b_old, pool); sn = F.build_sr(b_new, pool)
    so = so[so.date == tdate].set_index("stock_id").sort_index()[F.SR_FEATS]; sn = sn[sn.date == tdate].set_index("stock_id").sort_index()[F.SR_FEATS]
    if not np.allclose(so.values.astype(float), sn.values.astype(float), equal_nan=True): bad.append("SR 特徵")
    print(f"比對 {len(cols)} 個欄位 × {len(o)} 檔 + 池內 {len(pool)} 檔支撐壓力特徵:", "全部相同 ✓" if not bad else f"不同 ✗ {bad}")
    sys.exit(1 if bad else 0)

if __name__ == "__main__":
    main()
