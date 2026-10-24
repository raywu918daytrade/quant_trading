"""M1 M 頭破底再空: 1 分 K 出現雙頭 (M 頭) 且跌破頸線 (兩頭之間的最低點) 才進場, 跟現行 09:01 進場比.
M 頭定義 (1 分 K): 高點 1 = 局部最高 (前後 3 分鐘內最高), 之後有回檔, 再出現高點 2 (局部最高, 與高點 1 差 ≤ 1%, 相隔 ≥ 5 分鐘),
  兩頭之間最低 low = 頸線, 頸線比兩頭低 ≥ depth (0.5% / 1%). 高點 2 之後第一根 close < 頸線 = 破底, 下一分鐘 open ×(1−0.1%) 做空.
cutoff 前沒破底 → 不做 / 或在 cutoff 照進. 出場同現行 (+8% 強制 / 13:24 / 收盤). 樣本同 28. 輸出 stats/m1_double_top.md"""
import numpy as np, pandas as pd
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
ROOT = Path(__file__).resolve().parent.parent; OUT = ROOT/"stats"
BASE, SLIP, K, PEN, CAP, PER_STOCK, MAX_N, PRE_K = 0.0023, 0.001, 0.08, 0.07, 3_000_000, 600_000, 10, 0.06
CUTOFFS = {"10:00": 60, "11:00": 120, "12:00": 180, "13:00": 240}; DEPTHS = (0.005, 0.01); W, TOL, SEP = 3, 0.01, 5

def find_mtop(t, h, l, c, depth):
    n = len(t); peaks = [i for i in range(n) if h[i] == h[max(0, i-W):i+W+1].max() and (i == 0 or h[i] > h[i-1] or i < W)]
    peaks = [i for i in peaks if t[i] < 265]
    for a in range(len(peaks)):
        i = peaks[a]
        for b in range(a+1, len(peaks)):
            j = peaks[b]
            if t[j] - t[i] < SEP: continue
            if abs(h[j] - h[i])/h[i] > TOL: 
                if h[j] > h[i]*(1+TOL): break      # 創新高, 高點 1 作廢, 換下一個
                continue
            neck = l[i:j+1].min()
            if neck > min(h[i], h[j])*(1-depth): continue
            brk = np.where((c < neck) & (np.arange(n) > j) & (t < 265))[0]
            if len(brk): return int(brk[0]), neck, t[i], t[j]
            break
    return None, None, None, None

def simulate_month(args):
    f, trades = args
    df = pd.read_parquet(f); ts = pd.to_datetime(df["date"]); df["day"] = ts.dt.normalize(); df["t"] = ts.dt.hour*60 + ts.dt.minute - 540
    df = df[(df.t>=0)&(df.t<=270)]; key = set(zip(trades.stock_id, trades.date)); df = df[[(s, d) in key for s, d in zip(df.stock_id, df.day)]]
    out = []
    for (sid, day), g in df.groupby(["stock_id","day"], sort=False):
        tr = trades[(trades.stock_id==sid)&(trades.date==day)].iloc[0]; pc = tr.prev_close
        g = g.sort_values("t"); t = g.t.values; o = g.open.values; h = g.high.values; l = g.low.values; c = g.close.values
        auction = c[t==270][-1] if (t==270).any() else c[-1]
        def exit_from(t_entry):
            m = (t >= t_entry) & (t < 264)
            if (h[m] >= pc*(1+K)).any(): return pc*(1+K)*(1+SLIP*3), "forced"
            i264 = np.where(t == 264)[0]
            if len(i264) and t_entry <= 264 and c[i264[0]] >= pc*(1+PRE_K): return c[i264[0]]*(1+SLIP), "pre"
            if auction >= pc*1.0975: return (tr.next_open*(1+PEN) if np.isfinite(tr.next_open) else auction), "lock"
            return auction, "close"
        res = {"stock_id": sid, "date": day, "rank": tr["rank"], "entry": tr.entry}
        ex, _ = exit_from(4); res["g|現行"] = tr.entry/ex - 1; res["in|現行"] = 1.0
        for depth in DEPTHS:
            bi, neck, t1, t2 = find_mtop(t, h, l, c, depth); dl = f"{depth*100:.1f}%"
            res[f"brk_t|{dl}"] = t[bi] if bi is not None else -1
            for lab, cut in CUTOFFS.items():
                for mode in ("沒就不做", "沒就照進"):
                    name = f"M頭{dl}破底@{lab} {mode}"
                    if bi is not None and t[bi] < cut: idx = bi + 1
                    elif mode == "沒就照進": w = np.where(t >= cut)[0]; idx = w[0] if len(w) else None
                    else: idx = None
                    if idx is None or idx >= len(t) or t[idx] >= 265: res[f"g|{name}"] = np.nan; res[f"in|{name}"] = 0.0; continue
                    entry = o[idx]*(1-SLIP)
                    if entry <= pc*0.905: res[f"g|{name}"] = np.nan; res[f"in|{name}"] = 0.0; continue
                    ex, _ = exit_from(t[idx]); res[f"g|{name}"] = entry/ex - 1; res[f"in|{name}"] = 1.0
        out.append(res)
    return pd.DataFrame(out)

if __name__ == "__main__":
    R = pd.read_parquet(ROOT/"data/processed/trailing_stop_trades_sr.parquet")[["stock_id","date","entry","prev_close","next_open","rank"]]; R["ym"] = R.date.dt.strftime("%Y_%m")
    jobs = [(str(ROOT/"data/raw/m1"/f"{ym}.parquet"), g) for ym, g in R.groupby("ym") if (ROOT/"data/raw/m1"/f"{ym}.parquet").exists()]
    with ProcessPoolExecutor(6) as ex: parts = list(ex.map(simulate_month, jobs))
    S = pd.concat(parts, ignore_index=True); S.to_parquet(ROOT/"data/processed/m1_double_top_trades.parquet", index=False); print("simulated", len(S), flush=True)
    all_days = pd.Series(sorted(S.date.unique())); names = [c[2:] for c in S.columns if c.startswith("g|")]
    def alloc(col):
        recs = []
        for dt, gg in S[S[f"in|{col}"]==1].sort_values(["date","rank"]).groupby("date", sort=False):
            cash = CAP; amts=[]; nets=[]
            for _, r in gg.iterrows():
                lc = r.entry*1000
                if cash < lc: continue
                lots = int(min(PER_STOCK, cash)//lc)
                if lots < 1: continue
                a = lots*lc; cash -= a; amts.append(a); nets.append(r[f"g|{col}"]-BASE)
                if len(amts) >= MAX_N: break
            recs.append((dt, len(amts), sum(amts)/CAP, sum(a*n for a, n in zip(amts, nets))/CAP))
        return pd.DataFrame(recs, columns=["date","n","util","ret"]).set_index("date").reindex(all_days).fillna({"n":0,"util":0,"ret":0.0})
    L = []; P = lambda s="": L.append(str(s))
    P("# M1 M 頭破底再空 (2020~2026, sr 模型每日前 10 檔, M1 逐分鐘, 300 萬)\n")
    P("- M 頭: 1 分 K 兩個局部高點 (前後 3 分鐘內最高) 相差 ≤ 1%、相隔 ≥ 5 分鐘, 中間最低點 = 頸線, 頸線比頭低 ≥ 0.5% 或 1%。第二頭之後 close 跌破頸線 = 破底, 下一分鐘開盤 (−0.1%) 做空。cutoff 前沒破底 → 不做 / 在 cutoff 照進。出場、成本、配置同現行。\n")
    rows = []
    for nm in names:
        D = alloc(nm); s = D.ret; g = S[S[f"in|{nm}"]==1][f"g|{nm}"] - BASE
        rows.append({"規則": nm, "進場率%": S[f"in|{nm}"].mean()*100, "日均檔數": D.n.mean(), "資金使用率": D.util.mean(), "每筆淨%": g.mean()*100, "勝率": (g>0).mean(), "日淨%": s.mean()*100, "Sharpe": s.mean()/s.std()*np.sqrt(245) if s.std()>0 else np.nan, "MDD%": ((1+s).cumprod()/(1+s).cumprod().cummax()-1).min()*100, "年化(萬)": s.mean()*245*CAP/1e4})
    P("## 1. 全期間"); P(pd.DataFrame(rows).set_index("規則").round(3).to_markdown()); P("")
    yr = pd.DataFrame({nm: (S[S[f"in|{nm}"]==1].assign(net=lambda x: x[f"g|{nm}"]-BASE).groupby(S.date.dt.year).net.mean()*100) for nm in names if "沒就不做" in nm or nm == "現行"}); P("## 2. 各年每筆淨% (沒就不做版)"); P(yr.round(2).to_markdown()); P("")
    P("## 3. M 頭破底本身有沒有資訊 (頸線深度 1%)")
    bt = S["brk_t|1.0%"]
    for lab, cut in CUTOFFS.items():
        m = (bt >= 0) & (bt < cut); nm = f"M頭1.0%破底@{lab} 沒就不做"
        P(f"- {lab} 前破底 {int(m.sum())} 筆 ({m.mean()*100:.0f}%): 現行 09:01 進場每筆淨 {(S.loc[m,'g|現行']-BASE).mean()*100:.3f}% (勝率 {((S.loc[m,'g|現行']-BASE)>0).mean():.3f}) / 破底後才進場 {(S.loc[m, f'g|{nm}']-BASE).mean()*100:.3f}% (勝率 {((S.loc[m, f'g|{nm}']-BASE)>0).mean():.3f})；沒破底的 {int((~m).sum())} 筆 現行每筆淨 {(S.loc[~m,'g|現行']-BASE).mean()*100:.3f}% (勝率 {((S.loc[~m,'g|現行']-BASE)>0).mean():.3f})")
    P(f"- 破底時間分布 (有破底的): 中位 {bt[bt>=0].median():.0f} 分 (09:00 起), 25% {bt[bt>=0].quantile(.25):.0f} 分, 75% {bt[bt>=0].quantile(.75):.0f} 分")
    (OUT/"m1_double_top.md").write_text("\n".join(L)); print("\n".join(L))
