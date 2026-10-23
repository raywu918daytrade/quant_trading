"""進場時機測試: 5 分 K「紅轉綠且綠 K 實體 > 紅 K 實體」(動能反轉) 出現才做空, 與現行 09:01~09:03 進場比較.
交易樣本 = data/processed/trailing_stop_trades_sr.parquet (定案策略 sr 模型每日前 10 檔, 2020~2026, 含 entry/prev_close/next_open/rank).
變體:
  現行                : entry = e01_vwap, 出場同現行 (+8% 強制 / 13:24 +6% / 收盤)
  型態@cutoff, 沒就不做 : 從 09:05 的 5 分 K 起找第一個 (紅K, 綠K, 綠實體>紅實體) 的組合 (綠 K 收盤 ≤ 進場前最高? 不加), 在綠 K 收盤那根的下一根開盤價 ×(1+0.1%) 進場;
                        cutoff 前沒出現 → 當天不做這檔. cutoff ∈ {09:30, 10:00, 11:00}
  型態@cutoff, 沒就照進 : 同上, 但沒出現就在 cutoff 那根開盤進場
  加碼條件: 綠 K 收盤 < 開盤價 (已經轉弱) 的版本
出場: 進場後的分鐘 high ≥ 昨收+8% → 強制; 13:24 預先; 收盤. 成本 0.23%. 300 萬配置依原排名 (沒進場的檔跳過, 錢給下一檔).
輸出 stats/m5_pattern_entry.md
"""
import sys, glob, numpy as np, pandas as pd
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
ROOT = Path(__file__).resolve().parent.parent; OUT = ROOT/"stats"
BASE, SLIP, K, PEN, CAP, PER_STOCK, MAX_N, PRE_K = 0.0023, 0.001, 0.08, 0.07, 3_000_000, 600_000, 10, 0.06
CUTOFFS = {"09:30": 30, "10:00": 60, "11:00": 120}

def simulate_month(args):
    f, trades = args
    df = pd.read_parquet(f); ts = pd.to_datetime(df["date"]); df["day"] = ts.dt.normalize(); df["t"] = ts.dt.hour*60 + ts.dt.minute - 540
    df = df[(df.t>=0)&(df.t<=270)]
    key = set(zip(trades.stock_id, trades.date)); df = df[[(s, d) in key for s, d in zip(df.stock_id, df.day)]]
    out = []
    for (sid, day), g in df.groupby(["stock_id","day"], sort=False):
        tr = trades[(trades.stock_id==sid)&(trades.date==day)].iloc[0]; pc = tr.prev_close; day_open = tr.day_open
        g = g.sort_values("t"); t = g.t.values; o = g.open.values; h = g.high.values; c = g.close.values
        auction = c[t==270][-1] if (t==270).any() else c[-1]
        # 5 分 K
        b = (t // 5); k = pd.DataFrame({"b": b, "o": o, "h": h, "c": c, "t": t}).groupby("b").agg(o=("o","first"), h=("h","max"), c=("c","last"), t=("t","first")).reset_index()
        k = k[k.t < 265]
        def exit_from(t_entry, entry):
            # 進場後出場
            m = t >= t_entry
            if (h[m & (t < 264)] >= pc*(1+K)).any() if m.any() else False: return pc*(1+K)*(1+SLIP*3), "forced"
            i264 = np.where(t == 264)[0]
            if len(i264) and t_entry <= 264 and c[i264[0]] >= pc*(1+PRE_K): return c[i264[0]]*(1+SLIP), "pre"
            if auction >= pc*1.0975: return (tr.next_open*(1+PEN) if np.isfinite(tr.next_open) else auction), "lock"
            return auction, "close"
        res = {"stock_id": sid, "date": day, "rank": tr["rank"]}
        # 現行
        ex, rs = exit_from(4, tr.entry); res["g|現行"] = tr.entry/ex - 1; res["in|現行"] = 1.0
        # 型態
        ko, kc, kt = k.o.values, k.c.values, k.t.values
        found = None
        for i in range(1, len(k)):
            red = kc[i-1] > ko[i-1]; green = kc[i] < ko[i]
            if red and green and (ko[i]-kc[i]) > (kc[i-1]-ko[i-1]): found = i; break
        found_t = kt[found] + 5 if found is not None else None    # 下一根開盤
        found_below_open = found is not None and kc[found] < day_open
        for lab, cut in CUTOFFS.items():
            for mode in ("沒就不做", "沒就照進"):
                for extra in ("", "+綠K收<開盤"):
                    name = f"型態@{lab} {mode}{extra}"
                    ok = found is not None and found_t <= cut and (not extra or found_below_open)
                    if ok: t_e = found_t
                    elif mode == "沒就照進": t_e = cut
                    else: res[f"g|{name}"] = np.nan; res[f"in|{name}"] = 0.0; continue
                    idx = np.where(t >= t_e)[0]
                    if not len(idx): res[f"g|{name}"] = np.nan; res[f"in|{name}"] = 0.0; continue
                    entry = o[idx[0]]*(1-SLIP)   # 做空賣出: 滑價往下
                    if entry <= pc*0.905: res[f"g|{name}"] = np.nan; res[f"in|{name}"] = 0.0; continue
                    ex, rs = exit_from(t[idx[0]], entry); res[f"g|{name}"] = entry/ex - 1; res[f"in|{name}"] = 1.0
        out.append(res)
    return pd.DataFrame(out)

if __name__ == "__main__":
    R = pd.read_parquet(ROOT/"data/processed/trailing_stop_trades_sr.parquet")[["stock_id","date","entry","prev_close","next_open","rank"]]
    d1 = pd.read_parquet(ROOT/"data/processed/d1.parquet", columns=["stock_id","date","open"]); d1["date"] = pd.to_datetime(d1.date)
    R = R.merge(d1.rename(columns={"open": "day_open"}), on=["stock_id","date"], how="left"); R["ym"] = R.date.dt.strftime("%Y_%m")
    jobs = [(str(ROOT/"data/raw/m1"/f"{ym}.parquet"), g) for ym, g in R.groupby("ym") if (ROOT/"data/raw/m1"/f"{ym}.parquet").exists()]
    with ProcessPoolExecutor(6) as ex: parts = list(ex.map(simulate_month, jobs))
    S = pd.concat(parts, ignore_index=True).merge(R[["stock_id","date","entry"]], on=["stock_id","date"])
    S.to_parquet(ROOT/"data/processed/m5_pattern_trades.parquet", index=False); print("simulated", len(S), flush=True)
    all_days = pd.Series(sorted(S.date.unique()))
    names = [c[2:] for c in S.columns if c.startswith("g|")]
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
        D = pd.DataFrame(recs, columns=["date","n","util","ret"]).set_index("date"); return D.reindex(all_days).fillna({"n":0,"util":0,"ret":0.0})
    L = []; P = lambda s="": L.append(str(s))
    P("# 5 分 K 紅轉綠 (綠實體 > 紅實體) 進場時機測試 (2020~2026, sr 模型每日前 10 檔, M1 逐分鐘)\n")
    P("- 型態: 第 i−1 根 5 分 K 收 > 開 (紅), 第 i 根收 < 開 (綠) 且綠實體 > 紅實體; 在第 i+1 根開盤 (−0.1% 滑價) 做空. cutoff 前沒出現 → 不做 / 或在 cutoff 照進. 出場規則同現行. 300 萬配置: 沒進場的檔跳過, 資金給下一檔.\n")
    rows = []
    for nm in names:
        D = alloc(nm); s = D.ret; g = S[S[f"in|{nm}"]==1][f"g|{nm}"] - BASE
        rows.append({"規則": nm, "進場率%": S[f"in|{nm}"].mean()*100, "日均檔數": D.n.mean(), "資金使用率": D.util.mean(), "每筆淨%": g.mean()*100, "勝率": (g>0).mean(), "日淨%": s.mean()*100,
                     "Sharpe": s.mean()/s.std()*np.sqrt(245) if s.std()>0 else np.nan, "MDD%": ((1+s).cumprod()/(1+s).cumprod().cummax()-1).min()*100, "年化(萬)": s.mean()*245*CAP/1e4})
    tab = pd.DataFrame(rows).set_index("規則"); P("## 1. 全期間"); P(tab.round(3).to_markdown()); P("")
    # 各年每筆
    yr = pd.DataFrame({nm: (S[S[f"in|{nm}"]==1].assign(net=lambda x: x[f"g|{nm}"]-BASE).groupby(S.date.dt.year).net.mean()*100) for nm in names})
    P("## 2. 各年每筆淨%"); P(yr.round(2).to_markdown()); P("")
    # 型態出現 vs 未出現 (以 10:00 cutoff) 的現行報酬: 型態有沒有預測力
    nm = "型態@10:00 沒就不做"; m = S[f"in|{nm}"]==1
    P("## 3. 型態本身有沒有資訊: 10:00 前出現型態的交易 vs 沒出現的, 用『現行 09:01 進場』的報酬比")
    P(f"- 出現型態 {int(m.sum())} 筆: 現行每筆淨 {(S.loc[m,'g|現行']-BASE).mean()*100:.3f}% (勝率 {((S.loc[m,'g|現行']-BASE)>0).mean():.3f})；未出現 {int((~m).sum())} 筆: {(S.loc[~m,'g|現行']-BASE).mean()*100:.3f}% (勝率 {((S.loc[~m,'g|現行']-BASE)>0).mean():.3f})")
    P(f"- 出現型態的交易改在型態後進場: {(S.loc[m, f'g|{nm}']-BASE).mean()*100:.3f}% vs 同一批交易 09:01 進場 {(S.loc[m,'g|現行']-BASE).mean()*100:.3f}% → 等型態的代價/好處 {((S.loc[m, f'g|{nm}']-S.loc[m,'g|現行'])).mean()*100:+.3f}%/筆")
    (OUT/"m5_pattern_entry.md").write_text("\n".join(L)); print("\n".join(L))
