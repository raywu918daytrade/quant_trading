"""開盤爆量測試: 第一根 M1 (09:00, 含開盤競價) / 第一根 M5 (09:00~09:04) 的成交量, 相對前 1 日 / 前 5 日同時段、或前 5 日全日量, 跟當天做空結果的關係.
樣本 = trailing_stop_trades_sr.parquet (sr 模型每日前 10 檔, 2020~2026). 量用 M1 (張).
比值:
  r1_5d  = 第一根 M1 量 / 前 5 日第一根 M1 量平均   (09:01 進場前就知道)
  r5_1d  = 第一根 M5 量 / 前 1 日第一根 M5 量
  r5_5d  = 第一根 M5 量 / 前 5 日第一根 M5 量平均   (要等到 09:05 才知道 → 進場改 09:05 開盤)
  r5_day = 第一根 M5 量 / 前 5 日全日量平均        (開盤 5 分鐘就做掉平常一天的幾成)
分析: 依比值分組看現行報酬/勝率/強制回補率/盤中最高漲幅; 當濾網 (比值 > 門檻不做) 的 300 萬結果.
輸出 stats/open_volume_test.md"""
import numpy as np, pandas as pd
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
ROOT = Path(__file__).resolve().parent.parent; OUT = ROOT/"stats"
BASE, SLIP, K, PEN, CAP, PER_STOCK, MAX_N, PRE_K = 0.0023, 0.001, 0.08, 0.07, 3_000_000, 600_000, 10, 0.06

def load(ym, ids):
    f = ROOT/"data/raw/m1"/f"{ym}.parquet"
    if not f.exists(): return None
    d = pd.read_parquet(f, filters=[("stock_id", "in", list(ids))]); ts = pd.to_datetime(d["date"]); d["day"] = ts.dt.normalize(); d["t"] = ts.dt.hour*60 + ts.dt.minute - 540
    return d[(d.t >= 0) & (d.t <= 270)]

def month(args):
    ym, trades = args
    ids = set(trades.stock_id); y, m = map(int, ym.split("_")); prev = f"{y}_{m-1:02d}" if m > 1 else f"{y-1}_12"
    parts = [x for x in (load(prev, ids), load(ym, ids)) if x is not None]
    if not parts: return pd.DataFrame()
    d = pd.concat(parts, ignore_index=True)
    agg = d.groupby(["stock_id","day"]).apply(lambda g: pd.Series({"v1": g.volume[g.t == 0].sum(), "v5": g.volume[g.t <= 4].sum(), "vday": g.volume.sum()})).reset_index().sort_values(["stock_id","day"])
    for c in ("v1","v5","vday"):
        agg[f"{c}_p1"] = agg.groupby("stock_id")[c].shift(1); agg[f"{c}_p5"] = agg.groupby("stock_id")[c].transform(lambda s: s.shift(1).rolling(5, min_periods=3).mean())
    key = set(zip(trades.stock_id, trades.date)); cur = d[[(s, dd) in key for s, dd in zip(d.stock_id, d.day)]]
    out = []
    for (sid, day), g in cur.groupby(["stock_id","day"], sort=False):
        tr = trades[(trades.stock_id == sid) & (trades.date == day)].iloc[0]; pc = tr.prev_close
        g = g.sort_values("t"); t = g.t.values; o = g.open.values; h = g.high.values; c = g.close.values
        auction = c[t == 270][-1] if (t == 270).any() else c[-1]
        def exit_from(t_e):
            if (h[(t >= t_e) & (t < 264)] >= pc*(1+K)).any(): return pc*(1+K)*(1+SLIP*3), "forced"
            i = np.where(t == 264)[0]
            if len(i) and c[i[0]] >= pc*(1+PRE_K): return c[i[0]]*(1+SLIP), "pre"
            if auction >= pc*1.0975: return (tr.next_open*(1+PEN) if np.isfinite(tr.next_open) else auction), "lock"
            return auction, "close"
        a = agg[(agg.stock_id == sid) & (agg.day == day)]
        if a.empty: continue
        a = a.iloc[0]
        ex, rs = exit_from(4); res = {"stock_id": sid, "date": day, "rank": tr["rank"], "entry": tr.entry, "g01": tr.entry/ex - 1, "reason": rs, "maxup": h[(t >= 4) & (t < 270)].max()/pc - 1 if ((t >= 4) & (t < 270)).any() else np.nan}
        i5 = np.where(t >= 5)[0]
        if len(i5):
            e5 = o[i5[0]]*(1-SLIP); ex5, _ = exit_from(t[i5[0]]); res["entry05"] = e5; res["g05"] = e5/ex5 - 1 if e5 > pc*0.905 else np.nan
        else: res["entry05"] = np.nan; res["g05"] = np.nan
        res.update({"v1": a.v1, "v5": a.v5, "r1_5d": a.v1/a.v1_p5 if a.v1_p5 > 0 else np.nan, "r5_1d": a.v5/a.v5_p1 if a.v5_p1 > 0 else np.nan, "r5_5d": a.v5/a.v5_p5 if a.v5_p5 > 0 else np.nan, "r5_day": a.v5/a.vday_p5 if a.vday_p5 > 0 else np.nan})
        out.append(res)
    return pd.DataFrame(out)

if __name__ == "__main__":
    R = pd.read_parquet(ROOT/"data/processed/trailing_stop_trades_sr.parquet")[["stock_id","date","entry","prev_close","next_open","rank"]]; R["ym"] = R.date.dt.strftime("%Y_%m")
    with ProcessPoolExecutor(6) as ex: parts = list(ex.map(month, [(ym, g) for ym, g in R.groupby("ym")]))
    S = pd.concat([p for p in parts if len(p)], ignore_index=True); S.to_parquet(ROOT/"data/processed/open_volume_trades.parquet", index=False); print("simulated", len(S), flush=True)
    all_days = pd.Series(sorted(S.date.unique()))
    def alloc(mask, gcol, ecol="entry"):
        recs = []
        for dt, gg in S[mask & S[gcol].notna()].sort_values(["date","rank"]).groupby("date", sort=False):
            cash = CAP; amts = []; nets = []
            for _, r in gg.iterrows():
                lc = r[ecol]*1000
                if cash < lc: continue
                lots = int(min(PER_STOCK, cash)//lc)
                if lots < 1: continue
                a = lots*lc; cash -= a; amts.append(a); nets.append(r[gcol]-BASE)
                if len(amts) >= MAX_N: break
            recs.append((dt, len(amts), sum(a*n for a, n in zip(amts, nets))/CAP))
        D = pd.DataFrame(recs, columns=["date","n","ret"]).set_index("date").reindex(all_days); return D.fillna({"n": 0, "ret": 0.0})
    def stats(name, mask, gcol, ecol="entry"):
        D = alloc(mask, gcol, ecol); s = D.ret; g = S.loc[mask & S[gcol].notna(), gcol] - BASE
        return {"規則": name, "交易占比%": (mask & S[gcol].notna()).mean()*100, "日均檔數": D.n.mean(), "每筆淨%": g.mean()*100, "勝率": (g > 0).mean(), "日淨%": s.mean()*100, "Sharpe": s.mean()/s.std()*np.sqrt(245), "MDD%": ((1+s).cumprod()/(1+s).cumprod().cummax()-1).min()*100, "年化(萬)": s.mean()*245*CAP/1e4}
    L = []; P = lambda s="": L.append(str(s))
    P("# 開盤爆量 vs 做空結果 (2020~2026, sr 模型每日前 10 檔, M1 逐分鐘, 300 萬)\n")
    P(f"- 樣本 {len(S)} 筆。r1_5d = 09:00 那根 M1 量 / 前 5 日同根平均 (09:01 進場前可知)；r5_1d / r5_5d = 09:00~09:04 五分鐘量 / 前 1 日 / 前 5 日同時段；r5_day = 五分鐘量 / 前 5 日全日量平均。用 M5 的版本進場要延到 09:05 開盤。\n")
    P("## 1. 依比值分組: 現行 09:01 進場的結果 (爆量的那組是不是更常被軋)")
    for col, bins, labs in [("r1_5d", [0, .5, 1, 2, 3, 5, 10, 1e9], ["<0.5x","0.5~1x","1~2x","2~3x","3~5x","5~10x",">10x"]), ("r5_1d", [0, .5, 1, 2, 3, 5, 10, 1e9], ["<0.5x","0.5~1x","1~2x","2~3x","3~5x","5~10x",">10x"]),
                            ("r5_5d", [0, .5, 1, 2, 3, 5, 10, 1e9], ["<0.5x","0.5~1x","1~2x","2~3x","3~5x","5~10x",">10x"]), ("r5_day", [0, .05, .1, .2, .3, .5, 1e9], ["<5%","5~10%","10~20%","20~30%","30~50%",">50%"])]:
        b = pd.cut(S[col], bins, labels=labs); net = S.g01 - BASE
        T = pd.DataFrame({"筆數": net.groupby(b).size(), "占比%": net.groupby(b).size()/len(S)*100, "每筆淨%": net.groupby(b).mean()*100, "勝率": (net > 0).groupby(b).mean(), "強制回補%": (S.reason == "forced").groupby(b).mean()*100, "鎖漲停%": (S.reason == "lock").groupby(b).mean()*100, "盤中最高漲幅中位%": S.maxup.groupby(b).median()*100, "09:05進場每筆淨%": (S.g05 - BASE).groupby(b).mean()*100})
        P(f"\n### {col}"); P(T.round(3).to_markdown())
    P("\n## 2. 當濾網: 比值 > 門檻的不做 (資金給下一檔; 用 M5 的版本全部改 09:05 開盤進場)")
    rows = [stats("現行 09:01", S.g01.notna(), "g01"), stats("現行但 09:05 進場 (無濾網, 看延後成本)", S.g05.notna(), "g05", "entry05")]
    for th in (2, 3, 5, 10): rows.append(stats(f"r1_5d ≤ {th}x, 09:01 進場", S.r1_5d <= th, "g01"))
    for th in (2, 3, 5, 10): rows.append(stats(f"r5_5d ≤ {th}x, 09:05 進場", S.r5_5d <= th, "g05", "entry05"))
    for th in (2, 3, 5, 10): rows.append(stats(f"r5_1d ≤ {th}x, 09:05 進場", S.r5_1d <= th, "g05", "entry05"))
    for th in (.1, .2, .3, .5): rows.append(stats(f"r5_day ≤ {th*100:.0f}%, 09:05 進場", S.r5_day <= th, "g05", "entry05"))
    for th in (2, 3, 5): rows.append(stats(f"反向: 只做 r1_5d > {th}x, 09:01 進場", S.r1_5d > th, "g01"))
    P(pd.DataFrame(rows).set_index("規則").round(3).to_markdown()); P("")
    P("## 3. 各年每筆淨% (現行 vs r1_5d ≤ 3x vs r5_5d ≤ 3x@09:05)")
    yr = pd.DataFrame({"現行": (S.g01-BASE).groupby(S.date.dt.year).mean()*100, "r1_5d≤3x": (S.g01-BASE)[S.r1_5d <= 3].groupby(S.date.dt.year).mean()*100, "r5_5d≤3x@09:05": (S.g05-BASE)[S.r5_5d <= 3].groupby(S.date.dt.year).mean()*100, "全部@09:05": (S.g05-BASE).groupby(S.date.dt.year).mean()*100})
    P(yr.round(2).to_markdown()); P("")
    P("## 4. 相關")
    P(f"- 比值 vs 現行淨報酬 Spearman: " + ", ".join(f"{c} {S[[c,'g01']].corr(method='spearman').iloc[0,1]:+.3f}" for c in ("r1_5d","r5_1d","r5_5d","r5_day")))
    P(f"- 比值 vs 盤中最高漲幅 Spearman: " + ", ".join(f"{c} {S[[c,'maxup']].corr(method='spearman').iloc[0,1]:+.3f}" for c in ("r1_5d","r5_1d","r5_5d","r5_day")))
    (OUT/"open_volume_test.md").write_text("\n".join(L)); print("\n".join(L))
