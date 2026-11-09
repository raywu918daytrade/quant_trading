"""開盤 5 分紅 K → 延後放空 的混合版. 樣本 = trailing_stop_trades_sr.parquet (sr 每日前 10 檔, 2020~2026), M1 逐分鐘.
開盤 5 分 K = 09:00 開 o0 ~ 09:04 收 c4; 紅 K 幅度 = (c4 − o0) / 昨收.
規則: 紅 K 幅度 ≥ X 的檔延到 09:05 (或 09:10) 那一分鐘開盤 (−0.1% 滑價) 才空, 其他照現行 09:01 進.
出場同現行: +8% 強制回補 (+0.3%) / 13:24 ≥ +6% 回補 / 收盤競價 (收盤 ≥ +9.75% 鎖死 → 隔日開盤 +7%).
延後進場前已摸到 +8% 的不進 (資金也不補給別檔, 保守). 300 萬配置 (每檔 ≤ 60 萬, 依原排名, 最多 10 檔). 成本 0.23%.
-> stats/red_k_delay.md"""
import os, numpy as np, pandas as pd
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
ROOT = Path(os.environ.get("QT_ROOT", Path(__file__).resolve().parent.parent)); OUT = ROOT / "stats"
BASE, SLIP, K, PEN, CAP, PER_STOCK, MAX_N, PRE_K, SPLIT = 0.0023, 0.001, 0.08, 0.07, 3_000_000, 600_000, 10, 0.06, 2024
DELAYS = (5, 10)

def simulate_month(args):
    f, trades = args
    df = pd.read_parquet(f); df = df[df.stock_id.isin(set(trades.stock_id))]
    ts = pd.to_datetime(df["date"]); df["day"] = ts.dt.normalize(); df["t"] = ts.dt.hour * 60 + ts.dt.minute - 540
    df = df[(df.t >= 0) & (df.t <= 270)].drop_duplicates(["stock_id", "day", "t"], keep="last").sort_values(["stock_id", "day", "t"])
    groups = {k: g for k, g in df.groupby(["stock_id", "day"], sort=False)}
    out = []
    for tr in trades.itertuples():
        g = groups.get((tr.stock_id, tr.date))
        if g is None or len(g) < 30: continue
        pc = tr.prev_close; t = g.t.values; o = g.open.values; h = g.high.values; c = g.close.values
        auction = c[t == 270][-1] if (t == 270).any() else c[-1]
        def exit_from(te):
            m = (t >= te) & (t < 264)
            if (h[m] >= pc * (1 + K)).any(): return pc * (1 + K) * (1 + SLIP * 3)
            i264 = np.where(t == 264)[0]
            if len(i264) and c[i264[0]] >= pc * (1 + PRE_K): return c[i264[0]] * (1 + SLIP)
            if auction >= pc * 1.0975: return tr.next_open * (1 + PEN) if np.isfinite(tr.next_open) else auction
            return auction
        w5 = t < 5
        if not w5.any(): continue
        res = {"stock_id": tr.stock_id, "date": tr.date, "rank": tr.rank, "entry": tr.entry, "redk": (c[w5][-1] - o[w5][0]) / pc,
               "g0": tr.entry / exit_from(4) - 1}
        for d in DELAYS:
            idx = np.where(t >= d)[0]
            pre = (t < d) & (h >= pc * (1 + K))
            if not len(idx) or pre.any(): res[f"g{d}"] = np.nan; continue
            e = o[idx[0]] * (1 - SLIP); res[f"g{d}"] = e / exit_from(t[idx[0]]) - 1
        out.append(res)
    return pd.DataFrame(out)

def tstat(x):
    x = pd.Series(x).dropna(); return x.mean() / x.std() * np.sqrt(len(x)) if len(x) > 2 and x.std() > 0 else np.nan

if __name__ == "__main__":
    R = pd.read_parquet(ROOT / "data/processed/trailing_stop_trades_sr.parquet")[["stock_id", "date", "entry", "prev_close", "next_open", "rank"]]
    R["date"] = pd.to_datetime(R.date).astype("datetime64[ns]"); R["ym"] = R.date.dt.strftime("%Y_%m")
    jobs = [(str(ROOT / "data/raw/m1" / f"{ym}.parquet"), g) for ym, g in R.groupby("ym") if (ROOT / "data/raw/m1" / f"{ym}.parquet").exists()]
    with ProcessPoolExecutor(6) as ex: S = pd.concat(list(ex.map(simulate_month, jobs)), ignore_index=True)
    S["oos"] = S.date.dt.year >= SPLIT; all_days = pd.Series(sorted(S.date.unique()))

    def alloc(gser):
        X = S.assign(g=gser).sort_values(["date", "rank"]); recs = []
        for dt, gg in X.groupby("date", sort=False):
            cash = CAP; tot = 0.0; n = 0
            for r in gg.itertuples():
                lc = r.entry * 1000; lots = int(min(PER_STOCK, cash) // lc) if cash >= lc else 0
                if lots < 1: continue
                a = lots * lc; cash -= a; n += 1
                if np.isfinite(r.g): tot += a * (r.g - BASE)
                if n >= MAX_N: break
            recs.append((dt, n, tot / CAP))
        return pd.DataFrame(recs, columns=["date", "n", "ret"]).set_index("date").reindex(all_days).fillna({"n": 0, "ret": 0.0})

    RULES = [("現行 09:01 全進", S.g0)]
    for d in DELAYS:
        RULES.append((f"全部延到 09:{d:02d}", S[f"g{d}"]))
        for X in (0.0, 0.005, 0.01, 0.02):
            m = S.redk > X
            RULES.append((f"紅K>{X*100:.1f}% → 09:{d:02d}, 其他 09:01", S[f"g{d}"].where(m, S.g0)))
    L = []; P = lambda s="": L.append(str(s))
    P(f"# 開盤 5 分紅 K 延後放空 (sr 每日前 10 檔, {S.date.min():%Y-%m-%d}~{S.date.max():%Y-%m-%d}, {len(S)} 筆)"); P()
    P("- 紅 K 幅度 = (09:04 收 − 09:00 開) / 昨收; 紅 K ≥ X 的檔延到 09:05 / 09:10 開盤 (−0.1%) 才空, 其他照現行 09:01. 延後前已摸 +8% 的不進")
    P("- 出場同現行; 300 萬配置; 「差 t」= 改變的那幾筆, 延後 − 09:01 每筆差的 t 值 (|t|>2 才算不是運氣)")
    P(f"- 紅 K 占比: >0 {(S.redk > 0).mean()*100:.1f}%, >0.5% {(S.redk > 0.005).mean()*100:.1f}%, >1% {(S.redk > 0.01).mean()*100:.1f}%, >2% {(S.redk > 0.02).mean()*100:.1f}%"); P()
    rows = []
    for nm, gs in RULES:
        D = alloc(gs); s = D.ret; ch = (gs != S.g0) | gs.isna(); diff = (gs - S.g0)[ch].fillna(-(S.g0[ch] - BASE))   # 沒進場 → 少了那筆
        r = {"規則": nm, "改變筆數%": ch.mean() * 100, "改變那些 每筆差%": diff.mean() * 100, "差 t": tstat(diff),
             "差 t OOS": tstat(diff[S.oos[ch]]), "日淨%": s.mean() * 100, "Sharpe": s.mean() / s.std() * np.sqrt(245),
             "MDD%": ((1 + s).cumprod() / (1 + s).cumprod().cummax() - 1).min() * 100, "年化(萬)": s.mean() * 245 * CAP / 1e4}
        for lab, mm in (("IS", D.index.year < SPLIT), ("OOS", D.index.year >= SPLIT)):
            ss = s[mm]; r[f"日淨% {lab}"] = ss.mean() * 100; r[f"Sharpe {lab}"] = ss.mean() / ss.std() * np.sqrt(245)
        rows.append(r)
    P("## 規則比較"); P(pd.DataFrame(rows).set_index("規則").round(3).to_markdown()); P()
    P("## 紅 K 幅度分組: 09:01 vs 09:05 vs 09:10 每筆淨% (未進場不算)")
    S["bin"] = pd.cut(S.redk, [-1, -0.01, 0, 0.005, 0.01, 0.02, 1], labels=["黑 ≤−1%", "黑 −1~0%", "紅 0~0.5%", "紅 0.5~1%", "紅 1~2%", "紅 >2%"])
    G = S.groupby("bin", observed=True).agg(筆數=("g0", "size"), **{"09:01": ("g0", lambda x: (x - BASE).mean() * 100)},
                                             **{f"09:{d:02d}": (f"g{d}", lambda x: (x.dropna() - BASE).mean() * 100) for d in DELAYS},
                                             **{f"09:{d:02d} 沒進%": (f"g{d}", lambda x: x.isna().mean() * 100) for d in DELAYS})
    P(G.round(3).to_markdown()); P()
    P("## 各年 日淨% (現行 vs 最佳混合)")
    best = max(rows[1:], key=lambda r: r["日淨%"])["規則"]; gs = dict(RULES)[best]
    Y = pd.DataFrame({"現行": alloc(S.g0).ret, best: alloc(gs).ret}); P((Y.groupby(Y.index.year).mean() * 100).round(3).T.to_markdown()); P()
    (OUT / "red_k_delay.md").write_text("\n".join(L), encoding="utf-8"); print("\n".join(L))
