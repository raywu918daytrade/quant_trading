"""大長 K 後突破 + MACD 背離 (5 分 K, 盤中) 的統計. 樣本 = trailing_stop_trades_sr.parquet (sr 模型每日前 10 檔, 2020~2026).

定義 (5 分 K, MACD 12/26/9 用前一交易日 + 當日的 5 分 K 連續計算, 避免開盤 EMA 沒暖機):
  大長紅 K : 5 分 K 收 − 開 ≥ B × 昨收 (B = 1.5% / 2.5%), 且在 13:00 前
  突破     : 大長紅 K 之後 W = 12 根 (1 小時) 內, 第一根 close > 大長紅 K 的 high
  頂背離   : 突破那根的 MACD 值 < 大長紅 K 到突破前一根之間的 MACD 最高值 (價格新高, MACD 沒新高)
             MACD 值用 DIF (快慢線差) 或 OSC (柱狀體) 兩種
  → 做空: 突破那根的下一根 5 分 K 開盤 (第一分鐘開盤 −0.1% 滑價) 做空, 出場同現行 (+8% 強制 / 13:24 +6% / 收盤競價)
  對照   : 同樣大長紅 K + 突破但「沒有」背離 (MACD 也創高)
  鏡像 (回補訊號): 大長黑 K (開 − 收 ≥ B×昨收) 後 1 小時內 close < 其 low, 且 MACD 沒創低 (底背離)
             → 對 09:01 已進場的空單: 在下一根開盤回補 vs 抱到現行出場
每天每檔只取第一個事件. 基準 = 同一批候選在「同一根 5 分 K」開盤隨便空到現行出場的平均 (扣掉時間點效應).
輸出 stats/bigk_break_macd_div.md, data/processed/bigk_break_macd_div_events.parquet
"""
import os, numpy as np, pandas as pd
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
ROOT = Path(os.environ.get("QT_ROOT", Path(__file__).resolve().parent.parent)); OUT = ROOT/"stats"
BASE, SLIP, K, PEN, CAP, PER_STOCK, MAX_N, PRE_K = 0.0023, 0.001, 0.08, 0.07, 3_000_000, 600_000, 10, 0.06
BODIES = (0.015, 0.025); W = 12; MEASURES = ("DIF", "OSC"); LAST_SIG_T = 240   # 13:00 前的 K
OOS_YEAR = 2024

def macd(c):
    s = pd.Series(c); dif = s.ewm(span=12, adjust=False).mean() - s.ewm(span=26, adjust=False).mean()
    dea = dif.ewm(span=9, adjust=False).mean(); return dif.values, (dif - dea).values

def bars5(g):
    b = g.t.values // 5
    k = pd.DataFrame({"b": b, "o": g.open.values, "h": g.high.values, "l": g.low.values, "c": g.close.values, "t": g.t.values})
    return k.groupby("b", sort=True).agg(o=("o","first"), h=("h","max"), l=("l","min"), c=("c","last"), t=("t","first")).reset_index(drop=True)

def find_event(o, h, l, c, tb, m, pc, B, bull):
    """回傳 (突破那根 index, 是否背離) 或 None. bull=False: 大長紅 K 上破 + 頂背離; bull=True: 大長黑 K 下破 + 底背離."""
    n = len(c)
    for i in range(n):
        if tb[i] >= LAST_SIG_T: break
        body = (o[i] - c[i]) if bull else (c[i] - o[i])
        if body < B*pc: continue
        for j in range(i+1, min(n, i+1+W)):
            if tb[j] >= LAST_SIG_T + 20: break
            if (not bull and c[j] > h[i]) or (bull and c[j] < l[i]):
                div = (m[j] < m[i:j].max()) if not bull else (m[j] > m[i:j].min())
                return j, bool(div)
    return None

def simulate_month(args):
    files, trades = args
    sids = set(trades.stock_id)
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df = df[df.stock_id.isin(sids)]
    ts = pd.to_datetime(df["date"]); df["day"] = ts.dt.normalize(); df["t"] = ts.dt.hour*60 + ts.dt.minute - 540
    df = df[(df.t>=0)&(df.t<=270)].sort_values(["stock_id","day","t"])
    groups = {k: g for k, g in df.groupby(["stock_id","day"], sort=False)}
    days_of = df.groupby("stock_id").day.unique().to_dict()
    out = []
    for _, tr in trades.iterrows():
        sid, day, pc = tr.stock_id, tr.date, tr.prev_close
        g = groups.get((sid, day))
        if g is None or len(g) < 30: continue
        t = g.t.values; o = g.open.values; h = g.high.values; c = g.close.values
        auction = c[t==270][-1] if (t==270).any() else c[-1]
        def exit_from(t_entry):
            m = (t >= t_entry) & (t < 264)
            if (h[m] >= pc*(1+K)).any(): return pc*(1+K)*(1+SLIP*3), "forced"
            i264 = np.where(t == 264)[0]
            if len(i264) and t_entry <= 264 and c[i264[0]] >= pc*(1+PRE_K): return c[i264[0]]*(1+SLIP), "pre"
            if auction >= pc*1.0975: return (tr.next_open*(1+PEN) if np.isfinite(tr.next_open) else auction), "lock"
            return auction, "close"
        def short_at(t_e):
            idx = np.where(t >= t_e)[0]
            if not len(idx) or t[idx[0]] >= 265: return np.nan, np.nan, np.nan
            entry = o[idx[0]]*(1-SLIP)
            if entry <= pc*0.905: return np.nan, np.nan, np.nan
            ex, _ = exit_from(t[idx[0]])
            j30 = np.where(t <= t[idx[0]] + 30)[0][-1]
            return entry/ex - 1, o[idx[0]]/c[j30] - 1, t[idx[0]]
        # 5 分 K + MACD (前一交易日暖機)
        k = bars5(g[g.t < 270])
        prev_days = [d for d in days_of.get(sid, []) if d < day]
        if prev_days:
            pg = groups.get((sid, max(prev_days)))
            kp = bars5(pg[pg.t < 270]) if pg is not None else k.iloc[:0]
        else: kp = k.iloc[:0]
        dif_all, osc_all = macd(np.r_[kp.c.values, k.c.values]); np_ = len(kp)
        mv = {"DIF": dif_all[np_:], "OSC": osc_all[np_:]}
        ko, kh, kl, kc, kt = k.o.values, k.h.values, k.l.values, k.c.values, k.t.values
        res = {"stock_id": sid, "date": day, "rank": tr["rank"], "entry": tr.entry, "warm": np_}
        ex, _ = exit_from(4); res["g_hold"] = tr.entry/ex - 1
        # 基準: 每根 5 分 K 開盤隨便空
        for bi in range(len(kt)):
            if kt[bi] >= 265: break
            res[f"base|{kt[bi]//5}"] = short_at(kt[bi])[0]
        for B in BODIES:
            for meas in MEASURES:
                tag = f"{B*100:.1f}%|{meas}"
                ev = find_event(ko, kh, kl, kc, kt, mv[meas], pc, B, bull=False)
                if ev is None: res[f"cls|{tag}"] = "無"
                else:
                    j, div = ev; res[f"cls|{tag}"] = "背離" if div else "沒背離"
                    g_, f30, te = short_at(kt[j] + 5); res[f"g|{tag}"] = g_; res[f"f30|{tag}"] = f30; res[f"t|{tag}"] = te; res[f"bar|{tag}"] = (kt[j] + 5)//5
                evb = find_event(ko, kh, kl, kc, kt, mv[meas], pc, B, bull=True)
                if evb is None: res[f"bcls|{tag}"] = "無"
                else:
                    j, div = evb; res[f"bcls|{tag}"] = "背離" if div else "沒背離"
                    t_e = kt[j] + 5; idx = np.where(t >= t_e)[0]
                    pre = (t >= 4) & (t < t_e) & (t < 264)
                    if not len(idx) or t[idx[0]] >= 265 or (h[pre] >= pc*(1+K)).any(): res[f"bcov|{tag}"] = res["g_hold"]
                    else: res[f"bcov|{tag}"] = tr.entry/(o[idx[0]]*(1+SLIP)) - 1
                    res[f"bt|{tag}"] = t_e
        out.append(res)
    return pd.DataFrame(out)

def tstat(x):
    x = pd.Series(x).dropna(); return x.mean()/x.std()*np.sqrt(len(x)) if len(x) > 2 and x.std() > 0 else np.nan

if __name__ == "__main__":
    R = pd.read_parquet(ROOT/"data/processed/trailing_stop_trades_sr.parquet")[["stock_id","date","entry","prev_close","next_open","rank"]]
    R["date"] = pd.to_datetime(R.date); R["ym"] = R.date.dt.strftime("%Y_%m")
    M1 = ROOT/"data/raw/m1"; jobs = []
    for ym, g in R.groupby("ym"):
        f = M1/f"{ym}.parquet"
        if not f.exists(): continue
        y, m = map(int, ym.split("_")); py, pm = (y, m-1) if m > 1 else (y-1, 12); pf = M1/f"{py}_{pm:02d}.parquet"
        jobs.append(([str(pf), str(f)] if pf.exists() else [str(f)], g))
    with ProcessPoolExecutor(int(os.environ.get("QT_PROCS", 6))) as ex: parts = list(ex.map(simulate_month, jobs))
    S = pd.concat(parts, ignore_index=True); S["year"] = S.date.dt.year; S["oos"] = S.year >= OOS_YEAR
    S.to_parquet(ROOT/"data/processed/bigk_break_macd_div_events.parquet", index=False); print("simulated", len(S), flush=True)
    base_cols = [c for c in S.columns if c.startswith("base|")]
    base_mean = {int(c.split("|")[1]): (S[c] - BASE).mean() for c in base_cols}
    all_days = pd.Series(sorted(S.date.unique()))
    tags = [f"{B*100:.1f}%|{m}" for B in BODIES for m in MEASURES]

    def alloc(mask, gcol):
        recs = []
        for dt, gg in S[mask].sort_values(["date","rank"]).groupby("date", sort=False):
            cash = CAP; amts = []; nets = []
            for _, r in gg.iterrows():
                lc = r.entry*1000
                if cash < lc: continue
                lots = int(min(PER_STOCK, cash)//lc)
                if lots < 1: continue
                a = lots*lc; cash -= a; amts.append(a); nets.append(r[gcol] - BASE)
                if len(amts) >= MAX_N: break
            recs.append((dt, len(amts), sum(a*n for a, n in zip(amts, nets))/CAP))
        D = pd.DataFrame(recs, columns=["date","n","ret"]).set_index("date").reindex(all_days).fillna({"n": 0, "ret": 0.0}); return D

    L = []; P = lambda s="": L.append(str(s))
    P("# 大長 K 後突破 + MACD 背離 (5 分 K, 2020~2026, sr 模型每日前 10 檔, M1 逐分鐘)\n")
    P(f"- 樣本 {len(S)} 檔日 / {S.date.nunique()} 天；樣本內 2020~{OOS_YEAR-1}、樣本外 {OOS_YEAR}~。MACD 有前一交易日暖機的占 {(S.warm>0).mean()*100:.1f}%")
    P("- 大長紅 K: 5 分 K 收−開 ≥ 1.5% 或 2.5% 昨收 (13:00 前)；突破: 之後 1 小時內第一根 close > 長紅 K 的 high；")
    P("  頂背離: 突破那根的 MACD (DIF 或 OSC 柱) < 長紅 K 到突破前的 MACD 最高 → 下一根 5 分 K 開盤 (−0.1%) 做空，出場同現行。每檔每天只取第一個事件。")
    P("- 「比同時間」= 事件每筆淨 − 同一批候選在同一根 5 分 K 開盤隨便空的平均 (扣掉越晚進場越差的時間效應)；t 是這個差的 t 值，|t|>2 才算不是運氣。")
    P("- 「09:01 單」= 同一檔同一天現行 09:01 進場的每筆淨 (看事件有沒有挑出本來就好/壞的單)\n")

    P("## 1. 頂背離做空: 事件本身 (每筆, 扣成本 0.23%)")
    rows = []
    for tag in tags:
        for cls in ("背離", "沒背離"):
            m = (S[f"cls|{tag}"] == cls) & S[f"g|{tag}"].notna()
            if not m.any(): continue
            E = S[m]; net = E[f"g|{tag}"] - BASE; exc = net - E[f"bar|{tag}"].map(base_mean)
            row = {"長K / MACD": tag.replace("|", " "), "類別": cls, "筆數": int(m.sum()), "占檔日%": m.mean()*100,
                   "進場時間中位": f"{9 + int(E[f't|{tag}'].median())//60}:{int(E[f't|{tag}'].median())%60:02d}",
                   "每筆淨%": net.mean()*100, "勝率": (net > 0).mean(), "30分內跌%": E[f"f30|{tag}"].mean()*100,
                   "比同時間%": exc.mean()*100, "t": tstat(exc)}
            for nm, mm in (("IS", ~E.oos), ("OOS", E.oos)):
                row[f"每筆淨% {nm}"] = net[mm].mean()*100; row[f"比同時間% {nm}"] = exc[mm].mean()*100; row[f"t {nm}"] = tstat(exc[mm])
            row["09:01 單%"] = (E.g_hold - BASE).mean()*100
            rows.append(row)
    P(pd.DataFrame(rows).round(3).to_markdown(index=False)); P("")
    P(f"- 參考: 全部候選 09:01 進場每筆淨 {(S.g_hold-BASE).mean()*100:.3f}% (勝率 {((S.g_hold-BASE)>0).mean():.3f})\n")

    P("## 2. 頂背離當進場條件 (300 萬配置, 只空出現頂背離的檔, 依原排名配) vs 現行")
    rows = []; mask_all = pd.Series(True, index=S.index)
    for nm, mask, gcol in [("現行 09:01 全進", mask_all, "g_hold")] + [(f"只空頂背離 {t.replace('|',' ')}", (S[f"cls|{t}"] == "背離") & S[f"g|{t}"].notna(), f"g|{t}") for t in tags]:
        D = alloc(mask, gcol); s = D.ret
        r = {"規則": nm, "日均檔數": D.n.mean(), "日淨%": s.mean()*100, "Sharpe": s.mean()/s.std()*np.sqrt(245) if s.std() > 0 else np.nan,
             "MDD%": ((1+s).cumprod()/(1+s).cumprod().cummax()-1).min()*100, "年化(萬)": s.mean()*245*CAP/1e4}
        for lab, mm in (("IS", D.index.year < OOS_YEAR), ("OOS", D.index.year >= OOS_YEAR)):
            ss = s[mm]; r[f"年化(萬) {lab}"] = ss.mean()*245*CAP/1e4; r[f"Sharpe {lab}"] = ss.mean()/ss.std()*np.sqrt(245) if ss.std() > 0 else np.nan
        rows.append(r)
    P(pd.DataFrame(rows).set_index("規則").round(3).to_markdown()); P("")

    P("## 3. 已在 09:01 空的單, 盤中出現頂背離 → 之後還會跌嗎 (加碼參考)")
    P("- 就是第 1 節的「比同時間」：> 0 代表頂背離後再空比同時間隨便空好。每筆淨看第 1 節。\n")

    P("## 4. 底背離 (大長黑 K 下破 + MACD 沒創低) 當回補訊號: 對 09:01 空單, 訊號出現就回補 vs 抱到現行出場")
    rows = []
    for tag in tags:
        for cls in ("背離", "沒背離"):
            m = (S[f"bcls|{tag}"] == cls) & S[f"bcov|{tag}"].notna()
            if not m.any(): continue
            E = S[m]; d = E[f"bcov|{tag}"] - E.g_hold
            row = {"長K / MACD": tag.replace("|", " "), "類別": cls, "筆數": int(m.sum()), "占檔日%": m.mean()*100,
                   "訊號時間中位": f"{9 + int(E[f'bt|{tag}'].median())//60}:{int(E[f'bt|{tag}'].median())%60:02d}",
                   "抱到底每筆淨%": (E.g_hold - BASE).mean()*100, "訊號回補每筆淨%": (E[f"bcov|{tag}"] - BASE).mean()*100,
                   "回補−抱%": d.mean()*100, "t": tstat(d)}
            for nm, mm in (("IS", ~E.oos), ("OOS", E.oos)): row[f"回補−抱% {nm}"] = d[mm].mean()*100; row[f"t {nm}"] = tstat(d[mm])
            rows.append(row)
    P(pd.DataFrame(rows).round(3).to_markdown(index=False)); P("")

    P("## 5. 各年 頂背離做空 每筆淨% / 比同時間%")
    yr = {}
    for tag in tags:
        m = (S[f"cls|{tag}"] == "背離") & S[f"g|{tag}"].notna(); E = S[m]; net = E[f"g|{tag}"] - BASE
        yr[f"{tag.replace('|',' ')} 每筆淨"] = net.groupby(E.year).mean()*100
        yr[f"{tag.replace('|',' ')} 比同時間"] = (net - E[f"bar|{tag}"].map(base_mean)).groupby(E.year).mean()*100
        yr[f"{tag.replace('|',' ')} 筆數"] = net.groupby(E.year).size()
    P(pd.DataFrame(yr).round(2).to_markdown()); P("")
    P("## 6. 基準: 同時間隨便空 (每根 5 分 K 開盤空到現行出場, 每筆淨%)")
    bm = pd.Series(base_mean).sort_index(); bm.index = [f"{9 + (i*5)//60}:{(i*5)%60:02d}" for i in bm.index]
    P((bm*100).round(3).to_frame("每筆淨%").iloc[::3].T.to_markdown()); P("")
    OUT.mkdir(exist_ok=True); (OUT/"bigk_break_macd_div.md").write_text("\n".join(L)); print("\n".join(L))
