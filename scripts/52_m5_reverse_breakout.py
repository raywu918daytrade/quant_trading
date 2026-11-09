"""M5 反向突破 多空都做: 往上突破就空, 往下跌破就做多 (破底翻 / 假突破), 跟 fade 現行「跌破才空」對照.
樣本 = 當沖池 (上市櫃非 ETF, 股價 ≥5, 前 20 日均成交額 ≥5 億, ATR20 ≥3%), 2020~2026 有 M1 的月份.

規則 (5 分 K 由 M1 合成, 訊號 K 要在 12:00 前收完, 下一根 5 分 K 第一分鐘開盤進場, 滑價 0.1%):
  A 開盤區間 (ORB) : 09:00~09:05 第一根 5 分 K 的高低 = 區間
      A空: 之後第一根 close > 區間高 → 空 ; A多: 第一根 close < 區間低 → 做多
  B 大長 K         : 長紅 K (收−開 ≥ 2.5% 昨收) 後 1 小時內 close > 其 high → 空 ;
                     長黑 K (開−收 ≥ 2.5% 昨收) 後 1 小時內 close < 其 low → 做多
  出場: 收盤 (13:30 競價) / 停損版: 空單碰到訊號 K 的 high、多單碰到訊號 K 的 low 就停 (+0.1% 滑價)
        都有 ±8% (相對昨收) 強制出場. 成本 0.23% 來回.
  順勢 (= 突破跟著做) 只列每筆淨, 用來看反向是不是只是把順勢倒過來.
  對照 fade 現行: 跳空 ≥2%, 第一次 close 跌破開盤 5 分區間低就空, 碰到區間高停損, 否則收盤回補 (這裡用 M5 收盤判斷, 跟實盤可能略不同).
每檔每天每個規則只取第一個訊號. 跳空分組: ≥+2% / ≤−2% / 不限.
輸出 stats/m5_reverse_breakout.md, data/processed/m5_reverse_breakout_trades.parquet
"""
import os, sys, glob, numpy as np, pandas as pd
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
ROOT = Path(os.environ.get("QT_ROOT", Path(__file__).resolve().parent.parent)); OUT = ROOT/"stats"
sys.path.insert(0, str(Path(__file__).resolve().parent))
BASE, SLIP, K = 0.0023, 0.001, 0.08
BIG, W, SIG_END = 0.025, 12, 180     # 長 K 門檻, 突破視窗 (根), 訊號 K 開始時間 < 12:00 前 (分鐘, 09:00 起)
POOL_DV, POOL_ATR, OOS_YEAR = 5e8, 0.03, 2024

def bars5(t, o, h, l, c):
    b = t // 5
    k = pd.DataFrame({"b": b, "o": o, "h": h, "l": l, "c": c, "t": t}).groupby("b", sort=True).agg(
        o=("o","first"), h=("h","max"), l=("l","min"), c=("c","last"), t=("t","first")).reset_index(drop=True)
    return k[k.t < 265]

def trade(t, o, h, l, c, auction, pc, t_e, side, stop):
    """side +1 = 空, −1 = 多. 回傳 (毛報酬, 出場原因, 進場時間)."""
    idx = np.where(t >= t_e)[0]
    if not len(idx) or t[idx[0]] >= 265: return np.nan, None, np.nan
    i = idx[0]; e_raw = o[i]
    if side == 1:
        entry = e_raw*(1-SLIP)
        if e_raw <= pc*0.905: return np.nan, None, np.nan
    else:
        entry = e_raw*(1+SLIP)
        if e_raw >= pc*1.095: return np.nan, None, np.nan
    m = slice(i, None); tt = t[m]; hh = h[m]; ll = l[m]; ok = tt < 265
    if side == 1:
        hit_f = np.where(ok & (hh >= pc*(1+K)))[0]; hit_s = np.where(ok & (hh >= stop))[0] if stop is not None else np.array([], int)
    else:
        hit_f = np.where(ok & (ll <= pc*(1-K)))[0]; hit_s = np.where(ok & (ll <= stop))[0] if stop is not None else np.array([], int)
    jf = hit_f[0] if len(hit_f) else 10**9; js = hit_s[0] if len(hit_s) else 10**9
    if js < jf or (js == jf and js < 10**9):
        ex = (max(stop, o[i+js]) if side == 1 else min(stop, o[i+js])) if js > 0 else (max(stop, e_raw) if side == 1 else min(stop, e_raw))
        ex = ex*(1+SLIP) if side == 1 else ex*(1-SLIP); why = "stop"
    elif jf < 10**9:
        ex = pc*(1+K)*(1+3*SLIP) if side == 1 else pc*(1-K)*(1-3*SLIP); why = "forced"
    else: ex = auction; why = "close"
    g = entry/ex - 1 if side == 1 else ex/entry - 1
    return g, why, t[i]

def find_orb(k):
    """回傳 {'up': (j, 訊號K), 'dn': (j, 訊號K)} — 第一根 close 突破 / 跌破第一根 5 分 K 區間."""
    if len(k) < 3 or k.t.iloc[0] != 0: return {}
    hi, lo = k.h.iloc[0], k.l.iloc[0]; out = {}
    for j in range(1, len(k)):
        if k.t.iloc[j] >= SIG_END: break
        if "up" not in out and k.c.iloc[j] > hi: out["up"] = j
        if "dn" not in out and k.c.iloc[j] < lo: out["dn"] = j
        if len(out) == 2: break
    return out

def find_bigk(k, pc):
    out = {}; o, h, l, c, tb = k.o.values, k.h.values, k.l.values, k.c.values, k.t.values; n = len(c)
    for side, key in ((1, "up"), (-1, "dn")):
        for i in range(n):
            if tb[i] >= SIG_END: break
            body = (c[i]-o[i]) if side == 1 else (o[i]-c[i])
            if body < BIG*pc: continue
            for j in range(i+1, min(n, i+1+W)):
                if tb[j] >= SIG_END: break
                if (side == 1 and c[j] > h[i]) or (side == -1 and c[j] < l[i]): out[key] = j; break
            if key in out: break
    return out

def simulate_month(args):
    f, days = args
    df = pd.read_parquet(f); df = df[df.stock_id.isin(set(days.stock_id))]
    ts = pd.to_datetime(df["date"]); df["day"] = ts.dt.normalize(); df["t"] = ts.dt.hour*60 + ts.dt.minute - 540
    df = df[(df.t >= 0) & (df.t <= 270)].sort_values(["stock_id","day","t"])
    groups = {key: g for key, g in df.groupby(["stock_id","day"], sort=False)}
    out = []
    for r in days.itertuples(index=False):
        g = groups.get((r.stock_id, r.date))
        if g is None or len(g) < 30: continue
        pc = r.prev_close; t = g.t.values; o = g.open.values; h = g.high.values; l = g.low.values; c = g.close.values
        auction = c[t == 270][-1] if (t == 270).any() else c[-1]
        k = bars5(t, o, h, l, c)
        res = {"stock_id": r.stock_id, "date": r.date, "gap": r.gap}
        for rule, ev in (("A", find_orb(k)), ("B", find_bigk(k, pc))):
            for key, side, lab in (("up", 1, "空"), ("dn", -1, "多")):
                if key not in ev: continue
                j = ev[key]; t_e = k.t.iloc[j] + 5
                stop = k.h.iloc[j] if side == 1 else k.l.iloc[j]
                for sv, sp in (("收盤", None), ("停損", stop)):
                    gr, why, te = trade(t, o, h, l, c, auction, pc, t_e, side, sp)
                    res[f"g|{rule}{lab}|{sv}"] = gr; res[f"why|{rule}{lab}|{sv}"] = why
                res[f"t|{rule}{lab}"] = te
                gm, _, _ = trade(t, o, h, l, c, auction, pc, t_e, -side, None); res[f"g|{rule}{lab}|順勢"] = gm
        # fade 現行: 跌破開盤區間低就空, 碰區間高停損
        orb = find_orb(k)
        if "dn" in orb:
            gr, why, te = trade(t, o, h, l, c, auction, pc, k.t.iloc[orb["dn"]] + 5, 1, k.h.iloc[0]); res["g|fade現行"] = gr; res["t|fade現行"] = te
        out.append(res)
    return pd.DataFrame(out)

def tstat(x):
    x = pd.Series(x).dropna(); return x.mean()/x.std()*np.sqrt(len(x)) if len(x) > 2 and x.std() > 0 else np.nan

if __name__ == "__main__":
    import features as F
    fs = sorted(glob.glob(str(ROOT/"data/raw/d1/*.parquet")))
    d1 = pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)
    d1["date"] = pd.to_datetime(d1["date"]); d1 = d1.drop_duplicates(["stock_id","date"]).sort_values(["stock_id","date"]).reset_index(drop=True)
    info = pd.read_parquet(ROOT/"data/raw/meta/info.parquet"); d1 = d1.merge(info[["stock_id","market_type"]], on="stock_id", how="left")
    fb = F.build_B(F.d1_base(d1))
    pool = fb[F.universe_mask(fb) & (fb.dv20_prev >= POOL_DV) & (fb.atr20 >= POOL_ATR) & fb.open.notna() & fb.prev_close.notna()][["stock_id","date","gap","prev_close"]].copy()
    pool["ym"] = pool.date.dt.strftime("%Y_%m")
    M1 = ROOT/"data/raw/m1"
    jobs = [(str(M1/f"{ym}.parquet"), g) for ym, g in pool.groupby("ym") if (M1/f"{ym}.parquet").exists()]
    print(f"pool {len(pool)} stock-days, {len(jobs)} months with M1", flush=True)
    with ProcessPoolExecutor(int(os.environ.get("QT_PROCS", 6))) as ex: parts = list(ex.map(simulate_month, jobs))
    S = pd.concat(parts, ignore_index=True); S["year"] = S.date.dt.year; S["oos"] = S.year >= OOS_YEAR
    S.to_parquet(ROOT/"data/processed/m5_reverse_breakout_trades.parquet", index=False); print("simulated", len(S), flush=True)
    all_days = pd.Series(sorted(S.date.unique()))
    GAPS = {"跳空≥+2%": S.gap >= 0.02, "跳空≤−2%": S.gap <= -0.02, "不限跳空": pd.Series(True, index=S.index)}

    def summarize(col, mask, label):
        E = S[mask & S[col].notna()]; net = E[col] - BASE
        if len(E) < 5: return None
        daily = net.groupby(E.date).mean().reindex(all_days).fillna(0)
        r = {"規則": label, "筆數": len(E), "日均筆數": len(E)/len(all_days), "每筆淨%": net.mean()*100, "勝率": (net > 0).mean(), "t": tstat(net)}
        for nm, mm in (("IS", ~E.oos), ("OOS", E.oos)): r[f"每筆淨% {nm}"] = net[mm].mean()*100; r[f"t {nm}"] = tstat(net[mm])
        r["日均淨% (當天平均分)"] = daily.mean()*100; r["Sharpe"] = daily.mean()/daily.std()*np.sqrt(245) if daily.std() > 0 else np.nan
        r["MDD%"] = ((1+daily).cumprod()/(1+daily).cumprod().cummax()-1).min()*100
        return r

    L = []; P = lambda s="": L.append(str(s))
    P("# M5 反向突破, 多空都做 (往上突破就空 / 往下跌破就多) — 當沖池 2020~2026\n")
    P(f"- 當沖池 {len(S)} 檔日 / {S.date.nunique()} 天 (上市櫃非 ETF, 股價≥5, 20 日均額≥5 億, ATR20≥3%)；樣本內 2020~{OOS_YEAR-1}、樣本外 {OOS_YEAR}~")
    P("- A = 開盤 5 分區間 (09:00~09:05 那根) 第一次收盤突破/跌破；B = 長紅/長黑 K (實體 ≥2.5% 昨收) 後 1 小時內收盤突破/跌破。訊號 K 在 12:00 前，下一根 5 分 K 開盤進場 (0.1% 滑價)。")
    P("- 收盤 = 抱到 13:30；停損 = 空單碰訊號 K 高、多單碰訊號 K 低就出。都含 ±8% 強制出場、成本 0.23%。順勢 = 同訊號跟著突破方向做、抱到收盤。")
    P("- 每筆淨的 t：|t|>2 才算不是運氣。日均淨/Sharpe = 每天把錢平均分給當天所有訊號 (沒訊號那天 0)。\n")
    for gl, gm in GAPS.items():
        P(f"## {gl}")
        rows = []
        for rule in ("A", "B"):
            for lab in ("空", "多"):
                for sv in ("收盤", "停損", "順勢"):
                    r = summarize(f"g|{rule}{lab}|{sv}", gm, f"{rule}{lab} {'反向' if sv != '順勢' else ''}{sv}")
                    if r: rows.append(r)
            # 多空合併 (反向, 收盤 / 停損)
            for sv in ("收盤", "停損"):
                tmp = pd.concat([S.loc[gm, ["date","oos", f"g|{rule}空|{sv}"]].rename(columns={f"g|{rule}空|{sv}": "g"}),
                                 S.loc[gm, ["date","oos", f"g|{rule}多|{sv}"]].rename(columns={f"g|{rule}多|{sv}": "g"})]).dropna()
                if len(tmp) < 5: continue
                net = tmp.g - BASE; daily = net.groupby(tmp.date).mean().reindex(all_days).fillna(0)
                r = {"規則": f"{rule}多空合併 反向{sv}", "筆數": len(tmp), "日均筆數": len(tmp)/len(all_days), "每筆淨%": net.mean()*100, "勝率": (net > 0).mean(), "t": tstat(net)}
                for nm, mm in (("IS", ~tmp.oos), ("OOS", tmp.oos)): r[f"每筆淨% {nm}"] = net[mm].mean()*100; r[f"t {nm}"] = tstat(net[mm])
                r["日均淨% (當天平均分)"] = daily.mean()*100; r["Sharpe"] = daily.mean()/daily.std()*np.sqrt(245) if daily.std() > 0 else np.nan
                r["MDD%"] = ((1+daily).cumprod()/(1+daily).cumprod().cummax()-1).min()*100; rows.append(r)
        if gl == "跳空≥+2%":
            r = summarize("g|fade現行", gm, "fade 現行 (跌破區間低空, 區間高停損)")
            if r: rows.append(r)
        P(pd.DataFrame(rows).round(3).to_markdown(index=False)); P("")
    P("## 各年每筆淨% (不限跳空, 反向收盤版)")
    yr = {}
    for rule in ("A", "B"):
        for lab in ("空", "多"):
            col = f"g|{rule}{lab}|收盤"; E = S[S[col].notna()]; yr[f"{rule}{lab}"] = (E[col] - BASE).groupby(E.year).mean()*100
            yr[f"{rule}{lab} 筆數"] = E.groupby(E.year).size()
    E = S[(S.gap >= 0.02) & S["g|fade現行"].notna()]; yr["fade現行 (跳空≥2%)"] = (E["g|fade現行"] - BASE).groupby(E.year).mean()*100
    P(pd.DataFrame(yr).round(2).to_markdown()); P("")
    P("## 進場時間分布 (不限跳空, 中位 / 25% / 75%, 09:00 起分鐘)")
    for rule in ("A", "B"):
        for lab in ("空", "多"):
            x = S[f"t|{rule}{lab}"].dropna() if f"t|{rule}{lab}" in S else pd.Series(dtype=float)
            if len(x): P(f"- {rule}{lab}: {x.median():.0f} / {x.quantile(.25):.0f} / {x.quantile(.75):.0f}")
    OUT.mkdir(exist_ok=True); (OUT/"m5_reverse_breakout.md").write_text("\n".join(L)); print("\n".join(L))
