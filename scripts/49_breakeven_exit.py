"""賠了一筆之後, 其他部位改成「把累計虧損賺回來就出場」(回本即平), 會不會比抱到收盤好? 樣本/出場/成本同 46~48 (sr 每日前 10 名, M1 逐分鐘, 300 萬, 每支 60 萬).
現行所有部位 09:01 同時進場, 所以「下一筆」有兩種讀法:
  盤中 (A): 當天第一支 +8% 強制回補 (實現虧損) 之後
    A1 整本 (已實現 + 浮動, 已扣成本) 回到 ≥ 0 → 其餘全平
    A2 每支還在場的, 自己的浮盈金額 ≥ 已實現虧損 → 平那一支 (其他照現行)
    A3 只看 rank 順序的下一支還在場的: 它的浮盈 ≥ 已實現虧損 → 平它
  跨日 (B): 之前賠的錢, 今天整本 (已扣成本) 賺到目標 → 全平, 沒到就照現行抱到收盤
    B1 昨天賠 → 目標 = 昨天虧損
    B2 帳上在回撤 (累計損益 < 前高) → 目標 = 回撤金額 (回到前高就平)
    B3 連續虧損日 → 目標 = 這段連虧的總額
  B 的目標不是挑出來的參數, 但仍分 2020~23 / 2024~ 列.
平倉價 = 觸發分鐘 T 的下一分鐘開盤 ×(1+0.1%), T 用分 K 收盤判斷; 13:24 之後不動作. 輸出 stats/breakeven_exit.md"""
import importlib.util, numpy as np, pandas as pd
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
ROOT = Path(__file__).resolve().parent.parent; OUT = ROOT/"stats"
spec = importlib.util.spec_from_file_location("ff", ROOT/"scripts/46_first_loss_flatten.py"); ff = importlib.util.module_from_spec(spec); spec.loader.exec_module(ff)
BASE, SLIP, K, PEN, CAP, PER_STOCK, PRE_K, SPLIT = ff.BASE, ff.SLIP, ff.K, ff.PEN, ff.CAP, ff.PER_STOCK, ff.PRE_K, ff.SPLIT
T0, TPRE, TAUC, NT = ff.T0, ff.TPRE, ff.TAUC, ff.NT
A_RULES = ("A1 首支回補後整本回本→全平", "A2 首支回補後, 每支浮盈≥已實現虧損→平那支", "A3 首支回補後, 下一支浮盈≥已實現虧損→平它")
B_RULES = ("B1 昨天賠→今天賺回昨天虧損就全平", "B2 在回撤→今天賺回回撤就全平", "B3 連虧→今天賺回連虧總額就全平")

def sim_day(trs, bars):
    n = len(trs); E = trs.entry.values; PC = trs.prev_close.values; NO = trs.next_open.values
    O, H, C, HAS = (np.vstack([bars[s][k] for s in trs.stock_id]) for k in range(4))
    auc = np.array([C[i, TAUC] if HAS[i, TAUC] else C[i, :TAUC+1][HAS[i, :TAUC+1]][-1] if HAS[i].any() else E[i] for i in range(n)])
    tex = np.full(n, 999); gx = np.zeros(n); why = np.empty(n, object)
    for i in range(n):
        hit = np.where(np.nan_to_num(H[i, T0:TPRE], nan=-1) >= PC[i]*(1+K))[0]
        if len(hit): tex[i] = T0 + hit[0]; gx[i] = E[i]/(PC[i]*(1+K)*(1+SLIP*3)) - 1; why[i] = "forced"; continue
        if HAS[i, TPRE] and C[i, TPRE] >= PC[i]*(1+PRE_K): tex[i] = TPRE; gx[i] = E[i]/(C[i, TPRE]*(1+SLIP)) - 1; why[i] = "pre"; continue
        if auc[i] >= PC[i]*1.0975 and np.isfinite(NO[i]): gx[i] = E[i]/NO[i] - 1 - PEN; why[i] = "lock"
        else: gx[i] = E[i]/auc[i] - 1; why[i] = "close"
        tex[i] = TAUC
    lots = []; cash = CAP
    for e in E:
        lc = e*1000; k = int(min(PER_STOCK, cash)//lc) if cash >= lc else 0; a = k*lc; cash -= a; lots.append(a)
    W = np.array(lots)/CAP; act = W > 0
    tt = np.arange(NT)
    U = np.where(tt[None, :] >= tex[:, None], gx[:, None], E[:, None]/C - 1)              # 每分鐘浮動 (已出場 = 實現) 毛報酬
    UN = W[:, None]*(U - BASE)                                                              # 每支占 300 萬的淨損益
    book = UN.sum(0)
    FX = E[:, None]/(np.r_[O[:, 1:].T, O[:, -1:].T].T*(1+SLIP)) - 1                        # FX[i, T] = 在 T+1 開盤回補的毛報酬
    live = tt[None, :] < tex[:, None]
    flat = (W[:, None]*(np.where(live, FX, gx[:, None]) - BASE)).sum(0)                    # flat[T] = T 觸發全平的日淨
    flat[TPRE-1:] = np.nan
    base_ret = (W*(gx - BASE)).sum()
    out = {"現行": base_ret}; f = np.where((why == "forced") & act)[0]
    if not len(f) or tex[f].min()+1 >= TPRE:
        for nm in A_RULES: out[nm] = base_ret
        return out, -1, book[T0:TPRE], flat[T0:TPRE]
    Tf = int(tex[f].min())
    # A1
    w = np.where(book[Tf:TPRE-1] >= 0)[0]
    out[A_RULES[0]] = flat[Tf + w[0]] if len(w) else base_ret
    # A2 / A3: 已實現虧損 = 到 T 為止已出場部位的淨損益 (取負)
    real = lambda T: (W*(gx - BASE))[(tex <= T) & act].sum()
    REAL = np.array([real(T) for T in range(NT)])
    g2 = gx.copy(); g3 = gx.copy()
    nxt = [i for i in range(n) if act[i] and tex[i] > Tf][:1]
    for i in range(n):
        if not act[i] or tex[i] <= Tf: continue
        ok = np.where((UN[i, Tf:TPRE-1] >= -REAL[Tf:TPRE-1]) & (REAL[Tf:TPRE-1] < 0) & (tt[Tf:TPRE-1] < tex[i]))[0]
        if len(ok):
            T = Tf + ok[0]; g2[i] = FX[i, T]
            if i in nxt: g3[i] = FX[i, T]
    out[A_RULES[1]] = (W*(g2 - BASE)).sum(); out[A_RULES[2]] = (W*(g3 - BASE)).sum()
    return out, Tf, book[T0:TPRE], flat[T0:TPRE]

def simulate_month(args):
    f, trades = args
    df = pd.read_parquet(f); ts = pd.to_datetime(df["date"]); df["day"] = ts.dt.normalize(); df["t"] = ts.dt.hour*60 + ts.dt.minute - 540
    df = df[(df.t >= 0) & (df.t <= 270)]; key = set(zip(trades.stock_id, trades.date)); df = df[[(s, d) in key for s, d in zip(df.stock_id, df.day)]]
    df = df.drop_duplicates(["stock_id","day","t"], keep="last")
    bars = {(sid, day): ff.grid(g) for (sid, day), g in df.groupby(["stock_id","day"], sort=False)}
    rows, paths = [], {}
    for day, trs in trades.sort_values(["date","rank"]).groupby("date", sort=False):
        trs = trs[[(s, day) in bars for s in trs.stock_id]].reset_index(drop=True)
        if trs.empty: continue
        o, Tf, bk, fl = sim_day(trs, {s: bars[(s, day)] for s in trs.stock_id})
        rows.append({"date": day, "Tf": Tf, **o}); paths[day] = (bk, fl)
    return pd.DataFrame(rows), paths

def tm(x): return f"{9 + x//60:02d}:{x % 60:02d}"

if __name__ == "__main__":
    R = pd.read_parquet(ROOT/"data/processed/trailing_stop_trades_sr.parquet")[["stock_id","date","entry","prev_close","next_open","rank"]]
    R["date"] = pd.to_datetime(R.date).astype("datetime64[ns]"); R["ym"] = R.date.dt.strftime("%Y_%m")
    jobs = [(str(ROOT/"data/raw/m1"/f"{ym}.parquet"), g) for ym, g in R.groupby("ym") if (ROOT/"data/raw/m1"/f"{ym}.parquet").exists()]
    with ProcessPoolExecutor(6) as ex: parts = list(ex.map(simulate_month, jobs))
    D = pd.concat([p[0] for p in parts], ignore_index=True).set_index("date").sort_index(); PATHS = {k: v for p in parts for k, v in p[1].items()}
    days = D.index; D_IS, D_OOS = days[days.year < SPLIT], days[days.year >= SPLIT]
    # 跨日規則: 依序走, 目標取決於「規則自己」過去的損益
    TRIG = {nm: pd.Series(-1, index=days) for nm in B_RULES}
    for nm in B_RULES:
        r = []; cum = 0.0; peak = 0.0; streak = 0.0; prev = 0.0
        for d in days:
            tgt = -prev if nm.startswith("B1") and prev < 0 else (peak - cum) if nm.startswith("B2") and cum < peak else -streak if nm.startswith("B3") and streak < 0 else None
            x = D.at[d, "現行"]
            if tgt is not None:
                bk, fl = PATHS[d]; w = np.where(bk[:-1] >= tgt)[0]
                if len(w): x = fl[w[0]]; TRIG[nm][d] = T0 + int(w[0])
            r.append(x); prev = x; cum += x; peak = max(peak, cum); streak = streak + x if x < 0 else 0.0
        D[nm] = r
    names = ["現行", *A_RULES, *B_RULES]
    trig = {"現行": pd.Series(-1, index=days), **{nm: D.Tf.where(D[nm] != D["現行"], -1) for nm in A_RULES}, **TRIG}
    eligible = {**{nm: D.Tf >= 0 for nm in A_RULES}}
    def stats(nm, idx):
        r = D.loc[idx, nm]; eq = (1+r).cumprod(); mo = r.groupby(r.index.to_period("M")).apply(lambda x: (1+x).prod()-1); d0 = D.loc[idx, "現行"]; dd = r - d0
        tg = trig[nm].loc[idx] >= 0
        return {"觸發(平倉)天數%": tg.mean()*100, "日勝率%": (r > 0).mean()*100, "日淨%": r.mean()*100, "Sharpe": r.mean()/r.std()*np.sqrt(245), "MDD%": (eq/eq.cummax()-1).min()*100,
                "最差日%": r.min()*100, "最差月%": mo.min()*100, "年化(萬)": r.mean()*245*CAP/1e4, "日差 vs 現行%": dd.mean()*100,
                "t": dd.mean()/dd.std()*np.sqrt(len(dd)) if dd.std() > 0 else np.nan,
                "觸發日 現行%": d0[tg].mean()*100 if tg.any() else np.nan, "觸發日 規則%": r[tg].mean()*100 if tg.any() else np.nan}
    L = []; P = lambda x="": L.append(str(x))
    P(f"# 賠了之後「回本即平」? (sr 每日前 10 名, {days.min().date()}~{days.max().date()}, M1 逐分鐘, 300 萬, 每支 60 萬)\n")
    P("- 現行: 09:01 同時進場做空, +8% 強制回補 / 13:24 ≥ +6% 回補 / 收盤 (漲停鎖死隔日開盤 +7% 罰), 每筆扣 0.23%")
    P("- 盤中 A: 當天第一支 +8% 強制回補 (= 第一筆實現虧損) 之後才啟動；跨日 B: 之前賠的錢, 今天整本淨損益 (分 K 收盤算) 達到目標就全平")
    P("- 平倉價 = 觸發後下一分鐘開盤 ×(1+0.1%)；t = 每日差 (規則 − 現行) 的 t 值, |t| > 2 才算不是運氣")
    tf = D.Tf[D.Tf >= 0]
    P(f"- 有強制回補 (A 能啟動) 的天數 {len(tf)} / {len(D)} ({len(tf)/len(D)*100:.1f}%), 首支回補時間中位 {tm(int(tf.median()))}\n")
    for lab, idx in ((f"樣本內 2020~{SPLIT-1}", D_IS), (f"樣本外 {SPLIT}~", D_OOS), ("全期", days)):
        P(f"## {lab} ({len(idx)} 天)"); P(pd.DataFrame({nm: stats(nm, idx) for nm in names}).T.round(3).to_markdown()); P()
    P("## 各年日淨%"); P((D[names].groupby(D.index.year).mean()*100).round(3).to_markdown()); P()
    # 診斷: B2 觸發那天, 平倉後到收盤現行多賺多少 (回本即平砍掉了多少尾巴)
    for nm in B_RULES:
        tg = TRIG[nm] >= 0
        P(f"- {nm}: 平倉 {int(tg.sum())} 天, 平倉時間中位 {tm(int(TRIG[nm][tg].median())) if tg.any() else '—'}；這些天如果抱到收盤 (現行) 平均 {D['現行'][tg].mean()*100:+.3f}%, 回本即平 {D[nm][tg].mean()*100:+.3f}%")
    for nm in A_RULES:
        tg = trig[nm] >= 0
        P(f"- {nm}: 有動作 {int(tg.sum())} 天；這些天現行平均 {D['現行'][tg].mean()*100:+.3f}%, 規則 {D[nm][tg].mean()*100:+.3f}%")
    (OUT/"breakeven_exit.md").write_text("\n".join(L)); print("\n".join(L))
