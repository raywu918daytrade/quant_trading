"""上下半段: 10 點前有賺就先落袋, 賠的才用「10 點 ~ 收盤」(= 現行抱到收盤). 樣本/出場/成本同 46~49 (sr 每日前 10 名, M1 逐分鐘, 300 萬, 每支 60 萬).
個股規則 (沒觸發的照現行: +8% 強制回補 / 13:24 ≥ +6% / 收盤):
  S1 到 HH:MM 那支淨賺 (扣 0.23%) > 0 → 平; 賠的抱到收盤 (HH:MM = 09:30 / 10:00 / 10:30 / 11:00)
  S1b 10:00 淨賺 ≥ 1% 才平
  S2 10 點前碰停利 X% (預掛限價, 分 K 低點 ≤ 進場價 ×(1−X) 成交) → 平, 沒碰照現行
  S3 S2 + 10:00 還有賺的也平
  S4 對照 (反過來): 10:00 在賠的砍掉, 有賺的抱到收盤
整本規則:
  S5 10:00 整本淨 > 0 → 全平, 否則全部抱到收盤
  S6 10 點前整本淨 ≥ Y% (占 300 萬) → 全平
10 點後賠的單改成「賺回損失就平」(回本即平, 沒回本照現行到收盤):
  S7 10:00 有賺就平 + 賠的每支回到淨 ≥ 0 就平那支
  S7b 10:00 有賺就平 + 賠的那幾支合計回到淨 ≥ 0 就一起平
  S7c 10:00 有賺的照抱到收盤, 只有賠的回本就平 (單看 10 點後這一半)
  S7d 10 點前停利 4% + 10:00 賠的每支回本就平
市價平倉 = 判斷分鐘的下一分鐘開盤 ×(1+0.1%). X / Y / 時間只用 2020~23 挑, 2024~ 樣本外. 輸出 stats/split_10am.md"""
import importlib.util, numpy as np, pandas as pd
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
ROOT = Path(__file__).resolve().parent.parent; OUT = ROOT/"stats"
spec = importlib.util.spec_from_file_location("ff", ROOT/"scripts/46_first_loss_flatten.py"); ff = importlib.util.module_from_spec(spec); spec.loader.exec_module(ff)
BASE, SLIP, K, PEN, CAP, PER_STOCK, PRE_K, SPLIT = ff.BASE, ff.SLIP, ff.K, ff.PEN, ff.CAP, ff.PER_STOCK, ff.PRE_K, ff.SPLIT
T0, TPRE, TAUC, NT = ff.T0, ff.TPRE, ff.TAUC, ff.NT
T10 = 60
def rules():
    r = {"現行": None}
    for T, lab in ((30, "09:30"), (60, "10:00"), (90, "10:30"), (120, "11:00")): r[f"S1 {lab} 有賺就平, 賠的抱到收盤"] = ("s1", T, 0.0)
    r["S1b 10:00 賺 ≥1% 才平"] = ("s1", T10, 0.01)
    for x in (0.01, 0.02, 0.03, 0.04): r[f"S2 10 點前停利 {x*100:.0f}%"] = ("s2", x, False)
    for x in (0.02, 0.03): r[f"S3 10 點前停利 {x*100:.0f}% + 10:00 有賺就平"] = ("s2", x, True)
    r["S4 對照: 10:00 賠的砍掉, 賺的抱"] = ("s4", T10, None)
    r["S5 10:00 整本有賺就全平"] = ("s5", T10, 0.0)
    for y in (0.005, 0.01, 0.015, 0.02): r[f"S6 10 點前整本賺 ≥{y*100:.1f}% 就全平"] = ("s6", y, None)
    r["S7 10:00 有賺就平 + 賠的每支回本就平"] = ("s7", "one", "win")
    r["S7b 10:00 有賺就平 + 賠的合計回本就一起平"] = ("s7", "group", "win")
    r["S7c 10:00 賺的抱, 賠的每支回本就平"] = ("s7", "one", None)
    r["S7d 10 點前停利 4% + 賠的每支回本就平"] = ("s7", "one", "tp4")
    return r
RULES = rules()

def grid(g):
    o, h, c, has = ff.grid(g)
    l = np.full(NT, np.nan); l[g.t.values] = g.low.values
    return o, h, c, has, l

def sim_day(trs, bars):
    n = len(trs); E = trs.entry.values; PC = trs.prev_close.values; NO = trs.next_open.values
    O, H, C, HAS, Lo = (np.vstack([bars[s][k] for s in trs.stock_id]) for k in range(5))
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
    U = np.where(tt[None, :] >= tex[:, None], gx[:, None], E[:, None]/C - 1)
    book = (W[:, None]*(U - BASE)).sum(0)
    fx = lambda i, T: E[i]/(O[i, T+1]*(1+SLIP)) - 1                       # T 判斷, T+1 開盤回補
    live = lambda i, T: tex[i] > T
    res = {}; trig = {}
    for nm, rl in RULES.items():
        g = gx.copy(); tg = False
        if rl is None: res[nm] = g; trig[nm] = False; continue
        kind, p, q = rl
        if kind == "s1":
            for i in range(n):
                if live(i, p) and U[i, p] - BASE > q: g[i] = fx(i, p); tg = True
        elif kind == "s2":
            for i in range(n):
                ll = np.nan_to_num(Lo[i, T0:T10], nan=1e18); a = np.where(ll <= E[i]*(1-p))[0]
                if len(a) and T0 + a[0] < tex[i]: g[i] = 1/(1-p) - 1; tg = True
                elif q and live(i, T10) and U[i, T10] - BASE > 0: g[i] = fx(i, T10); tg = True
        elif kind == "s4":
            for i in range(n):
                if live(i, p) and U[i, p] - BASE < 0: g[i] = fx(i, p); tg = True
        elif kind in ("s5", "s6"):
            if kind == "s5": T = p if book[p] > q else None
            else:
                w = np.where(book[T0:T10+1] >= p)[0]; T = T0 + int(w[0]) if len(w) else None
            if T is not None:
                for i in range(n):
                    if live(i, T): g[i] = fx(i, T)
                tg = True
        elif kind == "s7":
            los = [i for i in range(n) if act[i] and live(i, T10) and U[i, T10] - BASE < 0]
            for i in range(n):
                if q == "win" and live(i, T10) and U[i, T10] - BASE > 0: g[i] = fx(i, T10); tg = True
                if q == "tp4":
                    ll = np.nan_to_num(Lo[i, T0:T10], nan=1e18); a = np.where(ll <= E[i]*0.96)[0]
                    if len(a) and T0 + a[0] < tex[i]: g[i] = 1/0.96 - 1; tg = True; los = [j for j in los if j != i]
            rng = np.arange(T10+1, TPRE-1)
            if p == "one":
                for i in los:
                    w = np.where((U[i, rng] - BASE >= 0) & (rng < tex[i]))[0]
                    if len(w): g[i] = fx(i, int(rng[w[0]])); tg = True
            elif los:
                grp = (W[los, None]*(U[los][:, rng] - BASE)).sum(0); w = np.where(grp >= 0)[0]
                if len(w):
                    T = int(rng[w[0]])
                    for i in los:
                        if live(i, T): g[i] = fx(i, T)
                    tg = True
        res[nm] = g; trig[nm] = tg
    diag = [{"u10": U[i, T10] - BASE, "after": gx[i] - fx(i, T10), "net": gx[i] - BASE, "w": W[i]} for i in range(n) if act[i] and live(i, T10)]
    return {nm: (W*(g - BASE)).sum() for nm, g in res.items()}, trig, diag, book[T10], (W*(gx - BASE)).sum()

def simulate_month(args):
    f, trades = args
    df = pd.read_parquet(f); ts = pd.to_datetime(df["date"]); df["day"] = ts.dt.normalize(); df["t"] = ts.dt.hour*60 + ts.dt.minute - 540
    df = df[(df.t >= 0) & (df.t <= 270)]; key = set(zip(trades.stock_id, trades.date)); df = df[[(s, d) in key for s, d in zip(df.stock_id, df.day)]]
    df = df.drop_duplicates(["stock_id","day","t"], keep="last")
    bars = {(sid, day): grid(g) for (sid, day), g in df.groupby(["stock_id","day"], sort=False)}
    rows, drows = [], []
    for day, trs in trades.sort_values(["date","rank"]).groupby("date", sort=False):
        trs = trs[[(s, day) in bars for s in trs.stock_id]].reset_index(drop=True)
        if trs.empty: continue
        o, tg, dg, b10, b0 = sim_day(trs, {s: bars[(s, day)] for s in trs.stock_id})
        rows.append({"date": day, "book10": b10, **{f"r|{k}": v for k, v in o.items()}, **{f"T|{k}": v for k, v in tg.items()}})
        for d in dg: drows.append({"date": day, **d})
    return pd.DataFrame(rows), pd.DataFrame(drows)

if __name__ == "__main__":
    R = pd.read_parquet(ROOT/"data/processed/trailing_stop_trades_sr.parquet")[["stock_id","date","entry","prev_close","next_open","rank"]]
    R["date"] = pd.to_datetime(R.date).astype("datetime64[ns]"); R["ym"] = R.date.dt.strftime("%Y_%m")
    jobs = [(str(ROOT/"data/raw/m1"/f"{ym}.parquet"), g) for ym, g in R.groupby("ym") if (ROOT/"data/raw/m1"/f"{ym}.parquet").exists()]
    with ProcessPoolExecutor(6) as ex: parts = list(ex.map(simulate_month, jobs))
    D = pd.concat([p[0] for p in parts], ignore_index=True).set_index("date").sort_index(); G = pd.concat([p[1] for p in parts], ignore_index=True)
    days = D.index; D_IS, D_OOS = days[days.year < SPLIT], days[days.year >= SPLIT]; names = list(RULES)
    DR = pd.DataFrame({nm: D[f"r|{nm}"] for nm in names})
    def stats(nm, idx):
        r = DR.loc[idx, nm]; eq = (1+r).cumprod(); mo = r.groupby(r.index.to_period("M")).apply(lambda x: (1+x).prod()-1); d0 = DR.loc[idx, "現行"]; dd = r - d0
        return {"觸發天數%": D.loc[idx, f"T|{nm}"].mean()*100, "日勝率%": (r > 0).mean()*100, "日淨%": r.mean()*100, "Sharpe": r.mean()/r.std()*np.sqrt(245),
                "MDD%": (eq/eq.cummax()-1).min()*100, "最差日%": r.min()*100, "最差月%": mo.min()*100, "年化(萬)": r.mean()*245*CAP/1e4,
                "日差 vs 現行%": dd.mean()*100, "t": dd.mean()/dd.std()*np.sqrt(len(dd)) if dd.std() > 0 else np.nan}
    L = []; P = lambda x="": L.append(str(x))
    P(f"# 上下半段: 10 點前有賺先平, 賠的抱到收盤? (sr 每日前 10 名, {days.min().date()}~{days.max().date()}, M1 逐分鐘, 300 萬, 每支 60 萬)\n")
    P("- 現行: 09:01 放空, +8% 強制回補 / 13:24 ≥ +6% 回補 / 其餘收盤回補, 每筆扣 0.23%")
    P("- 市價平倉 = 判斷分鐘的下一分鐘開盤 ×(1+0.1%)；停利限價 = 進場價 ×(1−X) 成交；t = 每日差 (規則 − 現行) 的 t 值, |t| > 2 才算不是運氣\n")
    P("## 1. 10:00 時賺 / 賠的單, 之後到收盤又怎麼走? (每支單, 已扣成本)")
    P("「10 點後再賺」= 抱到收盤 − 10:00 平掉, 正 = 抱著比較好\n")
    G["oos"] = G.date.dt.year >= SPLIT
    G["b"] = pd.cut(G.u10, [-1, -0.03, -0.01, 0, 0.01, 0.03, 1], labels=["賠 >3%", "賠 1~3%", "賠 0~1%", "賺 0~1%", "賺 1~3%", "賺 >3%"])
    T1 = G.groupby(["b", "oos"], observed=False).agg(筆數=("after", "size"), 十點淨=("u10", "mean"), 十點後再賺=("after", "mean"), 抱著較好比例=("after", lambda x: (x > 0).mean())).unstack()
    T1.columns = [f"{a} {'OOS' if b else 'IS'}" for a, b in T1.columns]
    for c in T1.columns:
        if not c.startswith("筆數"): T1[c] = T1[c]*100
    P(T1.round(2).to_markdown()); P()
    for lab, m in (("樣本內", ~G.oos), ("樣本外", G.oos)):
        x = G[m]; P(f"- {lab}: 10:00 還在場的單 {len(x)} 筆, 每筆全天淨 {x.net.mean()*100:.3f}%, 其中 10 點前 {x.u10.mean()*100:.3f}%、10 點後 {x.after.mean()*100:.3f}%")
    P()
    for lab, idx in ((f"樣本內 2020~{SPLIT-1}", D_IS), (f"樣本外 {SPLIT}~", D_OOS), ("全期", days)):
        P(f"## 2. 規則比較 — {lab} ({len(idx)} 天)"); P(pd.DataFrame({nm: stats(nm, idx) for nm in names}).T.round(3).to_markdown()); P()
    key = ["現行", "S1 10:00 有賺就平, 賠的抱到收盤", "S2 10 點前停利 2%", "S3 10 點前停利 2% + 10:00 有賺就平", "S4 對照: 10:00 賠的砍掉, 賺的抱", "S5 10:00 整本有賺就全平", "S6 10 點前整本賺 ≥1.0% 就全平", "S7 10:00 有賺就平 + 賠的每支回本就平", "S7c 10:00 賺的抱, 賠的每支回本就平"]
    G["be"] = G.u10 < 0
    for lab, m in (("樣本內", ~G.oos), ("樣本外", G.oos)):
        x = G[m & G.be]; P(f"- {lab}: 10:00 在賠的 {len(x)} 筆, 抱到收盤最後轉賺的 {(x.net > 0).mean()*100:.0f}%")
    P()
    P("## 3. 各年日淨%"); P((DR[key].groupby(DR.index.year).mean()*100).round(3).to_markdown())
    (OUT/"split_10am.md").write_text("\n".join(L)); print("\n".join(L))
