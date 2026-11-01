"""首支強制回補後, 把釋出的資金加到其他還在場的空單 (反向利用 46 的發現: 其他單之後平均再賺 0.6~0.8%).
樣本/出場/成本同 46_first_loss_flatten (sr 每日前 10 名, M1 逐分鐘, 300 萬).
加碼: 觸發分鐘 T 之後下一分鐘開盤 ×(1−0.1%) 再空, 跟原部位一起照現行規則出場; 加碼部分另付 0.23% 成本.
規則 (加碼金額 = 被回補那支原本的資金):
  F1 首支強制回補 → 平均加到其他還在場的
  F2 只加到當下在賺的 / F3 只加到當下在賠的
  F4 每一支強制回補都把資金再分給還在場的 (不只第一支)
  F5/F6 只在 10:00 / 11:00 前觸發才加
  F7 加到當下賺最多的 1 支 / F8 加到當下賠最多的 1 支
門檻沒有要挑的參數, 2020~23 / 2024~ 分開列. 輸出 stats/forced_recycle.md"""
import importlib.util, numpy as np, pandas as pd
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
ROOT = Path(__file__).resolve().parent.parent; OUT = ROOT/"stats"
spec = importlib.util.spec_from_file_location("ff", ROOT/"scripts/46_first_loss_flatten.py"); ff = importlib.util.module_from_spec(spec); spec.loader.exec_module(ff)
BASE, SLIP, K, PEN, CAP, PER_STOCK, PRE_K, SPLIT = ff.BASE, ff.SLIP, ff.K, ff.PEN, ff.CAP, ff.PER_STOCK, ff.PRE_K, ff.SPLIT
T0, TPRE, TAUC, NT = ff.T0, ff.TPRE, ff.TAUC, ff.NT
RULES = {"現行": None, "F1 首支回補→平均加到其他": ("first", "all", None), "F2 首支回補→只加在賺的": ("first", "win", None), "F3 首支回補→只加在賠的": ("first", "lose", None),
         "F4 每支回補都再分給在場的": ("every", "all", None), "F5 首支回補 (10:00 前)→加到其他": ("first", "all", 60), "F6 首支回補 (11:00 前)→加到其他": ("first", "all", 120),
         "F7 首支回補→加到賺最多的 1 支": ("first", "best", None), "F8 首支回補→加到賠最多的 1 支": ("first", "worst", None)}

def sim_day(trs, bars):
    n = len(trs); E = trs.entry.values; PC = trs.prev_close.values; NO = trs.next_open.values
    O = np.vstack([bars[s][0] for s in trs.stock_id]); H = np.vstack([bars[s][1] for s in trs.stock_id]); C = np.vstack([bars[s][2] for s in trs.stock_id]); HAS = np.vstack([bars[s][3] for s in trs.stock_id])
    auc = np.array([C[i, TAUC] if HAS[i, TAUC] else C[i, :TAUC+1][HAS[i, :TAUC+1]][-1] if HAS[i].any() else E[i] for i in range(n)])
    tex = np.full(n, 999); gx = np.zeros(n); why = np.empty(n, object); xpx = np.zeros(n)   # xpx = 有效回補價 (lock 含罰)
    for i in range(n):
        hit = np.where(np.nan_to_num(H[i, T0:TPRE], nan=-1) >= PC[i]*(1+K))[0]
        if len(hit): tex[i] = T0 + hit[0]; xpx[i] = PC[i]*(1+K)*(1+SLIP*3); why[i] = "forced"
        elif HAS[i, TPRE] and C[i, TPRE] >= PC[i]*(1+PRE_K): tex[i] = TPRE; xpx[i] = C[i, TPRE]*(1+SLIP); why[i] = "pre"
        else:
            tex[i] = TAUC
            if auc[i] >= PC[i]*1.0975 and np.isfinite(NO[i]): xpx[i] = E[i]/(E[i]/NO[i] - PEN); why[i] = "lock"
            else: xpx[i] = auc[i]; why[i] = "close"
        gx[i] = E[i]/xpx[i] - 1
    lots = []; cash = CAP
    for e in E:
        lc = e*1000; k = int(min(PER_STOCK, cash)//lc) if cash >= lc else 0; a = k*lc; cash -= a; lots.append(a)
    W = np.array(lots)/CAP; base_ret = (W*(gx - BASE)).sum()
    out = {}; trig = {}
    order = np.argsort(np.where(why == "forced", tex, 9999))
    for nm, rl in RULES.items():
        if rl is None: out[nm] = base_ret; trig[nm] = -1; continue
        scope, tgt, cut = rl; extra = 0.0; first_T = -1
        events = [(int(tex[j]), j) for j in order if why[j] == "forced"]
        if scope == "first": events = events[:1]
        for T, j in events:
            if T+1 >= TPRE or (cut is not None and T >= cut): continue
            live = [i for i in range(n) if tex[i] > T and W[i] > 0]
            u = {i: E[i]/C[i, T] - 1 for i in live}
            if tgt == "win": live = [i for i in live if u[i] > 0]
            elif tgt == "lose": live = [i for i in live if u[i] < 0]
            elif tgt == "best" and live: live = [max(live, key=u.get)]
            elif tgt == "worst" and live: live = [min(live, key=u.get)]
            if not live: continue
            if first_T < 0: first_T = T
            amt = W[j]/len(live)
            for i in live: extra += amt*((O[i, T+1]*(1-SLIP))/xpx[i] - 1 - BASE)
        out[nm] = base_ret + extra; trig[nm] = first_T
    return out, trig

def simulate_month(args):
    f, trades = args
    df = pd.read_parquet(f); ts = pd.to_datetime(df["date"]); df["day"] = ts.dt.normalize(); df["t"] = ts.dt.hour*60 + ts.dt.minute - 540
    df = df[(df.t >= 0) & (df.t <= 270)]; key = set(zip(trades.stock_id, trades.date)); df = df[[(s, d) in key for s, d in zip(df.stock_id, df.day)]]
    df = df.drop_duplicates(["stock_id","day","t"], keep="last")
    bars = {(sid, day): ff.grid(g) for (sid, day), g in df.groupby(["stock_id","day"], sort=False)}
    rows = []
    for day, trs in trades.sort_values(["date","rank"]).groupby("date", sort=False):
        trs = trs[[(s, day) in bars for s in trs.stock_id]].reset_index(drop=True)
        if trs.empty: continue
        o, t = sim_day(trs, {s: bars[(s, day)] for s in trs.stock_id})
        rows.append({"date": day, **{f"r|{k}": v for k, v in o.items()}, **{f"T|{k}": v for k, v in t.items()}})
    return pd.DataFrame(rows)

if __name__ == "__main__":
    R = pd.read_parquet(ROOT/"data/processed/trailing_stop_trades_sr.parquet")[["stock_id","date","entry","prev_close","next_open","rank"]]
    R["date"] = pd.to_datetime(R.date).astype("datetime64[ns]"); R["ym"] = R.date.dt.strftime("%Y_%m")
    jobs = [(str(ROOT/"data/raw/m1"/f"{ym}.parquet"), g) for ym, g in R.groupby("ym") if (ROOT/"data/raw/m1"/f"{ym}.parquet").exists()]
    with ProcessPoolExecutor(6) as ex: parts = list(ex.map(simulate_month, jobs))
    D = pd.concat(parts, ignore_index=True).set_index("date").sort_index(); days = D.index
    D_IS, D_OOS = days[days.year < SPLIT], days[days.year >= SPLIT]; names = list(RULES)
    DR = pd.DataFrame({nm: D[f"r|{nm}"] for nm in names})
    def stats(nm, idx):
        r = DR.loc[idx, nm]; eq = (1+r).cumprod(); mo = r.groupby(r.index.to_period("M")).apply(lambda x: (1+x).prod()-1); trg = D.loc[idx, f"T|{nm}"] >= 0; d0 = DR.loc[idx, "現行"]
        dd = r - d0
        return {"觸發天數%": trg.mean()*100, "日勝率%": (r > 0).mean()*100, "日淨%": r.mean()*100, "Sharpe": r.mean()/r.std()*np.sqrt(245), "MDD%": (eq/eq.cummax()-1).min()*100,
                "最差日%": r.min()*100, "最差 5 日平均%": r.nsmallest(5).mean()*100, "最差月%": mo.min()*100, "年化(萬)": r.mean()*245*CAP/1e4,
                "日差 vs 現行%": dd.mean()*100, "t": dd.mean()/dd.std()*np.sqrt(len(dd)) if dd.std() > 0 else np.nan,
                "觸發日 現行%": d0[trg].mean()*100 if trg.any() else np.nan, "觸發日 規則%": r[trg].mean()*100 if trg.any() else np.nan}
    L = []; P = lambda x="": L.append(str(x))
    P(f"# 首支強制回補後把資金加到其他空單 (sr 每日前 10 名, {days.min().date()}~{days.max().date()}, M1 逐分鐘, 300 萬)\n")
    P("- 加碼價 = 觸發後下一分鐘開盤 ×(1−0.1%)；加碼部分出場同原部位, 另付 0.23%；加碼金額 = 被回補那支原本的資金")
    P("- t = 每日差 (規則 − 現行) 的 t 值, |t| > 2 才算不是運氣\n")
    for lab, idx in ((f"樣本內 2020~{SPLIT-1}", D_IS), (f"樣本外 {SPLIT}~", D_OOS)):
        P(f"## {lab} ({len(idx)} 天)"); P(pd.DataFrame({nm: stats(nm, idx) for nm in names}).T.round(3).to_markdown()); P()
    P("## 各年日淨%"); P((DR.groupby(DR.index.year).mean()*100).round(3).to_markdown()); P()
    w = DR["現行"].nsmallest(10).index; X = DR.loc[w]*100; X.index = X.index.date
    P("## 現行最差 10 天, 各規則那天的日淨% (加碼會不會讓最差日更差)"); P(X.round(2).to_markdown())
    (OUT/"forced_recycle.md").write_text("\n".join(L)); print("\n".join(L))
