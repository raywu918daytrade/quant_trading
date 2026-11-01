"""空單先賺、之後又漲回進場價 → 在進場價加碼, 會不會更好? 樣本/出場/成本同 46/47 (sr 每日前 10 名, M1 逐分鐘, 300 萬).
做空, 進場價 E. 兩種「先 X 後回到進場價」:
  fav (先往有利走): 分 K 低點先碰 E×(1−X) (空單浮盈 X%), 之後高點又碰到 E → 加碼空
      fav 市價: 回到 E 那分鐘的下一分鐘開盤 ×(1−0.1%) 空
      fav 限價: 預掛 E 的限價空單, 之後高點 > E 才算成交 (成交價 = E, 不加滑價)
  adv (字面: 先漲再跌回): 分 K 高點先碰 E×(1+X) (空單浮虧 X%), 之後低點又碰到 E → 下一分鐘開盤 ×(1−0.1%) 市價加碼空
每支每天最多加一次, 加碼部位跟原部位一起照現行規則出場 (強制回補 / 13:24 / 收盤 / 鎖漲停罰), 另付 0.23% 成本.
加碼金額 = 原部位金額 × 倍數 (1× / 0.5×), 需要額外的當沖額度 (另列平均/最大額外資金). 日淨仍除以 300 萬.
X 與加碼時間限制只用 2020~23 挑, 2024~ 樣本外. 輸出 stats/breakeven_add.md"""
import importlib.util, numpy as np, pandas as pd
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
ROOT = Path(__file__).resolve().parent.parent; OUT = ROOT/"stats"
spec = importlib.util.spec_from_file_location("ff", ROOT/"scripts/46_first_loss_flatten.py"); ff = importlib.util.module_from_spec(spec); spec.loader.exec_module(ff)
BASE, SLIP, K, PEN, CAP, PER_STOCK, PRE_K, SPLIT = ff.BASE, ff.SLIP, ff.K, ff.PEN, ff.CAP, ff.PER_STOCK, ff.PRE_K, ff.SPLIT
T0, TPRE, TAUC, NT = ff.T0, ff.TPRE, ff.TAUC, ff.NT
XS = (0.005, 0.01, 0.02, 0.03, 0.04, 0.05)
MODES = ("fav 市價", "fav 限價", "adv 市價")
CUTS = {"全天": TPRE, "10:00 前": 60, "11:00 前": 120}
MULTS = (1.0, 0.5)

def grid(g):
    o, h, c, has = ff.grid(g)
    l = np.full(NT, np.nan); l[g.t.values] = g.low.values
    return o, h, c, has, l

def sim_day(trs, bars):
    n = len(trs); E = trs.entry.values; PC = trs.prev_close.values; NO = trs.next_open.values
    O, H, C, HAS, Lo = (np.vstack([bars[s][k] for s in trs.stock_id]) for k in range(5))
    auc = np.array([C[i, TAUC] if HAS[i, TAUC] else C[i, :TAUC+1][HAS[i, :TAUC+1]][-1] if HAS[i].any() else E[i] for i in range(n)])
    tex = np.full(n, 999); xpx = np.zeros(n); why = np.empty(n, object)
    for i in range(n):
        hit = np.where(np.nan_to_num(H[i, T0:TPRE], nan=-1) >= PC[i]*(1+K))[0]
        if len(hit): tex[i] = T0 + hit[0]; xpx[i] = PC[i]*(1+K)*(1+SLIP*3); why[i] = "forced"
        elif HAS[i, TPRE] and C[i, TPRE] >= PC[i]*(1+PRE_K): tex[i] = TPRE; xpx[i] = C[i, TPRE]*(1+SLIP); why[i] = "pre"
        else:
            tex[i] = TAUC
            if auc[i] >= PC[i]*1.0975 and np.isfinite(NO[i]): xpx[i] = E[i]/(E[i]/NO[i] - PEN); why[i] = "lock"
            else: xpx[i] = auc[i]; why[i] = "close"
    gx = E/xpx - 1
    lots = []; cash = CAP
    for e in E:
        lc = e*1000; k = int(min(PER_STOCK, cash)//lc) if cash >= lc else 0; a = k*lc; cash -= a; lots.append(a)
    W = np.array(lots)/CAP; base_ret = (W*(gx - BASE)).sum()
    ev = []
    for i in range(n):
        if W[i] <= 0: continue
        end = min(int(tex[i]), TPRE)                       # 只看出場前、13:24 前
        hh = np.nan_to_num(H[i, T0:end], nan=-1); ll = np.nan_to_num(Lo[i, T0:end], nan=1e18)
        for X in XS:
            for mode in MODES:
                if mode.startswith("fav"):
                    a = np.where(ll <= E[i]*(1-X))[0]
                    if not len(a): continue
                    b = np.where(hh[a[0]+1:] > E[i] if mode == "fav 限價" else hh[a[0]+1:] >= E[i])[0]
                else:
                    a = np.where(hh >= E[i]*(1+X))[0]
                    if not len(a): continue
                    b = np.where(ll[a[0]+1:] <= E[i])[0]
                if not len(b): continue
                t2 = T0 + a[0] + 1 + b[0]
                if mode == "fav 限價": fill = E[i]
                else:
                    if t2+1 >= TPRE or t2+1 > tex[i]: continue
                    fill = O[i, t2+1]*(1-SLIP)
                ev.append({"stock_id": trs.stock_id[i], "rank": trs["rank"][i], "X": X, "mode": mode, "t1": T0 + int(a[0]), "t2": int(t2), "w": W[i],
                           "add": fill/xpx[i] - 1 - BASE, "orig": gx[i] - BASE, "why": why[i]})
    return base_ret, ev, int((W > 0).sum())

def simulate_month(args):
    f, trades = args
    df = pd.read_parquet(f); ts = pd.to_datetime(df["date"]); df["day"] = ts.dt.normalize(); df["t"] = ts.dt.hour*60 + ts.dt.minute - 540
    df = df[(df.t >= 0) & (df.t <= 270)]; key = set(zip(trades.stock_id, trades.date)); df = df[[(s, d) in key for s, d in zip(df.stock_id, df.day)]]
    df = df.drop_duplicates(["stock_id","day","t"], keep="last")
    bars = {(sid, day): grid(g) for (sid, day), g in df.groupby(["stock_id","day"], sort=False)}
    drows, erows = [], []
    for day, trs in trades.sort_values(["date","rank"]).groupby("date", sort=False):
        trs = trs[[(s, day) in bars for s in trs.stock_id]].reset_index(drop=True)
        if trs.empty: continue
        b, ev, nn = sim_day(trs, {s: bars[(s, day)] for s in trs.stock_id})
        drows.append({"date": day, "base": b, "n": nn})
        for e in ev: e["date"] = day; erows.append(e)
    return pd.DataFrame(drows), pd.DataFrame(erows)

def tm(x): return f"{9 + x//60:02d}:{x % 60:02d}"

if __name__ == "__main__":
    R = pd.read_parquet(ROOT/"data/processed/trailing_stop_trades_sr.parquet")[["stock_id","date","entry","prev_close","next_open","rank"]]
    R["date"] = pd.to_datetime(R.date).astype("datetime64[ns]"); R["ym"] = R.date.dt.strftime("%Y_%m")
    jobs = [(str(ROOT/"data/raw/m1"/f"{ym}.parquet"), g) for ym, g in R.groupby("ym") if (ROOT/"data/raw/m1"/f"{ym}.parquet").exists()]
    with ProcessPoolExecutor(6) as ex: parts = list(ex.map(simulate_month, jobs))
    D = pd.concat([p[0] for p in parts], ignore_index=True).set_index("date").sort_index(); EV = pd.concat([p[1] for p in parts], ignore_index=True)
    EV.to_parquet(ROOT/"data/processed/breakeven_add_events.parquet", index=False)
    days = D.index; D_IS, D_OOS = days[days.year < SPLIT], days[days.year >= SPLIT]; EV["oos"] = EV.date.dt.year >= SPLIT
    ntr = {"IS": D.loc[D_IS, "n"].sum(), "OOS": D.loc[D_OOS, "n"].sum()}

    rules = {"現行": None}
    for mode in MODES:
        for X in XS:
            for cl, cut in CUTS.items():
                for m in MULTS: rules[f"{mode} {X*100:g}% {cl} {m:g}×"] = (mode, X, cut, m)
    DR = {"現行": D.base}; XC = {}
    for nm, rl in rules.items():
        if rl is None: continue
        mode, X, cut, m = rl; e = EV[(EV["mode"] == mode) & (EV.X == X) & (EV.t2 < cut)]
        ex = (m*e.w*e.add).groupby(e.date).sum().reindex(days, fill_value=0.0)
        DR[nm] = D.base + ex; XC[nm] = (m*e.w).groupby(e.date).sum().reindex(days, fill_value=0.0)*CAP/1e4
    DR = pd.DataFrame(DR)
    def stats(nm, idx, lab):
        r = DR.loc[idx, nm]; eq = (1+r).cumprod(); d0 = DR.loc[idx, "現行"]; dd = r - d0
        o = {"日淨%": r.mean()*100, "Sharpe": r.mean()/r.std()*np.sqrt(245), "MDD%": (eq/eq.cummax()-1).min()*100, "最差日%": r.min()*100,
             "最差 5 日平均%": r.nsmallest(5).mean()*100, "年化(萬)": r.mean()*245*CAP/1e4, "日差%": dd.mean()*100,
             "t": dd.mean()/dd.std()*np.sqrt(len(dd)) if dd.std() > 0 else np.nan}
        if nm != "現行":
            mode, X, cut, m = rules[nm]; e = EV[(EV["mode"] == mode) & (EV.X == X) & (EV.t2 < cut) & EV.date.isin(idx)]
            o.update({"觸發占筆數%": len(e)/ntr[lab]*100, "加碼段每筆淨%": e.add.mean()*100, "加碼段勝率%": (e.add > 0).mean()*100,
                      "額外資金 平均(萬)": XC[nm].loc[idx].mean(), "額外資金 最大(萬)": XC[nm].loc[idx].max()})
        return o
    L = []; P = lambda x="": L.append(str(x))
    P(f"# 空單先賺、回到進場價時加碼 (sr 每日前 10 名, {days.min().date()}~{days.max().date()}, M1 逐分鐘, 300 萬)\n")
    P(f"- {int(D.n.sum())} 筆 / {len(days)} 天 (樣本內 2020~{SPLIT-1} {len(D_IS)} 天, 樣本外 {SPLIT}~ {len(D_OOS)} 天)")
    P("- fav = 先往有利方向 (下跌) 走 X% 再漲回進場價；adv = 字面「先漲 X% 再跌回進場價」")
    P("- 市價 = 回到進場價那分鐘的下一分鐘開盤 ×(1−0.1%)；限價 = 預掛進場價, 高點 > 進場價才算成交；加碼段另付 0.23%, 跟原部位一起出場")
    P("- 加碼要用額外當沖額度, 日淨仍除以 300 萬；t = 每日差 (規則 − 現行) 的 t 值, |t| > 2 才算不是運氣\n")

    # 1. 加碼段本身: 全天, 1×
    P("## 1. 加碼那一段本身賺不賺 (全天都可加, 每筆淨% = 加碼價到出場價, 已扣 0.23%)")
    rows = []
    for mode in MODES:
        for X in XS:
            e = EV[(EV["mode"] == mode) & (EV.X == X)]
            row = {"規則": f"{mode} {X*100:g}%"}
            for lab, m in (("IS", ~e.oos), ("OOS", e.oos)):
                x = e[m]; row.update({f"{lab} 觸發占筆數%": len(x)/ntr[lab]*100, f"{lab} 加碼段淨%": x.add.mean()*100, f"{lab} 勝率%": (x.add > 0).mean()*100,
                                      f"{lab} 這些單原部位淨%": x.orig.mean()*100})
            row["加碼時間中位"] = tm(int(e.t2.median())); rows.append(row)
    P(pd.DataFrame(rows).set_index("規則").round(3).to_markdown())
    P(f"\n對照: 現行每筆淨 (全部單) 樣本內約 0.61%, 樣本外約 0.52% (見 first_loss_flatten.md)\n")

    # 2. 規則比較
    names = list(rules)
    TI = pd.DataFrame({nm: stats(nm, D_IS, "IS") for nm in names}).T; TO = pd.DataFrame({nm: stats(nm, D_OOS, "OOS") for nm in names}).T
    for lab, T in ((f"樣本內 2020~{SPLIT-1}", TI), (f"樣本外 {SPLIT}~", TO)):
        P(f"## 2. {lab}: 1× 全天 (每種 X)"); keep = ["現行"] + [k for k in names[1:] if " 全天 1×" in k]; P(T.loc[keep].round(3).to_markdown()); P()
    cand = TI.drop(index="現行"); picks = {"樣本內日淨最好": cand["日淨%"].idxmax(), "樣本內 Sharpe 最好": cand.Sharpe.idxmax(), "樣本內 t 最大": cand.t.idxmax()}
    P("## 3. 用樣本內挑 (所有 X × 時間限制 × 倍數), 看樣本外")
    for k, nm in picks.items():
        P(f"- {k}: **{nm}** → IS 日淨 {TI['日淨%'][nm]:.3f}% (差 {TI['日差%'][nm]:+.3f}%, t {TI.t[nm]:.1f}), OOS 日淨 {TO['日淨%'][nm]:.3f}% vs 現行 {TO['日淨%']['現行']:.3f}% "
          f"(差 {TO['日差%'][nm]:+.3f}%, t {TO.t[nm]:.1f}), OOS Sharpe {TO.Sharpe[nm]:.2f} vs {TO.Sharpe['現行']:.2f}, MDD {TO['MDD%'][nm]:.2f}% vs {TO['MDD%']['現行']:.2f}%")
    sel = list(dict.fromkeys(["現行"] + list(picks.values())))
    P("\n樣本內:"); P(TI.loc[sel].round(3).to_markdown()); P("\n樣本外:"); P(TO.loc[sel].round(3).to_markdown())
    P("\n## 4. 時間限制 (fav 市價 / fav 限價 / adv 市價, 1×) 樣本內 → 樣本外 日差% 與 t")
    rows = []
    for nm in names[1:]:
        mode, X, cut, m = rules[nm]
        if m != 1.0: continue
        rows.append({"規則": nm, "IS 日差%": TI['日差%'][nm], "IS t": TI.t[nm], "OOS 日差%": TO['日差%'][nm], "OOS t": TO.t[nm], "OOS MDD%": TO['MDD%'][nm]})
    P(pd.DataFrame(rows).set_index("規則").round(3).to_markdown())
    P("\n## 5. 各年日淨% (樣本內挑出的規則)"); P((DR[sel].groupby(DR.index.year).mean()*100).round(3).to_markdown())
    w = DR["現行"].nsmallest(10).index; X_ = DR.loc[w, sel]*100; X_.index = X_.index.date
    P("\n## 6. 現行最差 10 天, 加碼後那天的日淨%"); P(X_.round(2).to_markdown())
    (OUT/"breakeven_add.md").write_text("\n".join(L)); print("\n".join(L))
