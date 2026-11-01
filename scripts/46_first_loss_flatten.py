"""第一筆賠錢就全部平倉, 會不會少賠? 樣本 = trailing_stop_trades_sr.parquet (sr 每日前 10 名, 2020~2026), M1 逐分鐘重跑.
現行: 09:01 進場做空, 個股碰 +8% 強制回補 / 13:24 仍 ≥ +6% 回補 / 收盤漲停鎖死隔日開盤 +7% 罰 / 其餘收盤集合競價回補.
「全部平倉」規則: 當天某分鐘 T 觸發 → 還沒出場的部位在 T+1 分鐘開盤 ×(1+0.1%) 回補 (13:24 以後觸發不動作).
觸發定義:
  A. 首支碰 +8% 強制回補
  B. 首支浮虧 ≥ X% (分 K 收盤價相對進場價)
  C. 整本浮動損益 (已出場的算實現) ≤ −Y% (占 300 萬)
  D. A/B 的變形: 只平「當下也在賠」的, 有賺的留著
  E. 對照: 個股浮虧 ≥ X% 只停那一支 (不連坐) → 拆出「連坐」本身的效果
門檻只用 2020~23 挑, 2024~ 樣本外. 也統計同一天虧損的連動程度. 輸出 stats/first_loss_flatten.md"""
import numpy as np, pandas as pd
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
ROOT = Path(__file__).resolve().parent.parent; OUT = ROOT/"stats"
BASE, SLIP, K, PEN, CAP, PER_STOCK, MAX_N, PRE_K, SPLIT = 0.0023, 0.001, 0.08, 0.07, 3_000_000, 600_000, 10, 0.06, 2024
T0, TPRE, TAUC, NT = 4, 264, 270, 271
XS = (0.02, 0.03, 0.04, 0.05, 0.06)            # 個股浮虧門檻
YS = (0.003, 0.005, 0.0075, 0.01, 0.015, 0.02)  # 整本浮虧門檻 (占資金)

def rules():
    r = {"現行": None, "A 首支強制回補→全平": ("forced", None, "all"), "A' 首支強制回補→只平在賠的": ("forced", None, "losing")}
    for x in XS:
        r[f"B 首支浮虧≥{x*100:.0f}%→全平"] = ("stock", x, "all")
        r[f"D 首支浮虧≥{x*100:.0f}%→只平在賠的"] = ("stock", x, "losing")
        r[f"E 個股浮虧≥{x*100:.0f}% 只停那支"] = ("single", x, None)
    for y in YS: r[f"C 整本浮虧≥{y*100:.2g}%→全平"] = ("book", y, "all")
    for cut, lab in ((60, "10:00"), (120, "11:00")): r[f"A 首支強制回補 ({lab} 前才算)→全平"] = ("forced", cut, "all")
    return r
RULES = rules()

def grid(g):
    """一支股票一天 → 0..270 分鐘的 o/h/c (沒成交的分鐘: c 前值, o=前一分鐘 c, h=nan)."""
    o = np.full(NT, np.nan); h = np.full(NT, np.nan); c = np.full(NT, np.nan)
    t = g.t.values; o[t] = g.open.values; h[t] = g.high.values; c[t] = g.close.values
    cs = pd.Series(c).ffill().bfill().values; os_ = np.where(np.isnan(o), np.r_[cs[0], cs[:-1]], o)
    return os_, h, cs, ~np.isnan(c)

def sim_day(trs, bars):
    """trs: 當天的交易 (依 rank), bars: {(sid): (o,h,c,has)}. 回傳每支在各規則下的 gross + 診斷."""
    n = len(trs); E = trs.entry.values; PC = trs.prev_close.values; NO = trs.next_open.values
    O = np.vstack([bars[s][0] for s in trs.stock_id]); H = np.vstack([bars[s][1] for s in trs.stock_id]); C = np.vstack([bars[s][2] for s in trs.stock_id])
    HAS = np.vstack([bars[s][3] for s in trs.stock_id])
    auc = np.array([C[i, TAUC] if HAS[i, TAUC] else C[i, :TAUC+1][HAS[i, :TAUC+1]][-1] if HAS[i].any() else E[i] for i in range(n)])
    # 現行出場
    tex = np.full(n, 999); gx = np.zeros(n); why = np.empty(n, object)
    for i in range(n):
        hit = np.where(np.nan_to_num(H[i, T0:TPRE], nan=-1) >= PC[i]*(1+K))[0]
        if len(hit): tex[i] = T0 + hit[0]; gx[i] = E[i]/(PC[i]*(1+K)*(1+SLIP*3)) - 1; why[i] = "forced"; continue
        if HAS[i, TPRE] and C[i, TPRE] >= PC[i]*(1+PRE_K): tex[i] = TPRE; gx[i] = E[i]/(C[i, TPRE]*(1+SLIP)) - 1; why[i] = "pre"; continue
        if auc[i] >= PC[i]*1.0975 and np.isfinite(NO[i]): gx[i] = E[i]/NO[i] - 1 - PEN; why[i] = "lock"
        else: gx[i] = E[i]/auc[i] - 1; why[i] = "close"
        tex[i] = TAUC
    tt = np.arange(NT)
    U = np.where(tt[None, :] >= tex[:, None], gx[:, None], E[:, None]/C - 1)     # 每分鐘 每支 浮動 (已出場=實現) 毛報酬
    lots = []; cash = CAP
    for e in E:
        lc = e*1000; k = int(min(PER_STOCK, cash)//lc) if cash >= lc else 0; a = k*lc; cash -= a; lots.append(a)
    W = np.array(lots)/CAP; book = (W[:, None]*U).sum(0)
    def flat_at(T, only_losing):
        g = gx.copy()
        if T is None or T+1 >= TPRE: return g
        for i in range(n):
            if tex[i] <= T: continue
            if only_losing and U[i, T] >= 0: continue
            g[i] = E[i]/(O[i, T+1]*(1+SLIP)) - 1
        return g
    res = {}; trig = {}
    for nm, rl in RULES.items():
        if rl is None: res[nm] = gx; trig[nm] = -1; continue
        kind, p, mode = rl
        if kind == "forced":
            f = tex[why == "forced"]; T = int(f.min()) if len(f) else None
            if T is not None and p is not None and T >= p: T = None
        elif kind == "stock":
            m = (E[:, None]/C - 1 <= -p) & (tt[None, :] < tex[:, None]) & (tt[None, :] >= T0) & (tt[None, :] < TPRE)
            # 強制回補那分鐘本身也算觸發 (浮虧 ≥ 8% 一定 ≥ X)
            T = min([int(np.where(m[i])[0][0]) for i in range(n) if m[i].any()] + [int(x) for x in tex[why == "forced"]], default=None)
        elif kind == "book":
            w = np.where((book[T0:TPRE] <= -p))[0]; T = T0 + int(w[0]) if len(w) else None
        elif kind == "single":
            g = gx.copy()
            for i in range(n):
                w = np.where((E[i]/C[i, T0:TPRE] - 1 <= -p) & (np.arange(T0, TPRE) < tex[i]))[0]
                if len(w):
                    T1 = T0 + int(w[0])
                    if T1+1 < TPRE: g[i] = E[i]/(O[i, T1+1]*(1+SLIP)) - 1
            res[nm] = g; trig[nm] = -1; continue
        res[nm] = flat_at(T, mode == "losing"); trig[nm] = -1 if T is None or T+1 >= TPRE else T
    # 診斷: 首支強制回補時點 T, 其他還在場的部位「T 之後」到現行出場的報酬 (負 = 平掉比較好)
    f = tex[why == "forced"]; Tf = int(f.min()) if len(f) else -1
    rest = []
    if Tf >= 0 and Tf+1 < TPRE:
        for i in range(n):
            if tex[i] > Tf: rest.append((gx[i] - (E[i]/(O[i, Tf+1]*(1+SLIP)) - 1), U[i, Tf]))
    return res, trig, W, why, Tf, rest, book

def simulate_month(args):
    f, trades = args
    df = pd.read_parquet(f); ts = pd.to_datetime(df["date"]); df["day"] = ts.dt.normalize(); df["t"] = ts.dt.hour*60 + ts.dt.minute - 540
    df = df[(df.t >= 0) & (df.t <= 270)]; key = set(zip(trades.stock_id, trades.date)); df = df[[(s, d) in key for s, d in zip(df.stock_id, df.day)]]
    df = df.drop_duplicates(["stock_id","day","t"], keep="last")
    bars = {(sid, day): grid(g) for (sid, day), g in df.groupby(["stock_id","day"], sort=False)}
    trows, drows, rrows = [], [], []
    for day, trs in trades.sort_values(["date","rank"]).groupby("date", sort=False):
        trs = trs[[(s, day) in bars for s in trs.stock_id]].reset_index(drop=True)
        if trs.empty: continue
        res, trig, W, why, Tf, rest, book = sim_day(trs, {s: bars[(s, day)] for s in trs.stock_id})
        for i in range(len(trs)):
            trows.append({"stock_id": trs.stock_id[i], "date": day, "rank": trs["rank"][i], "w": W[i], "why": why[i], "g_stored": trs.g[i], **{f"g|{k}": v[i] for k, v in res.items()}})
        drows.append({"date": day, "n": len(trs), "t_first_forced": Tf, "book_min": book[T0:TPRE].min(), "t_book_min": T0 + int(book[T0:TPRE].argmin()), **{f"T|{k}": v for k, v in trig.items()}})
        for d, u in rest: rrows.append({"date": day, "Tf": Tf, "after": d, "u_at_T": u})
    return pd.DataFrame(trows), pd.DataFrame(drows), pd.DataFrame(rrows)

def tm(x): return "—" if x < 0 else f"{9 + (x)//60:02d}:{x % 60:02d}"

if __name__ == "__main__":
    R = pd.read_parquet(ROOT/"data/processed/trailing_stop_trades_sr.parquet")
    R = R[["stock_id","date","entry","prev_close","next_open","rank","gross|現行 (無移動停利)"]].rename(columns={"gross|現行 (無移動停利)": "g"})
    R["date"] = pd.to_datetime(R.date).astype("datetime64[ns]"); R["ym"] = R.date.dt.strftime("%Y_%m")
    jobs = [(str(ROOT/"data/raw/m1"/f"{ym}.parquet"), g) for ym, g in R.groupby("ym") if (ROOT/"data/raw/m1"/f"{ym}.parquet").exists()]
    with ProcessPoolExecutor(6) as ex: parts = list(ex.map(simulate_month, jobs))
    S = pd.concat([p[0] for p in parts], ignore_index=True); D = pd.concat([p[1] for p in parts], ignore_index=True).set_index("date").sort_index()
    RR = pd.concat([p[2] for p in parts], ignore_index=True)
    S.to_parquet(ROOT/"data/processed/first_loss_flatten_trades.parquet", index=False); print("simulated", len(S), "trades", len(D), "days", flush=True)
    S["oos"] = S.date.dt.year >= SPLIT; days = D.index; D_IS, D_OOS = days[days.year < SPLIT], days[days.year >= SPLIT]
    names = list(RULES)
    def daily(nm): return (S.w*(S[f"g|{nm}"] - BASE)).groupby(S.date).sum().reindex(days, fill_value=0.0)
    DR = pd.DataFrame({nm: daily(nm) for nm in names})
    def stats(nm, idx):
        r = DR.loc[idx, nm]; eq = (1+r).cumprod(); mo = r.groupby(r.index.to_period("M")).apply(lambda x: (1+x).prod()-1)
        s = S[S.date.isin(idx)]; net = s[f"g|{nm}"] - BASE; trg = (D.loc[idx, f"T|{nm}"] >= 0)
        d0 = DR.loc[idx, "現行"]
        return {"觸發天數%": trg.mean()*100, "每筆淨%": net.mean()*100, "每筆勝率%": (net > 0).mean()*100, "日勝率%": (r > 0).mean()*100, "日淨%": r.mean()*100,
                "Sharpe": r.mean()/r.std()*np.sqrt(245), "MDD%": (eq/eq.cummax()-1).min()*100, "最差日%": r.min()*100, "最差 5 日平均%": r.nsmallest(5).mean()*100,
                "最差月%": mo.min()*100, "年化(萬)": r.mean()*245*CAP/1e4, "觸發日 現行日淨%": d0[trg].mean()*100 if trg.any() else np.nan, "觸發日 規則日淨%": r[trg].mean()*100 if trg.any() else np.nan}
    L = []; P = lambda x="": L.append(str(x))
    P(f"# 第一筆賠錢就全部平倉? (sr 每日前 10 名, {days.min().date()}~{days.max().date()}, M1 逐分鐘, 300 萬)\n")
    chk = (S["g|現行"] - S.g_stored).abs()
    P(f"- {len(S)} 筆 / {len(days)} 天 (樣本內 2020~{SPLIT-1} {len(D_IS)} 天, 樣本外 {SPLIT}~ {len(D_OOS)} 天)；重跑現行 vs 原紀錄每筆差 < 0.1% 的占 {(chk < 1e-3).mean()*100:.1f}%")
    P("- 平倉價 = 觸發後下一分鐘開盤 ×(1+0.1%)；每筆淨 = 毛 − 0.23%；日淨 = 當天損益 / 300 萬\n")

    # ---- 1. 連動程度 ----
    P("## 1. 同一天的虧損有多連動?")
    S["net0"] = S["g|現行"] - BASE; S["loss"] = S.net0 < 0; p = S.loss.mean()
    nl = S.groupby("date").agg(n=("loss","size"), k=("loss","sum"))
    from math import comb
    obs = nl.k.clip(upper=10).value_counts(normalize=True).sort_index()
    exp = pd.Series({k: np.mean([comb(n, k)*p**k*(1-p)**(n-k) if k <= n else 0 for n in nl.n]) for k in range(11)})
    P(f"- 每筆賠錢率 {p*100:.1f}%。如果各支互不相關, 一天賠 ≥ 7 支的機率應只有 {exp[7:].sum()*100:.1f}%, 實際 {obs[obs.index >= 7].sum()*100:.1f}%；賠 ≤ 1 支: 應 {exp[:2].sum()*100:.1f}%, 實際 {obs[obs.index <= 1].sum()*100:.1f}%")
    P("\n一天賠幾支的分布 (實際 vs 互不相關的理論值):"); P(pd.DataFrame({"實際 天數%": obs*100, "互不相關 理論%": exp*100}).fillna(0).round(1).T.to_markdown())
    icc = S.groupby("date").net0.mean().var()/S.net0.var()
    # 同日兩兩相關: Var(日平均) = σ²/n + (n−1)/n ρσ²
    nbar = nl.n.mean(); rho = (icc - 1/nbar)/(1 - 1/nbar)
    P(f"\n- 同日兩兩相關係數 ρ ≈ {rho:.2f} (日平均報酬的變異有 {icc*100:.0f}% 來自「哪一天」)")
    # 條件機率: Σ k(k−1) / Σ k(n−1)
    kk = nl.k; nn = nl.n; P(f"- 已知某支賠錢, 同一天另一支也賠的機率 {(kk*(kk-1)).sum()/(kk*(nn-1)).sum()*100:.1f}% (無條件 {p*100:.1f}%)；已知某支被 +8% 強制回補, 同天其他支賠錢機率 "
      f"{S[S.date.isin(S[S.why=='forced'].date) & (S.why != 'forced')].loss.mean()*100:.1f}%")
    for nm2, idx in (("全部", days), ("樣本內", D_IS), ("樣本外", D_OOS)):
        r = DR.loc[idx, "現行"]; w = r.nsmallest(int(len(r)*.05))
        P(f"- {nm2}: 最差 5% 的日子 ({len(w)} 天) 平均日淨 {w.mean()*100:.2f}%, 那些天平均賠 {nl.loc[w.index, 'k'].mean():.1f} 支 / {nl.loc[w.index, 'n'].mean():.1f} 支；這 5% 的天數吃掉全部虧損日金額的 {w.sum()/r[r<0].sum()*100:.0f}%")

    # ---- 2. 首支強制回補之後, 其他支怎麼走 ----
    P("\n## 2. 首支碰 +8% 強制回補之後, 其他還在場的部位後來怎樣?")
    ft = D[D.t_first_forced >= 0]
    P(f"- 有強制回補的天數 {len(ft)} / {len(D)} ({len(ft)/len(D)*100:.1f}%)；首支強制回補時間中位 {tm(int(ft.t_first_forced.median()))}, 25% {tm(int(ft.t_first_forced.quantile(.25)))}, 75% {tm(int(ft.t_first_forced.quantile(.75)))}")
    RR["oos"] = RR.date.dt.year >= SPLIT
    for nm2, m in (("樣本內", ~RR.oos), ("樣本外", RR.oos)):
        x = RR[m]; P(f"- {nm2}: 其他部位 {len(x)} 筆, 在觸發時點平均浮動 {x.u_at_T.mean()*100:+.2f}%；如果不平, 觸發之後到收盤再多賺 {x.after.mean()*100:+.3f}%/筆 (正 = 留著比較好), "
                     f"留著比較好的占 {(x.after > 0).mean()*100:.0f}%；觸發時在賠的 {int((x.u_at_T<0).sum())} 筆之後平均 {x[x.u_at_T<0].after.mean()*100:+.3f}%, 在賺的 {int((x.u_at_T>=0).sum())} 筆 {x[x.u_at_T>=0].after.mean()*100:+.3f}%")
    RR["tb"] = pd.cut(RR.Tf, [0, 30, 60, 120, 180, 270], labels=["09:30 前","09:30~10:00","10~11","11~12","12 點後"])
    tb = RR.groupby(["tb","oos"], observed=False).after.agg(["size","mean"]).unstack(); tb.columns = [f"{a} {'OOS' if b else 'IS'}" for a, b in tb.columns]
    for c in tb.columns:
        if c.startswith("mean"): tb[c] = tb[c]*100
    P("\n依首支強制回補的時間 (其他部位觸發後的平均報酬%, 正 = 留著比較好):"); P(tb.round(3).to_markdown())

    # ---- 3. 規則比較 ----
    P("\n## 3. 各規則 vs 現行 (日淨 / 勝率 / 回撤 / 最差日)")
    P("「觸發日 現行/規則日淨」= 只看有觸發那些天, 規則把那些天變好還是變壞。")
    cols = ["觸發天數%","每筆淨%","每筆勝率%","日勝率%","日淨%","Sharpe","MDD%","最差日%","最差 5 日平均%","最差月%","年化(萬)","觸發日 現行日淨%","觸發日 規則日淨%"]
    TI = pd.DataFrame({nm: stats(nm, D_IS) for nm in names}).T[cols]; TO = pd.DataFrame({nm: stats(nm, D_OOS) for nm in names}).T[cols]
    P(f"\n### 樣本內 2020~{SPLIT-1}"); P(TI.round(3).to_markdown()); P(f"\n### 樣本外 {SPLIT}~"); P(TO.round(3).to_markdown())
    cand = TI.drop(index="現行"); best = cand.Sharpe.idxmax(); best2 = cand["日淨%"].idxmax(); bestm = cand["MDD%"].idxmax()
    P(f"\n- 樣本內 Sharpe 最好: **{best}** (IS Sharpe {TI.Sharpe[best]:.2f} vs 現行 {TI.Sharpe['現行']:.2f}) → 樣本外 Sharpe {TO.Sharpe[best]:.2f} vs 現行 {TO.Sharpe['現行']:.2f}, 日淨 {TO['日淨%'][best]:.3f}% vs {TO['日淨%']['現行']:.3f}%, MDD {TO['MDD%'][best]:.2f}% vs {TO['MDD%']['現行']:.2f}%")
    P(f"- 樣本內日淨最好: **{best2}** → 樣本外日淨 {TO['日淨%'][best2]:.3f}% vs 現行 {TO['日淨%']['現行']:.3f}%")
    P(f"- 樣本內 MDD 最小: **{bestm}** (IS {TI['MDD%'][bestm]:.2f}% vs {TI['MDD%']['現行']:.2f}%) → 樣本外 MDD {TO['MDD%'][bestm]:.2f}% vs {TO['MDD%']['現行']:.2f}%, 日淨 {TO['日淨%'][bestm]:.3f}%")
    # 差異的 t 值 (配對日)
    P("\n### 樣本外每日差 (規則 − 現行) 的 t 值 (|t| > 2 才算不是運氣)")
    rows = []
    for nm in names[1:]:
        dd = (DR.loc[D_OOS, nm] - DR.loc[D_OOS, "現行"]); di = (DR.loc[D_IS, nm] - DR.loc[D_IS, "現行"])
        rows.append({"規則": nm, "IS 日差%": di.mean()*100, "IS t": di.mean()/di.std()*np.sqrt(len(di)) if di.std() > 0 else np.nan,
                     "OOS 日差%": dd.mean()*100, "OOS t": dd.mean()/dd.std()*np.sqrt(len(dd)) if dd.std() > 0 else np.nan, "OOS 年化差(萬)": dd.mean()*245*CAP/1e4})
    P(pd.DataFrame(rows).set_index("規則").round(3).to_markdown())

    # ---- 4. 各年 ----
    P("\n## 4. 各年日淨% (主要規則)")
    keyr = ["現行", "A 首支強制回補→全平", "A' 首支強制回補→只平在賠的", "B 首支浮虧≥4%→全平", "D 首支浮虧≥4%→只平在賠的", "C 整本浮虧≥1%→全平", "E 個股浮虧≥4% 只停那支", best]
    keyr = list(dict.fromkeys(k for k in keyr if k in DR))
    P((DR[keyr].groupby(DR.index.year).mean()*100).round(3).to_markdown())

    # ---- 5. 現行最差 10 天, 規則能救多少 ----
    P("\n## 5. 現行最差 10 天, 各規則那天的日淨%")
    w = DR["現行"].nsmallest(10).index
    X = DR.loc[w, keyr]*100; X.insert(0, "賠幾支/幾支", [f"{int(nl.k[d])}/{int(nl.n[d])}" for d in w]); X.insert(1, "首支強制回補", [tm(int(D.t_first_forced[d])) for d in w])
    X.index = X.index.date; P(X.round(2).to_markdown())
    (OUT/"first_loss_flatten.md").write_text("\n".join(L)); print("\n".join(L))
