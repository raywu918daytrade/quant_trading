"""賠錢單的共同點 → 能不能當濾網避開? 樣本 = trailing_stop_trades_sr.parquet (sr 每日前 10 名, 2020~2026, 現行出場).
全部特徵都是進場 (09:01) 前可知: D1 (昨日以前) + 今日開盤跳空 + 進場價 vs 開盤 + 大盤開盤/前日狀態 + 排名/星期/月份.
做法:
  1. 賠錢單 vs 賺錢單 的特徵中位數與效果量 (差 / 合併標準差), 先找「最不一樣」的特徵
  2. 每個特徵切五分位 (切點用 2020~23 決定), 看各組每筆淨/勝率; 同時列 2020~23 (樣本內) 與 2024~ (樣本外)
  3. 自動挑候選濾網: 樣本內最差那一組 (每筆淨比整體低, t < −2) → 排掉後在樣本內/樣本外各跑一次 300 萬配置 (排掉的資金給下一名)
  4. LightGBM 用 2020~23 學「這筆會不會賠」, 2024~ 排掉預測最危險的 5/10/20%
每筆淨 = 毛 − 0.23%. 輸出 stats/loser_profile.md"""
import glob, numpy as np, pandas as pd
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent; OUT = ROOT/"stats"; BASE, CAP, PER_STOCK, MAX_N = 0.0023, 3_000_000, 600_000, 10
SPLIT = 2024   # >= SPLIT 為樣本外

R = pd.read_parquet(ROOT/"data/processed/trailing_stop_trades_sr.parquet")
R = R[["stock_id","date","entry","prev_close","rank","gross|現行 (無移動停利)","reason|現行 (無移動停利)"]].rename(columns={"gross|現行 (無移動停利)": "g", "reason|現行 (無移動停利)": "reason"})
R["net"] = R.g - BASE; R["date"] = pd.to_datetime(R.date).astype("datetime64[ns]"); R["loss"] = R.net < 0

info = pd.read_parquet(ROOT/"data/raw/meta/info.parquet")[["stock_id","market_type"]]
d1 = pd.concat([pd.read_parquet(f, columns=["stock_id","date","open","high","low","close","volume"]) for f in sorted(glob.glob(str(ROOT/"data/raw/d1/*.parquet")))], ignore_index=True)
d1["date"] = pd.to_datetime(d1.date).astype("datetime64[ns]"); d1 = d1.drop_duplicates(["stock_id","date"]).sort_values(["stock_id","date"]).reset_index(drop=True)
g = d1.groupby("stock_id"); pc = g.close.shift(1)
d1["ret"] = d1.close/pc - 1; d1["gap"] = d1.open/pc - 1; d1["intra"] = d1.close/d1.open - 1
d1["dv20"] = (d1.close*d1.volume).groupby(d1.stock_id).transform(lambda s: s.shift(1).rolling(20).mean())
# 大盤 (流動股平均)
liq = d1[d1.dv20 >= 1e8]
mk = liq.groupby("date").agg(mkt_ret=("ret","mean"), mkt_gap=("gap","mean"), breadth=("ret", lambda s: (s > 0).mean()), gap_breadth=("gap", lambda s: (s > 0).mean())).sort_index()
mk["mkt_pret"] = mk.mkt_ret.shift(1); mk["mkt_ret5"] = mk.mkt_ret.rolling(5).sum().shift(1); mk["mkt_ret20"] = mk.mkt_ret.rolling(20).sum().shift(1)
mk["mkt_vol20"] = mk.mkt_ret.rolling(20).std().shift(1); mk["p_breadth"] = mk.breadth.shift(1)
mk = mk[["mkt_gap","gap_breadth","mkt_pret","mkt_ret5","mkt_ret20","mkt_vol20","p_breadth"]]
# 個股 (只算有交易的股票)
s = d1[d1.stock_id.isin(R.stock_id.unique())].copy().reset_index(drop=True); g = s.groupby("stock_id"); pc = g.close.shift(1)
s["p_ret"] = g.ret.shift(1)
s["ret5"] = pc/g.close.shift(6) - 1; s["ret20"] = pc/g.close.shift(21) - 1
s["ma20_gap"] = pc/g.close.transform(lambda x: x.shift(1).rolling(20).mean()) - 1
s["ma60_gap"] = pc/g.close.transform(lambda x: x.shift(1).rolling(60).mean()) - 1
hi20 = g.high.transform(lambda x: x.shift(1).rolling(20).max()); lo20 = g.low.transform(lambda x: x.shift(1).rolling(20).min()); s["pos20"] = (pc - lo20)/(hi20 - lo20)
hi60 = g.high.transform(lambda x: x.shift(1).rolling(60).max()); s["vs_hi60"] = pc/hi60 - 1
tr = pd.concat([s.high-s.low, (s.high-pc).abs(), (s.low-pc).abs()], axis=1).max(axis=1); s["atr20"] = tr.groupby(s.stock_id).transform(lambda x: x.shift(1).rolling(20).mean())/pc
s["p_range"] = g.high.shift(1)/g.low.shift(1) - 1
s["p_upper_wick"] = (g.high.shift(1) - np.maximum(g.open.shift(1), pc))/pc   # 昨日上影線
s["p_intra"] = g.intra.shift(1)
v20 = g.volume.transform(lambda x: x.shift(2).rolling(20).mean()); s["p_volratio"] = g.volume.shift(1)/v20
up = (s.ret > 0).astype(int); run = (up != up.groupby(s.stock_id).shift(1)).cumsum()
s["up_streak"] = up.groupby([s.stock_id, run]).cumsum().groupby(s.stock_id).shift(1)   # 到昨天為止連漲幾天
s["p_limit"] = (s.p_ret >= .095).astype(float)
s["n_limit5"] = (s.ret >= .095).astype(int).groupby(s.stock_id).transform(lambda x: x.shift(1).rolling(5).sum())
F = s[["stock_id","date","open","gap","dv20","p_ret","ret5","ret20","ma20_gap","ma60_gap","pos20","vs_hi60","atr20","p_range","p_upper_wick","p_intra","p_volratio","up_streak","p_limit","n_limit5"]]
T = R.merge(F, on=["stock_id","date"], how="left").merge(mk, left_on="date", right_index=True, how="left").merge(info, on="stock_id", how="left")
T["ent_vs_open"] = T.entry/T.open - 1        # 09:00 開盤到 09:01 進場這 1 分鐘
T["gap_vs_mkt"] = T.gap - T.mkt_gap
T["price"] = T.prev_close; T["dv20_e"] = T.dv20/1e8
T["dow"] = T.date.dt.day_of_week + 1; T["month"] = T.date.dt.month; T["is_tpex"] = (T.market_type == "tpex").astype(float)
T["n_list"] = T.groupby("date").stock_id.transform("size")
T["oos"] = T.date.dt.year >= SPLIT

FEAT = {"gap": "今日跳空", "ent_vs_open": "開盤→進場 1 分鐘漲跌", "gap_vs_mkt": "個股跳空 − 大盤跳空", "price": "股價 (昨收)", "dv20_e": "20 日均成交額 (億)",
        "atr20": "ATR20%", "p_ret": "昨日漲跌", "p_intra": "昨日日內 (收/開)", "p_range": "昨日振幅", "p_upper_wick": "昨日上影線", "p_volratio": "昨日量 / 20 日均量",
        "ret5": "前 5 日累計", "ret20": "前 20 日累計", "ma20_gap": "昨收 vs MA20", "ma60_gap": "昨收 vs MA60", "pos20": "20 日區間位置 (0 低 1 高)", "vs_hi60": "昨收 vs 60 日高",
        "up_streak": "連漲天數", "n_limit5": "前 5 日漲停次數", "mkt_gap": "大盤跳空", "gap_breadth": "全市場開高家數比", "mkt_pret": "大盤昨日漲跌",
        "mkt_ret5": "大盤前 5 日累計", "mkt_ret20": "大盤前 20 日累計", "mkt_vol20": "大盤 20 日波動", "p_breadth": "昨日上漲家數比", "rank": "模型排名", "n_list": "當日名單檔數",
        "dow": "星期", "month": "月份", "is_tpex": "上櫃"}
PCT = {"gap","ent_vs_open","gap_vs_mkt","atr20","p_ret","p_intra","p_range","p_upper_wick","ret5","ret20","ma20_gap","ma60_gap","vs_hi60","mkt_gap","mkt_pret","mkt_ret5","mkt_ret20","mkt_vol20"}
DISCRETE = {"dow","month","is_tpex","rank","up_streak","n_limit5","n_list"}

DAYS = pd.DatetimeIndex(sorted(T.date.unique())); D_IS, D_OOS = DAYS[DAYS.year < SPLIT], DAYS[DAYS.year >= SPLIT]
def alloc(S):
    recs = []
    for dt, gg in S.sort_values(["date","rank"]).groupby("date", sort=False):
        cash = CAP; tot = 0.0; n = 0
        for e, nt in zip(gg.entry.values, gg.net.values):
            lc = e*1000
            if cash < lc: continue
            lots = int(min(PER_STOCK, cash)//lc)
            if lots < 1: continue
            a = lots*lc; cash -= a; tot += a*nt; n += 1
            if n >= MAX_N: break
        recs.append((dt, tot/CAP))
    return pd.DataFrame(recs, columns=["date","ret"]).set_index("date").ret
def st(S, days):
    r = alloc(S).reindex(days, fill_value=0.0); eq = (1+r).cumprod(); mo = r.groupby(r.index.to_period("M")).apply(lambda x: (1+x).prod()-1)
    return {"日淨%": r.mean()*100, "Sharpe": r.mean()/r.std()*np.sqrt(245), "MDD%": (eq/eq.cummax()-1).min()*100, "最差月%": mo.min()*100, "年化(萬)": r.mean()*245*CAP/1e4}
def tstat(x): return x.mean()/x.std()*np.sqrt(len(x)) if len(x) > 2 else np.nan

L = []; P = lambda x="": L.append(str(x))
IS, OOS = T[~T.oos], T[T.oos]
P(f"# 賠錢單的共同點 (sr 每日前 10 名, 2020~{T.date.dt.year.max()}, 現行進出場)\n")
P(f"- 交易 {len(T)} 筆 / {T.date.nunique()} 天；賠錢 (淨 < 0) {T.loss.sum()} 筆 = {T.loss.mean()*100:.1f}%，平均每筆賠 {T.net[T.loss].mean()*100:.2f}%；賺錢平均 {T.net[~T.loss].mean()*100:.2f}%")
P(f"- 樣本內 2020~{SPLIT-1}: {len(IS)} 筆, 每筆淨 {IS.net.mean()*100:.3f}%；樣本外 {SPLIT}~: {len(OOS)} 筆, 每筆淨 {OOS.net.mean()*100:.3f}%")
P("- 所有特徵都在 09:01 進場前可知；每筆淨 = 做空毛報酬 − 0.23%")

P("\n## 0. 賠錢單長什麼樣 (出場原因 × 虧損大小)")
lb = pd.cut(T.net[T.loss], [-1, -.07, -.04, -.02, -.01, 0], labels=["≤−7% (多為 +8% 強制回補)","−7~−4%","−4~−2%","−2~−1%","−1~0%"])
X = T[T.loss].assign(b=lb).groupby("b", observed=False).agg(筆數=("net","size"), 合計淨_單位pct=("net", lambda x: x.sum()*100)); X["占賠錢筆數%"] = X.筆數/T.loss.sum()*100; X["占總虧損%"] = X.合計淨_單位pct/X.合計淨_單位pct.sum()*100
P(X.round(1).to_markdown())
P(f"\n出場原因 (賠錢單): {T[T.loss].reason.value_counts().to_dict()}；(賺錢單): {T[~T.loss].reason.value_counts().to_dict()}")

P("\n## 1. 賠錢 vs 賺錢: 特徵中位數與效果量 (|效果量| 越大越不一樣；> 0.1 才算有點差)")
rows = []
for c, nm in FEAT.items():
    a, b = T.loc[T.loss, c].dropna(), T.loc[~T.loss, c].dropna(); k = 100 if c in PCT else 1
    es = (a.mean() - b.mean())/np.sqrt((a.var() + b.var())/2)
    ai, bi = IS.loc[IS.loss, c].dropna(), IS.loc[~IS.loss, c].dropna(); ao, bo = OOS.loc[OOS.loss, c].dropna(), OOS.loc[~OOS.loss, c].dropna()
    rows.append({"特徵": nm, "賠錢 中位": a.median()*k, "賺錢 中位": b.median()*k, "效果量": es,
                 "效果量 2020~23": (ai.mean()-bi.mean())/np.sqrt((ai.var()+bi.var())/2), f"效果量 {SPLIT}~": (ao.mean()-bo.mean())/np.sqrt((ao.var()+bo.var())/2)})
E = pd.DataFrame(rows).set_index("特徵"); E = E.reindex(E.效果量.abs().sort_values(ascending=False).index); P(E.round(3).to_markdown())

P("\n## 2. 各特徵分組 (切點用 2020~23 的五分位, 樣本外用同一組切點)")
P("每格 = 每筆淨% / 勝率；t = 該組每筆淨跟整體平均差的 t 值 (樣本內)。只列 2 組以上差異明顯的特徵 (樣本內最好與最差組差 > 0.3%)")
cands = []
for c, nm in FEAT.items():
    if c in DISCRETE and T[c].nunique() <= 12:
        key = T[c].clip(upper=10 if c in {"rank","n_list"} else (5 if c in {"up_streak","n_limit5"} else None)).map(lambda v: "NA" if pd.isna(v) else f"{int(v):02d}")
    else:
        q = IS[c].quantile([.2, .4, .6, .8]).values; q = np.unique(q)
        if len(q) < 2: continue
        bins = [-np.inf, *q, np.inf]; k = 100 if c in PCT else 1
        labs = [f"<{q[0]*k:.2f}"] + [f"{q[i]*k:.2f}~{q[i+1]*k:.2f}" for i in range(len(q)-1)] + [f">{q[-1]*k:.2f}"]
        labs = [f"Q{i+1} {x}" for i, x in enumerate(labs)]   # Q1 = 最小
        key = pd.cut(T[c], bins, labels=labs, ordered=False).astype(str)
    T["_k"] = key; mu_is = IS.net.mean()
    gi = T[~T.oos].groupby("_k").net; go = T[T.oos].groupby("_k").net
    tab = pd.DataFrame({"筆數 IS": gi.size(), "每筆淨% IS": gi.mean()*100, "勝率 IS": gi.apply(lambda x: (x > 0).mean()), "t IS": gi.apply(lambda x: (x.mean()-mu_is)/x.std()*np.sqrt(len(x)) if len(x) > 2 else np.nan),
                        "筆數 OOS": go.size(), "每筆淨% OOS": go.mean()*100, "勝率 OOS": go.apply(lambda x: (x > 0).mean())})
    tab = tab[tab["筆數 IS"] >= 30]; tab.index.name = "組"
    if tab.empty or tab["每筆淨% IS"].max() - tab["每筆淨% IS"].min() < 0.3: continue
    P(f"\n### {nm}" + (" (%)" if c in PCT else "")); P(tab.round(3).to_markdown())
    w = tab["t IS"].idxmin()
    if tab.loc[w, "t IS"] < -2: cands.append((nm, c, w, T._k == w))

P("\n## 3. 候選濾網: 樣本內最差那組 (t < −2) 排掉, 300 萬配置 (排掉的資金給下一名)")
P("樣本內好看但樣本外沒改善 = 過擬合，不能用。")
base_is, base_oos = st(IS, D_IS), st(OOS, D_OOS)
rows = [{"規則": "現行", "排掉占比%": 0, **{f"{k} IS": v for k, v in base_is.items()}, **{f"{k} OOS": v for k, v in base_oos.items()}}]
for nm, c, w, m in cands:
    keep = ~m; a, b = st(T[keep & ~T.oos], D_IS), st(T[keep & T.oos], D_OOS)
    rows.append({"規則": f"排 {nm} = {w}", "排掉占比%": m.mean()*100, **{f"{k} IS": v for k, v in a.items()}, **{f"{k} OOS": v for k, v in b.items()}})
Cd = pd.DataFrame(rows).set_index("規則"); Cd["OOS 日淨變化"] = Cd["日淨% OOS"] - base_oos["日淨%"]; Cd["OOS Sharpe 變化"] = Cd["Sharpe OOS"] - base_oos["Sharpe"]
P(Cd[["排掉占比%","日淨% IS","Sharpe IS","MDD% IS","日淨% OOS","Sharpe OOS","MDD% OOS","最差月% OOS","OOS 日淨變化","OOS Sharpe 變化"]].round(3).to_markdown())
ok = Cd[(Cd["OOS 日淨變化"] > 0) & (Cd["OOS Sharpe 變化"] > 0) & (Cd["日淨% IS"] > base_is["日淨%"])].index.tolist()
P(f"\n樣本內、樣本外日淨和 Sharpe 都變好的: {ok or '無'}")

P("\n## 4. 多特徵模型: LightGBM 用 2020~23 學「會不會賠」, 2024~ 排掉預測最危險的前 x%")
try:
    import lightgbm as lgb
    cols = [c for c in FEAT if c not in {"month"}]
    prm = dict(objective="binary", learning_rate=0.03, num_leaves=15, min_data_in_leaf=100, bagging_fraction=0.8, bagging_freq=1, feature_fraction=0.8, lambda_l2=5, verbose=-1, seed=0)
    m = lgb.train(prm, lgb.Dataset(IS[cols].astype(float), IS.loss.astype(int)), num_boost_round=300); T["p_loss"] = m.predict(T[cols].astype(float))
    def roc_auc_score(y, p):
        y = np.asarray(y).astype(bool); r = pd.Series(np.asarray(p)).rank().values; n1 = y.sum(); n0 = len(y) - n1
        return (r[y].sum() - n1*(n1+1)/2)/(n1*n0)
    P(f"- AUC 樣本內 {roc_auc_score(IS.loss, T.p_loss[~T.oos]):.3f}、樣本外 {roc_auc_score(OOS.loss, T.p_loss[T.oos]):.3f} (0.5 = 猜)")
except Exception as ex:
    T["p_loss"] = np.nan; P(f"- LightGBM 無法執行: {ex}")
if T.p_loss.notna().any():
    imp = pd.Series(m.feature_importance(importance_type="gain"), index=[FEAT[c] for c in cols]).sort_values(ascending=False); P(f"- 重要特徵前 8: {', '.join(imp.index[:8])}")
    O = T[T.oos].copy(); O["dec"] = pd.qcut(O.p_loss.rank(method="first"), 5, labels=["最安全 20%","2","3","4","最危險 20%"])
    gd = O.groupby("dec", observed=False).net; P("\n樣本外依預測風險五分組:"); P(pd.DataFrame({"筆數": gd.size(), "每筆淨%": gd.mean()*100, "勝率": gd.apply(lambda x: (x > 0).mean()), "≤−7% 比例%": gd.apply(lambda x: (x <= -.07).mean()*100)}).round(3).to_markdown())
    thr = {x: T.p_loss[~T.oos].quantile(1 - x) for x in (.05, .10, .20)}   # 門檻用樣本內分佈定
    rows = [{"規則": "現行", "排掉占比% OOS": 0, **st(OOS, D_OOS)}]
    for x, t in thr.items():
        keep = O.p_loss < t; rows.append({"規則": f"排預測最危險 {int(x*100)}% (樣本內門檻)", "排掉占比% OOS": (~keep).mean()*100, **st(O[keep], D_OOS)})
    P("\n樣本外 300 萬配置:"); P(pd.DataFrame(rows).set_index("規則").round(3).to_markdown())

P("\n## 5. 各年穩定度: 第 3 節候選的每筆淨% (排掉後) vs 現行")
yr = {"現行": T.groupby(T.date.dt.year).net.mean()*100}
for nm, c, w, m in cands: yr[f"排 {nm}={w}"] = T[~m].groupby(T[~m].date.dt.year).net.mean()*100
P(pd.DataFrame(yr).T.round(3).to_markdown())

(OUT/"loser_profile.md").write_text("\n".join(L)); print("\n".join(L))
