"""每天最多只做一支: 在 sr 每日前 10 名候選裡, 用什麼排序規則選第 1 名, 勝率/淨利最高?
樣本 = trailing_stop_trades_sr.parquet (sr 每日前 10 名, 2020~2026, 現行進出場), 特徵同 44_loser_profile (全部 09:01 進場前可知).
做法 (所有規則/方向/權重都只用 2020~23 決定, 2024~ 純樣本外):
  1. 同日比較 (IC): 每天把候選依特徵排序, 跟當天淨報酬的等級相關 (Spearman) 取平均 → 哪些特徵能「在同一天的候選裡」分出好壞
  2. 單一特徵選股: 每天選特徵最大 (或最小) 那支, 方向由樣本內決定
  3. 組合分數: 樣本內 IC 顯著的特徵, 當日百分位 × IC 正負號 加總
  4. LightGBM: (a) 44 的「會不會賠」模型, 選最安全 (b) 回歸預測淨報酬 (c) lambdarank 同日排序; 都用 2020~23 訓練,
     另做逐年滾動 (每年用之前所有年份重訓) 當穩健度檢查
每筆淨 = 毛 − 0.23%. 輸出 stats/pick_one.md"""
import glob, numpy as np, pandas as pd
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent; OUT = ROOT/"stats"; BASE, SPLIT, POS = 0.0023, 2024, 600_000

# ---------- 特徵 (與 44_loser_profile 相同) ----------
R = pd.read_parquet(ROOT/"data/processed/trailing_stop_trades_sr.parquet")
R = R[["stock_id","date","entry","prev_close","rank","gross|現行 (無移動停利)","reason|現行 (無移動停利)"]].rename(columns={"gross|現行 (無移動停利)": "g", "reason|現行 (無移動停利)": "reason"})
R["net"] = R.g - BASE; R["date"] = pd.to_datetime(R.date).astype("datetime64[ns]"); R["loss"] = R.net < 0
info = pd.read_parquet(ROOT/"data/raw/meta/info.parquet")[["stock_id","market_type"]]
d1 = pd.concat([pd.read_parquet(f, columns=["stock_id","date","open","high","low","close","volume"]) for f in sorted(glob.glob(str(ROOT/"data/raw/d1/*.parquet")))], ignore_index=True)
d1["date"] = pd.to_datetime(d1.date).astype("datetime64[ns]"); d1 = d1.drop_duplicates(["stock_id","date"]).sort_values(["stock_id","date"]).reset_index(drop=True)
g = d1.groupby("stock_id"); pc = g.close.shift(1)
d1["ret"] = d1.close/pc - 1; d1["gap"] = d1.open/pc - 1; d1["intra"] = d1.close/d1.open - 1
d1["dv20"] = (d1.close*d1.volume).groupby(d1.stock_id).transform(lambda s: s.shift(1).rolling(20).mean())
liq = d1[d1.dv20 >= 1e8]
mk = liq.groupby("date").agg(mkt_ret=("ret","mean"), mkt_gap=("gap","mean"), breadth=("ret", lambda s: (s > 0).mean()), gap_breadth=("gap", lambda s: (s > 0).mean())).sort_index()
mk["mkt_pret"] = mk.mkt_ret.shift(1); mk["mkt_ret5"] = mk.mkt_ret.rolling(5).sum().shift(1); mk["mkt_ret20"] = mk.mkt_ret.rolling(20).sum().shift(1)
mk["mkt_vol20"] = mk.mkt_ret.rolling(20).std().shift(1); mk["p_breadth"] = mk.breadth.shift(1)
mk = mk[["mkt_gap","gap_breadth","mkt_pret","mkt_ret5","mkt_ret20","mkt_vol20","p_breadth"]]
s = d1[d1.stock_id.isin(R.stock_id.unique())].copy().reset_index(drop=True); g = s.groupby("stock_id"); pc = g.close.shift(1)
s["p_ret"] = g.ret.shift(1)
s["ret5"] = pc/g.close.shift(6) - 1; s["ret20"] = pc/g.close.shift(21) - 1
s["ma20_gap"] = pc/g.close.transform(lambda x: x.shift(1).rolling(20).mean()) - 1
s["ma60_gap"] = pc/g.close.transform(lambda x: x.shift(1).rolling(60).mean()) - 1
hi20 = g.high.transform(lambda x: x.shift(1).rolling(20).max()); lo20 = g.low.transform(lambda x: x.shift(1).rolling(20).min()); s["pos20"] = (pc - lo20)/(hi20 - lo20)
hi60 = g.high.transform(lambda x: x.shift(1).rolling(60).max()); s["vs_hi60"] = pc/hi60 - 1
tr = pd.concat([s.high-s.low, (s.high-pc).abs(), (s.low-pc).abs()], axis=1).max(axis=1); s["atr20"] = tr.groupby(s.stock_id).transform(lambda x: x.shift(1).rolling(20).mean())/pc
s["p_range"] = g.high.shift(1)/g.low.shift(1) - 1
s["p_upper_wick"] = (g.high.shift(1) - np.maximum(g.open.shift(1), pc))/pc
s["p_intra"] = g.intra.shift(1)
v20 = g.volume.transform(lambda x: x.shift(2).rolling(20).mean()); s["p_volratio"] = g.volume.shift(1)/v20
up = (s.ret > 0).astype(int); run = (up != up.groupby(s.stock_id).shift(1)).cumsum()
s["up_streak"] = up.groupby([s.stock_id, run]).cumsum().groupby(s.stock_id).shift(1)
s["n_limit5"] = (s.ret >= .095).astype(int).groupby(s.stock_id).transform(lambda x: x.shift(1).rolling(5).sum())
F = s[["stock_id","date","open","gap","dv20","p_ret","ret5","ret20","ma20_gap","ma60_gap","pos20","vs_hi60","atr20","p_range","p_upper_wick","p_intra","p_volratio","up_streak","n_limit5"]]
T = R.merge(F, on=["stock_id","date"], how="left").merge(mk, left_on="date", right_index=True, how="left").merge(info, on="stock_id", how="left")
T["ent_vs_open"] = T.entry/T.open - 1; T["gap_vs_mkt"] = T.gap - T.mkt_gap
T["price"] = T.prev_close; T["dv20_e"] = T.dv20/1e8
T["dow"] = T.date.dt.day_of_week + 1; T["month"] = T.date.dt.month; T["is_tpex"] = (T.market_type == "tpex").astype(float)
T["n_list"] = T.groupby("date").stock_id.transform("size"); T["oos"] = T.date.dt.year >= SPLIT
T = T.sort_values(["date","rank"]).reset_index(drop=True)

FEAT = {"gap": "今日跳空", "ent_vs_open": "開盤→進場 1 分鐘漲跌", "gap_vs_mkt": "個股跳空 − 大盤跳空", "price": "股價 (昨收)", "dv20_e": "20 日均成交額",
        "atr20": "ATR20%", "p_ret": "昨日漲跌", "p_intra": "昨日日內 (收/開)", "p_range": "昨日振幅", "p_upper_wick": "昨日上影線", "p_volratio": "昨日量 / 20 日均量",
        "ret5": "前 5 日累計", "ret20": "前 20 日累計", "ma20_gap": "昨收 vs MA20", "ma60_gap": "昨收 vs MA60", "pos20": "20 日區間位置", "vs_hi60": "昨收 vs 60 日高",
        "up_streak": "連漲天數", "n_limit5": "前 5 日漲停次數", "rank": "模型排名", "is_tpex": "上櫃"}
MKT = ["mkt_gap","gap_breadth","mkt_pret","mkt_ret5","mkt_ret20","mkt_vol20","p_breadth","dow","n_list"]   # 同一天都一樣, 只給模型用 (互動)
LGB_COLS = list(FEAT) + MKT
DAYS = pd.DatetimeIndex(sorted(T.date.unique())); NDAY = {False: (DAYS.year < SPLIT).sum(), True: (DAYS.year >= SPLIT).sum()}

# 當日百分位 (0 = 最小, 1 = 最大), 同日只有 1 檔 → 0.5
for c in FEAT: T[f"pr_{c}"] = T.groupby("date")[c].rank(pct=True, method="average")
T["day_mean"] = T.groupby("date").net.transform("mean"); T["day_max"] = T.groupby("date").net.transform("max")
T["net_pr"] = T.groupby("date").net.rank(pct=True)
T["grade"] = (T.net_pr*5).clip(upper=4.999).astype(int)   # 同日五等級 (lambdarank 用)
T["win"] = (T.net > 0).astype(float); T["big_loss"] = (T.net <= -.07).astype(float); T["is_best"] = (T.net >= T.day_max - 1e-12).astype(float); T["top_half"] = (T.net_pr > .5).astype(float)
# 隨機選 1 支的期望值 = 當天候選平均
RAND = T.groupby("date").agg(net=("net","mean"), win=("win","mean"), big_loss=("big_loss","mean"), is_best=("is_best","mean"), top_half=("top_half","mean"), oos=("oos","first"), day_mean=("day_mean","first")).reset_index()

def pick(score):
    """每天選 score 最大的那支 (NaN 不選; 平手取模型排名較前). 回傳被選中的列."""
    x = T.assign(_s=score).dropna(subset=["_s"]).sort_values(["date","_s","rank"], ascending=[True, False, True])
    return x.drop_duplicates("date")
def stats(P, oos):
    rnd = "rank" not in P; P = P[P.oos == oos]; n = NDAY[oos]; net = P.net
    d = (P.net - P.day_mean); t = d.mean()/d.std()*np.sqrt(len(d)) if len(d) > 2 else np.nan
    cum = net.cumsum(); mdd = (cum - cum.cummax()).min()
    return {"天數": len(P), "勝率%": P.win.mean()*100, "每筆淨%": net.mean()*100, "比隨機多%": d.mean()*100, "t(比隨機)": t,
            "選到當日最好%": P.is_best.mean()*100, "落在當日前半%": P.top_half.mean()*100, "≤−7%占%": P.big_loss.mean()*100,
            "Sharpe": np.nan if rnd else net.mean()/net.std()*np.sqrt(245), "最大連續回落%": mdd*100,
            "年化(萬, 60萬一支)": net.sum()/max(n, 1)*245*POS/1e4}
def row(name, P): return {"規則": name, **{f"{k} IS": v for k, v in stats(P, False).items()}, **{f"{k} OOS": v for k, v in stats(P, True).items()}}
SHOW = ["勝率% IS","每筆淨% IS","t(比隨機) IS","勝率% OOS","每筆淨% OOS","比隨機多% OOS","t(比隨機) OOS","選到當日最好% OOS","落在當日前半% OOS","≤−7%占% OOS","Sharpe OOS","年化(萬, 60萬一支) OOS"]

L = []; P_ = lambda x="": L.append(str(x))
IS, OOS = T[~T.oos], T[T.oos]
P_(f"# 每天只做一支: 怎麼從候選名單選 (sr 每日前 10 名, 2020~{T.date.dt.year.max()}, 現行進出場)\n")
P_(f"- 候選 {len(T)} 筆 / {len(DAYS)} 天 (每天平均 {T.groupby('date').size().mean():.1f} 檔)；樣本內 2020~{SPLIT-1} {NDAY[False]} 天、樣本外 {SPLIT}~ {NDAY[True]} 天")
P_(f"- 全部候選每筆淨: 樣本內 {IS.net.mean()*100:.3f}% / 勝率 {(IS.net>0).mean()*100:.1f}%；樣本外 {OOS.net.mean()*100:.3f}% / 勝率 {(OOS.net>0).mean()*100:.1f}%")
P_("- 「隨機」= 當天候選平均；「t(比隨機)」= 選中那支比當天平均多多少的 t 值 (|t| > 2 才算不是運氣)")
P_(f"- 樣本外只有 {NDAY[True]} 天，每筆淨的標準誤約 {OOS.groupby('date').net.first().std()/np.sqrt(NDAY[True])*100:.2f}%，規則間差 0.3% 以內多半分不出來")

# ---------- 1. 同日 IC ----------
P_("\n## 1. 同一天的候選裡, 哪個特徵分得出好壞? (每日 Spearman IC 平均)")
P_("IC > 0 = 特徵越大越好空；t = IC 的 t 值。只用同一天內的相對大小, 大盤因子在這裡沒意義所以不列")
def daily_ic(df, c):
    x = df[[c, "net", "date"]].dropna()
    x = x[x.groupby("date")[c].transform("nunique") > 1]
    if x.empty: return pd.Series(dtype=float)
    r = x.groupby("date")[c].rank(); y = x.groupby("date").net.rank()
    z = pd.DataFrame({"d": x.date, "r": r, "y": y}).groupby("d").apply(lambda q: q.r.corr(q.y) if len(q) >= 3 else np.nan, include_groups=False).dropna()
    return z
icrows = []
for c, nm in FEAT.items():
    a, b = daily_ic(IS, c), daily_ic(OOS, c)
    if len(a) < 30 or len(b) < 30: continue
    icrows.append({"特徵": nm, "col": c, "IC IS": a.mean(), "t IS": a.mean()/a.std()*np.sqrt(len(a)), "IC OOS": b.mean(), "t OOS": b.mean()/b.std()*np.sqrt(len(b)),
                   "同號": np.sign(a.mean()) == np.sign(b.mean())})
IC = pd.DataFrame(icrows).set_index("特徵"); IC = IC.reindex(IC["t IS"].abs().sort_values(ascending=False).index)
P_(IC.drop(columns="col").round(3).to_markdown())

# ---------- 2. 單一特徵 ----------
P_("\n## 2. 每天選 1 支: 單一特徵規則 (方向用樣本內決定)")
rows = [row("模型排名第 1 (現行名單第 1)", pick(-T["rank"])), row("隨機選 1 支 (期望值)", RAND),
        row("事後最佳 (天花板, 做不到)", pick(T.net))]
for c, nm in FEAT.items():
    if c == "rank": continue
    hi, lo = pick(T[c]), pick(-T[c])
    best = (hi, "最大") if hi[~hi.oos].net.mean() >= lo[~lo.oos].net.mean() else (lo, "最小")
    rows.append(row(f"{nm} {best[1]}", best[0]))
S1 = pd.DataFrame(rows).set_index("規則")
fixed = S1.iloc[:3]; rest = S1.iloc[3:].sort_values("每筆淨% IS", ascending=False)
S1 = pd.concat([fixed, rest]); P_(S1[SHOW].round(2).to_markdown())

# ---------- 3. 組合分數 ----------
P_("\n## 3. 組合分數: 樣本內 IC 顯著 (|t| > 2) 的特徵, 當日百分位 × 正負號 加總")
sel = IC[IC["t IS"].abs() > 2]
def combo(cols_sign, w=None):
    sc = sum((T[f"pr_{c}"] - .5)*sg*(1 if w is None else w[c]) for c, sg in cols_sign.items()); return sc
cs = {r.col: np.sign(r["IC IS"]) for _, r in sel.iterrows()}
P_(f"- 選入特徵: {', '.join(f'{k}({FEAT[k]}, {'+' if v > 0 else '−'})' for k, v in cs.items()) or '無'}")
rows = [row("模型排名第 1", pick(-T["rank"]))]
if cs:
    rows.append(row("組合 (等權)", pick(combo(cs))))
    rows.append(row("組合 (IC 權重)", pick(combo(cs, {c: abs(IC.set_index('col').loc[c, 'IC IS']) for c in cs}))))
    top3 = list(sel.col[:3]); rows.append(row(f"組合 (樣本內最強 3 個: {', '.join(FEAT[c] for c in top3)})", pick(combo({c: cs[c] for c in top3}))))
    rows.append(row("組合等權 + 只在模型前 5 名裡選", pick(combo(cs).where(T["rank"] <= 5))))
    rows.append(row("組合等權 + 只在模型前 3 名裡選", pick(combo(cs).where(T["rank"] <= 3))))
S3 = pd.DataFrame(rows).set_index("規則"); P_(S3[SHOW].round(2).to_markdown())

# ---------- 4. LightGBM ----------
P_("\n## 4. LightGBM (2020~23 訓練 → 2024~ 選股) 與逐年滾動重訓")
try:
    import lightgbm as lgb
    X = lambda df: df[LGB_COLS].astype(float)
    base = dict(learning_rate=0.03, num_leaves=15, min_data_in_leaf=100, bagging_fraction=0.8, bagging_freq=1, feature_fraction=0.8, lambda_l2=5, verbose=-1, seed=0)
    def fit(kind, tr):
        if kind == "loss":   return lgb.train({**base, "objective": "binary"}, lgb.Dataset(X(tr), tr.loss.astype(int)), 300)
        if kind == "reg":    return lgb.train({**base, "objective": "huber", "alpha": 0.03}, lgb.Dataset(X(tr), tr.net - tr.day_mean), 300)
        if kind == "rank":
            tr = tr.sort_values(["date","rank"]); return lgb.train({**base, "objective": "lambdarank", "min_data_in_leaf": 50, "lambdarank_truncation_level": 3, "eval_at": [1]},
                                                                  lgb.Dataset(X(tr), tr.grade, group=tr.groupby("date", sort=True).size().values), 300)
    KINDS = {"loss": ("LGB 會不會賠 → 選最安全", -1), "reg": ("LGB 預測超額報酬 → 選最高", 1), "rank": ("LGB 同日排序 (lambdarank) → 選第 1", 1)}
    rows = [row("模型排名第 1", pick(-T["rank"]))]; imps = {}
    for k, (nm, sg) in KINDS.items():
        m = fit(k, IS); sc = sg*m.predict(X(T)); T[f"s_{k}"] = sc; rows.append(row(nm, pick(sc)))
        imps[nm] = pd.Series(m.feature_importance("gain"), index=LGB_COLS).sort_values(ascending=False).index[:6].tolist()
        # 逐年滾動: 2024 用 ~2023 訓練, 2025 用 ~2024, ...
        roll = pd.Series(np.nan, index=T.index)
        for y in sorted(T.date.dt.year.unique()):
            if y < SPLIT: continue
            mm = fit(k, T[T.date.dt.year < y]); ix = T.date.dt.year == y; roll[ix] = sg*mm.predict(X(T[ix]))
        T[f"r_{k}"] = roll; rows.append(row(nm + " (逐年重訓)", pick(roll.where(T.oos))))
    T["s_ens"] = sum(T.groupby("date")[f"s_{k}"].rank(pct=True) for k in KINDS); rows.append(row("三個 LGB 平均名次", pick(T.s_ens)))
    S4 = pd.DataFrame(rows).set_index("規則"); P_(S4[SHOW].round(2).to_markdown())
    P_("\n樣本內欄位是訓練資料本身 (一定好看), 只看 OOS。逐年重訓沒有樣本內數字。")
    for nm, v in imps.items(): P_(f"- {nm} 重要特徵: {', '.join(FEAT.get(c, c) for c in v)}")
    LGB_OK = True
except Exception as ex:
    P_(f"- LightGBM 無法執行: {ex}"); LGB_OK = False

# ---------- 5. 各年 ----------
P_("\n## 5. 各年每筆淨% (選中那支) — 看是不是每年都贏模型第 1 名")
cand = {"模型排名第 1": pick(-T["rank"]), "隨機選 1 支 (期望值)": RAND}
top_single = rest.index[:3].tolist()
for nm in top_single:
    c = next(k for k, v in FEAT.items() if nm.startswith(v + " ")); cand[nm] = pick(T[c] if nm.endswith("最大") else -T[c])
if cs: cand["組合 (等權)"] = pick(combo(cs))
if LGB_OK:
    for k, (nm, _) in KINDS.items(): cand[nm + " (逐年重訓)"] = pick(T[f"r_{k}"].where(T.oos, T[f"s_{k}"]))
yr = {nm: p.groupby(p.date.dt.year).net.mean()*100 for nm, p in cand.items()}
wr = {nm: p.groupby(p.date.dt.year).win.mean()*100 for nm, p in cand.items()}
P_(pd.DataFrame(yr).T.round(2).to_markdown()); P_("\n勝率%:"); P_(pd.DataFrame(wr).T.round(1).to_markdown())
P_("\n(LGB 逐年重訓那列的 2020~23 是樣本內模型, 只看 2024 以後)")

# ---------- 6. 該避開的: 每天「最不該選」 ----------
P_("\n## 6. 該避開的: 樣本內 IC 最強特徵的當日最差那支 vs 當日平均")
rows = []
for _, r in IC.head(5).iterrows():
    c = r.col; worst = pick(-np.sign(r["IC IS"])*T[c]); rows.append(row(f"每天 {r.name} {'最小' if r['IC IS'] > 0 else '最大'} 那支", worst))
P_(pd.DataFrame(rows).set_index("規則")[SHOW].round(2).to_markdown())

(OUT/"pick_one.md").write_text("\n".join(L)); print("\n".join(L))
