"""當沖池: 每日動態選 (前一日止) dv20>=5億 & ATR20>=3% & 上市櫃非ETF. 池內統計 + 產生 dataset_{A,B}_pool."""
import numpy as np, pandas as pd
from pathlib import Path
from scipy.stats import spearmanr
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
ROOT = Path(__file__).resolve().parent.parent; OUT = ROOT/"stats"
DV_TH, ATR_TH = 5e8, 0.03
d1 = pd.read_parquet(ROOT/"data/processed/d1_features.parquet").sort_values(["stock_id","date"])
g = d1.groupby("stock_id", sort=False)
d1["atr20"] = g.range.transform(lambda s: s.shift(1).rolling(20).mean())     # 前一日止
d1["dv20_prev"] = g.dollar_vol.transform(lambda s: s.shift(1).rolling(20).mean())
d1["dv5_prev"] = g.dollar_vol.transform(lambda s: s.shift(1).rolling(5).mean())
pool_cols = d1[["stock_id","date","atr20","dv20_prev","dv5_prev"]]
tu = pd.read_parquet(ROOT/"data/raw/meta/tick_universe.parquet")
tier_both = set(tu[tu.day_trade_tier=="both"].stock_id)

L = []; P = lambda s="": L.append(str(s))
P(f"# 當沖池統計 (每日動態: 前一日止 20日均成交額 ≥ {DV_TH/1e8:.0f} 億 且 ATR20 ≥ {ATR_TH*100:.0f}%, 上市櫃, 非ETF, 股價≥5)\n")
res = {}
for M in ["A","B"]:
    ds = pd.read_parquet(ROOT/f"data/processed/dataset_{M}.parquet").merge(pool_cols, on=["stock_id","date"], how="left")
    ds["in_pool"] = (ds.dv20_prev>=DV_TH)&(ds.atr20>=ATR_TH)
    pool = ds[ds.in_pool].copy()
    pool.to_parquet(ROOT/f"data/processed/dataset_{M}_pool.parquet", index=False)
    res[M] = (ds, pool)
    per_day = pool.groupby("date").size()
    P(f"- 資料集 {M}: 全體流動股 {len(ds)} 列 → 池內 {len(pool)} 列 ({pool.stock_id.nunique()} 檔曾入池)；每日池大小 median {per_day.median():.0f} (p10 {per_day.quantile(.1):.0f} / p90 {per_day.quantile(.9):.0f})")
ds, pool = res["B"]
last_pool = pool[pool.date==pool.date.max()]
P(f"- 最新日 ({pool.date.max().date()}) 池內 {len(last_pool)} 檔，其中在 tick_universe (tier=both) 的有 {last_pool.stock_id.isin(tier_both).sum()} 檔；tick_universe 561 檔中有 {len(tier_both & set(last_pool.stock_id))} 檔在池內")
P("")

# ---- 1. 池 vs 全體: 日內期望值 ----
P("## 1. 池內 vs 全體流動股 (模型B資料集, 09:00開→13:30收, %)")
def summ(x):
    y = x.y_intra; gp = x.gap
    return pd.Series({"n":len(x), "gap mean":gp.mean()*100, "intra mean":y.mean()*100, "intra median":y.median()*100, "intra std":y.std()*100,
                      "P(intra>0)":(y>0).mean(), "intra p5":y.quantile(.05)*100, "intra p95":y.quantile(.95)*100,
                      "做空每筆期望(毛)":-y.mean()*100, "盈虧比(空)":(-y[y<0].mean())/(y[y>0].mean())})
P(pd.DataFrame({"全體流動股":summ(ds), "當沖池":summ(pool), "池外":summ(ds[~ds.in_pool])}).T.round(3).to_markdown()); P("")
yr = pool.groupby(pool.date.dt.year).apply(lambda x: pd.Series({"n":len(x),"gap%":x.gap.mean()*100,"intra%":x.y_intra.mean()*100,"P(intra>0)":(x.y_intra>0).mean(),"intra std%":x.y_intra.std()*100}))
P("- 池內各年:"); P(yr.round(3).to_markdown()); P("")

# ---- 2. 池內再分層: 波動 × 成交額 ----
P("## 2. 池內分層 (09:00→收盤 平均 %, 看效應隨波動/成交額怎麼變)")
pool["atr_bin"] = pd.cut(pool.atr20, [0.03,0.04,0.05,0.07,1], labels=["3-4%","4-5%","5-7%",">7%"])
pool["dv_bin"] = pd.cut(pool.dv20_prev/1e8, [5,10,20,50,1e9], labels=["5-10億","10-20億","20-50億",">50億"])
pv = pool.pivot_table(index="atr_bin", columns="dv_bin", values="y_intra", aggfunc="mean", observed=True)*100
pn = pool.pivot_table(index="atr_bin", columns="dv_bin", values="y_intra", aggfunc="size", observed=True)
P("- ATR × 成交額 → 日內平均 %:"); P(pv.round(3).to_markdown()); P("- 樣本數:"); P(pn.to_markdown()); P("")
P("- 僅依 ATR 分層:"); P((pool.groupby("atr_bin", observed=True).y_intra.agg(["mean","median","std","size"])*[100,100,100,1]).round(3).to_markdown()); P("")

# ---- 3. 池內條件統計 (當沖決策用) ----
P("## 3. 池內條件期望值 (09:00→收盤, %)")
pool["gap_bin"] = pd.cut(pool.gap, [-1,-.05,-.03,-.01,.01,.03,.05,.07,1], labels=["<-5","-5~-3","-3~-1","-1~1","1~3","3~5","5~7",">7"])
gb = pool.groupby("gap_bin", observed=True).agg(n=("y_intra","size"), intra_mean=("y_intra","mean"), intra_med=("y_intra","median"), p_pos=("y_intra", lambda s:(s>0).mean()))
gb[["intra_mean","intra_med"]]*=100
P("- 依跳空 (%):"); P(gb.round(3).to_markdown()); P("")
cond = {
    "前一日漲停": pool.p_up_lim==1, "前一日跌停": pool.p_dn_lim==1,
    "前一日量比>2": pool.p_vol_ratio>2, "前5日累積>15%": pool.p_ret5>0.15, "前5日累積<-10%": pool.p_ret5<-0.10,
    "前20日累積>30%": pool.p_ret20>0.30, "收在20日高點區(pos20>0.9)": pool.pos20>0.9, "收在20日低點區(pos20<0.1)": pool.pos20<0.1,
    "大盤昨日>+1%": pool.mkt_pret>0.01, "大盤昨日<-1%": pool.mkt_pret<-0.01, "今日大盤平均跳空>+1%": pool.mkt_gap>0.01, "今日大盤平均跳空<-1%": pool.mkt_gap<-0.01,
    "週一": pool.dow==0, "週五": pool.dow==4,
}
rows = [{"條件":k, "n":int(m.sum()), "intra mean%":pool[m].y_intra.mean()*100, "intra median%":pool[m].y_intra.median()*100, "P(>0)":(pool[m].y_intra>0).mean(), "gap mean%":pool[m].gap.mean()*100} for k,m in cond.items()]
P(pd.DataFrame(rows).set_index("條件").round(3).to_markdown()); P("")

# ---- 4. 池內 09:30 之後 (模型A資料集) ----
dsA, poolA = res["A"]
P("## 4. 池內開盤 30 分之後 (模型A資料集, 09:30→13:30 收, %)")
P(f"- 池內 y_rest mean {poolA.y_rest.mean()*100:.3f}%, median {poolA.y_rest.median()*100:.3f}%, P(>0) {(poolA.y_rest>0).mean():.3f}  (全體流動股 {dsA.y_rest.mean()*100:.3f}%)")
poolA["o30_bin"] = pd.cut(poolA.o30_ret, [-1,-.05,-.03,-.015,-.005,.005,.015,.03,.05,1], labels=["<-5","-5~-3","-3~-1.5","-1.5~-.5","-.5~.5",".5~1.5","1.5~3","3~5",">5"])
poolA["rv_bin"] = pd.cut(poolA.o30_relvol, [0,.1,.2,.3,.5,10], labels=["<10%","10-20%","20-30%","30-50%",">50%"])
g1 = poolA.groupby("o30_bin", observed=True).agg(n=("y_rest","size"), rest_mean=("y_rest","mean"), rest_med=("y_rest","median"), p_pos=("y_rest", lambda s:(s>0).mean()))
g1[["rest_mean","rest_med"]]*=100
P("- 依開盤30分報酬 (%):"); P(g1.round(3).to_markdown()); P("")
g2 = poolA.groupby("rv_bin", observed=True).agg(n=("y_rest","size"), rest_mean=("y_rest","mean"), rest_med=("y_rest","median"), p_pos=("y_rest", lambda s:(s>0).mean()))
g2[["rest_mean","rest_med"]]*=100
P("- 依開盤30分量 / 20日均日量 (相對量):"); P(g2.round(3).to_markdown()); P("")
pv2 = poolA.pivot_table(index="o30_bin", columns="rv_bin", values="y_rest", aggfunc="mean", observed=True)*100
P("- 開盤30分報酬 × 相對量 → 其後平均 %:"); P(pv2.round(3).to_markdown()); P("")
P(f"- 池內 09:30 收在開盤30分區間位置 o30_pos: 高位(>0.8) 其後 {poolA[poolA.o30_pos>0.8].y_rest.mean()*100:.3f}%，低位(<0.2) 其後 {poolA[poolA.o30_pos<0.2].y_rest.mean()*100:.3f}%")
ics = {f: poolA.groupby("date").apply(lambda x: spearmanr(x[f], x.y_rest, nan_policy="omit")[0] if len(x)>20 else np.nan).mean()
       for f in ["gap","o30_ret","o30_relvol","o30_range","o30_pos","o30_hi_vs_prev","vol20","atr20","dv20_prev","p_ret5","p_ret20","p_vol_ratio","ma20_dev"]}
P("- 池內單變數日均 rank-IC vs 09:30→收盤:"); P(pd.Series(ics).sort_values().round(4).to_frame("IC").to_markdown()); P("")

(OUT/"daytrade_pool.md").write_text("\n".join(L)); print("\n".join(L))
fig, ax = plt.subplots(1,3, figsize=(17,5))
pv.plot(kind="bar", ax=ax[0]); ax[0].axhline(0,color="k",lw=.5); ax[0].set_title("pool: ATR x turnover -> intraday mean %")
gb.intra_mean.plot(kind="bar", ax=ax[1]); ax[1].axhline(0,color="k",lw=.5); ax[1].set_title("pool: gap bin -> intraday mean %")
g1.rest_mean.plot(kind="bar", ax=ax[2]); ax[2].axhline(0,color="k",lw=.5); ax[2].set_title("pool: o30 ret bin -> 09:30->close mean %")
plt.tight_layout(); plt.savefig(OUT/"daytrade_pool.png", dpi=110)
