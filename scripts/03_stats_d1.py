"""D1 統計: 報酬分布 / 跳空 / 日內 / 波動 / 流動性 / 自相關 / 星期效應 / 漲跌停."""
import pandas as pd, numpy as np
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT/"stats"; OUT.mkdir(exist_ok=True)
pd.set_option("display.width", 200)

df = pd.read_parquet(ROOT/"data/processed/d1.parquet")
df["is_etf"] = df.stock_id.str.startswith("00")
df = df.sort_values(["stock_id","date"])
g = df.groupby("stock_id", sort=False)
df["prev_close"] = g.close.shift(1)
df["ret"] = df.close/df.prev_close - 1                 # 收對收
df["gap"] = df.open/df.prev_close - 1                  # 隔夜跳空
df["intra"] = df.close/df.open - 1                     # 開到收
df["range"] = (df.high-df.low)/df.prev_close           # 日振幅
df["up_lim"] = df.ret >= 0.095                          # 近似漲停 (10%)
df["dn_lim"] = df.ret <= -0.095
df["dollar_vol"] = df.close*df.volume
df["ret_fwd1"] = g.ret.shift(-1)
df["intra_fwd1"] = g.intra.shift(-1)
df["gap_fwd1"] = g.gap.shift(-1)
df["ret5"] = g.close.shift(0)/g.close.shift(5) - 1
df["ret20"] = g.close.shift(0)/g.close.shift(20) - 1
df["vol20"] = g.ret.transform(lambda s: s.rolling(20).std())
df["dv20"] = g.dollar_vol.transform(lambda s: s.rolling(20).mean())
df["vol_ratio"] = df.volume / g.volume.transform(lambda s: s.rolling(20).mean())
df.to_parquet(ROOT/"data/processed/d1_features.parquet", index=False)

lines = []
P = lambda s="": lines.append(str(s))
P("# D1 統計摘要 (2023-10 ~ 2026-10)\n")

# ---- 1. 覆蓋 ----
P("## 1. 資料覆蓋")
per_day = df.groupby("date").stock_id.nunique()
P(f"- 交易日數: {df.date.nunique()}，股票數: {df.stock_id.nunique()} (ETF/00開頭: {df[df.is_etf].stock_id.nunique()})")
P(f"- 每日有資料的檔數: min {per_day.min()} / median {per_day.median():.0f} / max {per_day.max()}")
P(f"- market_type 分布 (股票-日):\n{df.market_type.value_counts(dropna=False).to_string()}")
P(f"- open 缺值: {df.open.isna().sum()}，volume=0: {(df.volume==0).sum()}")
P("")

stk = df[~df.is_etf & df.prev_close.notna() & df.ret.notna()].copy()
clean = stk[stk.market_type.isin(["twse","tpex"])]

# ---- 2. 報酬分布 ----
P("## 2. 日報酬分布 (上市+上櫃, 排除ETF/興櫃)")
def desc(s):
    s = s.dropna()
    q = s.quantile([.01,.05,.25,.5,.75,.95,.99])
    return pd.Series({"n":len(s),"mean%":s.mean()*100,"std%":s.std()*100,"skew":s.skew(),"kurt":s.kurt(),
        "p1%":q[.01]*100,"p5%":q[.05]*100,"p25%":q[.25]*100,"p50%":q[.5]*100,"p75%":q[.75]*100,"p95%":q[.95]*100,"p99%":q[.99]*100,
        "P(>0)":(s>0).mean()})
tab = pd.DataFrame({k:desc(clean[k]) for k in ["ret","gap","intra","range"]}).T
P(tab.round(3).to_markdown()); P("")
P("### 興櫃 (emerging) 對照")
P(pd.DataFrame({k:desc(stk[stk.market_type=="emerging"][k]) for k in ["ret","gap","intra","range"]}).T.round(3).to_markdown()); P("")

# ---- 3. 漲跌停 ----
P("## 3. 漲跌停頻率 (|ret|>=9.5%, 上市上櫃)")
P(f"- 漲停比例: {clean.up_lim.mean()*100:.3f}% ，跌停比例: {clean.dn_lim.mean()*100:.3f}%")
lim_next = clean.groupby("up_lim")[["ret_fwd1","gap_fwd1","intra_fwd1"]].mean()*100
P("- 漲停後隔日 (mean %):"); P(lim_next.round(3).to_markdown())
dlim_next = clean.groupby("dn_lim")[["ret_fwd1","gap_fwd1","intra_fwd1"]].mean()*100
P("- 跌停後隔日 (mean %):"); P(dlim_next.round(3).to_markdown()); P("")

# ---- 4. 跳空 vs 日內 ----
P("## 4. 跳空 → 當日日內報酬 (開→收) 的關係 (上市上櫃)")
clean["gap_bin"] = pd.cut(clean.gap, [-1,-.05,-.03,-.01,-.003,.003,.01,.03,.05,1])
gb = clean.groupby("gap_bin", observed=True).agg(n=("intra","size"), intra_mean=("intra","mean"), intra_med=("intra","median"),
        p_intra_pos=("intra", lambda s:(s>0).mean()), ret_fwd1=("ret_fwd1","mean"))
gb[["intra_mean","intra_med","ret_fwd1"]] *= 100
P(gb.round(3).to_markdown()); P("")
P(f"- corr(gap, intra) = {clean[['gap','intra']].corr().iloc[0,1]:.4f}  (負值=跳空傾向回補)")
P(f"- corr(ret, ret_fwd1) = {clean[['ret','ret_fwd1']].corr().iloc[0,1]:.4f}")
P(f"- corr(intra, intra_fwd1) = {clean[['intra','intra_fwd1']].corr().iloc[0,1]:.4f}")
P(f"- corr(ret5, ret_fwd1) = {clean[['ret5','ret_fwd1']].corr().iloc[0,1]:.4f}")
P(f"- corr(ret20, ret_fwd1) = {clean[['ret20','ret_fwd1']].corr().iloc[0,1]:.4f}")
P(f"- corr(vol_ratio, |ret_fwd1|) = {clean[['vol_ratio']].assign(a=clean.ret_fwd1.abs()).corr().iloc[0,1]:.4f}")
P("")

# ---- 5. 自相關 (pooled, 以每檔 demean) ----
P("## 5. 日報酬自相關 (每檔去均值後 pooled)")
ac = {}
for lag in [1,2,3,5,10]:
    s = clean.groupby("stock_id").ret.shift(lag)
    ac[lag] = clean.ret.corr(s)
P(pd.Series(ac, name="autocorr").round(4).to_markdown()); P("")

# ---- 6. 星期效應 ----
P("## 6. 星期效應 (上市上櫃, mean %)")
clean["dow"] = clean.date.dt.dayofweek
dow = clean.groupby("dow")[["ret","gap","intra","range"]].mean()*100
dow.index = ["Mon","Tue","Wed","Thu","Fri"][:len(dow)]
P(dow.round(3).to_markdown()); P("")

# ---- 7. 流動性 ----
P("## 7. 流動性 (20日均成交金額, 億元)")
last = clean[clean.date==clean.date.max()]
dv = (last.dv20/1e8).describe(percentiles=[.1,.25,.5,.75,.9,.95,.99])
P(dv.round(3).to_markdown())
for th in [0.5,1,3,10]:
    P(f"- 20日均成交金額 >= {th} 億: {(last.dv20>=th*1e8).sum()} 檔")
P("")

# ---- 8. 波動 vs 流動性分組 ----
P("## 8. 依成交金額分組的日內統計 (最新日分組, 全期間 mean %)")
last_dv = last.set_index("stock_id").dv20
clean["dv_q"] = pd.qcut(clean.stock_id.map(last_dv), 5, labels=["Q1 小","Q2","Q3","Q4","Q5 大"])
q = clean.groupby("dv_q", observed=True).agg(n=("ret","size"), ret_std=("ret","std"), range_mean=("range","mean"),
        intra_std=("intra","std"), gap_std=("gap","std"), up_lim=("up_lim","mean"))
for c in ["ret_std","range_mean","intra_std","gap_std","up_lim"]: q[c]*=100
P(q.round(3).to_markdown()); P("")

# ---- 9. 市場層級: 等權指數 ----
P("## 9. 市場 (等權) 時間序列")
mkt = clean.groupby("date").agg(ret=("ret","mean"), intra=("intra","mean"), gap=("gap","mean"), range=("range","mean"),
        breadth=("ret", lambda s:(s>0).mean()), up_lim=("up_lim","sum"), dn_lim=("dn_lim","sum"))
mkt.to_csv(OUT/"d1_market_daily.csv")
yr = mkt.groupby(mkt.index.year).agg(ret_mean=("ret","mean"), ret_std=("ret","std"), intra_mean=("intra","mean"),
        gap_mean=("gap","mean"), range_mean=("range","mean"), up_lim_per_day=("up_lim","mean"), dn_lim_per_day=("dn_lim","mean"))
for c in ["ret_mean","ret_std","intra_mean","gap_mean","range_mean"]: yr[c]*=100
P(yr.round(3).to_markdown()); P("")
P(f"- 等權日報酬 lag1 自相關: {mkt.ret.autocorr(1):.4f}；|ret| lag1 自相關 (波動群聚): {mkt.ret.abs().autocorr(1):.4f}")
P(f"- 等權日內 vs 跳空 corr: {mkt[['gap','intra']].corr().iloc[0,1]:.4f}")

# ---- charts ----
fig, ax = plt.subplots(2,2, figsize=(13,9))
clean.ret.clip(-.1,.1).hist(bins=100, ax=ax[0,0]); ax[0,0].set_title("daily ret (clipped ±10%)"); ax[0,0].set_yscale("log")
clean.gap.clip(-.1,.1).hist(bins=100, ax=ax[0,1]); ax[0,1].set_title("overnight gap"); ax[0,1].set_yscale("log")
(1+mkt.ret).cumprod().plot(ax=ax[1,0], label="close-close"); (1+mkt.intra).cumprod().plot(ax=ax[1,0], label="open-close"); (1+mkt.gap).cumprod().plot(ax=ax[1,0], label="gap only"); ax[1,0].legend(); ax[1,0].set_title("equal-weight cumulative")
mkt.range.rolling(20).mean().plot(ax=ax[1,1]); ax[1,1].set_title("avg daily range (20d MA)")
plt.tight_layout(); plt.savefig(OUT/"d1_overview.png", dpi=110)

fig, ax = plt.subplots(figsize=(8,4))
gb.intra_mean.plot(kind="bar", ax=ax); ax.set_title("gap bin -> mean intraday ret (%)"); ax.axhline(0,color="k",lw=.5)
plt.tight_layout(); plt.savefig(OUT/"d1_gap_vs_intra.png", dpi=110)

(OUT/"d1_summary.md").write_text("\n".join(lines))
print("\n".join(lines))
