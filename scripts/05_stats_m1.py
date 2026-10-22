"""M1 統計: 分鐘 profile / 開盤區間 / 開盤30分→後續 / 分時段報酬 / 收盤競價."""
import glob, numpy as np, pandas as pd
from pathlib import Path
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
ROOT = Path(__file__).resolve().parent.parent; OUT = ROOT/"stats"
lines = []; P = lambda s="": lines.append(str(s))
P("# M1 統計摘要 (2023-10 ~ 2026-10)\n")

m = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(str(ROOT/"data/processed/m1_daily/*.parquet")))], ignore_index=True)
d1 = pd.read_parquet(ROOT/"data/processed/d1_features.parquet")
df = m.merge(d1[["stock_id","date","market_type","prev_close","open","close","high","low","volume","ret","gap","intra","range","dv20","vol20","ret_fwd1","gap_fwd1","intra_fwd1","up_lim","dn_lim"]], on=["stock_id","date"], how="inner")
df["is_etf"] = df.stock_id.str.startswith("00")
df = df[~df.is_etf & df.prev_close.notna()]
df["liquid"] = df.dv20 >= 1e8
m_all = df; df = df[df.liquid & df.market_type.isin(["twse","tpex"])].copy()
P(f"- M1 stock-days: {len(m)}；與 D1 合併後 {len(m_all)}；流動性(≥1億)+上市櫃子集: {len(df)} ({df.stock_id.nunique()} 檔)")
P(f"- n_bars (liquid) median {df.n_bars.median():.0f}, p10 {df.n_bars.quantile(.1):.0f}；第一根在 09:00 的比例 {(df.t_first==0).mean():.3f}；最後一根 13:30 的比例 {(df.t_last==270).mean():.3f}")
P(f"- 13:30 後有 bar 的 stock-days: {m.after_n.notna().sum()} / {len(m)}")
P("")

# ---- 1. 分鐘 profile ----
P("## 1. 分鐘級 profile (全市場聚合)")
prof = pd.concat([pd.read_parquet(f) for f in glob.glob(str(ROOT/"data/processed/m1_profile/*.parquet"))])
pr = prof.groupby("tod").agg(n=("n","sum"), vol=("vol","sum"), dv=("dv","sum"), abs_lr=("abs_lr", "mean"), range_sum=("range_sum","sum"))
pr = pr[(pr.index>=540)&(pr.index<=810)]
pr["vol_share%"] = pr.vol/pr.vol.sum()*100; pr["mean_range%"] = pr.range_sum/pr.n*100
pr["hhmm"] = [f"{t//60:02d}:{t%60:02d}" for t in pr.index]
top = pr.sort_values("vol_share%", ascending=False).head(8)[["hhmm","vol_share%","abs_lr","mean_range%"]]
P("- 成交量占比最高的分鐘:"); P(top.round(4).to_markdown(index=False))
blk = pr.copy(); blk["blk"] = pd.cut(blk.index, [539,545,570,600,660,720,780,805,809,810], labels=["09:00-05","09:06-30","09:31-10:00","10-11","11-12","12-13","13:01-25","13:26-29","13:30 競價"])
b = blk.groupby("blk", observed=True).agg(vol_share=("vol_share%","sum"), abs_lr_mean=("abs_lr","mean"), range_mean=("mean_range%","mean"), minutes=("n","size"))
b["vol_share_per_min"] = b.vol_share/b.minutes
P("- 分時段:"); P(b.round(4).to_markdown()); P("")

# ---- 2. 開盤區間 ----
P("## 2. 開盤區間 (liquid 子集)")
df["day_range"] = df.m1_high - df.m1_low
for w in ["o05","o15","o30","o60"]:
    df[f"{w}_rng_frac"] = (df[f"{w}_h"]-df[f"{w}_l"])/df.day_range
    df[f"{w}_vol_frac"] = df[f"{w}_v"]/df.m1_vol
    df[f"{w}_ret"] = df[f"{w}_c"]/df.m1_open - 1
tab = pd.DataFrame({w: {"range占全日比 median": df[f"{w}_rng_frac"].median(), "range占全日比 mean": df[f"{w}_rng_frac"].mean(),
                        "量占全日比 median": df[f"{w}_vol_frac"].median(),
                        "日高在此區間內 P": (df.t_high < int(w[1:])).mean(), "日低在此區間內 P": (df.t_low < int(w[1:])).mean()} for w in ["o05","o15","o30","o60"]}).T
P(tab.round(3).to_markdown()); P("")
P(f"- 日高發生在 09:00 整(開盤即高點) P = {(df.t_high==0).mean():.3f}；日低在 09:00 P = {(df.t_low==0).mean():.3f}")
P(f"- 日高在 13:30 收盤 P = {(df.t_high>=270).mean():.3f}；日低在 13:30 P = {(df.t_low>=270).mean():.3f}")
th = pd.cut(df.t_high, [-1,0,15,30,60,120,180,240,269,300], labels=["09:00","-09:15","-09:30","-10:00","-11:00","-12:00","-13:00","-13:29","13:30"]).value_counts(normalize=True).sort_index()
tl = pd.cut(df.t_low,  [-1,0,15,30,60,120,180,240,269,300], labels=["09:00","-09:15","-09:30","-10:00","-11:00","-12:00","-13:00","-13:29","13:30"]).value_counts(normalize=True).sort_index()
P("- 日高/日低 發生時段分布:"); P(pd.DataFrame({"t_high":th,"t_low":tl}).round(3).to_markdown()); P("")

# ---- 3. 開盤30分報酬 → 之後 ----
P("## 3. 開盤後 N 分報酬 → 剩餘時段報酬 (liquid 子集, 動能 or 反轉?)")
for w, lab in [("o05","5分"),("o15","15分"),("o30","30分"),("o60","60分")]:
    df[f"rest_{w}"] = df.m1_close/df[f"{w}_c"] - 1
    r = df[[f"{w}_ret", f"rest_{w}"]].dropna()
    ic = r.corr().iloc[0,1]
    # cross-sectional (每日) rank corr 平均
    cs = df.groupby("date").apply(lambda x: x[f"{w}_ret"].corr(x[f"rest_{w}"], method="spearman")).mean()
    P(f"- {lab}: corr(開盤{lab}報酬, 其後→收盤報酬) pooled = {ic:.4f}，日均橫截面 rank-IC = {cs:.4f}")
df["o30_bin"] = pd.cut(df.o30_ret, [-1,-.05,-.03,-.015,-.005,.005,.015,.03,.05,1])
gb = df.groupby("o30_bin", observed=True).agg(n=("rest_o30","size"), rest_mean=("rest_o30","mean"), rest_med=("rest_o30","median"), p_pos=("rest_o30", lambda s:(s>0).mean()), ret_fwd1=("ret_fwd1","mean"))
for c in ["rest_mean","rest_med","ret_fwd1"]: gb[c]*=100
P("- 依開盤30分報酬分組 → 09:30→收盤 報酬 (%):"); P(gb.round(3).to_markdown()); P("")
# gap + o30 雙向
df["gap_bin"] = pd.cut(df.gap, [-1,-.03,-.01,.01,.03,1], labels=["gap<-3%","-3~-1","-1~1","1~3","gap>3%"])
df["o30_bin2"] = pd.cut(df.o30_ret, [-1,-.02,-.005,.005,.02,1], labels=["o30<-2%","-2~-.5","-.5~.5",".5~2","o30>2%"])
pv = df.pivot_table(index="gap_bin", columns="o30_bin2", values="rest_o30", aggfunc="mean", observed=True)*100
P("- gap × 開盤30分報酬 → 09:30→收盤 平均報酬 (%):"); P(pv.round(3).to_markdown())
pvn = df.pivot_table(index="gap_bin", columns="o30_bin2", values="rest_o30", aggfunc="size", observed=True)
P("- 樣本數:"); P(pvn.to_markdown()); P("")
# 開盤30分突破
df["brk_up"] = df.m1_close > df.o30_h; df["brk_dn"] = df.m1_close < df.o30_l
P(f"- 收盤站上開盤30分高點 P = {df.brk_up.mean():.3f}；收盤跌破開盤30分低點 P = {df.brk_dn.mean():.3f}；都沒有 (收在區間內) P = {(~df.brk_up & ~df.brk_dn).mean():.3f}")
P(f"- 開盤30分量占比 > 30% 時, 其後報酬 mean = {df[df.o30_vol_frac>.3].rest_o30.mean()*100:.3f}% (n={int((df.o30_vol_frac>.3).sum())}) vs 其他 {df[df.o30_vol_frac<=.3].rest_o30.mean()*100:.3f}%")
P("")

# ---- 4. 分時段報酬 ----
P("## 4. 分時段平均報酬 (liquid 子集, %)")
seg = pd.DataFrame({
    "09:00開(vs昨收) gap": df.gap, "09:00→09:30": df.o30_c/df.m1_open-1, "09:30→10:00": df.c10/df.o30_c-1,
    "10→11": df.c11/df.c10-1, "11→12": df.c12/df.c11-1, "12→13": df.c13/df.c12-1, "13:00→13:30收": df.m1_close/df.c13-1,
})
# 13:29 price: use last bar before 810
pre = df.c13  # placeholder
st = pd.DataFrame({"mean%": seg.mean()*100, "median%": seg.median()*100, "std%": seg.std()*100, "P(>0)": (seg>0).mean(), "skew": seg.skew()})
P(st.round(3).to_markdown()); P("")
yr = seg.groupby(df.date.dt.year).mean()*100
P("- 各年分時段 mean %:"); P(yr.round(3).to_markdown()); P("")

# ---- 5. 收盤競價 / 波動 ----
P("## 5. 收盤競價與波動 (liquid 子集)")
df["ca_share"] = df.close_auction_v/df.m1_vol
P(f"- 13:30 收盤競價量占全日: median {df.ca_share.median():.3f}, mean {df.ca_share.mean():.3f}")
df["ca_ret"] = df.close_auction_c/df.c13 - 1
P(f"- 13:00→收盤競價報酬 與 隔日 gap corr = {df[['ca_ret','gap_fwd1']].corr().iloc[0,1]:.4f}；與隔日 intra corr = {df[['ca_ret','intra_fwd1']].corr().iloc[0,1]:.4f}")
df["rv_d"] = np.sqrt(df.rv)
P(f"- 日內 realized vol (sqrt(sum lr^2)) median {df.rv_d.median()*100:.3f}%，與 (H-L)/prev_close corr = {df[['rv_d','range']].corr().iloc[0,1]:.3f}")
P(f"- 日內 rv_d 與 隔日 |ret| corr = {df.rv_d.corr(df.ret_fwd1.abs()):.4f}；vol20 與 隔日 |ret| corr = {df.vol20.corr(df.ret_fwd1.abs()):.4f}")
P(f"- VWAP 位置: 收盤 > VWAP 的 P = {(df.m1_close>df.vwap).mean():.3f}；(close/vwap-1) mean = {(df.m1_close/df.vwap-1).mean()*100:.3f}%")
P(f"- corr(close/vwap-1, ret_fwd1) = {(df.m1_close/df.vwap-1).corr(df.ret_fwd1):.4f}")

df.drop(columns=[c for c in ["o30_bin","gap_bin","o30_bin2"] if c in df]).to_parquet(ROOT/"data/processed/m1d_features_liquid.parquet", index=False)
m_all.to_parquet(ROOT/"data/processed/m1d_features_all.parquet", index=False)

# ---- charts ----
fig, ax = plt.subplots(2,2, figsize=(14,9))
pr["vol_share%"].plot(ax=ax[0,0]); ax[0,0].set_title("volume share by minute (%)"); ax[0,0].set_yscale("log")
pr["abs_lr"].plot(ax=ax[0,1]); ax[0,1].set_title("mean |1-min log ret| by minute")
pd.DataFrame({"t_high":th,"t_low":tl}).plot(kind="bar", ax=ax[1,0]); ax[1,0].set_title("time of day high / low")
gb.rest_mean.plot(kind="bar", ax=ax[1,1]); ax[1,1].axhline(0,color="k",lw=.5); ax[1,1].set_title("o30 ret bin -> 09:30->close mean ret (%)")
plt.tight_layout(); plt.savefig(OUT/"m1_overview.png", dpi=110)
(OUT/"m1_summary.md").write_text("\n".join(lines)); print("\n".join(lines))
