"""組訓練資料集: 每檔每日一列, 特徵皆為 09:30 (模型A) / 09:00 (模型B) 當下可得."""
import numpy as np, pandas as pd
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
d1 = pd.read_parquet(ROOT/"data/processed/d1_features.parquet").sort_values(["stock_id","date"])
m1 = pd.read_parquet(ROOT/"data/processed/m1d_features_all.parquet")
m1 = m1[["stock_id","date","m1_open","m1_close","n_bars","rv","vwap","t_first",
         "o05_h","o05_l","o05_c","o05_v","o15_h","o15_l","o15_c","o15_v","o30_h","o30_l","o30_c","o30_v","o30_dv","o30_n","m1_vol","last30_v","close_auction_v"]]

g = d1.groupby("stock_id", sort=False)
# ---- 前一日可得的 D1 特徵 (全部 shift(1) 或用 rolling 到前一日) ----
f = d1[["stock_id","date","market_type","open","close","prev_close"]].copy()
# 注意: d1_features 裡的 dv20/vol20 含當日 -> 這裡全部重算為「到前一日為止」
f["dv20"] = g.dollar_vol.transform(lambda s: s.shift(1).rolling(20).mean())
f["vol20"] = g.ret.transform(lambda s: s.shift(1).rolling(20).std())
f["gap"] = d1.gap
for c in ["ret","intra","range","gap","vol_ratio","up_lim","dn_lim","ret5","ret20"]:
    f[f"p_{c}"] = g[c].shift(1)
f["p_rv_d"] = None
f["ret2"] = g.close.shift(1)/g.close.shift(3) - 1
f["ret60"] = g.close.shift(1)/g.close.shift(61) - 1
f["hi20"] = g.high.transform(lambda s: s.shift(1).rolling(20).max()); f["lo20"] = g.low.transform(lambda s: s.shift(1).rolling(20).min())
f["pos20"] = (f.prev_close - f.lo20)/(f.hi20 - f.lo20)
f["ma5_dev"] = f.prev_close / g.close.transform(lambda s: s.shift(1).rolling(5).mean()) - 1
f["ma20_dev"] = f.prev_close / g.close.transform(lambda s: s.shift(1).rolling(20).mean()) - 1
f["ma60_dev"] = f.prev_close / g.close.transform(lambda s: s.shift(1).rolling(60).mean()) - 1
f["vol60"] = g.ret.transform(lambda s: s.shift(1).rolling(60).std())
f["vol5"] = g.ret.transform(lambda s: s.shift(1).rolling(5).std())
f["dv20_log"] = np.log1p(f.dv20)
f["dv_ratio5"] = g.dollar_vol.transform(lambda s: s.shift(1).rolling(5).mean()) / f.dv20
f["avgvol20_shares"] = g.volume.transform(lambda s: s.shift(1).rolling(20).mean())
f["uplim_cnt5"] = g.up_lim.transform(lambda s: s.shift(1).rolling(5).sum())
f["price_log"] = np.log(f.prev_close)
f["lo_price"] = (f.prev_close < 10).astype(int)
f["gap_abs"] = f.gap.abs()
# 興櫃 flag
f["is_tpex"] = (f.market_type=="tpex").astype(int)
# ---- 目標 ----
f["y_intra"] = d1.intra                                     # 模型B: 開→收

# ---- 合併 M1 開盤 30 分特徵 (模型A) ----
df = f.merge(m1, on=["stock_id","date"], how="left")
df["o30_ret"] = df.o30_c/df.open - 1
df["o15_ret"] = df.o15_c/df.open - 1
df["o05_ret"] = df.o05_c/df.open - 1
df["o30_range"] = (df.o30_h - df.o30_l)/df.prev_close
df["o30_pos"] = (df.o30_c - df.o30_l)/(df.o30_h - df.o30_l)                 # 0~1
df["o30_hi_vs_prev"] = df.o30_h/df.prev_close - 1
df["o30_lo_vs_prev"] = df.o30_l/df.prev_close - 1
df["o30_relvol"] = df.o30_v*1000 / df.avgvol20_shares                        # 開盤30分量 / 20日均日量
df["o05_relvol"] = df.o05_v*1000 / df.avgvol20_shares
df["o30_vwap_dev"] = df.o30_c / (df.o30_dv/df.o30_v) - 1
df["o30_n"] = df.o30_n
df["o30_momo"] = df.o30_c/df.o15_c - 1                                       # 09:15→09:30
df["y_rest"] = df.m1_close/df.o30_c - 1                                      # 模型A: 09:30→收盤
# ---- 市場橫截面特徵 (當日全市場平均, 只用已實現資訊) ----
liq = df[(df.dv20>=1e8)]
mk = liq.groupby("date").agg(mkt_gap=("gap","mean"), mkt_o30=("o30_ret","mean"), mkt_pret=("p_ret","mean"), mkt_ret5=("p_ret5","mean"),
                             mkt_breadth=("gap", lambda s:(s>0).mean()), mkt_uplim=("p_up_lim","mean"))
df = df.merge(mk, left_on="date", right_index=True, how="left")
df["gap_rel"] = df.gap - df.mkt_gap; df["o30_rel"] = df.o30_ret - df.mkt_o30
df["dow"] = df.date.dt.dayofweek
df = df.drop(columns=["p_rv_d"])

# ---- universe filter ----
univ = (df.dv20>=1e8) & df.market_type.isin(["twse","tpex"]) & ~df.stock_id.str.startswith("00") & df.prev_close.notna() & (df.prev_close>=5)
dsA = df[univ & df.o30_c.notna() & (df.o30_n>=5) & (df.n_bars>=60) & (df.t_first<=5) & df.y_rest.notna()].copy()
dsB = df[univ & df.y_intra.notna() & df.open.notna()].copy()
dsA.to_parquet(ROOT/"data/processed/dataset_A.parquet", index=False)
dsB.to_parquet(ROOT/"data/processed/dataset_B.parquet", index=False)
print("A:", dsA.shape, dsA.date.min().date(), dsA.date.max().date(), "| B:", dsB.shape)
print("y_rest mean/std %:", dsA.y_rest.mean()*100, dsA.y_rest.std()*100, "| y_intra:", dsB.y_intra.mean()*100, dsB.y_intra.std()*100)
