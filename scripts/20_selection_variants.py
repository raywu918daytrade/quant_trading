"""選股規則比較 (2020~2026, M1 精確成交): 純模型 / 純 ATR / 混合排名 / ATR 前 30 內用模型 / 模型前 30 內用 ATR.
其餘規則與 18 相同 (跳空≤4%, 大盤跳空≥−1%, 股價≤300, 非週一, +8% 強制回補, 300 萬每檔 60 萬填滿). 輸出 stats/selection_variants.md"""
import sys, glob, numpy as np, pandas as pd
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent)); import features as F
ROOT = Path(__file__).resolve().parent.parent; OUT = ROOT/"stats"
BASE, SLIP, K, PEN, CAP, PER_STOCK, MAX_N, MAX_PX, GAP_MAX, MKT_MIN = 0.0023, 0.001, 0.08, 0.07, 3_000_000, 600_000, 10, 300, 0.04, -0.01
fs = sorted(glob.glob(str(ROOT/"data/raw/d1/*.parquet")))
d1 = pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True); d1["date"] = pd.to_datetime(d1.date)
d1 = d1.drop_duplicates(["stock_id","date"]).sort_values(["stock_id","date"]).reset_index(drop=True)
info = pd.read_parquet(ROOT/"data/raw/meta/info.parquet"); d1 = d1.merge(info[["stock_id","market_type"]], on="stock_id", how="left")
f = F.build_B(F.d1_base(d1)); nxt = d1[["stock_id","date","open"]].sort_values(["stock_id","date"]); nxt["next_open"] = nxt.groupby("stock_id").open.shift(-1)
f = f.merge(nxt[["stock_id","date","next_open"]], on=["stock_id","date"], how="left")
pool = f[F.universe_mask(f) & (f.dv20_prev>=5e8) & (f.atr20>=0.03) & f.y_intra.notna() & f.open.notna()].copy()
pr = pd.concat([pd.read_parquet(ROOT/"data/processed/preds_B_pool_ext.parquet")[["date","stock_id","pred"]],
                pd.read_parquet(ROOT/"data/processed/preds_B_pool.parquet").query("date>='2024-10-01'")[["date","stock_id","pred"]]], ignore_index=True)
ext = pd.concat([pd.read_parquet(x) for x in sorted(glob.glob(str(ROOT/"data/processed/m1_entry_ext/*.parquet")))], ignore_index=True)
d = pool.merge(pr, on=["stock_id","date"]).merge(ext, on=["stock_id","date"])
d = d[d.e01_vwap.notna()].copy(); d["entry"] = d.e01_vwap; d["exit"] = d.close_auction_c.fillna(d.m1_close); d = d[d.exit.notna()].copy()
d["lock"] = d.exit >= d.prev_close*1.0975; trig = (d.e01_maxhigh_after >= d.prev_close*(1+K)) & (d.entry < d.prev_close*(1+K))
d["gross"] = np.where(trig, d.entry/(d.prev_close*(1+K)*(1+SLIP*3))-1, np.where(d.lock & d.next_open.notna(), d.entry/d.next_open-1-PEN, d.entry/d.exit-1))
d["net"] = d.gross - BASE; d["dow"] = d.date.dt.dayofweek
e = d[(d.gap<=GAP_MAX)&(d.mkt_gap>=MKT_MIN)&(d.entry>d.prev_close*0.905)&(d.entry<=MAX_PX)&(d.dow!=0)].copy()
g = e.groupby("date")
e["r_model"] = g.pred.rank(method="first"); e["r_atr"] = g.atr20.rank(ascending=False, method="first")
e["r_blend"] = (g.pred.rank(pct=True) + g.atr20.rank(ascending=False, pct=True)); e["r_blend"] = e.groupby("date").r_blend.rank(method="first")
e["r_vol60"] = g.vol60.rank(ascending=False, method="first")
def within(primary_rank, n_pre, secondary):
    sub = e[e[primary_rank]<=n_pre].copy(); sub["rank"] = sub.groupby("date")[secondary].rank(ascending=(secondary=="pred"), method="first"); return sub
VARIANTS = {
    "純模型 (現行)": e.assign(rank=e.r_model),
    "純 ATR20 (最高)": e.assign(rank=e.r_atr),
    "純 vol60 (最高)": e.assign(rank=e.r_vol60),
    "混合: 模型與 ATR 排名平均": e.assign(rank=e.r_blend),
    "ATR 前 30 內用模型挑": within("r_atr", 30, "pred"),
    "ATR 前 20 內用模型挑": within("r_atr", 20, "pred"),
    "模型前 30 內用 ATR 挑": within("r_model", 30, "atr20"),
}
def alloc(y):
    recs = []
    for dt, gg in y.sort_values(["date","rank"]).groupby("date", sort=False):
        cash = CAP; amts = []; nets = []
        for _, r in gg.iterrows():
            lc = r.entry*1000
            if cash < lc: continue
            lots = int(min(PER_STOCK, cash)//lc)
            if lots < 1: continue
            a = lots*lc; cash -= a; amts.append(a); nets.append(r.net)
            if len(amts) >= MAX_N: break
        recs.append((dt, len(amts), sum(a*n for a, n in zip(amts, nets))/CAP))
    return pd.DataFrame(recs, columns=["date","n","ret"]).set_index("date").ret
all_days = pd.Series(sorted(d.date.unique()))
L = []; P = lambda s="": L.append(str(s))
P("# 選股規則比較 2020~2026 (M1 精確成交, 300 萬配置, 其餘規則同定案)\n")
P("- 每種規則: 合格池內依該排名往下填 (每檔上限 60 萬, 最多 10 檔)。不交易日計 0。每筆淨% = 選入股票的平均 (未加權)。\n")
summary = []; yearly = {}
for name, y in VARIANTS.items():
    top = y[y["rank"]<=MAX_N]; R = alloc(top).reindex(all_days, fill_value=0.0)
    def st(s, t):
        return {"每筆淨%": t.net.mean()*100, "勝率": (t.net>0).mean(), "日淨%": s.mean()*100, "Sharpe": s.mean()/s.std()*np.sqrt(245), "MDD%": ((1+s).cumprod()/(1+s).cumprod().cummax()-1).min()*100,
                "最差月%": s.groupby(s.index.to_period("M")).apply(lambda x: ((1+x).prod()-1)*100).min(), "負月比例": (s.groupby(s.index.to_period("M")).apply(lambda x: (1+x).prod()-1)<0).mean()}
    summary.append({"規則": name, **st(R, top), "2020~24Q3 每筆%": top[top.date<"2024-10-01"].net.mean()*100, "2024Q4~ 每筆%": top[top.date>="2024-10-01"].net.mean()*100})
    yearly[name] = pd.Series({yr: st(R[R.index.year==yr], top[top.date.dt.year==yr]) for yr in sorted(R.index.year.unique())})
P("## 1. 全期間 (2020-01 ~ 2026-10)"); P(pd.DataFrame(summary).set_index("規則").round(3).to_markdown()); P("")
for metric in ["每筆淨%", "Sharpe", "MDD%"]:
    tab = pd.DataFrame({name: {yr: v[metric] for yr, v in ys.items()} for name, ys in yearly.items()})
    P(f"## 各年 {metric}"); P(tab.round(2).to_markdown()); P("")
# 穩定度: 各年 Sharpe 的最小值與平均
rows = [{"規則": name, "年Sharpe 平均": np.mean([v["Sharpe"] for v in ys.values]), "年Sharpe 最小": min(v["Sharpe"] for v in ys.values), "年每筆% 最小": min(v["每筆淨%"] for v in ys.values),
         "贏過純模型的年數": sum(1 for yr, v in ys.items() if v["每筆淨%"] > yearly["純模型 (現行)"][yr]["每筆淨%"])} for name, ys in yearly.items()]
P("## 穩定度"); P(pd.DataFrame(rows).set_index("規則").round(3).to_markdown())
(OUT/"selection_variants.md").write_text("\n".join(L)); print("\n".join(L))
