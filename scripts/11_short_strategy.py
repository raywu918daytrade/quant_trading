"""當沖池做空策略回測 (真實一點的成交假設)."""
import glob, numpy as np, pandas as pd
from pathlib import Path
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
ROOT = Path(__file__).resolve().parent.parent; OUT = ROOT/"stats"
COSTS = [0.0015, 0.0023, 0.0030, 0.0040]; BASE = 0.0023
PERIODS = {"探索 24Q4-25Q2": ("2024-10-01","2025-06-30"), "驗證 25Q3-26Q1": ("2025-07-01","2026-03-31"), "鎖定 26Q2+": ("2026-04-01","2026-12-31")}

ent = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(str(ROOT/"data/processed/m1_entry/*.parquet")))])
m1d = pd.concat([pd.read_parquet(f, columns=["stock_id","date","m1_close","close_auction_c","m1_high"]) for f in sorted(glob.glob(str(ROOT/"data/processed/m1_daily/*.parquet")))])
B = pd.read_parquet(ROOT/"data/processed/dataset_B_pool.parquet"); A = pd.read_parquet(ROOT/"data/processed/dataset_A_pool.parquet")
pB = pd.read_parquet(ROOT/"data/processed/preds_B_pool.parquet").rename(columns={"pred":"predB"})[["stock_id","date","predB"]]
pA = pd.read_parquet(ROOT/"data/processed/preds_A_pool.parquet").rename(columns={"pred":"predA"})[["stock_id","date","predA"]]
tu = pd.read_parquet(ROOT/"data/raw/meta/tick_universe.parquet"); tier_both = set(tu[tu.day_trade_tier=="both"].stock_id)

d = B[["stock_id","date","prev_close","open","gap","atr20","dv20_prev","mkt_gap","mkt_pret","p_up_lim","p_ret5","pos20","p_vol_ratio"]].merge(
    A[["stock_id","date","o30_ret","o30_relvol","o30_pos","o30_range","o30_c","o30_hi_vs_prev"]], on=["stock_id","date"], how="left")
d = d.merge(pB, on=["stock_id","date"], how="left").merge(pA, on=["stock_id","date"], how="left").merge(ent, on=["stock_id","date"], how="left").merge(m1d, on=["stock_id","date"], how="left")
d["exit"] = d.close_auction_c.fillna(d.m1_close)
d["in_tier"] = d.stock_id.isin(tier_both)
d = d[d.date >= "2024-10-01"].copy()      # 模型預測從 2024Q4 起, 統一期間
d["period"] = None
for k,(a,b) in PERIODS.items(): d.loc[(d.date>=a)&(d.date<=b), "period"] = k
d["rankB"] = d.groupby("date").predB.rank(method="first"); d["rankA"] = d.groupby("date").predA.rank(method="first")
d["pctB"] = d.groupby("date").predB.rank(pct=True); d["pctA"] = d.groupby("date").predA.rank(pct=True)
d["rank_atr"] = d.groupby("date").atr20.rank(ascending=False, method="first")

def trades(sel, entry_col, extra_filter=None):
    x = d[sel & d[entry_col].notna() & d.exit.notna()].copy()
    x["entry"] = x[entry_col]
    x = x[x.entry > x.prev_close*0.905]                      # 不在跌停價放空
    if extra_filter == "above_prev": x = x[x.entry >= x.prev_close]   # 平盤以上才可放空
    if extra_filter == "tier": x = x[x.in_tier]
    x["gross"] = x.entry/x.exit - 1                           # 做空: 賣高買低
    hi = x["e00_maxhigh_after"] if entry_col=="e00_vwap" else x["e30_maxhigh_after"]
    x["hit_limit_up"] = hi >= x.prev_close*1.095
    x["mae"] = hi/x.entry - 1                                 # 最大不利幅度
    return x

def nw_t(s, lag=5):
    s = s.dropna().values; n = len(s)
    if n < 10: return np.nan
    e = s - s.mean(); v = (e**2).sum()/n
    for l in range(1, lag+1):
        w = 1 - l/(lag+1); v += 2*w*(e[l:]*e[:-l]).sum()/n
    return s.mean()/np.sqrt(v/n)

def summarize(x, cost=BASE):
    x = x.copy(); x["net"] = x.gross - cost
    daily = x.groupby("date").net.mean()
    pf = x.net[x.net>0].sum()/(-x.net[x.net<0].sum()) if (x.net<0).any() else np.nan
    return {"交易數":len(x), "日均檔數":len(x)/x.date.nunique(), "每筆淨%":x.net.mean()*100, "每筆中位%":x.net.median()*100, "勝率":(x.net>0).mean(),
            "PF":pf, "日淨%":daily.mean()*100, "日std%":daily.std()*100, "Sharpe":daily.mean()/daily.std()*np.sqrt(245), "NW-t":nw_t(daily),
            "MDD%":((1+daily).cumprod()/(1+daily).cumprod().cummax()-1).min()*100, "觸漲停%":x.hit_limit_up.mean()*100, "觸漲停筆均淨%":x[x.hit_limit_up].net.mean()*100 if x.hit_limit_up.any() else np.nan,
            "最差1%筆":x.net.quantile(.01)*100}

STRATS = {
    "S1 模型B 最看空10檔 (09:00)":        (lambda: d.rankB<=10, "e00_vwap"),
    "S2 模型B 最看空20檔 (09:00)":        (lambda: d.rankB<=20, "e00_vwap"),
    "S3 模型A 最看空10檔 (09:30)":        (lambda: d.rankA<=10, "e30_vwap"),
    "S4 模型A 最看空20檔 (09:30)":        (lambda: d.rankA<=20, "e30_vwap"),
    "S5 規則: 跳空>+3% (09:00)":          (lambda: d.gap>0.03, "e00_vwap"),
    "S6 規則: 開盤30分放量>50%且漲>1.5% (09:30)": (lambda: (d.o30_relvol>0.5)&(d.o30_ret>0.015), "e30_vwap"),
    "S7 基準: ATR 最高20檔 (09:00)":      (lambda: d.rank_atr<=20, "e00_vwap"),
    "S8 模型B底20% ∩ 跳空>0 ∩ 大盤跳空>-0.5%": (lambda: (d.pctB<=0.2)&(d.gap>0)&(d.mkt_gap>-0.005), "e00_vwap"),
    "S9 模型B 最看空20檔, 大盤跳空<-1%不做":  (lambda: (d.rankB<=20)&(d.mkt_gap>=-0.01), "e00_vwap"),
    "S10 模型A底20% ∩ 開盤30分放量>30%":   (lambda: (d.pctA<=0.2)&(d.o30_relvol>0.3), "e30_vwap"),
}
L = []; P = lambda s="": L.append(str(s))
P("# 當沖池做空策略回測\n")
P("## 假設")
P("- 池: 前一日止 20日均成交額≥5億 & ATR20≥3%, 上市櫃非ETF, 股價≥5。期間 2024-10-01 ~ 2026-10-22 (模型預測皆為按季 walk-forward 樣本外)。")
P("- 進場價 = 09:00~09:02 (或 09:30~09:32) 三分鐘成交量加權均價 (VWAP)；出場價 = 13:30 收盤競價價 (無則用最後一根 close)。做空報酬 = 進場/出場 - 1。")
P(f"- 來回成本基準 {BASE*100:.2f}% (手續費折扣後約 0.08% + 當沖證交稅 0.15%)，並列 0.15/0.23/0.30/0.40% 敏感度。不含滑價以外的借券費。")
P("- 排除進場價在跌停附近 (< 昨收×0.905) 的交易。'平盤以上' 變體: 只做進場價 ≥ 昨收的交易 (模擬平盤下不得放空)。'tier' 變體: 只做 tick_universe 中 day_trade_tier=both 的股票 (目前快照, 有存活偏差, 僅供參考)。")
P("- 觸漲停% = 進場後最高價曾 ≥ 昨收×1.095 的交易比例 (漲停鎖死可能無法回補, 此處仍假設能以收盤價回補, 實際風險更大)。")
P("- 期間切分: 探索 2024Q4~2025Q2 / 驗證 2025Q3~2026Q1 / 鎖定 2026Q2 之後。鎖定期數字只應看一次。NW-t = 日報酬均值的 Newey-West(lag5) t 值。")
P("")

all_tr = {}
rows = []
for name,(sel,ecol) in STRATS.items():
    x = trades(sel(), ecol); all_tr[name] = x
    r = summarize(x); r["策略"] = name; rows.append(r)
tab = pd.DataFrame(rows).set_index("策略")
P(f"## 1. 全期間總覽 (成本 {BASE*100:.2f}%)"); P(tab.round(3).to_markdown()); P("")

P("## 2. 分期間 (日淨% / Sharpe / NW-t / 日均檔數)")
rows = []
for name,x in all_tr.items():
    r = {"策略":name}
    for k in PERIODS:
        xx = x[x.period==k]
        if len(xx)==0: continue
        s = summarize(xx); r[f"{k} 日淨%"] = s["日淨%"]; r[f"{k} Sharpe"] = s["Sharpe"]; r[f"{k} NW-t"] = s["NW-t"]
    rows.append(r)
P(pd.DataFrame(rows).set_index("策略").round(2).to_markdown()); P("")

P("## 3. 成本敏感度 (日淨%, 全期間)")
rows = []
for name,x in all_tr.items():
    r = {"策略":name}
    for c in COSTS: r[f"{c*100:.2f}%"] = summarize(x, c)["日淨%"]
    rows.append(r)
P(pd.DataFrame(rows).set_index("策略").round(3).to_markdown()); P("")

P("## 4. 執行限制變體 (S2 / S4 / S6 / S8)")
rows = []
for name in ["S2 模型B 最看空20檔 (09:00)","S4 模型A 最看空20檔 (09:30)","S6 規則: 開盤30分放量>50%且漲>1.5% (09:30)","S8 模型B底20% ∩ 跳空>0 ∩ 大盤跳空>-0.5%"]:
    sel, ecol = STRATS[name]
    for var, lab in [(None,"原始"),("above_prev","平盤以上"),("tier","tier=both")]:
        s = summarize(trades(sel(), ecol, var)); s["策略"] = name; s["變體"] = lab; rows.append(s)
P(pd.DataFrame(rows).set_index(["策略","變體"])[["交易數","日均檔數","每筆淨%","勝率","PF","日淨%","Sharpe","NW-t","MDD%","觸漲停%"]].round(3).to_markdown()); P("")

P("## 5. 停損變體 (若進場後最高價曾超過進場價 +X%, 視為在 +X%+0.3% 滑價處停損; 不知道順序, 偏保守)")
rows = []
for name in ["S2 模型B 最看空20檔 (09:00)","S4 模型A 最看空20檔 (09:30)","S8 模型B底20% ∩ 跳空>0 ∩ 大盤跳空>-0.5%"]:
    x = all_tr[name]
    for sl in [None, 0.02, 0.03, 0.05]:
        y = x.copy()
        if sl is not None:
            stopped = y.mae >= sl
            y.loc[stopped, "gross"] = -(sl + 0.003)
        s = summarize(y); s["策略"] = name; s["停損"] = "無" if sl is None else f"{sl*100:.0f}%"; s["停損比例%"] = (y.mae>=sl).mean()*100 if sl else 0
        rows.append(s)
P(pd.DataFrame(rows).set_index(["策略","停損"])[["每筆淨%","勝率","PF","日淨%","Sharpe","MDD%","停損比例%","最差1%筆"]].round(3).to_markdown()); P("")

P("## 6. S2 交易的條件剖面 (每筆淨%, 看哪些情況該避開)")
x = all_tr["S2 模型B 最看空20檔 (09:00)"].copy(); x["net"] = x.gross - BASE
prof = {
    "大盤跳空<-1%": x.mkt_gap<-0.01, "大盤跳空 -1~0": (x.mkt_gap>=-0.01)&(x.mkt_gap<0), "大盤跳空 0~1%": (x.mkt_gap>=0)&(x.mkt_gap<0.01), "大盤跳空>1%": x.mkt_gap>=0.01,
    "個股跳空<0": x.gap<0, "個股跳空 0~3%": (x.gap>=0)&(x.gap<0.03), "個股跳空>3%": x.gap>=0.03,
    "前一日漲停": x.p_up_lim==1, "ATR 3-5%": x.atr20<0.05, "ATR 5-7%": (x.atr20>=0.05)&(x.atr20<0.07), "ATR>7%": x.atr20>=0.07,
    "成交額 5-20億": x.dv20_prev<20e8, "成交額>20億": x.dv20_prev>=20e8, "週一": x.date.dt.dayofweek==0, "週五": x.date.dt.dayofweek==4,
}
rows = [{"條件":k, "n":int(m.sum()), "每筆淨%":x[m].net.mean()*100, "中位%":x[m].net.median()*100, "勝率":(x[m].net>0).mean(), "觸漲停%":x[m].hit_limit_up.mean()*100} for k,m in prof.items()]
P(pd.DataFrame(rows).set_index("條件").round(3).to_markdown()); P("")

(OUT/"short_strategy.md").write_text("\n".join(L)); print("\n".join(L))
fig, ax = plt.subplots(1,2, figsize=(16,5))
for name in ["S2 模型B 最看空20檔 (09:00)","S4 模型A 最看空20檔 (09:30)","S5 規則: 跳空>+3% (09:00)","S6 規則: 開盤30分放量>50%且漲>1.5% (09:30)","S7 基準: ATR 最高20檔 (09:00)","S8 模型B底20% ∩ 跳空>0 ∩ 大盤跳空>-0.5%"]:
    x = all_tr[name]; daily = (x.gross-BASE).groupby(x.date).mean()
    ((1+daily).cumprod()-1).plot(ax=ax[0], label=name[:22])
ax[0].legend(fontsize=7); ax[0].axhline(0,color="k",lw=.5); ax[0].set_title("short strategies: cumulative net (cost 0.23%)")
for k,(a,b) in PERIODS.items(): ax[0].axvline(pd.Timestamp(a), color="grey", ls="--", lw=.6)
x = all_tr["S2 模型B 最看空20檔 (09:00)"]; (x.gross-BASE).clip(-.12,.12).hist(bins=80, ax=ax[1]); ax[1].set_title("S2 per-trade net return"); ax[1].axvline(0,color="k",lw=.5)
plt.tight_layout(); plt.savefig(OUT/"short_strategy.png", dpi=110)
