"""漲停鎖死的真實代價: 收盤仍鎖漲停 -> 假設當日回補不到, 隔日 09:00~09:02 VWAP 回補 + 0.2% 額外成本."""
import glob, numpy as np, pandas as pd
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent; OUT = ROOT/"stats"
BASE = 0.0023; PENALTY = 0.002
PERIODS = {"探索": ("2024-10-01","2025-06-30"), "驗證": ("2025-07-01","2026-03-31"), "鎖定": ("2026-04-01","2026-12-31")}
ent = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(str(ROOT/"data/processed/m1_entry/*.parquet")))])
m1d = pd.concat([pd.read_parquet(f, columns=["stock_id","date","m1_close","close_auction_c","m1_high","last30_l","last30_h","close_auction_v","m1_vol"]) for f in sorted(glob.glob(str(ROOT/"data/processed/m1_daily/*.parquet")))])
B = pd.read_parquet(ROOT/"data/processed/dataset_B_pool.parquet")[["stock_id","date","prev_close","open","gap","atr20","mkt_gap"]]
pB = pd.read_parquet(ROOT/"data/processed/preds_B_pool.parquet").rename(columns={"pred":"predB"})[["stock_id","date","predB"]]
d1 = pd.read_parquet(ROOT/"data/processed/d1.parquet", columns=["stock_id","date","open","close"]).sort_values(["stock_id","date"])
d1["next_open"] = d1.groupby("stock_id").open.shift(-1); d1["next_date"] = d1.groupby("stock_id").date.shift(-1)
nxt = ent[["stock_id","date","e00_vwap"]].rename(columns={"date":"next_date","e00_vwap":"next_e00_vwap"})
d = B.merge(pB, on=["stock_id","date"]).merge(ent, on=["stock_id","date"], how="left").merge(m1d, on=["stock_id","date"], how="left")
d = d.merge(d1[["stock_id","date","next_open","next_date"]], on=["stock_id","date"], how="left").merge(nxt, on=["stock_id","next_date"], how="left")
d = d[d.date>="2024-10-01"].copy()
d["exit"] = d.close_auction_c.fillna(d.m1_close)
d["rankB"] = d.groupby("date").predB.rank(method="first"); d["pctB"] = d.groupby("date").predB.rank(pct=True)
d["period"] = None
for k,(a,b) in PERIODS.items(): d.loc[(d.date>=a)&(d.date<=b),"period"] = k
lim = d.prev_close*1.0975
d["touch_lim"] = d.e00_maxhigh_after >= d.prev_close*1.095
d["close_at_lim"] = d.exit >= lim                                  # 收盤鎖漲停
d["last30_locked"] = d.last30_l >= lim                             # 最後 30 分鐘都鎖在漲停
d["cover_next"] = d.next_e00_vwap.fillna(d.next_open)

STR = {"S1 模型B 最看空10檔": d.rankB<=10, "S2 模型B 最看空20檔": d.rankB<=20,
       "S8 模型B底20% ∩ 跳空>0 ∩ 大盤跳空>-0.5%": (d.pctB<=0.2)&(d.gap>0)&(d.mkt_gap>-0.005),
       "S9 模型B 最看空20檔, 大盤跳空<-1%不做": (d.rankB<=20)&(d.mkt_gap>=-0.01)}
L=[]; P=lambda s="": L.append(str(s))
P("# 漲停鎖死真實代價\n")
P(f"- 定義: 收盤鎖漲停 = 13:30 出場價 ≥ 昨收×1.0975；最後30分鎖死 = 13:00 後最低價仍 ≥ 昨收×1.0975。")
P(f"- 情境 '保守': 收盤鎖漲停的交易視為當日回補不到，改以隔日 09:00~09:02 VWAP 回補 (無則隔日開盤價)，並加 {PENALTY*100:.1f}% 額外成本 (違約/借券/處理費)。")
P(f"- 情境 '極端': 進場後只要曾觸及漲停 (≥ 昨收×1.095) 就視為回補不到，同樣隔日回補。這是上界，實際不會這麼糟。")
P("")
rows=[]; detail=[]
for name, sel in STR.items():
    x = d[sel & d.e00_vwap.notna() & d.exit.notna()].copy()
    x = x[x.e00_vwap > x.prev_close*0.905]
    x["gross"] = x.e00_vwap/x.exit - 1
    x["gross_cons"] = np.where(x.close_at_lim & x.cover_next.notna(), x.e00_vwap/x.cover_next - 1 - PENALTY, x.gross)
    x["gross_ext"]  = np.where(x.touch_lim & x.cover_next.notna(), x.e00_vwap/x.cover_next - 1 - PENALTY, x.gross)
    for col, lab in [("gross","原回測"),("gross_cons","保守: 收盤鎖漲停隔日回補"),("gross_ext","極端: 觸漲停即隔日回補")]:
        net = x[col]-BASE; daily = net.groupby(x.date).mean()
        r = {"策略":name,"情境":lab,"每筆淨%":net.mean()*100,"勝率":(net>0).mean(),"日淨%":daily.mean()*100,"Sharpe":daily.mean()/daily.std()*np.sqrt(245),
             "MDD%":((1+daily).cumprod()/(1+daily).cumprod().cummax()-1).min()*100,"最差1%筆":net.quantile(.01)*100,"最差筆":net.min()*100}
        for k in PERIODS:
            dd = net[x.period==k].groupby(x.date[x.period==k]).mean(); r[f"{k} 日淨%"] = dd.mean()*100
        rows.append(r)
    lk = x[x.close_at_lim]
    detail.append({"策略":name,"交易數":len(x),"觸漲停%":x.touch_lim.mean()*100,"收盤鎖漲停%":x.close_at_lim.mean()*100,"最後30分鎖死%":x.last30_locked.mean()*100,
                   "鎖漲停筆 原回測均淨%":(lk.gross-BASE).mean()*100,"鎖漲停筆 隔日回補均淨%":(lk.gross_cons-BASE).mean()*100,
                   "鎖漲停筆 隔日再漲停比例%":((lk.cover_next/lk.prev_close)>=1.19).mean()*100 if len(lk) else np.nan,
                   "鎖漲停筆 隔日開盤相對當日收盤均%":(lk.cover_next/lk.exit-1).mean()*100 if len(lk) else np.nan})
P("## 1. 鎖漲停發生率與代價"); P(pd.DataFrame(detail).set_index("策略").round(2).to_markdown()); P("")
P("## 2. 三種情境下的策略績效"); P(pd.DataFrame(rows).set_index(["策略","情境"]).round(3).to_markdown()); P("")
# 哪些條件下鎖漲停最常見 (S2)
x = d[(d.rankB<=20) & d.e00_vwap.notna() & d.exit.notna()].copy(); x["gross"] = x.e00_vwap/x.exit - 1
x["gap_bin"] = pd.cut(x.gap, [-1,0,0.03,0.06,1], labels=["跳空<0","0~3%","3~6%",">6%"])
x["atr_bin"] = pd.cut(x.atr20, [0,0.05,0.07,1], labels=["ATR3-5","5-7",">7"])
pv = x.pivot_table(index="gap_bin", columns="atr_bin", values="close_at_lim", aggfunc="mean", observed=True)*100
P("## 3. S2 交易中 '收盤鎖漲停' 比例 (%) — 跳空 × ATR"); P(pv.round(1).to_markdown())
pn = x.pivot_table(index="gap_bin", columns="atr_bin", values="gross", aggfunc=lambda s:(s.mean()-BASE)*100, observed=True)
P("- 對應每筆淨% (原回測):"); P(pn.round(2).to_markdown()); P("")
# 規避規則: 排除跳空>6% 之後
for th in [0.06, 0.04]:
    y = d[(d.rankB<=20)&(d.gap<=th)&(d.mkt_gap>=-0.01)&d.e00_vwap.notna()&d.exit.notna()].copy()
    y = y[y.e00_vwap>y.prev_close*0.905]; y["gross"]=y.e00_vwap/y.exit-1
    y["gc"] = np.where(y.close_at_lim & y.cover_next.notna(), y.e00_vwap/y.cover_next-1-PENALTY, y.gross)
    for col,lab in [("gross","原"),("gc","保守")]:
        net=y[col]-BASE; daily=net.groupby(y.date).mean()
        P(f"- S9 再排除 個股跳空>{th*100:.0f}%：{lab} 日淨 {daily.mean()*100:.3f}%，Sharpe {daily.mean()/daily.std()*np.sqrt(245):.2f}，收盤鎖漲停 {y.close_at_lim.mean()*100:.2f}%，日均檔數 {len(y)/y.date.nunique():.1f}")
(OUT/"limit_up_risk.md").write_text("\n".join(L)); print("\n".join(L))

# ---- 定案候選: S9 + 排除個股跳空>6%，分期間 (保守情境) ----
P("")
P("## 4. 定案候選 S9' = 模型B 最看空20檔, 大盤跳空≥-1%, 個股跳空≤6% — 分期間")
P("- 注意: 排除跳空>6% 這條規則是看過全期間 (含鎖定期) 之後才加的，鎖定期對它已不再是乾淨的樣本外。")
y = d[(d.rankB<=20)&(d.gap<=0.06)&(d.mkt_gap>=-0.01)&d.e00_vwap.notna()&d.exit.notna()].copy()
y = y[y.e00_vwap>y.prev_close*0.905]; y["gross"]=y.e00_vwap/y.exit-1
y["gc"] = np.where(y.close_at_lim & y.cover_next.notna(), y.e00_vwap/y.cover_next-1-PENALTY, y.gross)
rows=[]
for k in list(PERIODS)+["全期間"]:
    yy = y if k=="全期間" else y[y.period==k]
    for col,lab in [("gross","原"),("gc","保守")]:
        net=yy[col]-BASE; daily=net.groupby(yy.date).mean()
        rows.append({"期間":k,"情境":lab,"日淨%":daily.mean()*100,"Sharpe":daily.mean()/daily.std()*np.sqrt(245),"每筆淨%":net.mean()*100,"勝率":(net>0).mean(),
                     "MDD%":((1+daily).cumprod()/(1+daily).cumprod().cummax()-1).min()*100,"收盤鎖漲停%":yy.close_at_lim.mean()*100,"日均檔數":len(yy)/yy.date.nunique()})
P(pd.DataFrame(rows).set_index(["期間","情境"]).round(3).to_markdown())
yr = y.assign(net=y.gc-BASE).groupby(y.date.dt.to_period("Q")).apply(lambda g: pd.Series({"日淨%(保守)":g.net.groupby(g.date).mean().mean()*100, "鎖漲停%":g.close_at_lim.mean()*100, "n":len(g)}))
P("- 逐季 (保守情境):"); P(yr.round(3).to_markdown())
(OUT/"limit_up_risk.md").write_text("\n".join(L)); print("\n".join(L[-40:]))
