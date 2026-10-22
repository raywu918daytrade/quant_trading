"""集中度: 每日做 N 檔 / 加開高條件 / 分數門檻, 在定案規則下的取捨."""
import glob, numpy as np, pandas as pd
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent; OUT = ROOT/"stats"; BASE=0.0023; SLIP=0.001; K=0.08; PEN=0.07
PERIODS = {"探索": ("2024-10-01","2025-06-30"), "驗證": ("2025-07-01","2026-03-31"), "鎖定": ("2026-04-01","2026-12-31")}
ent = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(str(ROOT/"data/processed/m1_entry/*.parquet")))])
m1d = pd.concat([pd.read_parquet(f, columns=["stock_id","date","m1_close","close_auction_c"]) for f in sorted(glob.glob(str(ROOT/"data/processed/m1_daily/*.parquet")))])
B = pd.read_parquet(ROOT/"data/processed/dataset_B_pool.parquet")[["stock_id","date","prev_close","gap","mkt_gap","atr20"]]
pB = pd.read_parquet(ROOT/"data/processed/preds_B_pool.parquet")[["stock_id","date","pred"]]
d1 = pd.read_parquet(ROOT/"data/processed/d1.parquet", columns=["stock_id","date","open"]).sort_values(["stock_id","date"]); d1["next_open"]=d1.groupby("stock_id").open.shift(-1)
d = B.merge(pB, on=["stock_id","date"]).merge(ent, on=["stock_id","date"]).merge(m1d, on=["stock_id","date"]).merge(d1[["stock_id","date","next_open"]], on=["stock_id","date"])
d = d[(d.date>="2024-10-01")&d.e00_vwap.notna()].copy()
d["entry"]=d.e00_vwap; d["exit"]=d.close_auction_c.fillna(d.m1_close); d["lock"]=d.exit>=d.prev_close*1.0975
d["hi"]=d.e00_maxhigh_after/d.prev_close-1
# 定案出場: +8% 強制回補; 殘餘鎖死隔日+7%
trig = d.hi>=K; above = d.entry>=d.prev_close*(1+K)
d["gross"] = np.where(trig & ~above, d.entry/(d.prev_close*(1+K)*(1+SLIP*3))-1, np.where(d.lock, d.entry/d.next_open-1-PEN, d.entry/d.exit-1))
d["net"] = d.gross-BASE
elig = (d.gap<=0.04)&(d.mkt_gap>=-0.01)&(d.entry>d.prev_close*0.905)
d["period"]=None
for k,(a,b) in PERIODS.items(): d.loc[(d.date>=a)&(d.date<=b),"period"]=k
# 分數排名只在合格者中排
e = d[elig].copy(); e["rank"]=e.groupby("date").pred.rank(method="first"); e["pct"]=e.groupby("date").pred.rank(pct=True)
e["rank_up"] = e[e.gap>0].groupby("date").pred.rank(method="first"); e["rank_up"]=e.rank_up.fillna(1e9)

def stats(y, label):
    daily = y.net.groupby(y.date).mean(); days_total = e.date.nunique()
    r = {"規則":label,"日均檔數":len(y)/days_total,"有交易天數比":y.date.nunique()/days_total,"每筆淨%":y.net.mean()*100,"每筆中位%":y.net.median()*100,"勝率":(y.net>0).mean(),
         "日淨%":daily.mean()*100,"日std%":daily.std()*100,"Sharpe":daily.mean()/daily.std()*np.sqrt(245),"MDD%":((1+daily).cumprod()/(1+daily).cumprod().cummax()-1).min()*100,
         "最差日%":daily.min()*100,"日勝率":(daily>0).mean()}
    for k in PERIODS:
        dd = y[y.period==k].net.groupby(y[y.period==k].date).mean(); r[f"{k}日淨%"]=dd.mean()*100
    return r
rows=[]
for N in [3,5,8,10,15,20,30]: rows.append(stats(e[e["rank"]<=N], f"最看空 {N} 檔"))
for N in [5,10,15]: rows.append(stats(e[e.rank_up<=N], f"只做開高(gap>0) 最看空 {N} 檔"))
for th in [-0.004,-0.006,-0.008,-0.010]:
    y = e[(e.pred<=th)&(e["rank"]<=20)]; rows.append(stats(y, f"分數 ≤ {th*100:.1f}% 且在前20 (檔數浮動)"))
for th in [-0.006,-0.008]:
    y = e[(e.pred<=th)&(e["rank"]<=10)]; rows.append(stats(y, f"分數 ≤ {th*100:.1f}% 且在前10 (檔數浮動)"))
tab = pd.DataFrame(rows).set_index("規則")
L=["# 集中度取捨 (定案規則: 跳空≤4%, 大盤跳空≥-1%, +8%強制回補, 殘餘鎖死7%罰金, 成本0.23%)\n",
   "- 日淨% = 當天所選股票等權的平均淨報酬 = 滿倉資金的日報酬率。檔數少 → 每筆期望高但日波動大。日均檔數以全部有資格交易的天數為分母。\n",
   tab.round(3).to_markdown(), "",
   "## 模型分數 vs 實際 (合格交易, 依分數十分位)"]
e["dec"]=np.minimum((e.pct*10).apply(np.ceil), 10).astype(int)
dec = e.groupby("dec").agg(n=("net","size"), pred_mean=("pred","mean"), net_mean=("net","mean"), win=("net", lambda s:(s>0).mean()))
dec[["pred_mean","net_mean"]]*=100
L.append(dec.round(3).to_markdown())
L.append(f"\n- 合格股票每日平均 {e.groupby('date').size().mean():.0f} 檔；分數最低 5 檔 vs 第 6~10 檔 vs 11~20 檔 每筆淨: {e[e['rank']<=5].net.mean()*100:.3f}% / {e[(e['rank']>5)&(e['rank']<=10)].net.mean()*100:.3f}% / {e[(e['rank']>10)&(e['rank']<=20)].net.mean()*100:.3f}%")
(OUT/"concentration.md").write_text("\n".join(L)); print("\n".join(L))
