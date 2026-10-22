"""300 萬資金的真實配置: 以張為單位, 高價股放不進去. 比較股價上限 / 每檔預算 / 檔數."""
import glob, numpy as np, pandas as pd
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent; OUT = ROOT/"stats"; BASE=0.0023; SLIP=0.001; K=0.08; PEN=0.07
CAP = 3_000_000
ent = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(str(ROOT/"data/processed/m1_entry/*.parquet")))])
m1d = pd.concat([pd.read_parquet(f, columns=["stock_id","date","m1_close","close_auction_c"]) for f in sorted(glob.glob(str(ROOT/"data/processed/m1_daily/*.parquet")))])
B = pd.read_parquet(ROOT/"data/processed/dataset_B_pool.parquet")[["stock_id","date","prev_close","gap","mkt_gap","atr20","dv20_prev"]]
pB = pd.read_parquet(ROOT/"data/processed/preds_B_pool.parquet")[["stock_id","date","pred"]]
d1 = pd.read_parquet(ROOT/"data/processed/d1.parquet", columns=["stock_id","date","open"]).sort_values(["stock_id","date"]); d1["next_open"]=d1.groupby("stock_id").open.shift(-1)
d = B.merge(pB, on=["stock_id","date"]).merge(ent, on=["stock_id","date"]).merge(m1d, on=["stock_id","date"]).merge(d1[["stock_id","date","next_open"]], on=["stock_id","date"])
d = d[(d.date>="2024-10-01")&d.e00_vwap.notna()].copy()
d["entry"]=d.e00_vwap; d["exit"]=d.close_auction_c.fillna(d.m1_close); d["lock"]=d.exit>=d.prev_close*1.0975; d["hi"]=d.e00_maxhigh_after/d.prev_close-1
trig=d.hi>=K; above=d.entry>=d.prev_close*(1+K)
d["gross"]=np.where(trig&~above, d.entry/(d.prev_close*(1+K)*(1+SLIP*3))-1, np.where(d.lock, d.entry/d.next_open-1-PEN, d.entry/d.exit-1)); d["net"]=d.gross-BASE
e = d[(d.gap<=0.04)&(d.mkt_gap>=-0.01)&(d.entry>d.prev_close*0.905)].copy()
e["rank"]=e.groupby("date").pred.rank(method="first"); e["lot_cost"]=e.entry*1000
e["e00_amt"] = e.e00_vol*1000*e.entry   # 09:00~09:02 成交金額

L=[f"# 300 萬資金的真實配置 (張為單位)\n"]
# 股價分布 & 各價位績效
top10 = e[e["rank"]<=10]
L.append("## 1. 前 10 名候選的股價分布與績效")
top10["px_bin"]=pd.cut(top10.entry,[0,30,50,100,150,300,600,1e5],labels=["<30","30-50","50-100","100-150","150-300","300-600",">600"])
t = top10.groupby("px_bin",observed=True).agg(n=("net","size"), share=("net", lambda s: len(s)/len(top10)), net_mean=("net","mean"), win=("net", lambda s:(s>0).mean()), lock=("lock","mean"), e00_amt_med=("e00_amt","median"))
t[["share","net_mean","win","lock"]]*=100; t["e00_amt_med"]/=1e4
t.columns=["n","占比%","每筆淨%","勝率%","收盤鎖漲停%","09:00-02成交額中位(萬)"]
L.append(t.round(2).to_markdown()); L.append(f"- 前10名中一張 > 30 萬 (股價>300) 的比例 {(top10.lot_cost>300_000).mean()*100:.1f}%，> 15 萬 (股價>150) 的 {(top10.lot_cost>150_000).mean()*100:.1f}%\n")

def simulate(sub, n_target, budget_per, max_px=None, min_lots=1, label=""):
    """每日依排名往下挑, 每檔預算 budget_per, 張數=floor(budget/張價), 不足 min_lots 張跳過, 直到 n_target 檔或資金用完."""
    recs=[]
    for dt, g in sub.sort_values(["date","rank"]).groupby("date", sort=False):
        if max_px is not None: g=g[g.entry<=max_px]
        cash=CAP; picks=[]
        for _,r in g.iterrows():
            if len(picks)>=n_target or cash<r.lot_cost: continue
            lots=int(min(budget_per, cash)//r.lot_cost)
            if lots<min_lots: continue
            amt=lots*r.lot_cost; cash-=amt; picks.append((amt, r.net, r.e00_amt))
        if not picks: recs.append((dt,0,0,0,0,np.nan)); continue
        amts=np.array([p[0] for p in picks]); nets=np.array([p[1] for p in picks]); e00=np.array([p[2] for p in picks])
        recs.append((dt, len(picks), amts.sum()/CAP, (amts*nets).sum()/CAP, (amts*nets).sum()/amts.sum(), np.nanmax(amts/e00)))
    r=pd.DataFrame(recs, columns=["date","n","util","ret_on_cap","ret_on_deployed","max_amt_share_of_e00"]).set_index("date")
    s=r.ret_on_cap
    return {"配置":label,"日均檔數":r.n.mean(),"資金使用率":r.util.mean(),"日淨%(對300萬)":s.mean()*100,"日淨%(對已投入)":r.ret_on_deployed.mean()*100,"日std%":s.std()*100,
            "Sharpe":s.mean()/s.std()*np.sqrt(245),"MDD%":((1+s).cumprod()/(1+s).cumprod().cummax()-1).min()*100,"最差日%":s.min()*100,
            "日均獲利(元)":s.mean()*CAP,"單檔占09:00-02成交額 中位":r.max_amt_share_of_e00.median(),"占>10%天數比":(r.max_amt_share_of_e00>0.1).mean()}
rows=[]
rows.append(simulate(e,10,300_000,None,1,"10檔 每檔30萬 不限股價 (放不進就跳過)"))
rows.append(simulate(e,10,300_000,300,1,"10檔 每檔30萬 股價≤300"))
rows.append(simulate(e,10,300_000,150,1,"10檔 每檔30萬 股價≤150"))
rows.append(simulate(e,10,300_000,100,1,"10檔 每檔30萬 股價≤100"))
rows.append(simulate(e,10,300_000,150,2,"10檔 每檔30萬 股價≤150 至少2張"))
rows.append(simulate(e,8,375_000,150,1,"8檔 每檔37.5萬 股價≤150"))
rows.append(simulate(e,6,500_000,300,1,"6檔 每檔50萬 股價≤300"))
rows.append(simulate(e,5,600_000,300,1,"5檔 每檔60萬 股價≤300"))
rows.append(simulate(e,10,600_000,300,1,"排名往下填滿: 每檔上限60萬 最多10檔 股價≤300"))
rows.append(simulate(e,15,300_000,150,1,"15檔 每檔30萬 股價≤150"))
tab=pd.DataFrame(rows).set_index("配置")
L.append("## 2. 真實配置模擬 (每日 300 萬, 依模型排名往下挑, 張數無條件捨去, 剩餘現金報酬 0)")
L.append("- 日淨%(對300萬) 是真正的資金報酬率；日淨%(對已投入) 是選到的股票本身的表現。單檔占 09:00-02 成交額 = 該檔部位金額 / 進場視窗三分鐘成交金額，>10% 代表滑價會明顯。\n")
L.append(tab.round(3).to_markdown())
# 股價上限對 "池內候選" 績效的影響 (不考慮資金)
L.append("\n## 3. 股價上限本身對候選品質的影響 (前10名, 等權, 不考慮張數)")
rows=[]
for mp in [None,300,150,100,60]:
    s = e if mp is None else e[e.entry<=mp]; s=s.copy(); s["rk"]=s.groupby("date").pred.rank(method="first"); y=s[s.rk<=10]
    daily=y.net.groupby(y.date).mean()
    rows.append({"股價上限":"無" if mp is None else f"≤{mp}","每筆淨%":y.net.mean()*100,"勝率":(y.net>0).mean(),"日淨%":daily.mean()*100,"Sharpe":daily.mean()/daily.std()*np.sqrt(245),"MDD%":((1+daily).cumprod()/(1+daily).cumprod().cummax()-1).min()*100,"合格檔數/日":len(s)/s.date.nunique()})
L.append(pd.DataFrame(rows).set_index("股價上限").round(3).to_markdown())
(OUT/"capital_alloc.md").write_text("\n".join(L)); print("\n".join(L))
