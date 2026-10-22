"""在定案候選 (股價≤300, 排名前10) 上測單一條件過濾對勝率/期望/容量的影響, 再測組合並做 300 萬配置模擬."""
import glob, numpy as np, pandas as pd
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent; OUT = ROOT/"stats"; BASE=0.0023; SLIP=0.001; K=0.08; PEN=0.07; CAP=3_000_000
PERIODS = {"探索": ("2024-10-01","2025-06-30"), "驗證": ("2025-07-01","2026-03-31"), "鎖定": ("2026-04-01","2026-12-31")}
ent = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(str(ROOT/"data/processed/m1_entry/*.parquet")))])
m1d = pd.concat([pd.read_parquet(f, columns=["stock_id","date","m1_close","close_auction_c"]) for f in sorted(glob.glob(str(ROOT/"data/processed/m1_daily/*.parquet")))])
B = pd.read_parquet(ROOT/"data/processed/dataset_B_pool.parquet")[["stock_id","date","prev_close","gap","mkt_gap","mkt_pret","mkt_ret5","atr20","dv20_prev","p_up_lim","p_vol_ratio","p_ret5","pos20","p_intra","is_tpex"]]
pB = pd.read_parquet(ROOT/"data/processed/preds_B_pool.parquet")[["stock_id","date","pred"]]
d1 = pd.read_parquet(ROOT/"data/processed/d1.parquet", columns=["stock_id","date","open"]).sort_values(["stock_id","date"]); d1["next_open"]=d1.groupby("stock_id").open.shift(-1)
d = B.merge(pB, on=["stock_id","date"]).merge(ent, on=["stock_id","date"]).merge(m1d, on=["stock_id","date"]).merge(d1[["stock_id","date","next_open"]], on=["stock_id","date"])
d = d[(d.date>="2024-10-01")&d.e00_vwap.notna()].copy()
d["entry"]=d.e00_vwap; d["exit"]=d.close_auction_c.fillna(d.m1_close); d["lock"]=d.exit>=d.prev_close*1.0975; d["hi"]=d.e00_maxhigh_after/d.prev_close-1
trig=d.hi>=K; above=d.entry>=d.prev_close*(1+K)
d["gross"]=np.where(trig&~above, d.entry/(d.prev_close*(1+K)*(1+SLIP*3))-1, np.where(d.lock, d.entry/d.next_open-1-PEN, d.entry/d.exit-1)); d["net"]=d.gross-BASE
d["period"]=None
for k,(a,b) in PERIODS.items(): d.loc[(d.date>=a)&(d.date<=b),"period"]=k
e = d[(d.gap<=0.04)&(d.mkt_gap>=-0.01)&(d.entry>d.prev_close*0.905)&(d.entry<=300)].copy()
e["rank"]=e.groupby("date").pred.rank(method="first"); e["dow"]=e.date.dt.dayofweek
base = e[e["rank"]<=10].copy()
L=[f"# 過濾提高勝率 (基底: 跳空≤4%, 大盤跳空≥-1%, 股價≤300, 模型排名前10；{len(base)} 筆, 勝率 {(base.net>0).mean()*100:.1f}%, 每筆淨 {base.net.mean()*100:.3f}%)\n"]
conds = {
 "個股跳空 > 0 (開高)": base.gap>0, "個股跳空 > +1%": base.gap>0.01, "個股跳空 0~2%": (base.gap>0)&(base.gap<=0.02), "個股跳空 ≤ 0 (開平低)": base.gap<=0,
 "大盤跳空 ≥ 0": base.mkt_gap>=0, "大盤跳空 0~+1%": (base.mkt_gap>=0)&(base.mkt_gap<0.01), "大盤跳空 > +1%": base.mkt_gap>=0.01,
 "大盤昨日漲 (>0)": base.mkt_pret>0, "大盤昨日跌": base.mkt_pret<=0, "大盤近5日漲>2%": base.mkt_ret5>0.02,
 "前一日非漲停": base.p_up_lim==0, "前一日漲停": base.p_up_lim==1,
 "前一日量比 > 1.5": base.p_vol_ratio>1.5, "前一日量比 < 1": base.p_vol_ratio<1,
 "前5日累積 > +10%": base.p_ret5>0.10, "前5日累積 -5%~+10%": (base.p_ret5>-0.05)&(base.p_ret5<=0.10), "前5日累積 < -5%": base.p_ret5<=-0.05,
 "20日區間位置 > 0.8 (高檔)": base.pos20>0.8, "20日區間位置 < 0.3 (低檔)": base.pos20<0.3,
 "ATR 3~5%": base.atr20<0.05, "ATR 5~7%": (base.atr20>=0.05)&(base.atr20<0.07), "ATR > 7%": base.atr20>=0.07,
 "股價 50~150": (base.entry>=50)&(base.entry<150), "股價 < 50": base.entry<50, "股價 150~300": base.entry>=150,
 "成交額 5~20億": base.dv20_prev<20e8, "成交額 > 20億": base.dv20_prev>=20e8,
 "非週一": base.dow!=0, "週一": base.dow==0, "週二~四": base.dow.isin([1,2,3]),
 "模型分數 ≤ -0.3%": base.pred<=-0.003, "模型分數 ≤ -0.5%": base.pred<=-0.005,
 "前一日日內跌 (p_intra<0)": base.p_intra<0, "前一日日內漲 (p_intra>0)": base.p_intra>0,
 "上市 (twse)": base.is_tpex==0, "上櫃 (tpex)": base.is_tpex==1,
}
rows=[]
for k,m in conds.items():
    y=base[m]
    r={"條件":k,"保留比例":m.mean(),"每日檔數":len(y)/base.date.nunique(),"勝率":(y.net>0).mean(),"每筆淨%":y.net.mean()*100,"中位%":y.net.median()*100,"鎖漲停%":y.lock.mean()*100}
    for p in PERIODS: yy=y[y.period==p]; r[f"{p}勝率"]=(yy.net>0).mean(); r[f"{p}每筆%"]=yy.net.mean()*100
    rows.append(r)
tab=pd.DataFrame(rows).set_index("條件").sort_values("勝率",ascending=False)
L.append("## 1. 單一條件 (依勝率排序)"); L.append(tab.round(3).to_markdown()); L.append("")

# 組合
combos = {
 "A 基底 (無額外過濾)": pd.Series(True,index=base.index),
 "B 非週一": base.dow!=0,
 "C 非週一 ∩ 大盤跳空≥0": (base.dow!=0)&(base.mkt_gap>=0),
 "D 非週一 ∩ 前一日非漲停": (base.dow!=0)&(base.p_up_lim==0),
 "E 非週一 ∩ 大盤跳空≥0 ∩ 前一日非漲停": (base.dow!=0)&(base.mkt_gap>=0)&(base.p_up_lim==0),
 "F E ∩ 個股跳空>0": (base.dow!=0)&(base.mkt_gap>=0)&(base.p_up_lim==0)&(base.gap>0),
 "G E ∩ 股價≥50": (base.dow!=0)&(base.mkt_gap>=0)&(base.p_up_lim==0)&(base.entry>=50),
 "H 大盤跳空≥0 ∩ 前一日非漲停 (含週一)": (base.mkt_gap>=0)&(base.p_up_lim==0),
}
def alloc(y):
    recs=[]
    for dt,g in y.sort_values(["date","rank"]).groupby("date",sort=False):
        cash=CAP; amts=[]; nets=[]
        for _,r in g.iterrows():
            lc=r.entry*1000
            if cash<lc: continue
            lots=int(min(600_000,cash)//lc)
            if lots<1: continue
            a=lots*lc; cash-=a; amts.append(a); nets.append(r.net)
        if amts: a=np.array(amts); n=np.array(nets); recs.append((dt,len(a),a.sum()/CAP,(a*n).sum()/CAP))
        else: recs.append((dt,0,0,0))
    return pd.DataFrame(recs,columns=["date","n","util","ret"]).set_index("date")
rows=[]
for k,m in combos.items():
    y=base[m]; r=alloc(y); s=r.ret; days=base.date.nunique()
    row={"組合":k,"有交易天數比":(r.n>0).mean(),"日均檔數":r.n.mean(),"資金使用率":r.util.mean(),"勝率(筆)":(y.net>0).mean(),"每筆淨%":y.net.mean()*100,
         "日淨%(對300萬)":s.mean()*100,"日std%":s.std()*100,"Sharpe":s.mean()/s.std()*np.sqrt(245),"MDD%":((1+s).cumprod()/(1+s).cumprod().cummax()-1).min()*100,"日勝率":(s>0).mean(),"年化報酬(萬)":s.mean()*245*CAP/1e4}
    for p in PERIODS: pp=s[(s.index>=PERIODS[p][0])&(s.index<=PERIODS[p][1])]; row[f"{p}日淨%"]=pp.mean()*100
    rows.append(row)
L.append("## 2. 組合過濾 + 300 萬真實配置 (每檔上限 60 萬, 往下填滿)"); L.append(pd.DataFrame(rows).set_index("組合").round(3).to_markdown())
(OUT/"filters_winrate.md").write_text("\n".join(L)); print("\n".join(L))
