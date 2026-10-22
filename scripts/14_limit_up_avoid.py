"""漲停鎖死: 更重罰金情境 + 接近漲停強制回補規則 (S9')."""
import glob, numpy as np, pandas as pd
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent; OUT = ROOT/"stats"; BASE=0.0023; SLIP=0.001
ent = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(str(ROOT/"data/processed/m1_entry/*.parquet")))])
m1d = pd.concat([pd.read_parquet(f, columns=["stock_id","date","m1_close","close_auction_c","m1_high"]) for f in sorted(glob.glob(str(ROOT/"data/processed/m1_daily/*.parquet")))])
B = pd.read_parquet(ROOT/"data/processed/dataset_B_pool.parquet")[["stock_id","date","prev_close","gap","mkt_gap","atr20"]]
pB = pd.read_parquet(ROOT/"data/processed/preds_B_pool.parquet")[["stock_id","date","pred"]]
d1 = pd.read_parquet(ROOT/"data/processed/d1.parquet", columns=["stock_id","date","open"]).sort_values(["stock_id","date"])
d1["next_open"] = d1.groupby("stock_id").open.shift(-1)
d = B.merge(pB, on=["stock_id","date"]).merge(ent, on=["stock_id","date"]).merge(m1d, on=["stock_id","date"]).merge(d1[["stock_id","date","next_open"]], on=["stock_id","date"])
d = d[d.date>="2024-10-01"]; d["rank"] = d.groupby("date").pred.rank(method="first")
x = d[(d["rank"]<=20)&(d.gap<=0.06)&(d.mkt_gap>=-0.01)&d.e00_vwap.notna()].copy(); x = x[x.e00_vwap>x.prev_close*0.905]
x["entry"]=x.e00_vwap; x["exit"]=x.close_auction_c.fillna(x.m1_close)
x["lock"] = x.exit >= x.prev_close*1.0975
x["hi_vs_prev"] = x.e00_maxhigh_after/x.prev_close - 1
def stats(gross, label, lock_rate):
    net = pd.Series(gross, index=x.index)-BASE; daily = net.groupby(x.date).mean()
    return {"規則":label,"每筆淨%":net.mean()*100,"勝率":(net>0).mean(),"日淨%":daily.mean()*100,"Sharpe":daily.mean()/daily.std()*np.sqrt(245),
            "MDD%":((1+daily).cumprod()/(1+daily).cumprod().cummax()-1).min()*100,"最差1%筆":net.quantile(.01)*100,"仍會鎖死%":lock_rate*100}
L=["# 漲停鎖死: 罰金情境與漲停前強制回補 (S9')\n", f"- 樣本 {len(x)} 筆 / {x.date.nunique()} 天。收盤鎖漲停 {x.lock.mean()*100:.2f}%，盤中曾達昨收+7% 的 {(x.hi_vs_prev>=0.07).mean()*100:.1f}%，+8% 的 {(x.hi_vs_prev>=0.08).mean()*100:.1f}%，+9.5% 的 {(x.hi_vs_prev>=0.095).mean()*100:.1f}%\n"]
rows=[]
L.append("## 1. 鎖死罰金情境 (收盤鎖漲停 → 隔日開盤回補 + 罰金)")
for pen in [0.002, 0.03, 0.07]:
    g = np.where(x.lock, x.entry/x.next_open-1-pen, x.entry/x.exit-1)
    rows.append(stats(g, f"隔日回補 + 罰金 {pen*100:.1f}%", x.lock.mean()))
L.append(pd.DataFrame(rows).set_index("規則").round(3).to_markdown()); L.append("")
L.append("## 2. 漲停前強制回補: 盤中價格一到昨收×(1+k) 就市價回補 (含滑價 0.3%)。不知先後順序, 保守假設碰到就回補。殘餘鎖死 = 跳空開盤就已超過門檻者仍以隔日回補+7%罰金計")
rows=[]
for k in [0.05, 0.06, 0.07, 0.08, 0.09]:
    trig = x.hi_vs_prev >= k
    cover_px = x.prev_close*(1+k)*(1+SLIP*3)
    opened_above = x.entry >= x.prev_close*(1+k)       # 進場價已在門檻之上 -> 規則無法保護, 若鎖死照舊
    g = np.where(trig & ~opened_above, x.entry/cover_px-1, np.where(x.lock, x.entry/x.next_open-1-0.07, x.entry/x.exit-1))
    rows.append(stats(g, f"昨收 +{k*100:.0f}% 強制回補 (觸發率 {trig.mean()*100:.0f}%)", (x.lock & opened_above).mean()))
L.append(pd.DataFrame(rows).set_index("規則").round(3).to_markdown()); L.append("")
L.append("## 3. 強制回補 +8% 再搭配較低的跳空門檻 (避免開盤就在門檻之上)")
rows=[]
for gcap in [0.06, 0.05, 0.04, 0.03]:
    y = x[x.gap<=gcap]; trig = y.hi_vs_prev>=0.08; opened_above = y.entry>=y.prev_close*1.08
    cover_px = y.prev_close*1.08*(1+SLIP*3)
    g = np.where(trig & ~opened_above, y.entry/cover_px-1, np.where(y.lock, y.entry/y.next_open-1-0.07, y.entry/y.exit-1))
    net = pd.Series(g, index=y.index)-BASE; daily = net.groupby(y.date).mean()
    rows.append({"跳空上限":f"{gcap*100:.0f}%","每筆淨%":net.mean()*100,"勝率":(net>0).mean(),"日淨%":daily.mean()*100,"Sharpe":daily.mean()/daily.std()*np.sqrt(245),
                 "MDD%":((1+daily).cumprod()/(1+daily).cumprod().cummax()-1).min()*100,"仍會鎖死%":(y.lock&opened_above).mean()*100,"觸發回補%":trig.mean()*100,"日均檔數":len(y)/y.date.nunique()})
L.append(pd.DataFrame(rows).set_index("跳空上限").round(3).to_markdown())
(OUT/"limit_up_avoid.md").write_text("\n".join(L)); print("\n".join(L))
