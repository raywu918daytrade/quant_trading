"""S9' 的出場規則比較: 停利 / 停損 / 時間出場 / 移動停利 (皆用 M1 極值, 保守處理順序問題)."""
import glob, numpy as np, pandas as pd
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent; OUT = ROOT/"stats"; BASE=0.0023; PEN=0.002; SLIP=0.001
ent = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(str(ROOT/"data/processed/m1_entry/*.parquet")))])
m1d = pd.concat([pd.read_parquet(f, columns=["stock_id","date","m1_close","close_auction_c","m1_high","m1_low","t_high","t_low","c10","c11","c12","c13"]) for f in sorted(glob.glob(str(ROOT/"data/processed/m1_daily/*.parquet")))])
B = pd.read_parquet(ROOT/"data/processed/dataset_B_pool.parquet")[["stock_id","date","prev_close","gap","mkt_gap"]]
pB = pd.read_parquet(ROOT/"data/processed/preds_B_pool.parquet")[["stock_id","date","pred"]]
d1 = pd.read_parquet(ROOT/"data/processed/d1.parquet", columns=["stock_id","date","open"]).sort_values(["stock_id","date"])
d1["next_open"] = d1.groupby("stock_id").open.shift(-1)
d = B.merge(pB, on=["stock_id","date"]).merge(ent, on=["stock_id","date"]).merge(m1d, on=["stock_id","date"]).merge(d1[["stock_id","date","next_open"]], on=["stock_id","date"])
d = d[d.date>="2024-10-01"]
d["rank"] = d.groupby("date").pred.rank(method="first")
x = d[(d["rank"]<=20)&(d.gap<=0.06)&(d.mkt_gap>=-0.01)&d.e00_vwap.notna()].copy()
x = x[x.e00_vwap>x.prev_close*0.905]
x["entry"]=x.e00_vwap; x["exit"]=x.close_auction_c.fillna(x.m1_close)
x["lock"] = x.exit >= x.prev_close*1.0975
x["mfe"] = 1 - x.e00_minlow_after/x.entry      # 最大有利幅度 (做空: 往下)
x["mae"] = x.e00_maxhigh_after/x.entry - 1     # 最大不利幅度
base_gross = np.where(x.lock, x.entry/x.next_open-1-PEN, x.entry/x.exit-1)   # 保守: 鎖漲停隔日回補

def stats(gross, label):
    net = pd.Series(gross, index=x.index) - BASE; daily = net.groupby(x.date).mean()
    return {"規則":label,"每筆淨%":net.mean()*100,"勝率":(net>0).mean(),"PF":net[net>0].sum()/(-net[net<0].sum()),"日淨%":daily.mean()*100,
            "Sharpe":daily.mean()/daily.std()*np.sqrt(245),"MDD%":((1+daily).cumprod()/(1+daily).cumprod().cummax()-1).min()*100,"最差1%筆":net.quantile(.01)*100}
rows=[stats(base_gross,"基準: 收盤出 (鎖漲停隔日回補)")]
# 停利: 進場後最低價曾到 entry*(1-tp) -> 在 tp 處回補 (扣滑價). 沒到就照基準.
for tp in [0.02,0.03,0.04,0.05,0.07]:
    hit = x.mfe>=tp
    rows.append(stats(np.where(hit, tp-SLIP, base_gross), f"停利 {tp*100:.0f}% (觸及率 {hit.mean()*100:.0f}%)"))
# 停損: 最高價曾到 entry*(1+sl) -> 在 sl 處停損 (順序未知, 偏保守: 只要碰到就算停損)
for sl in [0.03,0.05,0.07]:
    hit = x.mae>=sl
    rows.append(stats(np.where(hit, -(sl+SLIP*3), base_gross), f"停損 {sl*100:.0f}% (觸及率 {hit.mean()*100:.0f}%)"))
# 停利+停損 (順序未知: 兩者都碰到時假設先碰到停損, 最保守)
for tp,sl in [(0.03,0.05),(0.05,0.05),(0.05,0.07)]:
    h_tp = x.mfe>=tp; h_sl = x.mae>=sl
    g = np.where(h_sl, -(sl+SLIP*3), np.where(h_tp, tp-SLIP, base_gross))
    rows.append(stats(g, f"停利{tp*100:.0f}%+停損{sl*100:.0f}% (兩者皆觸時算停損)"))
# 時間出場
for col,lab in [("c10","10:00"),("c11","11:00"),("c12","12:00"),("c13","13:00")]:
    px = x[col].fillna(x.exit)
    rows.append(stats(np.where(x.lock & (px>=x.prev_close*1.0975), x.entry/x.next_open-1-PEN, x.entry/px-1), f"時間出場 {lab}"))
tab = pd.DataFrame(rows).set_index("規則")
L=["# S9' 出場規則比較 (保守情境: 收盤鎖漲停隔日回補)\n",
   f"- 樣本: S9' 交易 {len(x)} 筆, {x.date.nunique()} 天。停利/停損用進場後 M1 極值判斷, 不知道先後順序, 停損採『碰到就算』、兩者皆碰到算停損, 均偏保守。滑價 0.1%。\n",
   tab.round(3).to_markdown(), "",
   "## 做空交易的日內路徑統計",
   f"- 最大有利幅度 MFE 中位數 {x.mfe.median()*100:.2f}%, 平均 {x.mfe.mean()*100:.2f}%；最大不利幅度 MAE 中位數 {x.mae.median()*100:.2f}%, 平均 {x.mae.mean()*100:.2f}%",
   f"- 最終獲利 (基準淨>0) 的交易中, MAE 曾超過 2% 的比例 {(x.mae[pd.Series(base_gross,index=x.index)-BASE>0]>=0.02).mean()*100:.0f}%，超過 3% 的 {(x.mae[pd.Series(base_gross,index=x.index)-BASE>0]>=0.03).mean()*100:.0f}% (這些若設停損會被掃掉)",
   f"- 日低發生時間: 09:00~09:30 {(x.t_low<30).mean()*100:.0f}%，09:30~11:00 {((x.t_low>=30)&(x.t_low<120)).mean()*100:.0f}%，11:00~13:00 {((x.t_low>=120)&(x.t_low<240)).mean()*100:.0f}%，13:00 後 {(x.t_low>=240).mean()*100:.0f}%",
   f"- 日高發生時間: 09:00~09:30 {(x.t_high<30).mean()*100:.0f}%，之後 {(x.t_high>=30).mean()*100:.0f}%"]
(OUT/"exit_rules.md").write_text("\n".join(L)); print("\n".join(L))
