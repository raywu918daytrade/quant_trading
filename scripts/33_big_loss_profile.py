"""大賠交易的 D1 特徵: 是不是「過去 5~10 日盤整後突然大漲」? 樣本 = trailing_stop_trades_sr.parquet (sr 模型每日前 10 檔, 2020~2026, 現行出場).
D1 (今日前) 特徵:
  rng10 / rng5  = 前 10 / 5 日 (不含昨日) 的 (最高−最低)/昨收 → 盤整程度 (小 = 盤整)
  bo10          = 昨收 / 前 2~11 日最高 − 1 → 昨天有沒有突破前 10 日區間 (>0 = 突破)
  p_ret         = 昨日漲幅;  gap = 今日開盤跳空;  up2 = 昨日+今日跳空合計 (昨收×(1+gap)/前日收 − 1)
  atr20_rel     = 昨日 ATR20 / 前 10 日區間 → 區間相對平常波動多窄
大賠 = 淨損 ≤ −7% (幾乎都是 +8% 強制回補). 輸出 stats/big_loss_profile.md"""
import numpy as np, pandas as pd
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent; OUT = ROOT/"stats"; BASE, CAP, PER_STOCK, MAX_N = 0.0023, 3_000_000, 600_000, 10
R = pd.read_parquet(ROOT/"data/processed/trailing_stop_trades_sr.parquet"); R = R[["stock_id","date","entry","prev_close","rank","gross|現行 (無移動停利)","reason|現行 (無移動停利)"]].rename(columns={"gross|現行 (無移動停利)": "g", "reason|現行 (無移動停利)": "reason"})
R["net"] = R.g - BASE; R["big"] = R.net <= -0.07
import glob
d1 = pd.concat([pd.read_parquet(f, columns=["stock_id","date","open","high","low","close"]) for f in sorted(glob.glob(str(ROOT/"data/raw/d1/*.parquet")))], ignore_index=True); d1["date"] = pd.to_datetime(d1.date); d1 = d1[d1.stock_id.isin(R.stock_id.unique())].drop_duplicates(["stock_id","date"]).sort_values(["stock_id","date"]).reset_index(drop=True)
g = d1.groupby("stock_id"); pc = g.close.shift(1)
d1["p_ret"] = pc/g.close.shift(2) - 1; d1["gap"] = d1.open/pc - 1; d1["up2"] = d1.open/g.close.shift(2) - 1
hi = d1.high.shift(2); lo = d1.low.shift(2)   # 前 2 日起 (不含昨日)
for n in (5, 10):
    d1[f"rng{n}"] = (g.high.transform(lambda s: s.shift(2).rolling(n).max()) - g.low.transform(lambda s: s.shift(2).rolling(n).min()))/pc
d1["bo10"] = pc/g.high.transform(lambda s: s.shift(2).rolling(10).max()) - 1
tr = g.apply(lambda x: pd.concat([x.high-x.low, (x.high-x.close.shift(1)).abs(), (x.low-x.close.shift(1)).abs()], axis=1).max(axis=1)).reset_index(level=0, drop=True)
d1["atr20"] = tr.groupby(d1.stock_id).transform(lambda s: s.shift(1).rolling(20).mean())/pc; d1["atr_rel"] = d1.atr20*10/d1.rng10
S = R.merge(d1[["stock_id","date","p_ret","gap","up2","rng5","rng10","bo10","atr20","atr_rel"]], on=["stock_id","date"], how="left")
all_days = pd.Series(sorted(S.date.unique()))
def alloc(mask):
    recs = []
    for dt, gg in S[mask].sort_values(["date","rank"]).groupby("date", sort=False):
        cash = CAP; amts = []; nets = []
        for _, r in gg.iterrows():
            lc = r.entry*1000
            if cash < lc: continue
            lots = int(min(PER_STOCK, cash)//lc)
            if lots < 1: continue
            a = lots*lc; cash -= a; amts.append(a); nets.append(r.net)
            if len(amts) >= MAX_N: break
        recs.append((dt, len(amts), sum(a*n for a, n in zip(amts, nets))/CAP))
    return pd.DataFrame(recs, columns=["date","n","ret"]).set_index("date").reindex(all_days).fillna({"n": 0, "ret": 0.0})
def stats(name, mask):
    D = alloc(mask); s = D.ret; n = S.net[mask]
    return {"規則": name, "交易占比%": mask.mean()*100, "日均檔數": D.n.mean(), "每筆淨%": n.mean()*100, "勝率": (n > 0).mean(), "大賠率%": (n <= -0.07).mean()*100, "日淨%": s.mean()*100, "Sharpe": s.mean()/s.std()*np.sqrt(245), "MDD%": ((1+s).cumprod()/(1+s).cumprod().cummax()-1).min()*100, "最差月%": s.groupby(s.index.to_period("M")).apply(lambda x: ((1+x).prod()-1)*100).min(), "年化(萬)": s.mean()*245*CAP/1e4}
L = []; P = lambda s="": L.append(str(s))
P("# 大賠交易的 D1 特徵 (2020~2026, sr 模型每日前 10 檔, 現行進出場)\n")
big = S[S.big]; P(f"- 樣本 {len(S)} 筆, 大賠 (淨損 ≤ −7%) {len(big)} 筆 = {S.big.mean()*100:.1f}%, 合計虧 {big.net.sum()*600000/1e4:.0f} 萬 (以每檔 60 萬估)；全部交易合計賺 {S.net.sum()*600000/1e4:.0f} 萬。大賠的出場原因: {big.reason.value_counts().to_dict()}")
P("\n## 1. 大賠 vs 其他交易的 D1 特徵中位數 (今日進場前可知)")
cols = {"rng5": "前5日區間%", "rng10": "前10日區間%", "atr_rel": "ATR20×10/前10日區間", "bo10": "昨收 vs 前10日高%", "p_ret": "昨日漲幅%", "gap": "今日跳空%", "up2": "昨+今合計%", "atr20": "ATR20%"}
T = pd.DataFrame({"大賠": S[S.big][list(cols)].median(), "小賠 (−7%~0)": S[(S.net < 0) & ~S.big][list(cols)].median(), "賺錢": S[S.net > 0][list(cols)].median(), "全部": S[list(cols)].median()}).rename(index=cols)
for c in T.index:
    if c != "ATR20×10/前10日區間": T.loc[c] *= 100
P(T.round(2).to_markdown())
P("\n## 2. 依『前 10 日區間』分組 (越小越盤整)")
def bucket_table(col, bins, labs, title):
    b = pd.cut(S[col], bins, labels=labs)
    T = pd.DataFrame({"筆數": S.groupby(b).size(), "占比%": S.groupby(b).size()/len(S)*100, "每筆淨%": S.net.groupby(b).mean()*100, "勝率": (S.net > 0).groupby(b).mean(), "大賠率%": S.big.groupby(b).mean()*100, "強制回補%": (S.reason == "forced").groupby(b).mean()*100, "大賠合計(萬)": S.net.where(S.big, 0).groupby(b).sum()*600000/1e4})
    P(f"\n### {title}"); P(T.round(2).to_markdown())
bucket_table("rng10", [0, .06, .09, .12, .16, .22, 9], ["<6%","6~9%","9~12%","12~16%","16~22%",">22%"], "前 10 日區間 (不含昨日) / 昨收")
bucket_table("rng5", [0, .04, .06, .08, .11, .15, 9], ["<4%","4~6%","6~8%","8~11%","11~15%",">15%"], "前 5 日區間 (不含昨日) / 昨收")
bucket_table("bo10", [-9, -.1, -.05, 0, .03, .07, 9], ["<−10%","−10~−5%","−5~0%","0~3% 小突破","3~7% 突破","> 7% 大突破"], "昨收 vs 前 10 日最高 (昨天有沒有突破盤整區)")
bucket_table("p_ret", [-9, 0, .03, .06, .095, 9], ["昨跌","0~3%","3~6%","6~9.5%","昨漲停"], "昨日漲幅")
bucket_table("up2", [-9, 0, .03, .06, .10, .14, 9], ["<0","0~3%","3~6%","6~10%","10~14%",">14%"], "昨日 + 今日跳空 合計")
P("\n## 3. 盤整 × 突破 交叉 (盤整 = 前 10 日區間 < 12%；突破 = 昨收 > 前 10 日最高)")
cons = S.rng10 < .12; bo = S.bo10 > 0
grp = np.select([cons & bo, cons & ~bo, ~cons & bo], ["盤整後突破", "盤整沒突破", "寬幅後突破"], "寬幅沒突破")
T = pd.DataFrame({"筆數": S.groupby(grp).size(), "占比%": S.groupby(grp).size()/len(S)*100, "每筆淨%": S.net.groupby(grp).mean()*100, "勝率": (S.net > 0).groupby(grp).mean(), "大賠率%": S.big.groupby(grp).mean()*100, "大賠合計(萬)": S.net.where(S.big, 0).groupby(grp).sum()*600000/1e4, "總損益(萬)": S.net.groupby(grp).sum()*600000/1e4})
P(T.round(2).to_markdown())
P("\n## 4. 當濾網 (排掉的檔資金給下一檔)")
rows = [stats("現行", S.net.notna())]
rows.append(stats("排 盤整後突破 (rng10<12% & bo10>0)", ~(cons & bo)))
rows.append(stats("排 盤整後突破 嚴格 (rng10<9% & bo10>3%)", ~((S.rng10 < .09) & (S.bo10 > .03))))
rows.append(stats("排 昨收突破前10日高 (bo10>0)", ~bo))
rows.append(stats("排 bo10>3%", ~(S.bo10 > .03)))
rows.append(stats("排 昨日漲停", ~(S.p_ret >= .095)))
rows.append(stats("排 昨+今 ≥ 10%", ~(S.up2 >= .10)))
rows.append(stats("排 昨+今 ≥ 14%", ~(S.up2 >= .14)))
rows.append(stats("排 前10日區間 < 9%", ~(S.rng10 < .09)))
rows.append(stats("排 ATR×10/區間 > 4 (區間比平常窄很多)", ~(S.atr_rel > 4)))
P(pd.DataFrame(rows).set_index("規則").round(3).to_markdown())
P("\n## 5. 各年每筆淨% (現行 vs 排盤整後突破)")
yr = pd.DataFrame({"現行": S.net.groupby(S.date.dt.year).mean()*100, "排盤整後突破": S.net[~(cons & bo)].groupby(S.date.dt.year).mean()*100, "現行 大賠率%": S.big.groupby(S.date.dt.year).mean()*100, "排後 大賠率%": S.big[~(cons & bo)].groupby(S.date.dt.year).mean()*100})
P(yr.round(2).to_markdown())
(OUT/"big_loss_profile.md").write_text("\n".join(L)); print("\n".join(L))
