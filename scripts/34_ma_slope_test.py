"""D1 均線斜率 (昨日為止) 跟做空結果: 斜率向上且變大 (加速) 的要不要避開? 樣本 = trailing_stop_trades_sr.parquet (sr 模型每日前 10 檔, 2020~2026, 現行進出場).
特徵 (都用昨收以前的 D1):
  s5  = MA5 昨 / MA5 3 日前 − 1  (3 日斜率)；s10 = MA10 昨 / MA10 5 日前 − 1；s20 = MA20 昨 / MA20 5 日前 − 1
  a5  = s5 − 3 日前的 s5 (斜率變化 = 加速度)；a10 = s10 − 5 日前的 s10；a20 同
  above20 = 昨收 / MA20 − 1
輸出 stats/ma_slope_test.md"""
import glob, numpy as np, pandas as pd
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent; OUT = ROOT/"stats"; BASE, CAP, PER_STOCK, MAX_N = 0.0023, 3_000_000, 600_000, 10
R = pd.read_parquet(ROOT/"data/processed/trailing_stop_trades_sr.parquet"); R = R[["stock_id","date","entry","prev_close","rank","gross|現行 (無移動停利)","reason|現行 (無移動停利)"]].rename(columns={"gross|現行 (無移動停利)": "g", "reason|現行 (無移動停利)": "reason"})
R["net"] = R.g - BASE; R["big"] = R.net <= -0.07
d1 = pd.concat([pd.read_parquet(f, columns=["stock_id","date","close"]) for f in sorted(glob.glob(str(ROOT/"data/raw/d1/*.parquet")))], ignore_index=True); d1["date"] = pd.to_datetime(d1.date)
d1 = d1[d1.stock_id.isin(R.stock_id.unique())].drop_duplicates(["stock_id","date"]).sort_values(["stock_id","date"]).reset_index(drop=True); g = d1.groupby("stock_id").close
pc = g.shift(1)
for n, k in ((5, 3), (10, 5), (20, 5)):
    ma = g.transform(lambda s: s.shift(1).rolling(n).mean()); s = ma/ma.groupby(d1.stock_id).shift(k) - 1
    d1[f"s{n}"] = s; d1[f"a{n}"] = s - s.groupby(d1.stock_id).shift(k)
    if n == 20: d1["above20"] = pc/ma - 1
S = R.merge(d1[["stock_id","date","s5","a5","s10","a10","s20","a20","above20"]], on=["stock_id","date"], how="left")
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
def bucket(col, bins, labs, title):
    b = pd.cut(S[col], bins, labels=labs)
    T = pd.DataFrame({"筆數": S.groupby(b).size(), "占比%": S.groupby(b).size()/len(S)*100, "每筆淨%": S.net.groupby(b).mean()*100, "勝率": (S.net > 0).groupby(b).mean(), "大賠率%": S.big.groupby(b).mean()*100, "強制回補%": (S.reason == "forced").groupby(b).mean()*100, "總損益(萬)": S.net.groupby(b).sum()*600000/1e4})
    P(f"\n### {title}"); P(T.round(2).to_markdown())
L = []; P = lambda s="": L.append(str(s))
P("# D1 均線斜率 vs 做空結果 (2020~2026, sr 模型每日前 10 檔, 現行進出場)\n")
P(f"- 樣本 {len(S)} 筆。斜率 s = MA 昨 / MA k 日前 − 1 (MA5 用 k=3, MA10/20 用 k=5)；加速度 a = 斜率 − k 日前的斜率 (>0 = 向上變大)。全部用昨收以前的資料。")
P("\n## 1. 大賠 / 小賠 / 賺錢 的中位數 (%)")
cols = ["s5","a5","s10","a10","s20","a20","above20"]
T = pd.DataFrame({"大賠": S[S.big][cols].median(), "小賠": S[(S.net < 0) & ~S.big][cols].median(), "賺錢": S[S.net > 0][cols].median(), "全部": S[cols].median()})*100; P(T.round(2).to_markdown())
P("\n## 2. 依斜率 / 加速度分組")
bucket("s5", [-9, -.03, 0, .03, .06, .10, 9], ["<−3%","−3~0%","0~3%","3~6%","6~10%",">10%"], "MA5 三日斜率")
bucket("a5", [-9, -.05, -.02, 0, .02, .05, 9], ["<−5%","−5~−2%","−2~0%","0~2% 加速","2~5% 加速",">5% 加速"], "MA5 斜率變化 (加速度)")
bucket("s10", [-9, -.03, 0, .03, .06, .10, 9], ["<−3%","−3~0%","0~3%","3~6%","6~10%",">10%"], "MA10 五日斜率")
bucket("a10", [-9, -.05, -.02, 0, .02, .05, 9], ["<−5%","−5~−2%","−2~0%","0~2% 加速","2~5% 加速",">5% 加速"], "MA10 斜率變化 (加速度)")
bucket("s20", [-9, -.03, 0, .03, .06, .10, 9], ["<−3%","−3~0%","0~3%","3~6%","6~10%",">10%"], "MA20 五日斜率")
bucket("a20", [-9, -.03, -.01, 0, .01, .03, 9], ["<−3%","−3~−1%","−1~0%","0~1% 加速","1~3% 加速",">3% 加速"], "MA20 斜率變化 (加速度)")
bucket("above20", [-9, -.05, 0, .05, .10, .20, 9], ["<−5%","−5~0%","0~5%","5~10%","10~20%",">20%"], "昨收 vs MA20")
P("\n## 3. 斜率向上 × 加速 交叉 (MA10)")
up = S.s10 > 0; acc = S.a10 > 0
grp = np.select([up & acc, up & ~acc, ~up & acc], ["向上且加速", "向上但減速", "向下但加速"], "向下且減速")
T = pd.DataFrame({"筆數": S.groupby(grp).size(), "占比%": S.groupby(grp).size()/len(S)*100, "每筆淨%": S.net.groupby(grp).mean()*100, "勝率": (S.net > 0).groupby(grp).mean(), "大賠率%": S.big.groupby(grp).mean()*100, "強制回補%": (S.reason == "forced").groupby(grp).mean()*100, "總損益(萬)": S.net.groupby(grp).sum()*600000/1e4}); P(T.round(2).to_markdown())
P("\n## 4. 當濾網 (排掉的檔資金給下一檔)")
rows = [stats("現行", S.net.notna())]
for nm, m in [("排 MA5 斜率 > 6%", S.s5 > .06), ("排 MA5 斜率 > 10%", S.s5 > .10), ("排 MA5 向上且加速 (s5>0 & a5>0)", (S.s5 > 0) & (S.a5 > 0)), ("排 MA5 加速 > 2%", S.a5 > .02),
              ("排 MA10 斜率 > 6%", S.s10 > .06), ("排 MA10 向上且加速", up & acc), ("排 MA10 加速 > 2%", S.a10 > .02), ("排 MA20 斜率 > 6%", S.s20 > .06), ("排 MA20 向上且加速", (S.s20 > 0) & (S.a20 > 0)),
              ("排 昨收 > MA20 +10%", S.above20 > .10), ("排 昨收 > MA20 +20%", S.above20 > .20), ("反向: 只做 MA10 向上且加速", ~(up & acc)), ("反向: 只做 MA5 斜率 > 6%", ~(S.s5 > .06))]:
    rows.append(stats(nm, ~m))
P(pd.DataFrame(rows).set_index("規則").round(3).to_markdown())
P("\n## 5. 各年每筆淨% (現行 vs 排 MA10 向上且加速 vs 排 MA5 斜率>6%)")
yr = pd.DataFrame({"現行": S.net.groupby(S.date.dt.year).mean()*100, "排MA10向上加速": S.net[~(up & acc)].groupby(S.date.dt.year).mean()*100, "排MA5斜率>6%": S.net[~(S.s5 > .06)].groupby(S.date.dt.year).mean()*100}); P(yr.round(2).to_markdown())
P("\n## 6. Spearman 相關 (特徵 vs 淨報酬)"); P("- " + ", ".join(f"{c} {S[[c,'net']].corr(method='spearman').iloc[0,1]:+.3f}" for c in cols))
(OUT/"ma_slope_test.md").write_text("\n".join(L)); print("\n".join(L))
