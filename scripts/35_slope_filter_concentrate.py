"""排掉『均線向上且加速』的股票後, 把 300 萬集中到 3~5 檔 (每檔 60/75/100/150 萬), 資金使用率不降, 會不會比現行好?
樣本 = sr 模型每日前 10 檔 (trailing_stop_trades_sr.parquet), 濾網後依原排名由前往後配, 每檔上限 per_stock, 配到 300 萬用完.
濾網: 無 / 排 MA10 向上且加速 / 排 MA5 向上且加速 / 排 MA20 向上且加速 / 排 MA5 斜率>6%. 特徵同 34. 輸出 stats/slope_filter_concentrate.md"""
import glob, numpy as np, pandas as pd
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent; OUT = ROOT/"stats"; BASE, CAP = 0.0023, 3_000_000
R = pd.read_parquet(ROOT/"data/processed/trailing_stop_trades_sr.parquet"); R = R[["stock_id","date","entry","rank","gross|現行 (無移動停利)"]].rename(columns={"gross|現行 (無移動停利)": "g"}); R["net"] = R.g - BASE
d1 = pd.concat([pd.read_parquet(f, columns=["stock_id","date","close"]) for f in sorted(glob.glob(str(ROOT/"data/raw/d1/*.parquet")))], ignore_index=True); d1["date"] = pd.to_datetime(d1.date)
d1 = d1[d1.stock_id.isin(R.stock_id.unique())].drop_duplicates(["stock_id","date"]).sort_values(["stock_id","date"]).reset_index(drop=True); g = d1.groupby("stock_id").close
for n, k in ((5, 3), (10, 5), (20, 5)):
    ma = g.transform(lambda s: s.shift(1).rolling(n).mean()); s = ma/ma.groupby(d1.stock_id).shift(k) - 1; d1[f"s{n}"] = s; d1[f"a{n}"] = s - s.groupby(d1.stock_id).shift(k)
S = R.merge(d1[["stock_id","date","s5","a5","s10","a10","s20","a20"]], on=["stock_id","date"], how="left")
all_days = pd.Series(sorted(S.date.unique()))
FILTERS = {"無濾網 (現行選股)": S.net.notna(), "排 MA10 向上且加速": ~((S.s10 > 0) & (S.a10 > 0)), "排 MA5 向上且加速": ~((S.s5 > 0) & (S.a5 > 0)), "排 MA20 向上且加速": ~((S.s20 > 0) & (S.a20 > 0)), "排 MA5 斜率 > 6%": ~(S.s5 > .06), "排 MA10 加速 > 2%": ~(S.a10 > .02)}
def alloc(mask, per_stock, max_n=10):
    recs = []
    for dt, gg in S[mask].sort_values(["date","rank"]).groupby("date", sort=False):
        cash = CAP; amts = []; nets = []
        for _, r in gg.iterrows():
            lc = r.entry*1000
            if cash < lc: continue
            lots = int(min(per_stock, cash)//lc)
            if lots < 1: continue
            a = lots*lc; cash -= a; amts.append(a); nets.append(r.net)
            if len(amts) >= max_n: break
        recs.append((dt, len(amts), sum(amts)/CAP, sum(a*n for a, n in zip(amts, nets))/CAP, min(nets)*max(amts)/CAP if nets else 0))
    return pd.DataFrame(recs, columns=["date","n","util","ret","worst1"]).set_index("date").reindex(all_days).fillna({"n": 0, "util": 0, "ret": 0.0, "worst1": 0})
L = []; P = lambda s="": L.append(str(s))
P("# 排『均線向上且加速』+ 資金集中 3~5 檔 (2020~2026, sr 模型每日前 10 檔, 300 萬)\n")
P("- 濾網後依原排名配資金, 每檔上限 per_stock, 300 萬配完為止 (日均檔數 = 實際配到的檔數)。現行 = 無濾網、每檔 60 萬。")
rows = []
for fname, fm in FILTERS.items():
    for ps in (600_000, 750_000, 1_000_000, 1_500_000):
        D = alloc(fm, ps); s = D.ret; mo = s.groupby(s.index.to_period("M")).apply(lambda x: ((1+x).prod()-1)*100)
        rows.append({"濾網": fname, "每檔上限(萬)": ps//10000, "日均檔數": D.n.mean(), "資金使用率%": D.util.mean()*100, "日淨%": s.mean()*100, "日淨標準差%": s.std()*100, "Sharpe": s.mean()/s.std()*np.sqrt(245), "MDD%": ((1+s).cumprod()/(1+s).cumprod().cummax()-1).min()*100, "最差日%": s.min()*100, "最差月%": mo.min(), "負月比例": (mo < 0).mean(), "年化(萬)": s.mean()*245*CAP/1e4})
T = pd.DataFrame(rows); P("\n## 1. 全期間 (濾網 × 每檔上限)"); P(T.set_index(["濾網","每檔上限(萬)"]).round(3).to_markdown())
P("\n## 2. 各年日淨% / Sharpe (現行 vs 排 MA10 向上且加速 各集中度)")
yr = {}
for fname in ("無濾網 (現行選股)", "排 MA10 向上且加速"):
    for ps in (600_000, 1_000_000, 1_500_000):
        s = alloc(FILTERS[fname], ps).ret; lab = f"{fname[:6]} {ps//10000}萬"
        yr[f"{lab} 日淨"] = s.groupby(s.index.year).mean()*100; yr[f"{lab} Sharpe"] = s.groupby(s.index.year).apply(lambda x: x.mean()/x.std()*np.sqrt(245))
P(pd.DataFrame(yr).round(2).to_markdown())
P("\n## 3. 濾網後每天剩幾檔可做 (前 10 名裡)")
for fname, fm in FILTERS.items():
    n = S[fm].groupby("date").size().reindex(all_days, fill_value=0); P(f"- {fname}: 日均 {n.mean():.1f} 檔, ≤2 檔的天數 {(n <= 2).mean()*100:.1f}%, ≤4 檔 {(n <= 4).mean()*100:.1f}%")
(OUT/"slope_filter_concentrate.md").write_text("\n".join(L)); print("\n".join(L))
