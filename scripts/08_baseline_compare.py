"""模型 vs 單純波動排序基準; 以及波動分組內的殘餘 IC."""
import numpy as np, pandas as pd
from pathlib import Path
from scipy.stats import spearmanr
import sys
ROOT = Path(__file__).resolve().parent.parent
SUF = "_pool" if len(sys.argv) > 1 and sys.argv[1] == "pool" else ""
COST = 0.0023
L = []; P = lambda s="": L.append(str(s))
P("# 模型 vs 基準比較" + ("  [當沖池]" if SUF else "") + "\n")
def ic_series(d, col):
    return d.groupby("date").apply(lambda x: spearmanr(x[col], x.y, nan_policy="omit")[0] if len(x)>20 else np.nan).dropna()
def bt(d, col):
    pct = d.groupby("date")[col].rank(pct=True)
    s = -d[pct<=0.1].groupby("date").y.mean() - COST
    ls = ((d[pct>0.9].groupby("date").y.mean() - d[pct<=0.1].groupby("date").y.mean())/2) - COST
    return s, ls
for M, Y in [("A","y_rest"),("B","y_intra")]:
    pr = pd.read_parquet(ROOT/f"data/processed/preds_{M}{SUF}.parquet")
    ds = pd.read_parquet(ROOT/f"data/processed/dataset_{M}{SUF}.parquet")[["stock_id","date","vol60","vol20","o30_range","gap","p_ret20"]]
    d = pr.merge(ds, on=["stock_id","date"])
    d["neg_vol60"] = -d.vol60; d["neg_vol20"] = -d.vol20
    d["neg_o30_range"] = -d.o30_range.fillna(0)
    d["vol_combo"] = -(d.groupby("date").vol60.rank(pct=True) + d.groupby("date").vol20.rank(pct=True))
    P(f"## 模型 {M} (目標 {Y})")
    rows = []
    for col, name in [("pred","LightGBM 模型"),("neg_vol60","基準: -vol60"),("neg_vol20","基準: -vol20"),("vol_combo","基準: -(rank vol60 + rank vol20)")] + ([("neg_o30_range","基準: -開盤30分振幅")] if M=="A" else []):
        ic = ic_series(d, col); s, ls = bt(d, col)
        rows.append({"預測訊號":name, "IC mean":ic.mean(), "IC-IR":ic.mean()/ic.std(), "P(IC>0)":(ic>0).mean(),
                     "Short bottom10% 日淨%":s.mean()*100, "Short Sharpe":s.mean()/s.std()*np.sqrt(245),
                     "L/S 10% 日淨%":ls.mean()*100, "L/S Sharpe":ls.mean()/ls.std()*np.sqrt(245)})
    P(pd.DataFrame(rows).set_index("預測訊號").round(4).to_markdown()); P("")
    # 波動分組內的殘餘 IC
    d["vq"] = d.groupby("date").vol60.transform(lambda s: pd.qcut(s.rank(method="first"), 5, labels=False))
    res = d.groupby(["date","vq"]).apply(lambda x: spearmanr(x.pred, x.y)[0] if len(x)>15 else np.nan).dropna()
    P(f"- 同日同 vol60 五分位內的模型 rank-IC: mean {res.mean():.4f}, IR {res.mean()/res.std():.3f}, P>0 {(res>0).mean():.3f}  (若接近 0 代表模型≈波動排序)")
    byq = res.groupby(level=1).mean()
    P("- 各波動分位內 IC (0=低波動, 4=高波動): " + ", ".join(f"Q{int(k)}={v:.4f}" for k,v in byq.items()))
    # 與 -vol60 的相關
    rc = d.groupby("date").apply(lambda x: spearmanr(x.pred, x.neg_vol60)[0]).mean()
    P(f"- 模型預測 與 -vol60 的日均 rank 相關: {rc:.3f}"); P("")
(ROOT/f"stats/model_vs_baseline{SUF}.md").write_text("\n".join(L)); print("\n".join(L))
