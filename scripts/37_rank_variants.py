"""提高勝率的模型端候選 (特徵同正式 sr 模型, walk-forward 2020Q1~2026Q4):
  cls   : 分類目標 (y = 開→收 跌幅 > 0.23% 成本, 即空單會賺) 用機率排序 → preds_B_pool_cls_ext.parquet
  radj  : 現行迴歸預測 / ATR20 (風險調整後排序) → preds_B_pool_radj_ext.parquet (不用重訓)
  mix   : 迴歸名次 + 分類名次 平均 → preds_B_pool_mix_ext.parquet
之後各用 24 (--preds --tag) 跑 M1 精確回測. 用法: python3 scripts/37_rank_variants.py"""
import sys, glob, numpy as np, pandas as pd, lightgbm as lgb
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent)); import features as F
ROOT = Path(__file__).resolve().parent.parent
d1 = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(str(ROOT/"data/raw/d1/*.parquet")))], ignore_index=True); d1["date"] = pd.to_datetime(d1.date)
d1 = d1.drop_duplicates(["stock_id","date"]).sort_values(["stock_id","date"]).reset_index(drop=True)
d1 = d1.merge(pd.read_parquet(ROOT/"data/raw/meta/info.parquet")[["stock_id","market_type"]], on="stock_id", how="left")
base = F.d1_base(d1); f = F.build_B(base); f = f.merge(pd.read_parquet(ROOT/"data/processed/sr_features.parquet"), on=["stock_id","date"], how="left")
pool = f[F.universe_mask(f) & (f.dv20_prev>=5e8) & (f.atr20>=0.03) & f.y_intra.notna() & f.open.notna()].copy().sort_values(["date","stock_id"]).reset_index(drop=True)
pool["y_cls"] = (pool.y_intra < -0.0023).astype(int)
FEATS = F.D1_FEATS + F.SR_FEATS
params = dict(objective="binary", learning_rate=0.03, num_leaves=31, min_data_in_leaf=100, feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1, lambda_l2=10.0, verbose=-1, num_threads=8, seed=42)
preds = []
for q in pd.period_range("2020Q1", "2026Q4", freq="Q"):
    ts, te = q.start_time, q.end_time; test = pool[(pool.date>=ts)&(pool.date<=te)]
    tr_all = pool[pool.date < ts - pd.Timedelta(days=1)]; vcut = ts - pd.DateOffset(months=2); tr, va = tr_all[tr_all.date < vcut], tr_all[tr_all.date >= vcut]
    if len(test)==0 or len(tr) < 5000: continue
    m = lgb.train(params, lgb.Dataset(tr[FEATS], tr.y_cls), num_boost_round=2000, valid_sets=[lgb.Dataset(va[FEATS], va.y_cls)], callbacks=[lgb.early_stopping(100, verbose=False)])
    preds.append(pd.DataFrame({"date":test.date.values, "stock_id":test.stock_id.values, "prob":m.predict(test[FEATS], num_iteration=m.best_iteration)})); print(q, m.best_iteration, flush=True)
C = pd.concat(preds, ignore_index=True)
# 24 的排名是 pred 由小到大 (rank 1 = 最看跌), 所以分類輸出要取負
C.assign(pred=-C.prob)[["date","stock_id","pred"]].to_parquet(ROOT/"data/processed/preds_B_pool_cls_ext.parquet", index=False)
reg = pd.read_parquet(ROOT/"data/processed/preds_B_pool_sr_ext.parquet").merge(pool[["stock_id","date","atr20"]], on=["stock_id","date"])
reg.assign(pred=reg.pred/reg.atr20)[["date","stock_id","pred"]].to_parquet(ROOT/"data/processed/preds_B_pool_radj_ext.parquet", index=False)
mx = reg.merge(C, on=["stock_id","date"]); mx["pred"] = (mx.groupby("date").pred.rank() + mx.groupby("date").prob.rank(ascending=False))/2
mx[["date","stock_id","pred"]].to_parquet(ROOT/"data/processed/preds_B_pool_mix_ext.parquet", index=False); print("saved cls/radj/mix")
