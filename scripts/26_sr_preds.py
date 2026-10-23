"""產生正式模型 (sr = 基準+支撐壓力) 2020Q1~2026Q4 的 walk-forward 預測 → data/processed/preds_B_pool_sr_ext.parquet (供 24/25 以 --preds 使用)."""
import sys, glob, pandas as pd, lightgbm as lgb
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent)); import features as F
ROOT = Path(__file__).resolve().parent.parent
d1 = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(str(ROOT/"data/raw/d1/*.parquet")))], ignore_index=True); d1["date"] = pd.to_datetime(d1.date)
d1 = d1.drop_duplicates(["stock_id","date"]).sort_values(["stock_id","date"]).reset_index(drop=True)
d1 = d1.merge(pd.read_parquet(ROOT/"data/raw/meta/info.parquet")[["stock_id","market_type"]], on="stock_id", how="left")
base = F.d1_base(d1); f = F.build_B(base)
sr = pd.read_parquet(ROOT/"data/processed/sr_features.parquet"); f = f.merge(sr, on=["stock_id","date"], how="left")
pool = f[F.universe_mask(f) & (f.dv20_prev>=5e8) & (f.atr20>=0.03) & f.y_intra.notna() & f.open.notna()].copy().sort_values(["date","stock_id"]).reset_index(drop=True)
pool["y_dm"] = pool.y_intra.clip(-0.1, 0.1); pool["y_dm"] = pool.y_dm - pool.groupby("date").y_dm.transform("mean")
FEATS = F.D1_FEATS + F.SR_FEATS
params = dict(objective="regression", learning_rate=0.03, num_leaves=31, min_data_in_leaf=100, feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1, lambda_l2=10.0, verbose=-1, num_threads=8, seed=42)
preds = []
for q in pd.period_range("2020Q1", "2026Q4", freq="Q"):
    ts, te = q.start_time, q.end_time; test = pool[(pool.date>=ts)&(pool.date<=te)]
    tr_all = pool[pool.date < ts - pd.Timedelta(days=1)]; vcut = ts - pd.DateOffset(months=2); tr, va = tr_all[tr_all.date < vcut], tr_all[tr_all.date >= vcut]
    if len(test)==0 or len(tr) < 5000: continue
    m = lgb.train(params, lgb.Dataset(tr[FEATS], tr.y_dm), num_boost_round=2000, valid_sets=[lgb.Dataset(va[FEATS], va.y_dm)], callbacks=[lgb.early_stopping(100, verbose=False)])
    preds.append(pd.DataFrame({"date":test.date.values, "stock_id":test.stock_id.values, "pred":m.predict(test[FEATS], num_iteration=m.best_iteration)})); print(q, m.best_iteration, flush=True)
pd.concat(preds, ignore_index=True).to_parquet(ROOT/"data/processed/preds_B_pool_sr_ext.parquet", index=False); print("saved")
