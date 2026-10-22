"""訓練影子模型 Bseq (基準 33 + K 線序列 45 特徵) 給指定季度使用, 存 models/lgbm_Bseq_pool_<Q>.txt.
訓練集 = 該季之前全部 (D1 從 2018 起), 最後 2 個月 early-stopping. 用法: python3 scripts/22_train_seq_model.py [2026Q4]"""
import sys, glob, numpy as np, pandas as pd, lightgbm as lgb
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent)); import features as F
ROOT = Path(__file__).resolve().parent.parent
Q = pd.Period(sys.argv[1] if len(sys.argv) > 1 else pd.Timestamp.today().to_period("Q"), freq="Q")
d1 = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(str(ROOT/"data/raw/d1/*.parquet")))], ignore_index=True); d1["date"] = pd.to_datetime(d1.date)
d1 = d1.drop_duplicates(["stock_id","date"]).sort_values(["stock_id","date"]).reset_index(drop=True)
d1 = d1.merge(pd.read_parquet(ROOT/"data/raw/meta/info.parquet")[["stock_id","market_type"]], on="stock_id", how="left")
base = F.d1_base(d1); f = F.build_B(base).merge(F.build_seq(base), on=["stock_id","date"], how="left")
pool = f[F.universe_mask(f) & (f.dv20_prev>=5e8) & (f.atr20>=0.03) & f.y_intra.notna() & f.open.notna()].copy()
pool["y_dm"] = pool.y_intra.clip(-0.1, 0.1); pool["y_dm"] = pool.y_dm - pool.groupby("date").y_dm.transform("mean")
FEATS = F.D1_FEATS + F.SEQ_FEATS
ts = Q.start_time; tr_all = pool[pool.date < ts - pd.Timedelta(days=1)]; vcut = ts - pd.DateOffset(months=2)
tr, va = tr_all[tr_all.date < vcut], tr_all[tr_all.date >= vcut]
params = dict(objective="regression", learning_rate=0.03, num_leaves=31, min_data_in_leaf=100, feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1, lambda_l2=10.0, verbose=-1, num_threads=8, seed=42)
m = lgb.train(params, lgb.Dataset(tr[FEATS], tr.y_dm), num_boost_round=2000, valid_sets=[lgb.Dataset(va[FEATS], va.y_dm)], callbacks=[lgb.early_stopping(100, verbose=False)])
out = ROOT/"models"/f"lgbm_Bseq_pool_{Q}.txt"; m.save_model(str(out))
print(f"{Q}: train {len(tr)} ({tr.date.min().date()}~{tr.date.max().date()}) valid {len(va)} best_iter {m.best_iteration} -> {out.name}")
