"""開盤量 PR 當模型特徵: 在正式模型 (D1 + SR) 上加 09:00 第一根 M1 量 (vol1) 或 09:00~09:04 五分鐘量 (vol5) 的 PR/比值特徵, walk-forward 2020Q1~2026Q4,
存 preds_B_pool_srvol{1,5}_ext.parquet, 之後用 24 (--preds, --tag) 跑 M1 精確回測比較. vol5 的進場時間其實要到 09:05, 這裡先用現行 09:01 進場 (樂觀上限).
特徵 (N = 5/20/60): pr{1,5}_N = 今日量在前 N 日同時段的百分位; lr{1,5}_20 = log(今日量 / 前 20 日平均); r{1,5}_day = 今日量 / 前 20 日全日量平均 (M1 張).
M1 只有部分股票 (每月 ~400 檔), 沒有的填 NaN. 用法: python3 scripts/32_open_volume_feature.py [build|train vol1|train vol5]"""
import sys, glob, numpy as np, pandas as pd
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
sys.path.insert(0, str(Path(__file__).resolve().parent)); import features as F
ROOT = Path(__file__).resolve().parent.parent; CACHE = ROOT/"data/processed/open_vol_features.parquet"; NS = (5, 20, 60)
VOL_FEATS = {k: [f"pr{k}_{n}" for n in NS] + [f"lr{k}_20", f"r{k}_day"] for k in ("1", "5")}

def month_vol(f):
    d = pd.read_parquet(f, columns=["stock_id","date","volume"]); ts = pd.to_datetime(d["date"]); d["day"] = ts.dt.normalize(); d["t"] = ts.dt.hour*60 + ts.dt.minute - 540
    d = d[(d.t >= 0) & (d.t <= 270)]; g = d.groupby(["stock_id","day"])
    return pd.DataFrame({"v1": g.apply(lambda x: x.volume[x.t == 0].sum()), "v5": g.apply(lambda x: x.volume[x.t <= 4].sum()), "vday": g.volume.sum()}).reset_index()

def build():
    with ProcessPoolExecutor(6) as ex: parts = list(ex.map(month_vol, sorted(glob.glob(str(ROOT/"data/raw/m1/*.parquet")))))
    v = pd.concat(parts, ignore_index=True).sort_values(["stock_id","day"]).reset_index(drop=True); out = []
    for sid, g in v.groupby("stock_id", sort=False):
        g = g.copy()
        for k in ("1", "5"):
            x = g[f"v{k}"].values; n = len(x)
            for N in NS:
                pr = np.full(n, np.nan)
                for i in range(n):
                    p = x[max(0, i-N):i]
                    if len(p) >= min(N, 5): pr[i] = (p < x[i]).mean()*100
                g[f"pr{k}_{N}"] = pr
            m20 = g[f"v{k}"].shift(1).rolling(20, min_periods=5).mean(); g[f"lr{k}_20"] = np.log((g[f"v{k}"]+1)/(m20+1))
            g[f"r{k}_day"] = g[f"v{k}"]/g.vday.shift(1).rolling(20, min_periods=5).mean()
        out.append(g)
    v = pd.concat(out, ignore_index=True).rename(columns={"day": "date"}); v.to_parquet(CACHE, index=False); print("features", len(v), v.date.min().date(), v.date.max().date())

def train(k):
    d1 = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(str(ROOT/"data/raw/d1/*.parquet")))], ignore_index=True); d1["date"] = pd.to_datetime(d1.date)
    d1 = d1.drop_duplicates(["stock_id","date"]).sort_values(["stock_id","date"]).reset_index(drop=True)
    d1 = d1.merge(pd.read_parquet(ROOT/"data/raw/meta/info.parquet")[["stock_id","market_type"]], on="stock_id", how="left")
    base = F.d1_base(d1); f = F.build_B(base); f = f.merge(pd.read_parquet(ROOT/"data/processed/sr_features.parquet"), on=["stock_id","date"], how="left")
    vf = pd.read_parquet(CACHE)[["stock_id","date"] + VOL_FEATS[k]]; f = f.merge(vf, on=["stock_id","date"], how="left")
    pool = f[F.universe_mask(f) & (f.dv20_prev>=5e8) & (f.atr20>=0.03) & f.y_intra.notna() & f.open.notna()].copy().sort_values(["date","stock_id"]).reset_index(drop=True)
    pool["y_dm"] = pool.y_intra.clip(-0.1, 0.1); pool["y_dm"] = pool.y_dm - pool.groupby("date").y_dm.transform("mean")
    p20 = pool[pool.date >= "2020-01-01"]; print(f"pool 2020+ {len(p20)} 筆, 有開盤量特徵 {p20[VOL_FEATS[k][0]].notna().mean()*100:.1f}%", flush=True)
    FEATS = F.D1_FEATS + F.SR_FEATS + VOL_FEATS[k]; import lightgbm as lgb
    params = dict(objective="regression", learning_rate=0.03, num_leaves=31, min_data_in_leaf=100, feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1, lambda_l2=10.0, verbose=-1, num_threads=8, seed=42)
    preds = []; imps = []
    for q in pd.period_range("2020Q1", "2026Q4", freq="Q"):
        ts, te = q.start_time, q.end_time; test = pool[(pool.date>=ts)&(pool.date<=te)]
        tr_all = pool[pool.date < ts - pd.Timedelta(days=1)]; vcut = ts - pd.DateOffset(months=2); tr, va = tr_all[tr_all.date < vcut], tr_all[tr_all.date >= vcut]
        if len(test)==0 or len(tr) < 5000: continue
        m = lgb.train(params, lgb.Dataset(tr[FEATS], tr.y_dm), num_boost_round=2000, valid_sets=[lgb.Dataset(va[FEATS], va.y_dm)], callbacks=[lgb.early_stopping(100, verbose=False)])
        preds.append(pd.DataFrame({"date":test.date.values, "stock_id":test.stock_id.values, "pred":m.predict(test[FEATS], num_iteration=m.best_iteration)}))
        imps.append(pd.Series(m.feature_importance("gain"), index=FEATS)); print(q, m.best_iteration, flush=True)
    pd.concat(preds, ignore_index=True).to_parquet(ROOT/"data/processed"/f"preds_B_pool_srvol{k}_ext.parquet", index=False)
    imp = pd.concat(imps, axis=1).mean(axis=1); imp = imp/imp.sum()*100; imp.sort_values(ascending=False).round(2).to_csv(ROOT/"stats"/f"open_vol_feature_importance_vol{k}.csv")
    print("importance top 15:\n", imp.sort_values(ascending=False).head(15).round(2).to_string()); print("vol feats:\n", imp[VOL_FEATS[k]].round(2).to_string())

if __name__ == "__main__":
    if sys.argv[1] == "build": build()
    else: train(sys.argv[2][-1])
