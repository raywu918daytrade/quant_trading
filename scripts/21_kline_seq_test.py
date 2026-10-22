"""K 線序列特徵對照實驗: 基準 33 特徵 vs 基準 + 最近 10 天 K 棒序列. 同一 pipeline 按季 walk-forward 2020Q1~2026Q4 各訓一組,
比每年 rank-IC 與定案策略 (M1 精確成交, 300 萬配置) 績效. 輸出 stats/kline_seq_test.md, data/processed/preds_B_pool_seq.parquet."""
import sys, glob, numpy as np, pandas as pd, lightgbm as lgb
from pathlib import Path
from scipy.stats import spearmanr
sys.path.insert(0, str(Path(__file__).resolve().parent)); import features as F
ROOT = Path(__file__).resolve().parent.parent; OUT = ROOT/"stats"
BASE, SLIP, K, PEN, CAP, PER_STOCK, MAX_N, MAX_PX, GAP_MAX, MKT_MIN = 0.0023, 0.001, 0.08, 0.07, 3_000_000, 600_000, 10, 300, 0.04, -0.01
TEST_Q = pd.period_range("2020Q1", "2026Q4", freq="Q")

fs = sorted(glob.glob(str(ROOT/"data/raw/d1/*.parquet")))
d1 = pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True); d1["date"] = pd.to_datetime(d1.date)
d1 = d1.drop_duplicates(["stock_id","date"]).sort_values(["stock_id","date"]).reset_index(drop=True)
d1 = d1.merge(pd.read_parquet(ROOT/"data/raw/meta/info.parquet")[["stock_id","market_type"]], on="stock_id", how="left")
base = F.d1_base(d1); f = F.build_B(base)
# ---- K 線序列特徵 (全部 shift ≥1, 只用前一日以前) ----
g = base.groupby("stock_id", sort=False)
seq = base[["stock_id","date"]].copy()
up_wick = (base.high - base[["open","close"]].max(axis=1)) / base.prev_close
lo_wick = (base[["open","close"]].min(axis=1) - base.low) / base.prev_close
SEQ = []
for k in range(1, 11):
    for name, s in [("ret", base.ret), ("vr", base.vol_ratio)] + ([("intra", base.intra), ("rng", base.range), ("uw", up_wick), ("lw", lo_wick), ("gap", base.gap)] if k <= 5 else []):
        col = f"k{k}_{name}"; seq[col] = s.groupby(base.stock_id, sort=False).shift(k).values; SEQ.append(col)
f = f.merge(seq, on=["stock_id","date"], how="left")
nxt = d1[["stock_id","date","open"]].sort_values(["stock_id","date"]); nxt["next_open"] = nxt.groupby("stock_id").open.shift(-1)
f = f.merge(nxt[["stock_id","date","next_open"]], on=["stock_id","date"], how="left")
pool = f[F.universe_mask(f) & (f.dv20_prev>=5e8) & (f.atr20>=0.03) & f.y_intra.notna() & f.open.notna()].copy().sort_values(["date","stock_id"]).reset_index(drop=True)
pool["y_clip"] = pool.y_intra.clip(-0.1, 0.1); pool["y_dm"] = pool.y_clip - pool.groupby("date").y_clip.transform("mean")
print(f"pool {len(pool)} rows; seq feats {len(SEQ)}", flush=True)
params = dict(objective="regression", learning_rate=0.03, num_leaves=31, min_data_in_leaf=100, feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1, lambda_l2=10.0, verbose=-1, num_threads=8, seed=42)

def walk(feats, tag):
    preds = []; imp = pd.Series(0.0, index=feats)
    for q in TEST_Q:
        ts, te = q.start_time, q.end_time; test = pool[(pool.date>=ts)&(pool.date<=te)]
        tr_all = pool[pool.date < ts - pd.Timedelta(days=1)]; vcut = ts - pd.DateOffset(months=2); tr, va = tr_all[tr_all.date < vcut], tr_all[tr_all.date >= vcut]
        if len(test)==0 or len(tr) < 5000: continue
        m = lgb.train(params, lgb.Dataset(tr[feats], tr.y_dm), num_boost_round=2000, valid_sets=[lgb.Dataset(va[feats], va.y_dm)], callbacks=[lgb.early_stopping(100, verbose=False)])
        preds.append(pd.DataFrame({"date":test.date.values, "stock_id":test.stock_id.values, "pred":m.predict(test[feats], num_iteration=m.best_iteration)}))
        imp += pd.Series(m.feature_importance("gain"), index=feats); print(f"{tag} {q}: train {len(tr)} best_iter {m.best_iteration}", flush=True)
    return pd.concat(preds, ignore_index=True), imp
pr_base, imp_base = walk(F.D1_FEATS, "base"); pr_seq, imp_seq = walk(F.D1_FEATS + SEQ, "seq")
pr_seq.to_parquet(ROOT/"data/processed/preds_B_pool_seq.parquet", index=False); pr_base.to_parquet(ROOT/"data/processed/preds_B_pool_base2020.parquet", index=False)

ext = pd.concat([pd.read_parquet(x) for x in sorted(glob.glob(str(ROOT/"data/processed/m1_entry_ext/*.parquet")))], ignore_index=True)
def strategy(pr):
    d = pool.merge(pr, on=["stock_id","date"]).merge(ext, on=["stock_id","date"]); d = d[d.e01_vwap.notna()].copy()
    d["entry"] = d.e01_vwap; d["exit"] = d.close_auction_c.fillna(d.m1_close); d = d[d.exit.notna()].copy()
    d["lock"] = d.exit >= d.prev_close*1.0975; trig = (d.e01_maxhigh_after >= d.prev_close*(1+K)) & (d.entry < d.prev_close*(1+K))
    d["gross"] = np.where(trig, d.entry/(d.prev_close*(1+K)*(1+SLIP*3))-1, np.where(d.lock & d.next_open.notna(), d.entry/d.next_open-1-PEN, d.entry/d.exit-1)); d["net"] = d.gross-BASE
    e = d[(d.gap<=GAP_MAX)&(d.mkt_gap>=MKT_MIN)&(d.entry>d.prev_close*0.905)&(d.entry<=MAX_PX)&(d.date.dt.dayofweek!=0)].copy()
    e["rank"] = e.groupby("date").pred.rank(method="first"); top = e[e["rank"]<=MAX_N]
    recs = []
    for dt, gg in top.sort_values(["date","rank"]).groupby("date", sort=False):
        cash = CAP; amts=[]; nets=[]
        for _, r in gg.iterrows():
            lc = r.entry*1000
            if cash < lc: continue
            lots = int(min(PER_STOCK, cash)//lc)
            if lots < 1: continue
            a = lots*lc; cash -= a; amts.append(a); nets.append(r.net)
        recs.append((dt, sum(a*n for a, n in zip(amts, nets))/CAP))
    s = pd.DataFrame(recs, columns=["date","ret"]).set_index("date").ret.reindex(pd.Series(sorted(d.date.unique())), fill_value=0.0)
    ic = d.groupby("date").apply(lambda x: spearmanr(x.pred, x.y_intra)[0] if len(x)>20 else np.nan).dropna()
    return s, top, ic
L = []; P = lambda s="": L.append(str(s))
P("# K 線序列特徵對照 (2020Q1~2026Q4 按季 walk-forward, 兩組模型同 pipeline 同訓練集)\n")
P(f"- 基準: 33 特徵 (features.D1_FEATS)。序列: 基準 + {len(SEQ)} 個 (最近 5 天每天的 ret/開到收/振幅/上影/下影/量比/跳空, 第 6~10 天的 ret/量比), 全部 shift≥1。")
P("- 策略: 定案規則 + M1 精確成交 + 300 萬配置, 不交易日計 0。\n")
res = {"基準 33 特徵": strategy(pr_base), "基準 + K 線序列": strategy(pr_seq)}
rows = []
for yr in range(2020, 2027):
    r = {"年": yr}
    for k, (s, top, ic) in res.items():
        ss = s[s.index.year==yr]; tt = top[top.date.dt.year==yr]; ii = ic[ic.index.year==yr]
        r[f"{k} IC"] = ii.mean(); r[f"{k} 每筆%"] = tt.net.mean()*100; r[f"{k} 日淨%"] = ss.mean()*100; r[f"{k} Sharpe"] = ss.mean()/ss.std()*np.sqrt(245)
    rows.append(r)
r = {"年": "全期間"}
for k, (s, top, ic) in res.items():
    r[f"{k} IC"] = ic.mean(); r[f"{k} 每筆%"] = top.net.mean()*100; r[f"{k} 日淨%"] = s.mean()*100; r[f"{k} Sharpe"] = s.mean()/s.std()*np.sqrt(245)
rows.append(r)
tab = pd.DataFrame(rows).set_index("年")
P("## 各年比較"); P(tab.round(3).to_markdown()); P("")
for k, (s, top, ic) in res.items():
    P(f"- {k}: 全期間 MDD {((1+s).cumprod()/(1+s).cumprod().cummax()-1).min()*100:.1f}%，負月比例 {(s.groupby(s.index.to_period('M')).apply(lambda x:(1+x).prod()-1)<0).mean()*100:.0f}%，IC-IR {ic.mean()/ic.std():.2f}，P(IC>0) {(ic>0).mean():.2f}")
both = res["基準 33 特徵"][2].to_frame("b").join(res["基準 + K 線序列"][2].to_frame("s")).dropna(); diff = both.s - both.b
P(f"- 每日 IC 差 (序列 − 基準): 平均 {diff.mean():.4f}, t-stat {diff.mean()/diff.std()*np.sqrt(len(diff)):.2f}, 序列較高的天數比 {(diff>0).mean():.2f}")
P(""); P("## 序列模型特徵重要度 (gain%, 前 25)"); P((imp_seq.sort_values(ascending=False).head(25)/imp_seq.sum()*100).round(2).to_frame("gain%").to_markdown())
P(""); P(f"- 序列特徵合計 gain 占比 {imp_seq[SEQ].sum()/imp_seq.sum()*100:.1f}%")
(OUT/"kline_seq_test.md").write_text("\n".join(L)); print("\n".join(L))
