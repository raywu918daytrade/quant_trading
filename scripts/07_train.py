"""LightGBM walk-forward (按季) 訓練 + 評估.  用法: python 07_train.py A|B"""
import sys, json, numpy as np, pandas as pd, lightgbm as lgb
from pathlib import Path
from scipy.stats import spearmanr
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
ROOT = Path(__file__).resolve().parent.parent; OUT = ROOT/"stats"; MD = ROOT/"models"
MODEL = sys.argv[1] if len(sys.argv) > 1 else "A"
SUF = "_pool" if len(sys.argv) > 2 and sys.argv[2] == "pool" else ""
TAG = MODEL + SUF

D1_FEATS = ["gap","gap_abs","gap_rel","p_ret","p_intra","p_range","p_gap","p_vol_ratio","p_up_lim","p_dn_lim","p_ret5","p_ret20","ret2","ret60",
            "pos20","ma5_dev","ma20_dev","ma60_dev","vol5","vol20","vol60","dv20_log","dv_ratio5","uplim_cnt5","price_log","lo_price","is_tpex",
            "mkt_gap","mkt_pret","mkt_ret5","mkt_breadth","mkt_uplim","dow"]
M1_FEATS = ["o30_ret","o15_ret","o05_ret","o30_range","o30_pos","o30_hi_vs_prev","o30_lo_vs_prev","o30_relvol","o05_relvol","o30_vwap_dev","o30_n","o30_momo","mkt_o30","o30_rel"]
if MODEL == "A":
    df = pd.read_parquet(ROOT/f"data/processed/dataset_A{SUF}.parquet"); FEATS = D1_FEATS + M1_FEATS; Y = "y_rest"; desc = "09:30 進場 → 13:30 出場 (D1 + 開盤30分 M1 特徵)"
else:
    df = pd.read_parquet(ROOT/f"data/processed/dataset_B{SUF}.parquet"); FEATS = D1_FEATS; Y = "y_intra"; desc = "09:00 開盤進場 → 13:30 出場 (D1 特徵 + 跳空)"
df = df.sort_values(["date","stock_id"]).reset_index(drop=True)
df["y_clip"] = df[Y].clip(-0.1, 0.1)
# 目標去掉當日市場平均 (訓練用), 評估仍用原始 y
df["y_dm"] = df.y_clip - df.groupby("date").y_clip.transform("mean")

folds = pd.period_range("2024Q4", "2026Q4", freq="Q")
params = dict(objective="regression", learning_rate=0.03, num_leaves=31, min_data_in_leaf=(100 if SUF else 300), feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1,
              lambda_l2=10.0, verbose=-1, num_threads=8, seed=42)
preds = []; imp = pd.Series(0.0, index=FEATS); fold_info = []
for q in folds:
    ts, te = q.start_time, q.end_time
    test = df[(df.date>=ts)&(df.date<=te)]
    if len(test)==0: continue
    tr_all = df[df.date < ts - pd.Timedelta(days=1)]
    vcut = ts - pd.DateOffset(months=2)
    tr, va = tr_all[tr_all.date < vcut], tr_all[tr_all.date >= vcut]
    dtr = lgb.Dataset(tr[FEATS], tr.y_dm); dva = lgb.Dataset(va[FEATS], va.y_dm)
    m = lgb.train(params, dtr, num_boost_round=2000, valid_sets=[dva], callbacks=[lgb.early_stopping(100, verbose=False)])
    p = m.predict(test[FEATS], num_iteration=m.best_iteration)
    preds.append(pd.DataFrame({"date":test.date.values, "stock_id":test.stock_id.values, "y":test[Y].values, "pred":p}))
    imp += pd.Series(m.feature_importance("gain"), index=FEATS)
    fold_info.append((str(q), len(tr), len(test), m.best_iteration))
    print(f"{q}: train {len(tr)} test {len(test)} best_iter {m.best_iteration}", flush=True)
    m.save_model(str(MD/f"lgbm_{TAG}_{q}.txt"))
pr = pd.concat(preds, ignore_index=True)
pr.to_parquet(ROOT/"data/processed"/f"preds_{TAG}.parquet", index=False)

# ---- 評估 ----
L = []; P = lambda s="": L.append(str(s))
P(f"# 模型 {TAG} 報告 — {desc}" + ("  [當沖池: dv20≥5億 & ATR20≥3%]" if SUF else "") + "\n")
P(f"- 目標: `{Y}`，特徵數 {len(FEATS)}，LightGBM 按季 walk-forward，訓練集=測試季之前全部 (最後 2 個月作 early-stopping 驗證)")
P("- folds (季, 訓練筆數, 測試筆數, best_iter): " + "; ".join(f"{a}:{b}/{c}/{d}" for a,b,c,d in fold_info)); P("")

ic = pr.groupby("date").apply(lambda x: spearmanr(x.pred, x.y)[0] if len(x)>20 else np.nan).dropna()
P("## 每日橫截面 rank-IC")
P(f"- mean {ic.mean():.4f}, std {ic.std():.4f}, IC-IR {ic.mean()/ic.std():.3f}, t-stat {ic.mean()/ic.std()*np.sqrt(len(ic)):.2f}, P(IC>0) {(ic>0).mean():.3f}, days {len(ic)}")
icq = ic.groupby(ic.index.to_period("Q")).agg(["mean","std","count"]); icq["IR"] = icq["mean"]/icq["std"]
P(icq.round(4).to_markdown()); P("")

pr["dec"] = pr.groupby("date").pred.transform(lambda s: pd.qcut(s.rank(method="first"), 10, labels=False)+1)
dec = pr.groupby("dec").y.agg(["mean","median","count"]); dec["mean"]*=100; dec["median"]*=100
dec["P(y>0)"] = pr.groupby("dec").y.apply(lambda s:(s>0).mean())
P("## 依預測十分位 (1=最看空, 10=最看多) 的實際報酬 (%)"); P(dec.round(3).to_markdown()); P("")

# ---- 回測: 每日等權 ----
COST_RT = 0.0023   # 現股當沖: 手續費 0.1425%*2*0.28折 + 證交稅 0.15% (當沖減半) ≈ 0.23%
def port(sel_long, sel_short=None):
    r = pr[sel_long].groupby("date").y.mean()
    if sel_short is not None:
        r = (r - pr[sel_short].groupby("date").y.mean())/2
    return r
pr["pct"] = pr.groupby("date").pred.rank(pct=True)
strat = {
    "Long top10%": port(pr.pct>0.9),
    "Long top5%":  port(pr.pct>0.95),
    "Long top2%":  port(pr.pct>0.98),
    "Short bottom10%": -port(pr.pct<=0.1),
    "L/S top-bottom 10%": port(pr.pct>0.9, pr.pct<=0.1),
    "L/S top-bottom 5%": port(pr.pct>0.95, pr.pct<=0.05),
    "All (universe mean)": port(pr.pct>-1),
}
rows = []
for k, r in strat.items():
    net = r - COST_RT
    rows.append({"策略":k, "日均毛報酬%": r.mean()*100, "日均淨報酬%": net.mean()*100, "日std%": r.std()*100,
                 "年化Sharpe(淨)": net.mean()/net.std()*np.sqrt(245), "勝率(淨)": (net>0).mean(), "最大回撤(淨)%": ((1+net).cumprod()/(1+net).cumprod().cummax()-1).min()*100,
                 "累積淨%": ((1+net).prod()-1)*100})
bt = pd.DataFrame(rows).set_index("策略")
P(f"## 回測 (每日等權, 進出各一次, 來回成本 {COST_RT*100:.2f}%)"); P(bt.round(3).to_markdown()); P("")
yr = pd.DataFrame({k:(v-COST_RT).groupby(v.index.year).mean()*100 for k,v in strat.items()})
P("- 各年日均淨報酬 %:"); P(yr.round(3).to_markdown()); P("")

P("## 特徵重要度 (gain, 各 fold 加總, 前 20)")
P((imp.sort_values(ascending=False).head(20)/imp.sum()*100).round(2).to_frame("gain%").to_markdown()); P("")
(OUT/f"model_{TAG}_report.md").write_text("\n".join(L)); print("\n".join(L))

fig, ax = plt.subplots(1,3, figsize=(18,5))
ic.rolling(20).mean().plot(ax=ax[0]); ax[0].axhline(0,color="k",lw=.5); ax[0].set_title(f"model {TAG}: daily rank-IC (20d MA)")
dec["mean"].plot(kind="bar", ax=ax[1]); ax[1].set_title("decile mean return (%)")
for k in ["Long top5%","L/S top-bottom 10%","All (universe mean)"]:
    ((1+strat[k]-COST_RT).cumprod()-1).plot(ax=ax[2], label=k+" net")
ax[2].legend(); ax[2].set_title("cumulative net return"); ax[2].axhline(0,color="k",lw=.5)
plt.tight_layout(); plt.savefig(OUT/f"model_{TAG}.png", dpi=110)
