"""把定案策略 (S9'' + 週一不做 + 300 萬配置) 往前延伸到 2020~2024Q3 做樣本外檢驗.

資料: data/raw/d1 需含 2018-01 起的月檔 (2018~2019 只當訓練). 特徵用 scripts/features.py (與訓練/每日腳本同一份).
模型: 池版模型 B, 參數同 07_train.py, 按季 walk-forward (訓練 = 測試季之前全部, 最後 2 個月 early-stopping). 測試季 2020Q1~2024Q3,
      模型存 models/ext/ (不動正式用的 models/lgbm_B_pool_*). 2024Q4 之後沿用現有 preds_B_pool.parquet (正式模型的樣本外預測).
成交假設 (D1 近似, 因 2020~2023 沒有 M1):
  進場 = open × (1 + ENTRY_ADJ); ENTRY_ADJ 由 2024Q4+ 有 M1 的期間校準 (策略交易的 09:01~09:03 VWAP / open - 1 的平均, 從 m1_entry 算).
  出場 = close (≈ 13:30 收盤競價). 強制回補: high ≥ 昨收×1.08 且 open < 昨收×1.08 → 以 昨收×1.08×(1+0.3%) 回補.
  鎖漲停: close ≥ 昨收×1.0975 且未觸發強制回補 → 隔日開盤回補 - 7% 罰金. 來回成本 0.23%.
規則: 池 dv20≥5億 & ATR20≥3% & 上市櫃非ETF & 價≥5; 跳空≤4%; 大盤平均跳空≥-1%; 不在跌停附近; 股價≤300; 非週一; 模型排名往下填,
      每檔上限 60 萬, 300 萬填滿, 最多 10 檔.
輸出 stats/extended_backtest.md, data/processed/preds_B_pool_ext.parquet.
"""
import sys, glob, json, numpy as np, pandas as pd, lightgbm as lgb
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent)); import features as F
ROOT = Path(__file__).resolve().parent.parent; OUT = ROOT/"stats"; MD = ROOT/"models/ext"; MD.mkdir(parents=True, exist_ok=True)
BASE, SLIP, K, PEN, CAP, PER_STOCK, MAX_N, MAX_PX, GAP_MAX, MKT_MIN = 0.0023, 0.001, 0.08, 0.07, 3_000_000, 600_000, 10, 300, 0.04, -0.01
POOL_DV, POOL_ATR = 5e8, 0.03
TEST_Q = pd.period_range("2020Q1", "2024Q3", freq="Q")
RETRAIN = "--retrain" in sys.argv

# ---------- 1. D1 + 特徵 ----------
fs = sorted(glob.glob(str(ROOT/"data/raw/d1/*.parquet")))
d1 = pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)
d1["date"] = pd.to_datetime(d1["date"]); d1 = d1.drop_duplicates(["stock_id","date"]).sort_values(["stock_id","date"]).reset_index(drop=True)
info = pd.read_parquet(ROOT/"data/raw/meta/info.parquet"); d1 = d1.merge(info[["stock_id","name","market_type"]], on="stock_id", how="left")
print(f"D1: {len(d1)} rows, {d1.date.min().date()} ~ {d1.date.max().date()}, {d1.stock_id.nunique()} stocks", flush=True)
f = F.build_B(F.d1_base(d1))
d1h = d1[["stock_id","date","high","low","close","open"]].sort_values(["stock_id","date"]); d1h["next_open"] = d1h.groupby("stock_id").open.shift(-1)
f = f.merge(d1h[["stock_id","date","high","next_open"]], on=["stock_id","date"], how="left")
pool = f[F.universe_mask(f) & (f.dv20_prev>=POOL_DV) & (f.atr20>=POOL_ATR) & f.y_intra.notna() & f.open.notna()].copy()
pool = pool.sort_values(["date","stock_id"]).reset_index(drop=True)
print(f"pool rows {len(pool)}, {pool.date.min().date()} ~ {pool.date.max().date()}, per-day median {pool.groupby('date').size().median():.0f}", flush=True)

# ---------- 2. walk-forward 訓練 2020Q1~2024Q3 ----------
pool["y_clip"] = pool.y_intra.clip(-0.1, 0.1); pool["y_dm"] = pool.y_clip - pool.groupby("date").y_clip.transform("mean")
params = dict(objective="regression", learning_rate=0.03, num_leaves=31, min_data_in_leaf=100, feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1,
              lambda_l2=10.0, verbose=-1, num_threads=8, seed=42)
pred_path = ROOT/"data/processed/preds_B_pool_ext.parquet"
if pred_path.exists() and not RETRAIN:
    pr_ext = pd.read_parquet(pred_path); print("loaded cached ext preds", len(pr_ext), flush=True)
else:
    preds = []
    for q in TEST_Q:
        ts, te = q.start_time, q.end_time
        test = pool[(pool.date>=ts)&(pool.date<=te)]
        tr_all = pool[pool.date < ts - pd.Timedelta(days=1)]; vcut = ts - pd.DateOffset(months=2)
        tr, va = tr_all[tr_all.date < vcut], tr_all[tr_all.date >= vcut]
        if len(test)==0 or len(tr) < 5000: print("skip", q, len(tr), len(test)); continue
        m = lgb.train(params, lgb.Dataset(tr[F.D1_FEATS], tr.y_dm), num_boost_round=2000, valid_sets=[lgb.Dataset(va[F.D1_FEATS], va.y_dm)], callbacks=[lgb.early_stopping(100, verbose=False)])
        p = m.predict(test[F.D1_FEATS], num_iteration=m.best_iteration)
        preds.append(pd.DataFrame({"date":test.date.values, "stock_id":test.stock_id.values, "y":test.y_intra.values, "pred":p}))
        m.save_model(str(MD/f"lgbm_B_pool_{q}.txt")); print(f"{q}: train {len(tr)} test {len(test)} best_iter {m.best_iteration}", flush=True)
    pr_ext = pd.concat(preds, ignore_index=True); pr_ext.to_parquet(pred_path, index=False)
pr_cur = pd.read_parquet(ROOT/"data/processed/preds_B_pool.parquet")[["date","stock_id","pred"]]
pr = pd.concat([pr_ext[["date","stock_id","pred"]], pr_cur[pr_cur.date>="2024-10-01"]], ignore_index=True)

# ---------- 3. 校準進場價調整 (2024Q4+ 有 M1) ----------
ent = pd.concat([pd.read_parquet(f_, columns=["stock_id","date","e01_vwap"]) for f_ in sorted(glob.glob(str(ROOT/"data/processed/m1_entry_delay/*.parquet")))])
ecol = "e01_vwap"   # 09:01~09:03 三分鐘 VWAP (可執行的進場假設)
cal = pool.merge(pr, on=["stock_id","date"]).merge(ent, on=["stock_id","date"])
cal = cal[(cal.date>="2024-10-01")&(cal.gap<=GAP_MAX)&(cal.mkt_gap>=MKT_MIN)&(cal.open<=MAX_PX)]
cal["rank"] = cal.groupby("date").pred.rank(method="first"); cal = cal[cal["rank"]<=MAX_N]
ENTRY_ADJ = float((cal[ecol]/cal.open - 1).mean())
print(f"entry adj ({ecol} vs open) over {len(cal)} strategy trades: {ENTRY_ADJ*100:+.3f}%", flush=True)

# ---------- 4. 策略模擬: D1 近似 與 M1 精確 ----------
def build_trades(mode):
    d = pool.merge(pr, on=["stock_id","date"], how="inner")
    if mode == "m1":
        ext = pd.concat([pd.read_parquet(f_) for f_ in sorted(glob.glob(str(ROOT/"data/processed/m1_entry_ext/*.parquet")))], ignore_index=True)
        d = d.merge(ext, on=["stock_id","date"], how="inner")
        d = d[d.e01_vwap.notna()].copy(); d["entry"] = d.e01_vwap; d["exit"] = d.close_auction_c.fillna(d.m1_close); hi = d.e01_maxhigh_after
        d = d[d.exit.notna()].copy()
    else:
        d["entry"] = d.open*(1+ENTRY_ADJ); d["exit"] = d.close; hi = d.high
    d["lock"] = d.exit >= d.prev_close*1.0975
    trig = (hi >= d.prev_close*(1+K)) & (d.entry < d.prev_close*(1+K))
    d["gross"] = np.where(trig, d.entry/(d.prev_close*(1+K)*(1+SLIP*3))-1, np.where(d.lock & d.next_open.notna(), d.entry/d.next_open-1-PEN, d.entry/d.exit-1))
    d["net"] = d.gross - BASE; d["forced"] = trig; d["dow"] = d.date.dt.dayofweek
    e = d[(d.gap<=GAP_MAX)&(d.mkt_gap>=MKT_MIN)&(d.entry>d.prev_close*0.905)&(d.entry<=MAX_PX)&(d.dow!=0)].copy()
    e["rank"] = e.groupby("date").pred.rank(method="first")
    return d, e, e[e["rank"]<=MAX_N].copy()

def alloc(y):
    recs = []
    for dt, g in y.sort_values(["date","rank"]).groupby("date", sort=False):
        cash = CAP; amts = []; nets = []
        for _, r in g.iterrows():
            lc = r.entry*1000
            if cash < lc: continue
            lots = int(min(PER_STOCK, cash)//lc)
            if lots < 1: continue
            a = lots*lc; cash -= a; amts.append(a); nets.append(r.net)
            if len(amts) >= MAX_N: break
        recs.append((dt, len(amts), sum(amts)/CAP, sum(a*n for a, n in zip(amts, nets))/CAP))
    return pd.DataFrame(recs, columns=["date","n","util","ret"]).set_index("date")

def stats(s, trades_, d):
    s = s.dropna()
    return {"交易日數": int(len(s)), "有交易天數": int((trades_.groupby("date").size().reindex(s.index).fillna(0)>0).sum()), "筆數": int(len(trades_)),
            "每筆淨%": trades_.net.mean()*100, "勝率": (trades_.net>0).mean(), "日淨%(對300萬)": s.mean()*100, "日std%": s.std()*100,
            "Sharpe": s.mean()/s.std()*np.sqrt(245) if s.std()>0 else np.nan, "MDD%": ((1+s).cumprod()/(1+s).cumprod().cummax()-1).min()*100,
            "年化報酬(萬)": s.mean()*245*CAP/1e4, "強制回補%": trades_.forced.mean()*100, "鎖漲停%": (trades_.lock & ~trades_.forced).mean()*100,
            "池/日": float(d[d.date.isin(s.index)].groupby("date").size().mean())}

MODES = [("D1 近似", "d1")]
if glob.glob(str(ROOT/"data/processed/m1_entry_ext/*.parquet")): MODES.append(("M1 精確", "m1"))
RES = {}
for label, mode in MODES:
    d, e, top = build_trades(mode); R = alloc(top)
    all_days = pd.Series(sorted(d.date.unique())); R = R.reindex(all_days, fill_value=0.0)
    RES[label] = (d, e, top, R)
d, e, top, R = RES["D1 近似"]

L = []; P = lambda s="": L.append(str(s))
P("# 定案策略延伸回測 2020 ~ 2026 (2020Q1~2024Q3 為新的樣本外期間)\n")
P(f"- 資料 D1 {d1.date.min().date()} ~ {d1.date.max().date()}；特徵與每日腳本同一份 (features.py)；池版模型 B 按季 walk-forward，2020Q1~2024Q3 新訓 (models/ext/)，2024Q4 起沿用正式模型預測。")
P(f"- 成交假設 (D1 近似): 進場 = 開盤價 × (1 {ENTRY_ADJ*100:+.3f}%) [由 2024Q4+ 策略交易的 {ecol} 校準]；出場 = 收盤；強制回補 high ≥ 昨收+8% → 昨收×1.08×1.003；鎖漲停 → 隔日開盤 −7%；成本 0.23%。")
P("- 規則: 跳空≤4%、大盤平均跳空≥−1%、非跌停附近、股價≤300、非週一、模型排名往下填 (每檔上限 60 萬, 300 萬, 最多 10 檔)。無交易日 (週一/大盤跳空<−1%/無候選) 報酬計 0。")
P("- 注意: 2020~2023 無當沖資格名單 (現沖券賣是否可先賣) 與無 M1，進場價為近似；2020Q1~2020Q2 的模型只用 2018~2019 兩年訓練，樣本較少。\n")
for label, (dd, ee, tt, RR) in RES.items():
    rows = []
    for yr, s_ in RR.ret.groupby(RR.index.year):
        rows.append({"年": str(yr), **stats(s_, tt[tt.date.dt.year==yr], dd)})
    rows.append({"年": "2020~2024Q3 合計", **stats(RR.ret[RR.index<"2024-10-01"], tt[tt.date<"2024-10-01"], dd)})
    rows.append({"年": "2024Q4~2026", **stats(RR.ret[RR.index>="2024-10-01"], tt[tt.date>="2024-10-01"], dd)})
    rows.append({"年": "全期間", **stats(RR.ret, tt, dd)})
    P(f"## 1. 各年績效 (300 萬, 每日) — 成交假設: {label}" + ("  [M1 期間: " + str(dd.date.min().date()) + " ~ " + str(dd.date.max().date()) + "]" if label.startswith("M1") else ""))
    P(pd.DataFrame(rows).set_index("年").round(3).to_markdown()); P("")
P("- 對照: 2024Q4~2026 用 M1 精算 (stats/filters_winrate.md 組合 B 非週一): 每筆淨 0.58%、勝率 58.7%、日淨 0.73%、Sharpe 4.5、MDD −13%。D1 近似與它的差距就是近似誤差。\n")

if "M1 精確" in RES: d, e, top, R = RES["M1 精確"]; P("(以下 2~5 節用 M1 精確成交假設)\n")
rows = []
for q, s in R.ret.groupby(R.index.to_period("Q")):
    t = top[top.date.dt.to_period("Q")==q]
    if len(t)==0: continue
    rows.append({"季": str(q), "筆數": len(t), "每筆淨%": t.net.mean()*100, "勝率": (t.net>0).mean(), "日淨%": s.mean()*100, "Sharpe": s.mean()/s.std()*np.sqrt(245) if s.std()>0 else np.nan,
                 "累積%": ((1+s).prod()-1)*100, "強制回補%": t.forced.mean()*100, "鎖漲停%": (t.lock&~t.forced).mean()*100})
P("## 2. 各季"); P(pd.DataFrame(rows).set_index("季").round(3).to_markdown()); P("")

# 模型 vs 基準 (ATR 最高) 與池內平均, 各年
from scipy.stats import spearmanr
ic = d.groupby("date").apply(lambda x: spearmanr(x.pred, x.y_intra)[0] if len(x)>20 else np.nan).dropna()
e2 = d[(d.gap<=GAP_MAX)&(d.mkt_gap>=MKT_MIN)&(d.entry>d.prev_close*0.905)&(d.entry<=MAX_PX)&(d.dow!=0)].copy()
e2["rank_atr"] = e2.groupby("date").atr20.rank(ascending=False, method="first")
rows = []
for yr in sorted(d.date.dt.year.unique()):
    m = top[top.date.dt.year==yr]; b = e2[(e2.date.dt.year==yr)&(e2.rank_atr<=MAX_N)]; a = e2[e2.date.dt.year==yr]
    rows.append({"年": yr, "日均 rank-IC": ic[ic.index.year==yr].mean(), "P(IC>0)": (ic[ic.index.year==yr]>0).mean(), "模型前10 每筆淨%": m.net.mean()*100, "ATR前10 每筆淨%": b.net.mean()*100,
                 "合格池平均 每筆淨%": a.net.mean()*100, "合格池 檔/日": len(a)/a.date.nunique() if len(a) else np.nan, "池內平均跳空%": d[d.date.dt.year==yr].gap.mean()*100, "池內日內平均%": d[d.date.dt.year==yr].y_intra.mean()*100})
P("## 3. 模型有沒有用: 各年 rank-IC、模型前 10 vs ATR 前 10 vs 合格池平均 (每筆淨 %, 同一套成交假設)"); P(pd.DataFrame(rows).set_index("年").round(3).to_markdown()); P("")

# 成本敏感度 (2020~2024Q3)
rows = []
for c in [0.0015, 0.0023, 0.0030, 0.0040]:
    t = top[top.date<"2024-10-01"]; rows.append({"來回成本": f"{c*100:.2f}%", "每筆淨%": (t.gross-c).mean()*100, "勝率": ((t.gross-c)>0).mean()})
P("## 4. 成本敏感度 (2020~2024Q3, 每筆)"); P(pd.DataFrame(rows).set_index("來回成本").round(3).to_markdown()); P("")
# 最差月份
mo = R.ret.groupby(R.index.to_period("M")).apply(lambda s: ((1+s).prod()-1)*100)
P("## 5. 月報酬 (%) 最差 8 個月與最好 5 個月"); P(mo.sort_values().head(8).round(2).to_frame("月報酬%").to_markdown()); P(""); P(mo.sort_values().tail(5).round(2).to_frame("月報酬%").to_markdown()); P("")
P(f"- 負報酬月份比例 {(mo<0).mean()*100:.0f}% ({int((mo<0).sum())}/{len(mo)})")
(OUT/"extended_backtest.md").write_text("\n".join(L)); print("\n".join(L))
R.to_csv(ROOT/"data/processed/extended_daily_returns.csv")
