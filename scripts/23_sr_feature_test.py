"""D1 支撐壓力特徵對照實驗.

支撐/壓力的定義 (只用前一日以前的資料, 回看 LOOK 天):
  轉折高點 (swing high): 某日 high 是前後各 W 天內的最高 → 視為一個壓力價位; 轉折低點同理為支撐.
  「摸到就反轉」強度: 以該價位 ±TOL 為帶, 回看期間內有幾天的 high (壓力) / low (支撐) 進入這個帶 = 觸碰次數 (touches).
今天 09:00 可得的特徵 (相對今天開盤價 open):
  sr_res_dist   最近一個在 open 之上的壓力價 / open − 1 (沒有 → 回看期間最高價)
  sr_res_touch  那個壓力價位的觸碰次數
  sr_res_age    距離那個轉折高點幾天
  sr_sup_dist   open / 最近一個在 open 之下的支撐價 − 1
  sr_sup_touch, sr_sup_age
  sr_near_res   open 是否在壓力帶內 (|dist| <= TOL)       sr_near_sup 同理
  sr_above_res  open 是否已突破回看期間「最強壓力」(觸碰最多) 之上
  sr_n_res, sr_n_sup  回看期間轉折高/低點數量
  sr_hi60_dist, sr_lo60_dist  open 對 60 日最高/最低 (前一日止)
對照: 基準 33 / 基準+SR / 基準+seq / 基準+seq+SR, 2020Q1~2026Q4 按季 walk-forward, 定案策略 M1 精確成交. 輸出 stats/sr_feature_test.md
"""
import sys, glob, numpy as np, pandas as pd, lightgbm as lgb
from pathlib import Path
from scipy.stats import spearmanr
sys.path.insert(0, str(Path(__file__).resolve().parent)); import features as F
ROOT = Path(__file__).resolve().parent.parent; OUT = ROOT/"stats"
BASE, SLIP, K, PEN, CAP, PER_STOCK, MAX_N, MAX_PX, GAP_MAX, MKT_MIN = 0.0023, 0.001, 0.08, 0.07, 3_000_000, 600_000, 10, 300, 0.04, -0.01
LOOK, W, TOL = 60, 3, 0.01
TEST_Q = pd.period_range("2020Q1", "2026Q4", freq="Q")

def sr_features(stock: pd.DataFrame) -> pd.DataFrame:
    """單檔 (已按日期排序). 以前一日為止的 LOOK 天 high/low 找轉折點並計算特徵; 用 numpy 逐日滾動 (O(n·LOOK))."""
    hi = stock.high.values; lo = stock.low.values; op = stock.open.values; n = len(stock)
    # 轉折點: 第 i 天 high 為 [i-W, i+W] 最高 (確認需要 W 天之後, 所以第 i 天的轉折在第 i+W 天才可知)
    is_sh = np.zeros(n, bool); is_sl = np.zeros(n, bool)
    for i in range(W, n-W):
        seg_h = hi[i-W:i+W+1]; seg_l = lo[i-W:i+W+1]
        if np.isfinite(hi[i]) and hi[i] == np.nanmax(seg_h): is_sh[i] = True
        if np.isfinite(lo[i]) and lo[i] == np.nanmin(seg_l): is_sl[i] = True
    cols = {k: np.full(n, np.nan) for k in ["sr_res_dist","sr_res_touch","sr_res_age","sr_sup_dist","sr_sup_touch","sr_sup_age","sr_near_res","sr_near_sup","sr_above_res","sr_n_res","sr_n_sup","sr_hi60_dist","sr_lo60_dist"]}
    for t in range(LOOK+W, n):
        o = op[t]
        if not np.isfinite(o): continue
        a, b = t-LOOK, t-W          # 可知的轉折點: 日期 < t-W (需 W 天確認); 觸碰計算用 [t-LOOK, t-1]
        wh = hi[a:t]; wl = lo[a:t]
        shs = np.where(is_sh[a:b])[0] + a; sls = np.where(is_sl[a:b])[0] + a
        res = [(hi[i], i) for i in shs if hi[i] > o]; sup = [(lo[i], i) for i in sls if lo[i] < o]
        cols["sr_n_res"][t] = len(shs); cols["sr_n_sup"][t] = len(sls)
        if res:
            p, i = min(res); touch = int(np.sum(np.abs(wh/p-1) <= TOL))
            cols["sr_res_dist"][t] = p/o-1; cols["sr_res_touch"][t] = touch; cols["sr_res_age"][t] = t-i; cols["sr_near_res"][t] = float(p/o-1 <= TOL)
        else:
            cols["sr_res_dist"][t] = np.nanmax(wh)/o-1; cols["sr_res_touch"][t] = 0; cols["sr_res_age"][t] = LOOK; cols["sr_near_res"][t] = 0.0
        if sup:
            p, i = max(sup); touch = int(np.sum(np.abs(wl/p-1) <= TOL))
            cols["sr_sup_dist"][t] = o/p-1; cols["sr_sup_touch"][t] = touch; cols["sr_sup_age"][t] = t-i; cols["sr_near_sup"][t] = float(o/p-1 <= TOL)
        else:
            cols["sr_sup_dist"][t] = o/np.nanmin(wl)-1; cols["sr_sup_touch"][t] = 0; cols["sr_sup_age"][t] = LOOK; cols["sr_near_sup"][t] = 0.0
        if len(shs):
            strongest = max(shs, key=lambda i: int(np.sum(np.abs(wh/hi[i]-1) <= TOL))); cols["sr_above_res"][t] = float(o > hi[strongest])
        else: cols["sr_above_res"][t] = 0.0
        cols["sr_hi60_dist"][t] = np.nanmax(wh)/o-1; cols["sr_lo60_dist"][t] = o/np.nanmin(wl)-1
    out = pd.DataFrame(cols, index=stock.index); out["stock_id"] = stock.stock_id.values; out["date"] = stock.date.values
    return out
SR_FEATS = ["sr_res_dist","sr_res_touch","sr_res_age","sr_sup_dist","sr_sup_touch","sr_sup_age","sr_near_res","sr_near_sup","sr_above_res","sr_n_res","sr_n_sup","sr_hi60_dist","sr_lo60_dist"]

if __name__ == "__main__":
    fs = sorted(glob.glob(str(ROOT/"data/raw/d1/*.parquet")))
    d1 = pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True); d1["date"] = pd.to_datetime(d1.date)
    d1 = d1.drop_duplicates(["stock_id","date"]).sort_values(["stock_id","date"]).reset_index(drop=True)
    d1 = d1.merge(pd.read_parquet(ROOT/"data/raw/meta/info.parquet")[["stock_id","market_type"]], on="stock_id", how="left")
    base = F.d1_base(d1); f = F.build_B(base).merge(F.build_seq(base), on=["stock_id","date"], how="left")
    nxt = d1[["stock_id","date","open"]].sort_values(["stock_id","date"]); nxt["next_open"] = nxt.groupby("stock_id").open.shift(-1)
    f = f.merge(nxt[["stock_id","date","next_open"]], on=["stock_id","date"], how="left")
    # SR 只需對曾入池的股票算 (省時間)
    pool_mask = F.universe_mask(f) & (f.dv20_prev>=5e8) & (f.atr20>=0.03) & f.y_intra.notna() & f.open.notna()
    ids = set(f.loc[pool_mask, "stock_id"].unique()); print(f"計算 SR 特徵: {len(ids)} 檔 ...", flush=True)
    sr_cache = ROOT/"data/processed/sr_features.parquet"
    if sr_cache.exists(): sr = pd.read_parquet(sr_cache)
    else:
        from concurrent.futures import ProcessPoolExecutor
        parts = [g for _, g in base[base.stock_id.isin(ids)].groupby("stock_id", sort=False)]
        with ProcessPoolExecutor(8) as ex: sr = pd.concat(list(ex.map(sr_features, parts, chunksize=20)), ignore_index=True)
        sr.to_parquet(sr_cache, index=False)
    f = f.merge(sr, on=["stock_id","date"], how="left")
    pool = f[pool_mask.reindex(f.index, fill_value=False) if len(pool_mask)==len(f) else (F.universe_mask(f) & (f.dv20_prev>=5e8) & (f.atr20>=0.03) & f.y_intra.notna() & f.open.notna())].copy()
    pool = pool.sort_values(["date","stock_id"]).reset_index(drop=True)
    pool["y_clip"] = pool.y_intra.clip(-0.1, 0.1); pool["y_dm"] = pool.y_clip - pool.groupby("date").y_clip.transform("mean")
    print(f"pool {len(pool)}; SR 覆蓋率 {pool.sr_res_dist.notna().mean():.2f}", flush=True)
    # 單變數看一眼
    L = []; P = lambda s="": L.append(str(s))
    P("# D1 支撐壓力特徵對照 (2020Q1~2026Q4 按季 walk-forward)\n")
    P(f"- 定義: 回看 {LOOK} 天, 轉折點 = 前後 {W} 天最高/最低, 價位帶 ±{TOL*100:.0f}%; 特徵見程式 docstring. 只用前一日以前的資料.")
    ics = {c: pool.groupby("date").apply(lambda x: spearmanr(x[c], x.y_intra, nan_policy="omit")[0] if x[c].notna().sum()>20 else np.nan).mean() for c in SR_FEATS}
    P("- 單變數日均 rank-IC vs 開→收 (負 = 值越大越看空):"); P(pd.Series(ics).round(4).to_frame("IC").to_markdown()); P("")
    nb = pool.groupby(pool.sr_near_res.fillna(-1)).y_intra.agg(["mean","size"]); nb["mean"]*=100
    P("- 開盤在壓力帶內 (sr_near_res=1) vs 否: 開→收平均 %"); P(nb.round(3).to_markdown()); P("")
    tb = pool.assign(tb=pd.cut(pool.sr_res_touch, [-1,0,1,2,4,100], labels=["0","1","2","3-4","5+"])).groupby("tb", observed=True).y_intra.agg(["mean","size"]); tb["mean"]*=100
    P("- 最近壓力的觸碰次數 → 開→收平均 %"); P(tb.round(3).to_markdown()); P("")
    params = dict(objective="regression", learning_rate=0.03, num_leaves=31, min_data_in_leaf=100, feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1, lambda_l2=10.0, verbose=-1, num_threads=8, seed=42)
    def walk(feats, tag):
        preds = []; imp = pd.Series(0.0, index=feats)
        for q in TEST_Q:
            ts, te = q.start_time, q.end_time; test = pool[(pool.date>=ts)&(pool.date<=te)]
            tr_all = pool[pool.date < ts - pd.Timedelta(days=1)]; vcut = ts - pd.DateOffset(months=2); tr, va = tr_all[tr_all.date < vcut], tr_all[tr_all.date >= vcut]
            if len(test)==0 or len(tr) < 5000: continue
            m = lgb.train(params, lgb.Dataset(tr[feats], tr.y_dm), num_boost_round=2000, valid_sets=[lgb.Dataset(va[feats], va.y_dm)], callbacks=[lgb.early_stopping(100, verbose=False)])
            preds.append(pd.DataFrame({"date":test.date.values, "stock_id":test.stock_id.values, "pred":m.predict(test[feats], num_iteration=m.best_iteration)}))
            imp += pd.Series(m.feature_importance("gain"), index=feats); print(f"{tag} {q}: best_iter {m.best_iteration}", flush=True)
        return pd.concat(preds, ignore_index=True), imp
    SETS = {"基準 33": F.D1_FEATS, "基準 + SR": F.D1_FEATS + SR_FEATS, "基準 + seq": F.D1_FEATS + F.SEQ_FEATS, "基準 + seq + SR": F.D1_FEATS + F.SEQ_FEATS + SR_FEATS}
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
    res = {}; imps = {}
    for k, feats in SETS.items():
        pr, imp = walk(feats, k); res[k] = strategy(pr); imps[k] = imp
    rows = []
    for yr in list(range(2020, 2027)) + ["全期間"]:
        r = {"年": yr}
        for k, (s, top, ic) in res.items():
            ss = s if yr == "全期間" else s[s.index.year==yr]; tt = top if yr == "全期間" else top[top.date.dt.year==yr]; ii = ic if yr == "全期間" else ic[ic.index.year==yr]
            r[f"{k} IC"] = ii.mean(); r[f"{k} 每筆%"] = tt.net.mean()*100; r[f"{k} 日淨%"] = ss.mean()*100; r[f"{k} Sharpe"] = ss.mean()/ss.std()*np.sqrt(245)
        rows.append(r)
    tab = pd.DataFrame(rows).set_index("年")
    P("## 四組模型各年比較 (M1 精確成交, 300 萬配置, 不交易日計 0)")
    for metric in ["每筆%", "Sharpe", "IC"]:
        P(f"### {metric}"); P(tab[[c for c in tab.columns if c.endswith(" " + metric)]].round(3).to_markdown()); P("")
    for k, (s, top, ic) in res.items():
        P(f"- {k}: MDD {((1+s).cumprod()/(1+s).cumprod().cummax()-1).min()*100:.1f}%，負月比例 {(s.groupby(s.index.to_period('M')).apply(lambda x:(1+x).prod()-1)<0).mean()*100:.0f}%，IC-IR {ic.mean()/ic.std():.2f}")
    b = res["基準 33"][2]
    for k in ["基準 + SR", "基準 + seq", "基準 + seq + SR"]:
        both = b.to_frame("b").join(res[k][2].to_frame("s")).dropna(); diff = both.s-both.b
        P(f"- 每日 IC 差 ({k} − 基準): 平均 {diff.mean():.4f}, t-stat {diff.mean()/diff.std()*np.sqrt(len(diff)):.2f}")
    imp = imps["基準 + seq + SR"]; P(""); P("## 基準+seq+SR 模型中 SR 特徵的重要度 (gain%)"); P((imp[SR_FEATS]/imp.sum()*100).sort_values(ascending=False).round(2).to_frame("gain%").to_markdown())
    P(f"- SR 特徵合計 {imp[SR_FEATS].sum()/imp.sum()*100:.1f}%, seq 合計 {imp[F.SEQ_FEATS].sum()/imp.sum()*100:.1f}%")
    (OUT/"sr_feature_test.md").write_text("\n".join(L)); print("\n".join(L))
