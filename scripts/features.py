"""共用特徵計算: 與 03_stats_d1 / 06_build_dataset 的模型 B 特徵完全一致 (daily_candidates 與訓練共用)."""
import numpy as np, pandas as pd

D1_FEATS = ["gap","gap_abs","gap_rel","p_ret","p_intra","p_range","p_gap","p_vol_ratio","p_up_lim","p_dn_lim","p_ret5","p_ret20","ret2","ret60",
            "pos20","ma5_dev","ma20_dev","ma60_dev","vol5","vol20","vol60","dv20_log","dv_ratio5","uplim_cnt5","price_log","lo_price","is_tpex",
            "mkt_gap","mkt_pret","mkt_ret5","mkt_breadth","mkt_uplim","dow"]

def d1_base(d1: pd.DataFrame) -> pd.DataFrame:
    """03_stats_d1 的衍生欄位 (需 stock_id,date,open,high,low,close,volume,market_type; 已排序)."""
    df = d1.sort_values(["stock_id","date"]).reset_index(drop=True).copy()
    g = df.groupby("stock_id", sort=False)
    df["prev_close"] = g.close.shift(1)
    df["ret"] = df.close/df.prev_close - 1
    df["gap"] = df.open/df.prev_close - 1
    df["intra"] = df.close/df.open - 1
    df["range"] = (df.high-df.low)/df.prev_close
    df["up_lim"] = (df.ret >= 0.095).astype(float)
    df["dn_lim"] = (df.ret <= -0.095).astype(float)
    df["dollar_vol"] = df.close*df.volume
    df["ret5"] = g.close.shift(0)/g.close.shift(5) - 1
    df["ret20"] = g.close.shift(0)/g.close.shift(20) - 1
    df["vol_ratio"] = df.volume / g.volume.transform(lambda s: s.rolling(20).mean())
    return df

def build_B(d1f: pd.DataFrame) -> pd.DataFrame:
    """06_build_dataset 的模型 B 特徵 (回傳所有股票日, 未套 universe 篩選). 另附 atr20 / dv20_prev 供選池."""
    d1 = d1f.sort_values(["stock_id","date"]).reset_index(drop=True)
    g = d1.groupby("stock_id", sort=False)
    f = d1[["stock_id","date","market_type","open","close","prev_close"]].copy()
    f["dv20"] = g.dollar_vol.transform(lambda s: s.shift(1).rolling(20).mean())
    f["vol20"] = g.ret.transform(lambda s: s.shift(1).rolling(20).std())
    f["gap"] = d1.gap
    for c in ["ret","intra","range","gap","vol_ratio","up_lim","dn_lim","ret5","ret20"]:
        f[f"p_{c}"] = g[c].shift(1)
    for c in ["p_up_lim","p_dn_lim"]: f[c] = f[c].astype(float)
    f["ret2"] = g.close.shift(1)/g.close.shift(3) - 1
    f["ret60"] = g.close.shift(1)/g.close.shift(61) - 1
    f["hi20"] = g.high.transform(lambda s: s.shift(1).rolling(20).max()); f["lo20"] = g.low.transform(lambda s: s.shift(1).rolling(20).min())
    f["pos20"] = (f.prev_close - f.lo20)/(f.hi20 - f.lo20)
    f["ma5_dev"] = f.prev_close / g.close.transform(lambda s: s.shift(1).rolling(5).mean()) - 1
    f["ma20_dev"] = f.prev_close / g.close.transform(lambda s: s.shift(1).rolling(20).mean()) - 1
    f["ma60_dev"] = f.prev_close / g.close.transform(lambda s: s.shift(1).rolling(60).mean()) - 1
    f["vol60"] = g.ret.transform(lambda s: s.shift(1).rolling(60).std())
    f["vol5"] = g.ret.transform(lambda s: s.shift(1).rolling(5).std())
    f["dv20_log"] = np.log1p(f.dv20)
    f["dv_ratio5"] = g.dollar_vol.transform(lambda s: s.shift(1).rolling(5).mean()) / f.dv20
    f["uplim_cnt5"] = g.up_lim.transform(lambda s: s.shift(1).rolling(5).sum())
    f["price_log"] = np.log(f.prev_close)
    f["lo_price"] = (f.prev_close < 10).astype(int)
    f["gap_abs"] = f.gap.abs()
    f["is_tpex"] = (f.market_type=="tpex").astype(int)
    f["y_intra"] = d1.intra
    # 市場橫截面 (與 06 相同: dv20>=1e8 的全部股票)
    liq = f[(f.dv20>=1e8)]
    mk = liq.groupby("date").agg(mkt_gap=("gap","mean"), mkt_pret=("p_ret","mean"), mkt_ret5=("p_ret5","mean"),
                                 mkt_breadth=("gap", lambda s:(s>0).mean()), mkt_uplim=("p_up_lim","mean"))
    f = f.merge(mk, left_on="date", right_index=True, how="left")
    f["mkt_uplim"] = f.mkt_uplim.astype(float)
    f["gap_rel"] = f.gap - f.mkt_gap
    f["dow"] = f.date.dt.dayofweek
    # 選池用 (09_daytrade_pool)
    f["atr20"] = g.range.transform(lambda s: s.shift(1).rolling(20).mean())
    f["dv20_prev"] = f.dv20
    return f

def universe_mask(f: pd.DataFrame) -> pd.Series:
    return (f.dv20>=1e8) & f.market_type.isin(["twse","tpex"]) & ~f.stock_id.str.startswith("00") & f.prev_close.notna() & (f.prev_close>=5)

# ---- 台股 tick ----
def tick_size(p):
    return 0.01 if p<10 else 0.05 if p<50 else 0.1 if p<100 else 0.5 if p<500 else 1.0 if p<1000 else 5.0
def round_down_tick(p):
    t = tick_size(p); return np.floor(p/t+1e-9)*t
def round_up_tick(p):
    t = tick_size(p); return np.ceil(p/t-1e-9)*t
def limit_up(prev_close):
    return round_down_tick(prev_close*1.10)

# ---- K 線序列特徵 (影子模型 Bseq 用; 2026-10-23 實驗 stats/kline_seq_test.md) ----
SEQ_FEATS = [f"k{k}_{n}" for k in range(1, 11) for n in (["ret","vr"] + (["intra","rng","uw","lw","gap"] if k <= 5 else []))]

def build_seq(d1f: pd.DataFrame) -> pd.DataFrame:
    """輸入 d1_base() 的輸出. 回傳 stock_id,date + SEQ_FEATS: 最近 10 天每天的 ret/量比, 最近 5 天的 開到收/振幅/上影/下影/跳空 (全部 shift>=1)."""
    b = d1f.sort_values(["stock_id","date"]).reset_index(drop=True)
    out = b[["stock_id","date"]].copy()
    uw = (b.high - b[["open","close"]].max(axis=1)) / b.prev_close
    lw = (b[["open","close"]].min(axis=1) - b.low) / b.prev_close
    src = {"ret": b.ret, "vr": b.vol_ratio, "intra": b.intra, "rng": b.range, "uw": uw, "lw": lw, "gap": b.gap}
    for col in SEQ_FEATS:
        k, n = col[1:].split("_", 1); out[col] = src[n].groupby(b.stock_id, sort=False).shift(int(k)).values
    return out

# ---- D1 支撐壓力特徵 (影子模型 Bsr 用; 2026-10-23 實驗 stats/sr_feature_test.md) ----
SR_LOOK, SR_W, SR_TOL = 60, 3, 0.01
SR_FEATS = ["sr_res_dist","sr_res_touch","sr_res_age","sr_sup_dist","sr_sup_touch","sr_sup_age","sr_near_res","sr_near_sup","sr_above_res","sr_n_res","sr_n_sup","sr_hi60_dist","sr_lo60_dist"]

def _sr_one(stock: pd.DataFrame) -> pd.DataFrame:
    """單檔 (已按日期排序, 需 open/high/low). 回看 SR_LOOK 天: 轉折高/低點 = 前後 SR_W 天極值 (需 SR_W 天後才確認);
    特徵相對今天開盤價, 只用前一日以前的資料."""
    LOOK, W, TOL = SR_LOOK, SR_W, SR_TOL
    hi = stock.high.values.astype(float); lo = stock.low.values.astype(float); op = stock.open.values.astype(float); n = len(stock)
    is_sh = np.zeros(n, bool); is_sl = np.zeros(n, bool)
    for i in range(W, n-W):
        if np.isfinite(hi[i]) and hi[i] == np.nanmax(hi[i-W:i+W+1]): is_sh[i] = True
        if np.isfinite(lo[i]) and lo[i] == np.nanmin(lo[i-W:i+W+1]): is_sl[i] = True
    cols = {k: np.full(n, np.nan) for k in SR_FEATS}
    for t in range(LOOK+W, n):
        o = op[t]
        if not np.isfinite(o): continue
        a, b = t-LOOK, t-W; wh = hi[a:t]; wl = lo[a:t]
        shs = np.where(is_sh[a:b])[0] + a; sls = np.where(is_sl[a:b])[0] + a
        res = [(hi[i], i) for i in shs if hi[i] > o]; sup = [(lo[i], i) for i in sls if lo[i] < o]
        cols["sr_n_res"][t] = len(shs); cols["sr_n_sup"][t] = len(sls)
        if res:
            p, i = min(res); cols["sr_res_dist"][t] = p/o-1; cols["sr_res_touch"][t] = int(np.sum(np.abs(wh/p-1) <= TOL)); cols["sr_res_age"][t] = t-i; cols["sr_near_res"][t] = float(p/o-1 <= TOL)
        else:
            cols["sr_res_dist"][t] = np.nanmax(wh)/o-1; cols["sr_res_touch"][t] = 0; cols["sr_res_age"][t] = LOOK; cols["sr_near_res"][t] = 0.0
        if sup:
            p, i = max(sup); cols["sr_sup_dist"][t] = o/p-1; cols["sr_sup_touch"][t] = int(np.sum(np.abs(wl/p-1) <= TOL)); cols["sr_sup_age"][t] = t-i; cols["sr_near_sup"][t] = float(o/p-1 <= TOL)
        else:
            cols["sr_sup_dist"][t] = o/np.nanmin(wl)-1; cols["sr_sup_touch"][t] = 0; cols["sr_sup_age"][t] = LOOK; cols["sr_near_sup"][t] = 0.0
        if len(shs):
            strongest = max(shs, key=lambda i: int(np.sum(np.abs(wh/hi[i]-1) <= TOL))); cols["sr_above_res"][t] = float(o > hi[strongest])
        else: cols["sr_above_res"][t] = 0.0
        cols["sr_hi60_dist"][t] = np.nanmax(wh)/o-1; cols["sr_lo60_dist"][t] = o/np.nanmin(wl)-1
    out = pd.DataFrame(cols, index=stock.index); out["stock_id"] = stock.stock_id.values; out["date"] = stock.date.values
    return out

def build_sr(d1f: pd.DataFrame, stock_ids=None, workers: int = 1) -> pd.DataFrame:
    """輸入含 stock_id,date,open,high,low 的 D1 (d1_base 輸出亦可). 回傳 stock_id,date + SR_FEATS. stock_ids 可限制只算部分股票."""
    b = d1f if stock_ids is None else d1f[d1f.stock_id.isin(set(stock_ids))]
    b = b.sort_values(["stock_id","date"]).reset_index(drop=True)
    parts = [g for _, g in b.groupby("stock_id", sort=False)]
    if workers > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(workers) as ex: outs = list(ex.map(_sr_one, parts, chunksize=20))
    else: outs = [_sr_one(g) for g in parts]
    return pd.concat(outs, ignore_index=True) if outs else pd.DataFrame(columns=["stock_id","date"] + SR_FEATS)

def shadow_feats(variant: str):
    """變體 → 額外特徵清單 (base 無)."""
    return {"seq": SEQ_FEATS, "sr": SR_FEATS}.get(variant, [])
