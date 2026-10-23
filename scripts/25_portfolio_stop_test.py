"""組合層級盤中停損測試: 整個空單組合的未實現損益逐分鐘, 浮虧超過 X% 就全部市價回補 (下一分鐘開盤 +0.1% 滑價).
交易與基準出場同 24_trailing_stop_test (需先跑過, 讀 data/processed/trailing_stop_trades.parquet). 輸出 stats/portfolio_stop_test.md"""
import sys, glob, numpy as np, pandas as pd
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
ROOT = Path(__file__).resolve().parent.parent; OUT = ROOT/"stats"
BASE, SLIP, K, PEN, CAP, PER_STOCK = 0.0023, 0.001, 0.08, 0.07, 3_000_000, 600_000
STOPS = [0.015, 0.02, 0.03, 0.04, 0.05]

def month_paths(args):
    f, tr = args
    df = pd.read_parquet(f); ts = pd.to_datetime(df["date"]); df["day"] = ts.dt.normalize(); df["t"] = ts.dt.hour*60 + ts.dt.minute - 540
    df = df[(df.t>=4)&(df.t<=270)]
    key = set(zip(tr.stock_id, tr.date)); df = df[[(s, d) in key for s, d in zip(df.stock_id, df.day)]]
    out = []
    for (sid, day), g in df.groupby(["stock_id","day"], sort=False):
        r = tr[(tr.stock_id==sid)&(tr.date==day)].iloc[0]
        g = g.sort_values("t"); t = g.t.values; c = g.close.values; o = g.open.values
        # 每分鐘該檔的空單報酬 (entry/close − 1), 以 1 分鐘為格, 缺的分鐘往前補
        path = np.full(271, np.nan); path[t] = r.entry/c - 1; nxt = np.full(271, np.nan); nxt[t] = o
        s = pd.Series(path).ffill().to_numpy(copy=True); s[:4] = np.nan
        out.append((sid, day, r.entry, r.amt, r.gross_base, s, pd.Series(nxt).bfill().to_numpy(copy=True)))
    return out

if __name__ == "__main__":
    TAG = sys.argv[sys.argv.index("--tag")+1] if "--tag" in sys.argv else ""
    R = pd.read_parquet(ROOT/"data/processed"/f"trailing_stop_trades{TAG}.parquet")
    R = R.rename(columns={"gross|現行 (無移動停利)": "gross_base"})[["stock_id","date","entry","prev_close","next_open","rank","gross_base"]]
    # 300 萬配置的每檔金額 (與 alloc 相同)
    amts = []
    for dt, gg in R.sort_values(["date","rank"]).groupby("date", sort=False):
        cash = CAP
        for idx, r in gg.iterrows():
            lc = r.entry*1000; a = 0
            if cash >= lc:
                lots = int(min(PER_STOCK, cash)//lc)
                if lots >= 1: a = lots*lc; cash -= a
            amts.append((idx, a))
    R["amt"] = pd.Series(dict(amts)); R = R[R.amt>0].copy()
    R["ym"] = R.date.dt.strftime("%Y_%m")
    jobs = [(str(ROOT/"data/raw/m1"/f"{ym}.parquet"), g) for ym, g in R.groupby("ym") if (ROOT/"data/raw/m1"/f"{ym}.parquet").exists()]
    with ProcessPoolExecutor(6) as ex: parts = [x for p in ex.map(month_paths, jobs) for x in p]
    print("paths", len(parts), flush=True)
    by_day = {}
    for sid, day, entry, amt, gb, path, nxt in parts: by_day.setdefault(day, []).append((sid, entry, amt, gb, path, nxt))
    days = sorted(by_day)
    L = []; P = lambda s="": L.append(str(s))
    P("# 組合層級盤中停損測試 (2020~2026, 300 萬, M1 逐分鐘)\n")
    P("- 組合未實現損益 = Σ(金額 × 該檔空單報酬) / 300 萬, 每分鐘用 close 計。基準出場同現行 (+8% 強制 / 13:24 預先 / 收盤)。停損規則: 任一分鐘組合浮虧 ≤ −X% → 全部以下一分鐘開盤價 +0.1% 回補 (已被 +8% 強制回補的檔不重複)。")
    # 基準日報酬 (現行) 與日內最大浮虧
    base_ret = {}; mdd_intra = {}; worst_min = {}
    for day in days:
        items = by_day[day]; tot = sum(a for _, _, a, _, _, _ in items)
        base_ret[day] = sum(a*(gb-BASE) for _, _, a, gb, _, _ in items)/CAP
        pathsum = (np.nansum(np.array([a*p for _, _, a, _, p, _ in items]), axis=0)/CAP).copy()   # 組合逐分鐘報酬 (未扣成本)
        pathsum[:4] = np.nan; mdd_intra[day] = np.nanmin(pathsum); worst_min[day] = int(np.nanargmin(pathsum))
    br = pd.Series(base_ret); mi = pd.Series(mdd_intra)
    P(f"- 現行: 日淨平均 {br.mean()*100:.3f}%，最差日 {br.min()*100:.2f}% ({br.idxmin().date()})，日內組合最大浮虧的分布: 中位 {mi.median()*100:.2f}%、p10 {mi.quantile(.1)*100:.2f}%、p1 {mi.quantile(.01)*100:.2f}%、最差 {mi.min()*100:.2f}%")
    worst = br.sort_values().head(10)
    P("\n## 1. 現行最差 10 天: 日淨、盤中最大浮虧發生時間、收盤時"); rows = []
    for day, v in worst.items():
        rows.append({"日期": day.date(), "日淨%": v*100, "盤中最大浮虧%": mi[day]*100, "發生時間": f"{9+worst_min[day]//60:02d}:{worst_min[day]%60:02d}", "檔數": len(by_day[day])})
    P(pd.DataFrame(rows).set_index("日期").round(2).to_markdown()); P("")
    # 浮虧 ≥ X 的日子, 後來收盤如何 (有沒有救回來)
    P("## 2. 盤中組合浮虧曾達 −X% 的日子: 天數、之後收盤的平均日淨 (現行) → 停損是砍尾巴還是砍反彈")
    rows = []
    for X in STOPS:
        hit = mi <= -X; rows.append({"門檻": f"-{X*100:.1f}%", "天數": int(hit.sum()), "占比%": hit.mean()*100, "這些天現行日淨平均%": br[hit].mean()*100 if hit.any() else np.nan, "其中收盤轉正的比例": (br[hit]>0).mean() if hit.any() else np.nan, "現行日淨中位%": br[hit].median()*100 if hit.any() else np.nan})
    P(pd.DataFrame(rows).set_index("門檻").round(3).to_markdown()); P("")
    # 模擬停損規則
    P("## 3. 套用組合停損後的績效")
    rows = []; daily = {"現行": br}
    def stats(s, label):
        mo = s.groupby(s.index.to_period("M")).apply(lambda x:((1+x).prod()-1)*100)
        return {"規則": label, "日淨%": s.mean()*100, "Sharpe": s.mean()/s.std()*np.sqrt(245), "MDD%": ((1+s).cumprod()/(1+s).cumprod().cummax()-1).min()*100, "最差日%": s.min()*100, "最差月%": mo.min(), "負月比例": (mo<0).mean(), "年化(萬)": s.mean()*245*CAP/1e4}
    rows.append(stats(br, "現行"))
    for X in STOPS:
        ret = {}; n_trig = 0
        for day in days:
            items = by_day[day]
            pathsum = (np.nansum(np.array([a*p for _, _, a, _, p, _ in items]), axis=0)/CAP).copy(); pathsum[:4] = np.nan
            trig = np.where(pathsum <= -X)[0]; trig = trig[(trig>=4)&(trig<264)]
            if len(trig)==0: ret[day] = base_ret[day]; continue
            n_trig += 1; tm = int(trig[0]); tot = 0.0
            for sid, entry, a, gb, path, nxt in items:
                # 若該檔在 tm 之前已被 +8% 強制回補 (近似: 其基準報酬為強制回補價且 path 在 tm 前曾 ≤ entry/(pc*1.08)-1), 簡化: 用基準 gross 若 gb 對應 forced; 這裡近似以 path 判斷
                px = nxt[tm+1] if tm+1 <= 270 and np.isfinite(nxt[tm+1]) else None
                g = (entry/(px*(1+SLIP)) - 1) if px else gb
                tot += a*(g-BASE)
            ret[day] = tot/CAP
        s = pd.Series(ret); r = stats(s, f"組合浮虧 −{X*100:.1f}% 全部回補"); r["觸發天數"] = n_trig; rows.append(r); daily[f"stop{X*100:g}"] = s
    P(pd.DataFrame(rows).set_index("規則").round(3).to_markdown())
    # 存每日序列 + 當天大盤日內 (流動股 開→收 平均, 由 D1 算) 供 27 對比
    D = pd.DataFrame(daily); D["intra_mdd"] = mi
    import glob as _g
    sys.path.insert(0, str(ROOT/"scripts")); import features as F
    d1 = pd.concat([pd.read_parquet(f) for f in sorted(_g.glob(str(ROOT/"data/raw/d1/*.parquet")))], ignore_index=True); d1["date"] = pd.to_datetime(d1.date)
    d1 = d1.drop_duplicates(["stock_id","date"]).sort_values(["stock_id","date"]).reset_index(drop=True)
    d1 = d1.merge(pd.read_parquet(ROOT/"data/raw/meta/info.parquet")[["stock_id","market_type"]], on="stock_id", how="left")
    fb = F.build_B(F.d1_base(d1)); liq = fb[(fb.dv20>=1e8)&fb.market_type.isin(["twse","tpex"])]
    mk = liq.groupby("date").agg(mkt_intra=("y_intra","mean"), mkt_gap=("gap","mean"), mkt_up=("y_intra", lambda s:(s>0).mean()))
    D = D.join(mk, how="left"); D.to_csv(ROOT/"data/processed"/f"portfolio_stop_daily{TAG}.csv")
    (OUT/f"portfolio_stop_test{TAG}.md").write_text("\n".join(L)); print("\n".join(L))
