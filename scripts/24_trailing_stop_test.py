"""移動停利 (做空) 測試, 用 M1 逐分鐘模擬 2020~2026 定案策略的每筆交易.

交易: 定案規則選出的每日前 10 檔 (模型預測 preds_B_pool_ext + preds_B_pool), 09:01~09:03 VWAP 進場 (e01_vwap).
基準出場 (現行): 分鐘 high >= 昨收+8% → 以 昨收×1.08×1.003 回補; 13:24 分鐘 close >= 昨收+6% → 該分鐘 close×1.001 回補; 否則 13:30 收盤競價.
移動停利變體 (從 09:04 起逐分鐘檢查, 訊號出現在第 t 分鐘 → 以第 t+1 分鐘 open × (1+0.1%) 回補; 其餘規則同基準):
  trail k%            : 追蹤進場後最低 close (lowest); close >= lowest × (1+k) → 回補
  trail k% after a%   : 同上, 但只在 (entry/lowest − 1) >= a (已有浮盈 a%) 之後才啟動
  MA N                : close > 過去 N 分鐘 close 均值 且 已過 N 分鐘 → 回補 (可加 after a%)
輸出 stats/trailing_stop_test.md
"""
import sys, glob, numpy as np, pandas as pd
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
sys.path.insert(0, str(Path(__file__).resolve().parent)); import features as F
ROOT = Path(__file__).resolve().parent.parent; OUT = ROOT/"stats"
BASE, SLIP, K, PEN, CAP, PER_STOCK, MAX_N, MAX_PX, GAP_MAX, MKT_MIN = 0.0023, 0.001, 0.08, 0.07, 3_000_000, 600_000, 10, 300, 0.04, -0.01
PRE_K = 0.06
RULES = {"現行 (無移動停利)": None,
         "trail 1%": ("trail", 0.01, 0.0), "trail 1.5%": ("trail", 0.015, 0.0), "trail 2%": ("trail", 0.02, 0.0), "trail 3%": ("trail", 0.03, 0.0),
         "trail 1.5% after +1%": ("trail", 0.015, 0.01), "trail 2% after +1%": ("trail", 0.02, 0.01), "trail 2% after +2%": ("trail", 0.02, 0.02), "trail 3% after +2%": ("trail", 0.03, 0.02),
         "MA10": ("ma", 10, 0.0), "MA20": ("ma", 20, 0.0), "MA30": ("ma", 30, 0.0), "MA20 after +1%": ("ma", 20, 0.01), "MA30 after +2%": ("ma", 30, 0.02)}

def build_trades():
    d1 = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(str(ROOT/"data/raw/d1/*.parquet")))], ignore_index=True); d1["date"] = pd.to_datetime(d1.date)
    d1 = d1.drop_duplicates(["stock_id","date"]).sort_values(["stock_id","date"]).reset_index(drop=True)
    d1 = d1.merge(pd.read_parquet(ROOT/"data/raw/meta/info.parquet")[["stock_id","market_type"]], on="stock_id", how="left")
    f = F.build_B(F.d1_base(d1)); nxt = d1[["stock_id","date","open"]].sort_values(["stock_id","date"]); nxt["next_open"] = nxt.groupby("stock_id").open.shift(-1)
    f = f.merge(nxt[["stock_id","date","next_open"]], on=["stock_id","date"], how="left")
    pool = f[F.universe_mask(f) & (f.dv20_prev>=5e8) & (f.atr20>=0.03) & f.y_intra.notna() & f.open.notna()].copy()
    if "--preds" in sys.argv: pr = pd.read_parquet(sys.argv[sys.argv.index("--preds")+1])[["date","stock_id","pred"]]
    else: pr = pd.concat([pd.read_parquet(ROOT/"data/processed/preds_B_pool_ext.parquet")[["date","stock_id","pred"]], pd.read_parquet(ROOT/"data/processed/preds_B_pool.parquet").query("date>='2024-10-01'")[["date","stock_id","pred"]]])
    ext = pd.concat([pd.read_parquet(x) for x in sorted(glob.glob(str(ROOT/"data/processed/m1_entry_ext/*.parquet")))], ignore_index=True)
    d = pool.merge(pr, on=["stock_id","date"]).merge(ext, on=["stock_id","date"]); d = d[d.e01_vwap.notna()].copy(); d["entry"] = d.e01_vwap
    e = d[(d.gap<=GAP_MAX)&(d.mkt_gap>=MKT_MIN)&(d.entry>d.prev_close*0.905)&(d.entry<=MAX_PX)&(d.date.dt.dayofweek!=0)].copy()
    e["rank"] = e.groupby("date").pred.rank(method="first")
    return e[e["rank"]<=MAX_N][["stock_id","date","entry","prev_close","next_open","rank"]].reset_index(drop=True)

def simulate_month(args):
    f, trades = args
    df = pd.read_parquet(f); ts = pd.to_datetime(df["date"]); df["day"] = ts.dt.normalize(); df["t"] = ts.dt.hour*60 + ts.dt.minute - 540
    df = df[(df.t>=0)&(df.t<=270)]
    key = set(zip(trades.stock_id, trades.date))
    df = df[[ (s, d) in key for s, d in zip(df.stock_id, df.day) ]]
    out = []
    for (sid, day), g in df.groupby(["stock_id","day"], sort=False):
        tr = trades[(trades.stock_id==sid)&(trades.date==day)].iloc[0]; entry = tr.entry; pc = tr.prev_close
        g = g.sort_values("t"); t = g.t.values; o = g.open.values; h = g.high.values; c = g.close.values
        auction = c[t==270][-1] if (t==270).any() else c[-1]
        res = {"stock_id": sid, "date": day, "entry": entry, "prev_close": pc, "next_open": tr.next_open}
        for name, rule in RULES.items():
            exit_px = None; reason = "close"; lowest = entry; csum = [];
            for i in range(len(t)):
                ti = t[i]
                if ti < 4: continue
                if ti >= 270: break
                if h[i] >= pc*(1+K): exit_px = pc*(1+K)*(1+SLIP*3); reason = "forced"; break
                if ti == 264 and c[i] >= pc*(1+PRE_K): exit_px = c[i]*(1+SLIP); reason = "pre"; break
                if ti >= 265: continue   # 13:25~13:29 無撮合
                if rule is not None:
                    kind, p, act = rule
                    lowest = min(lowest, c[i]); csum.append(c[i])
                    profit = entry/lowest - 1
                    sig = False
                    if kind == "trail": sig = (profit >= act) and (c[i] >= lowest*(1+p))
                    else:
                        if len(csum) > p: sig = (profit >= act) and (c[i] > np.mean(csum[-p:]))
                    if sig:
                        nxt_open = o[i+1] if i+1 < len(t) and t[i+1] < 265 else c[i]
                        exit_px = nxt_open*(1+SLIP); reason = "trail"; break
            if exit_px is None:
                lock = auction >= pc*1.0975
                if lock and np.isfinite(tr.next_open): gross = entry/tr.next_open - 1 - PEN; reason = "lock"
                else: gross = entry/auction - 1
            else: gross = entry/exit_px - 1
            res[f"gross|{name}"] = gross; res[f"reason|{name}"] = reason
        out.append(res)
    return pd.DataFrame(out)

if __name__ == "__main__":
    trades = build_trades(); print(f"trades {len(trades)} ({trades.date.min().date()}~{trades.date.max().date()})", flush=True)
    trades["ym"] = trades.date.dt.strftime("%Y_%m")
    jobs = [(str(ROOT/"data/raw/m1"/f"{ym}.parquet"), g) for ym, g in trades.groupby("ym") if (ROOT/"data/raw/m1"/f"{ym}.parquet").exists()]
    with ProcessPoolExecutor(6) as ex: parts = list(ex.map(simulate_month, jobs))
    R = pd.concat(parts, ignore_index=True).merge(trades[["stock_id","date","rank"]], on=["stock_id","date"])
    TAG = sys.argv[sys.argv.index("--tag")+1] if "--tag" in sys.argv else ""
    R.to_parquet(ROOT/"data/processed"/f"trailing_stop_trades{TAG}.parquet", index=False)
    print(f"simulated {len(R)} trades", flush=True)
    all_days = pd.Series(sorted(R.date.unique()))
    def alloc(y, col):
        recs = []
        for dt, gg in y.sort_values(["date","rank"]).groupby("date", sort=False):
            cash = CAP; amts=[]; nets=[]
            for _, r in gg.iterrows():
                lc = r.entry*1000
                if cash < lc: continue
                lots = int(min(PER_STOCK, cash)//lc)
                if lots < 1: continue
                a = lots*lc; cash -= a; amts.append(a); nets.append(r[col]-BASE)
            recs.append((dt, sum(a*n for a, n in zip(amts, nets))/CAP))
        return pd.DataFrame(recs, columns=["date","ret"]).set_index("date").ret.reindex(all_days, fill_value=0.0)
    L = []; P = lambda s="": L.append(str(s))
    P("# 移動停利測試 (做空, M1 逐分鐘, 2020~2026, 300 萬配置, 不交易日計 0)\n")
    P(f"- 樣本 {len(R)} 筆 / {R.date.nunique()} 天。進場 09:01~09:03 VWAP，基準出場 = +8% 強制 / 13:24 +6% 預先 / 13:30 收盤競價 (鎖漲停隔日 −7%)。移動停利訊號在第 t 分鐘出現, 以第 t+1 分鐘開盤價 +0.1% 回補。\n")
    rows = []; series = {}
    for name in RULES:
        col = f"gross|{name}"; s = alloc(R, col); series[name] = s; net = R[col]-BASE
        rs = R[f"reason|{name}"]
        rows.append({"規則": name, "每筆淨%": net.mean()*100, "勝率": (net>0).mean(), "中位%": net.median()*100, "日淨%": s.mean()*100, "Sharpe": s.mean()/s.std()*np.sqrt(245),
                     "MDD%": ((1+s).cumprod()/(1+s).cumprod().cummax()-1).min()*100, "最差月%": s.groupby(s.index.to_period("M")).apply(lambda x:((1+x).prod()-1)*100).min(),
                     "移動停利觸發%": (rs=="trail").mean()*100, "強制回補%": (rs=="forced").mean()*100, "鎖漲停%": (rs=="lock").mean()*100, "年化(萬)": s.mean()*245*CAP/1e4})
    tab = pd.DataFrame(rows).set_index("規則"); P("## 1. 全期間"); P(tab.round(3).to_markdown()); P("")
    yr = pd.DataFrame({name: s.groupby(s.index.year).mean()*100 for name, s in series.items()}); P("## 2. 各年日淨% (對 300 萬)"); P(yr.round(3).to_markdown()); P("")
    yrs = pd.DataFrame({name: s.groupby(s.index.year).apply(lambda x: x.mean()/x.std()*np.sqrt(245)) for name, s in series.items()}); P("## 3. 各年 Sharpe"); P(yrs.round(2).to_markdown()); P("")
    # 被移動停利砍掉的交易, 若持有到收盤會怎樣
    for name in ["trail 2%", "trail 2% after +1%", "MA20"]:
        m = R[f"reason|{name}"]=="trail"; held = R.loc[m, "gross|現行 (無移動停利)"]; stopped = R.loc[m, f"gross|{name}"]
        P(f"- {name}: 被停利的 {m.sum()} 筆 ({m.mean()*100:.0f}%), 停利時每筆 {stopped.mean()*100:+.3f}% vs 若持有到收盤 {held.mean()*100:+.3f}% (差 {(stopped-held).mean()*100:+.3f}%)；未被停利的 {(~m).sum()} 筆每筆 {R.loc[~m, 'gross|現行 (無移動停利)'].mean()*100:+.3f}%")
    (OUT/f"trailing_stop_test{TAG}.md").write_text("\n".join(L)); print("\n".join(L))
