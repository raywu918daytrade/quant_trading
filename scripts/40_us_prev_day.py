"""前一日美股漲跌 vs 現行策略 (sr, 10 檔 × 60 萬, 2020~2026) 的日淨/勝率.
美股: S&P 500 (^GSPC)、Nasdaq (^IXIC)、費半 (^SOX)、台積電 ADR (TSM), Yahoo 日線, 快取 data/raw/us/us_daily.parquet (加 --refresh 重抓).
對齊: 台股交易日 D 用「美國交易日 < D」的最後一個收盤 (美股收盤 = 台灣隔天 04:00/05:00, 在 09:00 開盤前已知); 週一對應上週五.
交易樣本同 36_drawdown_diagnosis (trailing_stop_trades_sr.parquet, 現行出場). 輸出 stats/us_prev_day.md"""
import sys, glob, time, numpy as np, pandas as pd, requests
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent; OUT = ROOT/"stats"; BASE, CAP, PER_STOCK, MAX_N = 0.0023, 3_000_000, 600_000, 10
US = {"^GSPC": "spx", "^IXIC": "ndx", "^SOX": "sox", "TSM": "tsm"}; NAMES = {"spx": "S&P 500", "ndx": "Nasdaq", "sox": "費半 SOX", "tsm": "台積電 ADR"}
CACHE = ROOT/"data/raw/us/us_daily.parquet"

def fetch_us():
    out = []
    for sym, key in US.items():
        r = requests.get(f"https://query1.finance.yahoo.com/v8/finance/chart/{sym}", params={"period1": int(pd.Timestamp("2019-10-01").timestamp()), "period2": int(time.time()), "interval": "1d"}, headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
        r.raise_for_status(); j = r.json()["chart"]["result"][0]
        ts = pd.to_datetime(j["timestamp"], unit="s", utc=True).tz_convert("America/New_York").tz_localize(None).normalize()
        out.append(pd.Series(j["indicators"]["adjclose"][0]["adjclose"] if "adjclose" in j["indicators"] else j["indicators"]["quote"][0]["close"], index=ts, name=key))
    u = pd.concat(out, axis=1).sort_index(); u = u[~u.index.duplicated(keep="last")]; u.index.name = "us_date"
    CACHE.parent.mkdir(parents=True, exist_ok=True); u.to_parquet(CACHE); return u

u = pd.read_parquet(CACHE) if CACHE.exists() and "--refresh" not in sys.argv else fetch_us()
u = u[u.index < pd.Timestamp.now().normalize()]   # 今天美股還沒收盤的那根不用
ur = u.pct_change(fill_method=None); ur.columns = [f"{c}_ret" for c in u.columns]; ur["sox_ret3"] = u.sox.pct_change(3, fill_method=None); ur = ur.reset_index()

R = pd.read_parquet(ROOT/"data/processed/trailing_stop_trades_sr.parquet"); R = R[["stock_id","date","entry","prev_close","rank","gross|現行 (無移動停利)"]].rename(columns={"gross|現行 (無移動停利)": "g"}); R["net"] = R.g - BASE
d1 = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(str(ROOT/"data/raw/d1/*.parquet")))], ignore_index=True); d1["date"] = pd.to_datetime(d1.date); d1 = d1.drop_duplicates(["stock_id","date"]).sort_values(["stock_id","date"]).reset_index(drop=True)
pc = d1.groupby("stock_id").close.shift(1); d1["gap"] = d1.open/pc - 1; d1["intra"] = d1.close/d1.open - 1
d1["dv20"] = (d1.close*d1.volume).groupby(d1.stock_id).transform(lambda s: s.shift(1).rolling(20).mean())
mk = d1[d1.dv20 >= 1e8].groupby("date").agg(mkt_gap=("gap","mean"), mkt_intra=("intra","mean")).sort_index().reset_index()
mk["date"] = mk.date.astype("datetime64[ns]"); ur = ur.assign(us_date=ur.us_date.astype("datetime64[ns]"))
mk = pd.merge_asof(mk, ur, left_on="date", right_on="us_date", allow_exact_matches=False).set_index("date")
mk["us_lag_days"] = (mk.index - mk.us_date).dt.days

def alloc(S):
    recs = []
    for dt, gg in S.sort_values(["date","rank"]).groupby("date", sort=False):
        cash = CAP; amts = []; nets = []
        for _, r in gg.iterrows():
            lc = r.entry*1000
            if cash < lc: continue
            lots = int(min(PER_STOCK, cash)//lc)
            if lots < 1: continue
            a = lots*lc; cash -= a; amts.append(a); nets.append(r.net)
            if len(amts) >= MAX_N: break
        recs.append((dt, len(amts), sum(a*n for a, n in zip(amts, nets))/CAP))
    return pd.DataFrame(recs, columns=["date","n","ret"]).set_index("date")
def st(name, s):
    mo = s.groupby(s.index.to_period("M")).apply(lambda x: ((1+x).prod()-1)*100); eq = (1+s).cumprod(); dd = eq/eq.cummax()-1
    return {"規則": name, "日淨%": s.mean()*100, "日勝率": (s[s != 0] > 0).mean(), "Sharpe": s.mean()/s.std()*np.sqrt(245), "MDD%": dd.min()*100, "最差日%": s.min()*100, "最差月%": mo.min(), "負月比例": (mo < 0).mean(), "交易日占比%": (s != 0).mean()*100, "年化(萬)": s.mean()*245*CAP/1e4}

D0 = alloc(R); s0 = D0.ret; M = mk.reindex(s0.index); df = pd.DataFrame({"ret": s0}).join(M)
T = R.merge(mk, left_on="date", right_index=True, how="left")
L = []; P = lambda s="": L.append(str(s))
P("# 前一日美股漲跌 vs 現行策略 (2020~2026, sr 10 檔 × 60 萬, 做空當沖)\n")
P(f"- 交易日 {len(s0)}，對到美股的 {df.spx_ret.notna().sum()}；美股資料 {u.index.min().date()} ~ {u.index.max().date()}")
P(f"- 對齊方式: 台股 D 日 ← 美國交易日 < D 的最後收盤 (週一 ← 上週五)。距離天數分布: {df.us_lag_days.value_counts().sort_index().to_dict()}")
P(f"- 整體: 日淨平均 {s0.mean()*100:.3f}%，日勝率 {(s0 > 0).mean():.3f}，每筆勝率 {(R.net > 0).mean():.3f}")

P("\n## 1. 美股前一日 vs 台股今日大盤 (流動股平均)")
P(pd.DataFrame({k: {"corr(美股, 台股跳空)": df[f"{k}_ret"].corr(df.mkt_gap), "corr(美股, 台股日內)": df[f"{k}_ret"].corr(df.mkt_intra), "corr(美股, 策略日淨)": df[f"{k}_ret"].corr(df.ret)} for k in NAMES}).rename(columns=NAMES).round(3).to_markdown())

BINS = [-1, -.02, -.01, -.005, 0, .005, .01, .02, 1]; LABS = ["<−2%","−2~−1%","−1~−0.5%","−0.5~0%","0~0.5%","0.5~1%","1~2%",">2%"]
BINS_W = [-1, -.03, -.02, -.01, 0, .01, .02, .03, 1]; LABS_W = ["<−3%","−3~−2%","−2~−1%","−1~0%","0~1%","1~2%","2~3%",">3%"]
P("\n## 2. 依前一日美股漲跌分組 (日 = 當天 10 檔合計日淨; 筆 = 每筆交易)")
for col, bins, labs, title in [("spx_ret", BINS, LABS, "S&P 500 前一日"), ("ndx_ret", BINS, LABS, "Nasdaq 前一日"), ("sox_ret", BINS_W, LABS_W, "費半 SOX 前一日"), ("tsm_ret", BINS_W, LABS_W, "台積電 ADR 前一日"), ("sox_ret3", [-1, -.06, -.03, 0, .03, .06, 1], ["<−6%","−6~−3%","−3~0%","0~3%","3~6%",">6%"], "費半 SOX 前 3 日累計")]:
    b = pd.cut(df[col], bins, labels=labs); bt = pd.cut(T[col], bins, labels=labs)
    G = pd.DataFrame({"交易日數": df.groupby(b, observed=False).size(), "日淨平均%": df.ret.groupby(b, observed=False).mean()*100, "日勝率": (df.ret > 0).groupby(b, observed=False).mean(), "最差日%": df.ret.groupby(b, observed=False).min()*100,
                      "合計(萬)": df.ret.groupby(b, observed=False).sum()*CAP/1e4, "每筆勝率": (T.net > 0).groupby(bt, observed=False).mean(), "每筆淨平均%": T.net.groupby(bt, observed=False).mean()*100,
                      "台股跳空平均%": df.mkt_gap.groupby(b, observed=False).mean()*100, "台股日內平均%": df.mkt_intra.groupby(b, observed=False).mean()*100})
    P(f"\n### {title}"); P(G.round(3).to_markdown())

P("\n## 3. 美股方向 × 台股今日跳空 (09:00 可知) 的日淨平均% (括號 = 天數)")
a = np.where(df.sox_ret > .01, "SOX > +1%", np.where(df.sox_ret < -.01, "SOX < −1%", "SOX ±1%")); g = pd.cut(df.mkt_gap, [-1, 0, .005, .01, 1], labels=["跳空<0","0~0.5%","0.5~1%",">1%"])
X = df.assign(a=a, g=g).groupby(["a","g"], observed=False).ret.agg(["mean","size"]); P(X.apply(lambda r: f"{r['mean']*100:.2f} ({int(r['size'])})", axis=1).unstack().reindex(["SOX < −1%","SOX ±1%","SOX > +1%"]).to_markdown())

P("\n## 4. 候選規則 (前一晚美股太強/太弱就不做) 的整體效果")
rows = [st("現行", s0)]
for name, cond in [("SPX 前日 > +1% 不做", M.spx_ret > .01), ("SPX 前日 > +1.5% 不做", M.spx_ret > .015), ("SPX 前日 < −1% 不做", M.spx_ret < -.01), ("Nasdaq 前日 > +1.5% 不做", M.ndx_ret > .015), ("Nasdaq 前日 < −1.5% 不做", M.ndx_ret < -.015),
                   ("SOX 前日 > +2% 不做", M.sox_ret > .02), ("SOX 前日 > +3% 不做", M.sox_ret > .03), ("SOX 前日 < −2% 不做", M.sox_ret < -.02), ("SOX 前日 < −3% 不做", M.sox_ret < -.03), ("TSM ADR 前日 > +3% 不做", M.tsm_ret > .03), ("TSM ADR 前日 < −3% 不做", M.tsm_ret < -.03)]:
    rows.append(st(name, s0.where(~cond.fillna(False), 0.0)))
P(pd.DataFrame(rows).set_index("規則").round(3).to_markdown())

P("\n## 5. 分年穩定度: SOX 前日漲跌三組的日淨平均%")
a = pd.Series(np.where(df.sox_ret > .01, "SOX > +1%", np.where(df.sox_ret < -.01, "SOX < −1%", "SOX ±1%")), index=df.index)
P(df.ret.groupby([df.index.year, a]).mean().mul(100).unstack().reindex(columns=["SOX < −1%","SOX ±1%","SOX > +1%"]).round(3).to_markdown())
(OUT/"us_prev_day.md").write_text("\n".join(L)); print("\n".join(L))
