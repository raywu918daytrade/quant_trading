"""13:24 尾盤回補規則回測 (做空, M1 逐分鐘, 2020~2026 定案策略每筆交易).

背景: 13:25~13:30 收盤集合競價期間沒有成交, 盤中「成交價 ≥ 昨收 +8% 立即回補」的規則在這五分鐘失效;
paper_trade.py 因此在 13:24 把「現價 ≥ 昨收 +6%」的部位先市價買回 (PRE_AUCTION_K), 避免收盤競價跳到漲停鎖死。這條門檻之前沒有單獨回測。
交易集與 24_trailing_stop_test.py 相同 (sr 模型前 10 檔, 09:01~09:03 VWAP 進, 跳空≤4%, 大盤≥−1%, 股價≤300, 非週一)。
每筆用 M1 算: 13:24 前是否觸 +8% (分鐘 high) / 13:24:00 的價格 (13:23 那根 bar 的 close) / 13:24 那根 bar 的 close (回補成交) / 13:30 收盤競價 / 收盤是否鎖漲停 (≥ 昨收 +9.75%, 隔日開盤 −7% 回補)。
比較: 不設尾盤規則 vs 門檻 3%~8%; 並看「13:24 已漲 k% 的股票, 13:24 → 收盤 的變化與鎖死率」。
用法: python3 scripts/43_pre_auction_test.py [--preds <parquet>] [--tag _x]   → stats/pre_auction_test.md
"""
import sys, glob, importlib, numpy as np, pandas as pd
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
ROOT = Path(__file__).resolve().parent.parent; OUT = ROOT/"stats"
sys.path.insert(0, str(Path(__file__).resolve().parent))
T24 = importlib.import_module("24_trailing_stop_test")
BASE, SLIP, K, PEN, CAP, PER_STOCK = T24.BASE, T24.SLIP, T24.K, T24.PEN, T24.CAP, T24.PER_STOCK
KS = [None, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08]

def scan_month(args):
    f, trades = args
    df = pd.read_parquet(f); ts = pd.to_datetime(df["date"]); df["day"] = ts.dt.normalize(); df["t"] = ts.dt.hour*60 + ts.dt.minute - 540
    df = df[(df.t>=0)&(df.t<=270)]
    key = set(zip(trades.stock_id, trades.date)); df = df[[(s, d) in key for s, d in zip(df.stock_id, df.day)]]
    out = []
    for (sid, day), g in df.groupby(["stock_id","day"], sort=False):
        tr = trades[(trades.stock_id==sid)&(trades.date==day)].iloc[0]; pc = tr.prev_close
        g = g.sort_values("t"); t = g.t.values; h = g.high.values; c = g.close.values
        pre = (t >= 4) & (t <= 264); forced = bool((h[pre] >= pc*(1+K)).any())
        b23 = c[t <= 263][-1] if (t <= 263).any() else np.nan           # 13:24:00 看到的現價
        b24 = c[t == 264][-1] if (t == 264).any() else b23               # 13:24 那分鐘的回補成交
        auction = c[t == 270][-1] if (t == 270).any() else c[-1]
        out.append({"stock_id": sid, "date": day, "entry": tr.entry, "prev_close": pc, "next_open": tr.next_open, "rank": tr["rank"],
                    "forced": forced, "p1324": b23, "x1324": b24, "auction": auction})
    return pd.DataFrame(out)

def gross(R, k):
    """每筆毛報酬 (做空). k=None 不設尾盤規則."""
    hold = np.where(R.lock & np.isfinite(R.next_open), R.entry/R.next_open - 1 - PEN, R.entry/R.auction - 1)
    g = np.where(R.forced, R.entry/(R.prev_close*(1+K)*(1+SLIP*3)) - 1, hold)
    reason = np.where(R.forced, "forced", np.where(R.lock, "lock", "close"))
    if k is not None:
        trig = (~R.forced) & (R.p1324 >= R.prev_close*(1+k))
        g = np.where(trig, R.entry/(R.x1324*(1+SLIP)) - 1, g); reason = np.where(trig, "pre", reason)
    return pd.Series(g, index=R.index), pd.Series(reason, index=R.index)

if __name__ == "__main__":
    if "--preds" not in sys.argv and (ROOT/"data/processed/preds_B_pool_sr_ext.parquet").exists():
        sys.argv += ["--preds", str(ROOT/"data/processed/preds_B_pool_sr_ext.parquet")]   # 現行正式 = sr 模型
    TAG = sys.argv[sys.argv.index("--tag")+1] if "--tag" in sys.argv else ""
    trades = T24.build_trades(); print(f"trades {len(trades)} ({trades.date.min().date()}~{trades.date.max().date()})", flush=True)
    trades["ym"] = trades.date.dt.strftime("%Y_%m")
    jobs = [(str(ROOT/"data/raw/m1"/f"{ym}.parquet"), g) for ym, g in trades.groupby("ym") if (ROOT/"data/raw/m1"/f"{ym}.parquet").exists()]
    with ProcessPoolExecutor(6) as ex: R = pd.concat(list(ex.map(scan_month, jobs)), ignore_index=True)
    R = R[R.p1324.notna() & R.auction.notna()].copy(); R["lock"] = R.auction >= R.prev_close*1.0975
    R["r1324"] = R.p1324/R.prev_close - 1; R["move"] = R.auction/R.p1324 - 1
    R.to_parquet(ROOT/"data/processed"/f"pre_auction_trades{TAG}.parquet", index=False)
    print(f"simulated {len(R)} trades", flush=True)
    all_days = pd.Series(sorted(R.date.unique()))
    def alloc(col):
        recs = []
        for dt, gg in R.sort_values(["date","rank"]).groupby("date", sort=False):
            cash = CAP; tot = 0.0
            for _, r in gg.iterrows():
                lc = r.entry*1000
                if cash < lc: continue
                lots = int(min(PER_STOCK, cash)//lc)
                if lots < 1: continue
                a = lots*lc; cash -= a; tot += a*(r[col]-BASE)
            recs.append((dt, tot/CAP))
        return pd.DataFrame(recs, columns=["date","ret"]).set_index("date").ret.reindex(all_days, fill_value=0.0)
    L = []; P = lambda s="": L.append(str(s))
    P("# 13:24 尾盤回補規則 (做空, M1 逐分鐘, 2020~2026, 300 萬配置, 不交易日計 0)\n")
    P(f"- 樣本 {len(R)} 筆 / {R.date.nunique()} 天 (sr 模型前 10 檔)。進場 09:01~09:03 VWAP；+8% 強制回補 (分鐘 high, 含滑價 0.3%)；13:24 規則用 13:24:00 的現價判斷、以 13:24 那分鐘的收盤價 +0.1% 回補；否則 13:30 收盤競價；收盤鎖漲停 (≥ 昨收 +9.75%) 隔日開盤 −7%。")
    P(f"- 13:24 前已觸 +8% 強制回補 {R.forced.mean()*100:.1f}%；未強制者收盤鎖漲停 {R[~R.forced].lock.mean()*100:.2f}% ({int(R[~R.forced].lock.sum())} 筆)\n")
    rows = []; series = {}; names = {}
    for k in KS:
        name = "不設尾盤規則" if k is None else f"13:24 ≥ 昨收 +{k*100:.0f}% 回補"; names[k] = name
        g, rs = gross(R, k); R[f"gross|{name}"] = g; net = g - BASE; s = alloc(f"gross|{name}"); series[name] = s
        rows.append({"規則": name, "每筆淨%": net.mean()*100, "勝率": (net>0).mean(), "日淨%": s.mean()*100, "Sharpe": s.mean()/s.std()*np.sqrt(245),
                     "MDD%": ((1+s).cumprod()/(1+s).cumprod().cummax()-1).min()*100, "最差1%筆": net.quantile(.01)*100, "最差筆%": net.min()*100,
                     "尾盤觸發%": (rs=="pre").mean()*100, "鎖漲停%": (rs=="lock").mean()*100, "年化(萬)": s.mean()*245*CAP/1e4})
    tab = pd.DataFrame(rows).set_index("規則"); P("## 1. 全期間"); P(tab.round(3).to_markdown()); P("")
    yr = pd.DataFrame({n: s.groupby(s.index.year).mean()*100 for n, s in series.items()}); P("## 2. 各年日淨% (對 300 萬)"); P(yr.round(3).to_markdown()); P("")
    # 條件分析: 13:24 已漲 x% 的交易, 之後到收盤怎麼走
    nf = R[~R.forced].copy()
    nf["bucket"] = pd.cut(nf.r1324*100, [-np.inf, 0, 2, 3, 4, 5, 6, 7, 8, 9.75, np.inf], labels=["≤0", "0~2", "2~3", "3~4", "4~5", "5~6", "6~7", "7~8", "8~9.75", "≥9.75 (已漲停)"])
    hold_net = R["gross|不設尾盤規則"] - BASE; cov_net = (nf.entry/(nf.x1324*(1+SLIP)) - 1) - BASE
    P("## 3. 13:24 現價相對昨收的位置 → 之後到收盤 (未觸 +8% 的交易)")
    P("「若 13:24 回補」= 以 13:24 分鐘收盤價 +0.1% 買回的每筆淨；「若抱到收盤」= 現行無尾盤規則 (含鎖漲停隔日 −7%)。差 > 0 代表回補比較好。")
    rows = []
    for b, gg in nf.groupby("bucket", observed=True):
        i = gg.index
        rows.append({"13:24 漲幅": b, "筆數": len(gg), "占比%": len(gg)/len(nf)*100, "13:24→收盤 平均%": gg.move.mean()*100, "中位%": gg.move.median()*100, "再漲>1%": (gg.move>0.01).mean()*100, "回落>1%": (gg.move<-0.01).mean()*100,
                     "收盤鎖漲停%": gg.lock.mean()*100, "若 13:24 回補 每筆淨%": cov_net.loc[i].mean()*100, "若抱到收盤 每筆淨%": hold_net.loc[i].mean()*100, "差%": (cov_net.loc[i]-hold_net.loc[i]).mean()*100, "差合計(單位%)": (cov_net.loc[i]-hold_net.loc[i]).sum()*100})
    P(pd.DataFrame(rows).set_index("13:24 漲幅").round(2).to_markdown()); P("")
    # 鎖漲停的交易 13:24 在哪
    lk = nf[nf.lock]
    if len(lk):
        P(f"## 4. 收盤鎖漲停的 {len(lk)} 筆, 13:24 時的漲幅分布")
        P(lk.r1324.mul(100).describe(percentiles=[.1,.25,.5,.75,.9]).round(2).to_frame("13:24 漲幅%").T.to_markdown()); P("")
        P(f"- 13:24 已 ≥ +6% 的占 {(lk.r1324>=0.06).mean()*100:.0f}%，≥ +7% 的占 {(lk.r1324>=0.07).mean()*100:.0f}%，< +4% 的占 {(lk.r1324<0.04).mean()*100:.0f}% (這些任何尾盤門檻都救不到)")
        P(f"- 鎖漲停交易每筆淨 (隔日 −7% 回補) 平均 {hold_net.loc[lk.index].mean()*100:+.2f}%，合計 {hold_net.loc[lk.index].sum()*100:+.1f} 單位%；全部交易淨合計 {hold_net.sum()*100:+.1f} 單位%\n")
    (OUT/f"pre_auction_test{TAG}.md").write_text("\n".join(L)); print("\n".join(L))
