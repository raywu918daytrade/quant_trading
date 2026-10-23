"""開盤量 PR 值: 今日第一根 M5 (09:00~09:04) 量在前 N 日同時段量裡的百分位 (PR 0~100, 100 = 比前 N 日全部都大); N = 5 / 20 / 60.
也算第一根 M1 (09:00) 的 PR (09:01 進場前可知). 報酬/出場沿用 30 存的 open_volume_trades.parquet (g01 = 09:01 進場, g05 = 09:05 進場).
輸出 stats/open_volume_pr.md"""
import numpy as np, pandas as pd
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
ROOT = Path(__file__).resolve().parent.parent; OUT = ROOT/"stats"
BASE, CAP, PER_STOCK, MAX_N = 0.0023, 3_000_000, 600_000, 10; NS = (5, 20, 60)

def load(ym, ids):
    f = ROOT/"data/raw/m1"/f"{ym}.parquet"
    if not f.exists(): return None
    d = pd.read_parquet(f, filters=[("stock_id", "in", list(ids))]); ts = pd.to_datetime(d["date"]); d["day"] = ts.dt.normalize(); d["t"] = ts.dt.hour*60 + ts.dt.minute - 540
    d = d[(d.t >= 0) & (d.t <= 4)]
    return d.groupby(["stock_id","day"]).apply(lambda g: pd.Series({"v1": g.volume[g.t == 0].sum(), "v5": g.volume.sum()})).reset_index()

def month(args):
    ym, trades = args; ids = set(trades.stock_id); y, m = map(int, ym.split("_"))
    yms = [f"{(y if m-k > 0 else y-1)}_{((m-k-1) % 12)+1:02d}" for k in range(3, -1, -1)]   # 前 3 個月 + 本月
    parts = [x for x in (load(z, ids) for z in yms) if x is not None]
    if not parts: return pd.DataFrame()
    agg = pd.concat(parts, ignore_index=True).sort_values(["stock_id","day"]); out = []
    for sid, g in agg.groupby("stock_id"):
        days = g.day.values; v1 = g.v1.values; v5 = g.v5.values
        for _, tr in trades[trades.stock_id == sid].iterrows():
            i = np.searchsorted(days, np.datetime64(tr.date))
            if i >= len(days) or days[i] != np.datetime64(tr.date): continue
            res = {"stock_id": sid, "date": tr.date}
            for n in NS:
                p1, p5 = v1[max(0, i-n):i], v5[max(0, i-n):i]
                res[f"pr1_{n}"] = (p1 < v1[i]).mean()*100 if len(p1) >= min(n, 5) else np.nan
                res[f"pr5_{n}"] = (p5 < v5[i]).mean()*100 if len(p5) >= min(n, 5) else np.nan
            out.append(res)
    return pd.DataFrame(out)

if __name__ == "__main__":
    S0 = pd.read_parquet(ROOT/"data/processed/open_volume_trades.parquet"); R = S0[["stock_id","date"]].copy(); R["ym"] = R.date.dt.strftime("%Y_%m")
    with ProcessPoolExecutor(6) as ex: parts = list(ex.map(month, [(ym, g) for ym, g in R.groupby("ym")]))
    S = S0.merge(pd.concat([p for p in parts if len(p)], ignore_index=True), on=["stock_id","date"], how="left"); S.to_parquet(ROOT/"data/processed/open_volume_pr_trades.parquet", index=False); print("merged", len(S), flush=True)
    all_days = pd.Series(sorted(S.date.unique()))
    def alloc(mask, gcol, ecol):
        recs = []
        for dt, gg in S[mask & S[gcol].notna()].sort_values(["date","rank"]).groupby("date", sort=False):
            cash = CAP; amts = []; nets = []
            for _, r in gg.iterrows():
                lc = r[ecol]*1000
                if cash < lc: continue
                lots = int(min(PER_STOCK, cash)//lc)
                if lots < 1: continue
                a = lots*lc; cash -= a; amts.append(a); nets.append(r[gcol]-BASE)
                if len(amts) >= MAX_N: break
            recs.append((dt, len(amts), sum(a*n for a, n in zip(amts, nets))/CAP))
        return pd.DataFrame(recs, columns=["date","n","ret"]).set_index("date").reindex(all_days).fillna({"n": 0, "ret": 0.0})
    def stats(name, mask, gcol, ecol):
        D = alloc(mask, gcol, ecol); s = D.ret; g = S.loc[mask & S[gcol].notna(), gcol] - BASE
        return {"規則": name, "交易占比%": (mask & S[gcol].notna()).mean()*100, "日均檔數": D.n.mean(), "每筆淨%": g.mean()*100, "勝率": (g > 0).mean(), "日淨%": s.mean()*100, "Sharpe": s.mean()/s.std()*np.sqrt(245), "MDD%": ((1+s).cumprod()/(1+s).cumprod().cummax()-1).min()*100, "年化(萬)": s.mean()*245*CAP/1e4}
    L = []; P = lambda s="": L.append(str(s))
    P("# 開盤量 PR 值 vs 做空結果 (2020~2026, sr 模型每日前 10 檔, 300 萬)\n")
    P("- pr5_N = 今日 09:00~09:04 五分鐘量在前 N 日同時段量的百分位 (100 = 比前 N 日都大, 0 = 比前 N 日都小)；pr1_N = 09:00 那一根 M1 的版本 (09:01 進場前可知)。用 M5 的版本進場延到 09:05 開盤。\n")
    P("## 1. 依 PR 分組: 現行 09:01 進場結果")
    bins = [-1, 20, 40, 60, 80, 95, 100]; labs = ["0~20","20~40","40~60","60~80","80~95","95~100"]
    for col in [f"pr5_{n}" for n in NS] + [f"pr1_{n}" for n in NS]:
        b = pd.cut(S[col], bins, labels=labs); net = S.g01 - BASE
        T = pd.DataFrame({"筆數": net.groupby(b).size(), "占比%": net.groupby(b).size()/S[col].notna().sum()*100, "每筆淨%": net.groupby(b).mean()*100, "勝率": (net > 0).groupby(b).mean(), "強制回補%": (S.reason == "forced").groupby(b).mean()*100, "鎖漲停%": (S.reason == "lock").groupby(b).mean()*100, "盤中最高漲幅中位%": S.maxup.groupby(b).median()*100, "09:05進場每筆淨%": (S.g05 - BASE).groupby(b).mean()*100})
        P(f"\n### {col}"); P(T.round(3).to_markdown())
    P("\n## 2. 當濾網 (PR > 門檻不做, 資金給下一檔)")
    rows = [stats("現行 09:01", S.g01.notna(), "g01", "entry"), stats("全部 09:05 進場 (無濾網)", S.g05.notna(), "g05", "entry05")]
    for n in NS:
        for th in (80, 95): rows.append(stats(f"pr1_{n} ≤ {th}, 09:01 進場", S[f"pr1_{n}"] <= th, "g01", "entry"))
    for n in NS:
        for th in (60, 80, 95): rows.append(stats(f"pr5_{n} ≤ {th}, 09:05 進場", S[f"pr5_{n}"] <= th, "g05", "entry05"))
    for n in NS: rows.append(stats(f"反向: 只做 pr5_{n} > 80, 09:05 進場", S[f"pr5_{n}"] > 80, "g05", "entry05"))
    P(pd.DataFrame(rows).set_index("規則").round(3).to_markdown()); P("")
    P("## 3. 各年每筆淨%")
    yr = pd.DataFrame({"現行": (S.g01-BASE).groupby(S.date.dt.year).mean()*100, "pr1_20≤95": (S.g01-BASE)[S.pr1_20 <= 95].groupby(S.date.dt.year).mean()*100, "pr5_20≤80@09:05": (S.g05-BASE)[S.pr5_20 <= 80].groupby(S.date.dt.year).mean()*100, "全部@09:05": (S.g05-BASE).groupby(S.date.dt.year).mean()*100})
    P(yr.round(2).to_markdown()); P("")
    P("## 4. Spearman 相關"); P("- PR vs 現行淨報酬: " + ", ".join(f"{c} {S[[c,'g01']].corr(method='spearman').iloc[0,1]:+.3f}" for c in [f"pr5_{n}" for n in NS] + [f"pr1_{n}" for n in NS]))
    P("- PR vs 盤中最高漲幅: " + ", ".join(f"{c} {S[[c,'maxup']].corr(method='spearman').iloc[0,1]:+.3f}" for c in [f"pr5_{n}" for n in NS]))
    (OUT/"open_volume_pr.md").write_text("\n".join(L)); print("\n".join(L))
