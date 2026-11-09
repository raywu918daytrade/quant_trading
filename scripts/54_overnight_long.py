"""隔夜做多: 收盤 (收盤競價) 買進 → 隔天開盤賣出. 全市場 universe (同 53), 2020-01 ~, D1.
不是當沖: 成本 = 手續費約 0.08% + 證交稅 0.3% = 0.38% (另列 0.23% 當對照, 等於假設可以零稅).
收盤漲停 (收 ≥ 昨收 ×1.095) 的買不到, 排除. 隔天要是下一個交易日 (停牌的跳過).
分組都只用當天收盤前就知道的資訊: 今天開→收 (盤中) 漲跌、今天漲跌、收盤在當天區間的位置、是不是 sr 前 10 (當天被現行策略放空的)。
組合: 每天挑「盤中跌最多」的 N 檔 (N = 5 / 10 / 20) 等權, 沒交易日計 0。另列 隔天收盤賣 (抱一整天) 對照。
-> stats/overnight_long.md"""
import os, importlib.util, numpy as np, pandas as pd
from pathlib import Path
ROOT = Path(os.environ.get("QT_ROOT", Path(__file__).resolve().parent.parent)); OUT = ROOT / "stats"; RAW = ROOT / "data/raw"
COST, COST_LO, SPLIT = 0.0038, 0.0023, 2024
spec = importlib.util.spec_from_file_location("qb", ROOT / "scripts/53_quiet_breakout.py"); qb = importlib.util.module_from_spec(spec); spec.loader.exec_module(qb)

def tstat(d):
    return d.mean() / d.std() * np.sqrt(len(d)) if len(d) > 2 and d.std() > 0 else np.nan

if __name__ == "__main__":
    U = qb.universe()
    d = pd.concat([pd.read_parquet(f) for f in sorted((RAW / "d1").glob("*.parquet"))], ignore_index=True)
    d["date"] = pd.to_datetime(d.date); d = d.sort_values(["stock_id", "date"])
    cal = np.sort(d.date.unique()); nxt = pd.Series(cal[1:], index=cal[:-1])
    g = d.groupby("stock_id", sort=False)
    d["n_date"] = g.date.shift(-1); d["n_open"] = g.open.shift(-1); d["n_close"] = g.close.shift(-1)
    d = d.merge(U, on=["stock_id", "date"], how="inner")
    d = d[d.n_date.eq(d.date.map(nxt)) & d.n_open.gt(0) & d.close.gt(0) & (d.close < d.pc * 1.095)].copy()
    d["on"] = (d.n_open / d.close - 1) * 100; d["cc"] = (d.n_close / d.close - 1) * 100
    d["intra"] = d.close / d.open - 1; d["ret"] = d.close / d.pc - 1
    d["loc"] = np.where(d.high > d.low, (d.close - d.low) / (d.high - d.low), 0.5)
    sr = pd.read_parquet(ROOT / "data/processed/trailing_stop_trades_sr.parquet")[["stock_id", "date"]]; sr["date"] = pd.to_datetime(sr.date).astype("datetime64[ns]"); sr["sr"] = True
    d = d.merge(sr, on=["stock_id", "date"], how="left"); d["sr"] = d.sr.fillna(False).astype(bool)
    d["oos"] = d.date.dt.year >= SPLIT; days = np.sort(d.date.unique())

    def row(lab, x):
        net = x.on - COST * 100; day = net.groupby(x.date).mean(); i = x[~x.oos]; o = x[x.oos]
        return {"分組": lab, "筆數": len(x), "日均筆": len(x) / len(days), "隔夜毛%": x.on.mean(), "淨% (0.38)": net.mean(), "淨% (0.23)": (x.on - COST_LO * 100).mean(),
                "勝率%": (net > 0).mean() * 100, "t": tstat(day), "淨 IS": (i.on - COST * 100).mean(), "t IS": tstat((i.on - COST * 100).groupby(i.date).mean()),
                "淨 OOS": (o.on - COST * 100).mean(), "t OOS": tstat((o.on - COST * 100).groupby(o.date).mean()), "抱到隔天收 淨%": (x.cc - COST * 100).mean()}
    L = []; P = lambda s="": L.append(str(s))
    P(f"# 隔夜做多: 收盤買 → 隔天開盤賣 (全市場 universe, {d.date.min():%Y-%m-%d}~{d.date.max():%Y-%m-%d}, {len(d)} 檔日 / {len(days)} 天)"); P()
    P("- universe 同 53 (上市櫃 4 碼非 ETF、昨收 ≥10、20 日均成交額 ≥1 億); 收盤漲停買不到排除")
    P("- 成本 0.38% (手續費 0.08% + 證交稅 0.3%, 非當沖); 0.23% 那欄只是對照; t 用每日平均算; IS 2020~23 / OOS 2024~"); P()
    rows = [row("全部", d)]
    for lab, lo, hi in [("盤中 ≤−5%", -1, -0.05), ("盤中 −5~−3%", -0.05, -0.03), ("盤中 −3~−1%", -0.03, -0.01), ("盤中 −1~1%", -0.01, 0.01), ("盤中 1~3%", 0.01, 0.03), ("盤中 ≥3%", 0.03, 1)]:
        rows.append(row(lab, d[(d.intra > lo) & (d.intra <= hi)]))
    for lab, lo, hi in [("今天跌停附近 ≤−9.5%", -1, -0.095), ("今天 −9.5~−5%", -0.095, -0.05), ("今天 −5~−2%", -0.05, -0.02), ("今天 −2~2%", -0.02, 0.02), ("今天 2~5%", 0.02, 0.05), ("今天 5~9.5%", 0.05, 0.095)]:
        rows.append(row(lab, d[(d.ret > lo) & (d.ret <= hi)]))
    rows.append(row("收在當天最低 10%", d[d["loc"] <= 0.1])); rows.append(row("收在當天最高 10%", d[d["loc"] >= 0.9]))
    rows.append(row("sr 前 10 (當天放空的)", d[d.sr])); rows.append(row("sr 前 10 且盤中 ≤−3%", d[d.sr & (d.intra <= -0.03)])); rows.append(row("sr 前 10 且盤中 ≥0", d[d.sr & (d.intra >= 0)]))
    P("## 分組 (每筆)"); P(pd.DataFrame(rows).round(3).to_markdown(index=False)); P()
    P("## 組合: 每天挑盤中跌最多的 N 檔, 等權 (日淨%, 成本 0.38%)"); rr = []
    for N in (5, 10, 20):
        for lab, pool in (("全市場", d), ("只限 sr 前 10", d[d.sr])):
            top = pool.sort_values(["date", "intra"]).groupby("date").head(N); s = pd.Series(0.0, index=days)
            dd = (top.on - COST * 100).groupby(top.date).mean(); s.loc[dd.index] = dd.values; s = s / 100
            r = {"組合": f"{lab} 前 {N}", "日淨%": s.mean() * 100, "Sharpe": s.mean() / s.std() * np.sqrt(245), "MDD%": ((1 + s).cumprod() / (1 + s).cumprod().cummax() - 1).min() * 100}
            for nm, m in (("IS", pd.DatetimeIndex(days).year < SPLIT), ("OOS", pd.DatetimeIndex(days).year >= SPLIT)):
                r[f"日淨% {nm}"] = s[m].mean() * 100; r[f"Sharpe {nm}"] = s[m].mean() / s[m].std() * np.sqrt(245)
            rr.append(r)
    P(pd.DataFrame(rr).round(3).to_markdown(index=False)); P()
    P("## 各年 隔夜淨% (每筆, 0.38%)")
    y = d.assign(year=d.date.dt.year, net=d.on - COST * 100)
    Y = pd.DataFrame({"全部": y.groupby("year").net.mean(), "盤中 ≤−5%": y[y.intra <= -0.05].groupby("year").net.mean(), "sr 前 10": y[y.sr].groupby("year").net.mean(),
                      "全部 毛%": y.groupby("year").on.mean()})
    P(Y.round(3).T.to_markdown()); P()
    (OUT / "overnight_long.md").write_text("\n".join(L), encoding="utf-8"); print("\n".join(L))
