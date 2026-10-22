"""M1 逐月串流 -> (a) 每檔每日日內特徵  (b) 每分鐘聚合 profile.  結果寫到 data/processed/."""
import glob, sys, numpy as np, pandas as pd
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
ROOT = Path(__file__).resolve().parent.parent
OUTD = ROOT/"data/processed/m1_daily"; OUTD.mkdir(parents=True, exist_ok=True)
OUTP = ROOT/"data/processed/m1_profile"; OUTP.mkdir(parents=True, exist_ok=True)

def process(f):
    ym = Path(f).stem
    if (OUTD/f"{ym}.parquet").exists() and (OUTP/f"{ym}.parquet").exists():
        return ym, "skip"
    df = pd.read_parquet(f)
    df["ts"] = pd.to_datetime(df["date"])
    df["date"] = df.ts.dt.normalize()
    df["tod"] = df.ts.dt.hour*60 + df.ts.dt.minute          # minutes since 00:00
    df["t"] = df.tod - 540                                   # minutes since 09:00
    df = df.sort_values(["stock_id","date","ts"])
    g = df.groupby(["stock_id","date"], sort=False)
    df["lr"] = np.log(df.close) - np.log(g.close.shift(1))
    df["dv"] = df.close*df.volume
    df["cumv"] = g.volume.cumsum()

    # ---- (b) minute-of-day profile ----
    prof = df.groupby("tod").agg(n=("close","size"), vol=("volume","sum"), dv=("dv","sum"),
                                 abs_lr=("lr", lambda s: s.abs().mean()), lr2=("lr", lambda s:(s**2).mean()),
                                 hl=("high", "sum"))
    prof["range_sum"] = df.assign(r=(df.high-df.low)/df.close).groupby("tod").r.sum()
    prof["ym"] = ym
    prof.reset_index().to_parquet(OUTP/f"{ym}.parquet", index=False)

    # ---- (a) per stock-day features ----
    def win(mask, name):
        s = df[mask].groupby(["stock_id","date"], sort=False)
        return pd.DataFrame({f"{name}_h": s.high.max(), f"{name}_l": s.low.min(), f"{name}_c": s.close.last(),
                             f"{name}_v": s.volume.sum(), f"{name}_dv": s.dv.sum(), f"{name}_n": s.close.size()})
    base = pd.DataFrame({
        "m1_open": g.open.first(), "m1_high": g.high.max(), "m1_low": g.low.min(), "m1_close": g.close.last(),
        "m1_vol": g.volume.sum(), "m1_dv": g.dv.sum(), "n_bars": g.close.size(),
        "t_first": g.t.first(), "t_last": g.t.last(),
        "rv": g.lr.apply(lambda s: np.nansum(s.values**2)),          # realized variance (log)
        "t_high": g.apply(lambda x: x.t.values[x.high.values.argmax()]),
        "t_low": g.apply(lambda x: x.t.values[x.low.values.argmin()]),
        "lr_abs_sum": g.lr.apply(lambda s: np.nansum(np.abs(s.values))),
    })
    base["vwap"] = base.m1_dv/base.m1_vol
    base = base.join(win(df.t < 5,  "o05")).join(win(df.t < 15, "o15")).join(win(df.t < 30, "o30")).join(win(df.t < 60, "o60"))
    base = base.join(win((df.t >= 240), "last30"))             # 13:00 以後
    base = base.join(win(df.tod == 810, "close_auction"))       # 13:30 bar
    base = base.join(win(df.tod > 810, "after"))                 # 13:30 之後的 bar (檢查用)
    # 10:00 / 11:00 / 12:00 / 13:00 的收盤價 (該時刻前最後一根)
    for hh, nm in [(600,"c10"),(660,"c11"),(720,"c12"),(780,"c13")]:
        base[nm] = df[df.tod < hh].groupby(["stock_id","date"], sort=False).close.last()
    base = base.reset_index()
    base.to_parquet(OUTD/f"{ym}.parquet", index=False)
    return ym, f"{len(df)} rows -> {len(base)} stock-days"

if __name__ == "__main__":
    fs = sorted(glob.glob(str(ROOT/"data/raw/m1/*.parquet")))
    if len(sys.argv) > 1: fs = [f for f in fs if Path(f).stem in sys.argv[1:]]
    with ProcessPoolExecutor(max_workers=4) as ex:
        for ym, msg in ex.map(process, fs):
            print(ym, msg, flush=True)
