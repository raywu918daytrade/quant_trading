"""從 raw M1 算進場視窗 VWAP 與進場後極值 (做空用)."""
import glob, numpy as np, pandas as pd
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT/"data/processed/m1_entry"; OUT.mkdir(parents=True, exist_ok=True)
def process(f):
    ym = Path(f).stem
    if (OUT/f"{ym}.parquet").exists(): return ym, "skip"
    df = pd.read_parquet(f)
    ts = pd.to_datetime(df["date"]); df["date"] = ts.dt.normalize(); df["t"] = ts.dt.hour*60 + ts.dt.minute - 540
    df = df[(df.t>=0)&(df.t<=270)]
    df["dv"] = df.close*df.volume
    key = ["stock_id","date"]
    out = []
    for name, lo, hi in [("e00",0,3),("e30",30,33),("e60",60,63)]:
        w = df[(df.t>=lo)&(df.t<hi)].groupby(key, sort=False)
        o = pd.DataFrame({f"{name}_vwap": w.dv.sum()/w.volume.sum(), f"{name}_vol": w.volume.sum(), f"{name}_n": w.close.size()})
        after = df[df.t>=hi].groupby(key, sort=False)
        o[f"{name}_maxhigh_after"] = after.high.max(); o[f"{name}_minlow_after"] = after.low.min()
        out.append(o)
    # 12:00 之後 / 13:00 之後 的 VWAP (提早出場用)
    for name, lo, hi in [("x12",180,183),("x13",240,243)]:
        w = df[(df.t>=lo)&(df.t<hi)].groupby(key, sort=False)
        out.append(pd.DataFrame({f"{name}_vwap": w.dv.sum()/w.volume.sum()}))
    res = pd.concat(out, axis=1).reset_index()
    res.to_parquet(OUT/f"{ym}.parquet", index=False)
    return ym, len(res)
if __name__ == "__main__":
    fs = sorted(glob.glob(str(ROOT/"data/raw/m1/*.parquet")))
    with ProcessPoolExecutor(4) as ex:
        for ym, msg in ex.map(process, fs): print(ym, msg, flush=True)
