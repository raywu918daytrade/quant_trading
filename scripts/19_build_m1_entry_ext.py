"""從 raw M1 (data/raw/m1/*.parquet) 算延伸回測需要的精確成交價: 每檔每日
  e01_vwap           09:01~09:03 三分鐘成交量加權均價 (可執行的進場假設, 與 13_entry_delay 相同)
  e01_maxhigh_after  09:04 起到收盤的最高價 (判斷強制回補 / 觸漲停)
  close_auction_c    13:30 收盤競價那根 bar 的 close (出場價); m1_close = 最後一根 (<=13:30) 的 close 備援
輸出 data/processed/m1_entry_ext/<YYYY_MM>.parquet, 只處理尚未存在的月份. 4 個 process 平行.
"""
import glob, pandas as pd
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT/"data/processed/m1_entry_ext"; OUT.mkdir(parents=True, exist_ok=True)

def process(f):
    ym = Path(f).stem
    if (OUT/f"{ym}.parquet").exists(): return ym, "skip"
    df = pd.read_parquet(f); ts = pd.to_datetime(df["date"]); df["date"] = ts.dt.normalize(); df["t"] = ts.dt.hour*60 + ts.dt.minute - 540
    df = df[(df.t>=0)&(df.t<=270)].copy(); df["dv"] = df.close*df.volume
    key = ["stock_id","date"]
    w = df[(df.t>=1)&(df.t<4)].groupby(key, sort=False)
    o = pd.DataFrame({"e01_vwap": w.dv.sum()/w.volume.sum(), "e01_vol": w.volume.sum()})
    o["e01_maxhigh_after"] = df[df.t>=4].groupby(key, sort=False).high.max()
    o["close_auction_c"] = df[df.t==270].groupby(key, sort=False).close.last()
    last = df.sort_values("t").groupby(key, sort=False).close.last(); o["m1_close"] = last
    o = o.reset_index(); o.to_parquet(OUT/f"{ym}.parquet", index=False)
    return ym, len(o)

if __name__ == "__main__":
    fs = sorted(glob.glob(str(ROOT/"data/raw/m1/*.parquet")))
    with ProcessPoolExecutor(4) as ex:
        for ym, msg in ex.map(process, fs): print(ym, msg, flush=True)
