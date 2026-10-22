"""進場延遲敏感度: 09:01~ / 09:02~ / 09:03~ / 09:05~ 三分鐘 VWAP 進場 vs 原 09:00~09:02."""
import glob, numpy as np, pandas as pd
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
ROOT = Path(__file__).resolve().parent.parent; OUT = ROOT/"stats"
E2 = ROOT/"data/processed/m1_entry_delay"; E2.mkdir(exist_ok=True)
WINS = {"e00":(0,3),"e01":(1,4),"e02":(2,5),"e03":(3,6),"e05":(5,8),"e10":(10,13)}
BASE=0.0023; PENALTY=0.002
def process(f):
    ym = Path(f).stem
    if (E2/f"{ym}.parquet").exists(): return ym
    df = pd.read_parquet(f); ts = pd.to_datetime(df["date"]); df["date"]=ts.dt.normalize(); df["t"]=ts.dt.hour*60+ts.dt.minute-540
    df = df[(df.t>=0)&(df.t<=270)]; df["dv"]=df.close*df.volume
    out=[]
    for n,(lo,hi) in WINS.items():
        w = df[(df.t>=lo)&(df.t<hi)].groupby(["stock_id","date"], sort=False)
        o = pd.DataFrame({f"{n}_vwap": w.dv.sum()/w.volume.sum(), f"{n}_first": w.open.first()})
        o[f"{n}_maxhigh_after"] = df[df.t>=hi].groupby(["stock_id","date"], sort=False).high.max()
        out.append(o)
    pd.concat(out, axis=1).reset_index().to_parquet(E2/f"{ym}.parquet", index=False); return ym
if __name__=="__main__":
    with ProcessPoolExecutor(4) as ex: list(ex.map(process, sorted(glob.glob(str(ROOT/"data/raw/m1/*.parquet")))))
    ent = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(str(E2/"*.parquet")))])
    m1d = pd.concat([pd.read_parquet(f, columns=["stock_id","date","m1_close","close_auction_c"]) for f in sorted(glob.glob(str(ROOT/"data/processed/m1_daily/*.parquet")))])
    B = pd.read_parquet(ROOT/"data/processed/dataset_B_pool.parquet")[["stock_id","date","prev_close","open","gap","mkt_gap"]]
    pB = pd.read_parquet(ROOT/"data/processed/preds_B_pool.parquet").rename(columns={"pred":"predB"})[["stock_id","date","predB"]]
    d1 = pd.read_parquet(ROOT/"data/processed/d1.parquet", columns=["stock_id","date","open"]).sort_values(["stock_id","date"])
    d1["next_open"] = d1.groupby("stock_id").open.shift(-1)
    d = B.merge(pB, on=["stock_id","date"]).merge(ent, on=["stock_id","date"], how="left").merge(m1d, on=["stock_id","date"], how="left").merge(d1[["stock_id","date","next_open"]], on=["stock_id","date"], how="left")
    d = d[d.date>="2024-10-01"].copy(); d["exit"]=d.close_auction_c.fillna(d.m1_close)
    d["rankB"]=d.groupby("date").predB.rank(method="first")
    base = (d.rankB<=20)&(d.gap<=0.06)&(d.mkt_gap>=-0.01)&d.exit.notna()
    d["close_at_lim"] = d.exit >= d.prev_close*1.0975
    L=[]; P=lambda s="": L.append(str(s))
    P("# 進場延遲敏感度 — S9' (模型B 最看空20檔, 大盤跳空≥-1%, 個股跳空≤6%)\n")
    P("- 進場價 = 該視窗三分鐘 VWAP；'first' 欄 = 該視窗第一根 K 的開盤價 (單點成交, 更悲觀)。出場 13:30 收盤競價。成本 0.23%。")
    P("- 保守情境 = 收盤鎖漲停者改隔日開盤回補 + 0.2%。\n")
    rows=[]
    for n,(lo,hi) in WINS.items():
        for pc,lab in [(f"{n}_vwap","VWAP"),(f"{n}_first","first")]:
            x = d[base & d[pc].notna()].copy(); x = x[x[pc]>x.prev_close*0.905]
            x["gross"]=x[pc]/x.exit-1
            x["gc"]=np.where(x.close_at_lim & x.next_open.notna(), x[pc]/x.next_open-1-PENALTY, x.gross)
            r={"進場視窗":f"09:{lo:02d}~09:{hi-1:02d}","價格":lab,"進場價/開盤價-1 (%)":(x[pc]/x.open-1).mean()*100}
            for col,sc in [("gross","原"),("gc","保守")]:
                net=x[col]-BASE; daily=net.groupby(x.date).mean()
                r[f"{sc} 每筆淨%"]=net.mean()*100; r[f"{sc} 日淨%"]=daily.mean()*100; r[f"{sc} Sharpe"]=daily.mean()/daily.std()*np.sqrt(245)
            r["勝率"]=(x.gc-BASE>0).mean(); r["n"]=len(x); rows.append(r)
    P(pd.DataFrame(rows).set_index(["進場視窗","價格"]).round(3).to_markdown()); P("")
    # 09:00 開盤到各時點的平均變動 (池內 S9' 候選)
    x = d[base].copy()
    P("- S9' 候選股在開盤後的平均價格路徑 (相對開盤價, %):")
    P(pd.Series({f"09:{hi-1:02d} VWAP":(x[f"{n}_vwap"]/x.open-1).mean()*100 for n,(lo,hi) in WINS.items()} | {"13:30 收盤":(x.exit/x.open-1).mean()*100}).round(3).to_frame("vs open %").to_markdown())
    (OUT/"entry_delay.md").write_text("\n".join(L)); print("\n".join(L))
