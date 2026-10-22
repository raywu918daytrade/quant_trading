"""把 signals/<date>.* 上傳到 HF dataset db/signals/，並更新 db/signals/latest.json (手機頁 web/mobile.html 讀這個).

用法:
  HF_TOKEN=... python3 scripts/publish_signals.py            # 上傳 signals/ 裡最新一天
  HF_TOKEN=... python3 scripts/publish_signals.py 2026-10-22 # 指定日期
token 只從環境變數 HF_TOKEN 讀，缺少就報錯；絕不印出。
"""
import os, sys, json, glob
from datetime import datetime, timezone, timedelta
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SIG = ROOT/"signals"
REPO = "raywu918daytrade/daytrade"
PREFIX = "db/signals"
# CSV 中文欄位 → JSON 英文鍵
COLS = {"stock_id":"stock_id","name":"name","昨收":"prev_close","開盤":"open","跳空%":"gap_pct","ATR20%":"atr20_pct",
        "20日均額(億)":"dv20_yi","模型分數%":"pred_pct","池內排名":"rank","張數":"lots","金額":"amount",
        "強制回補價(昨收+8%)":"force_cover","漲停價":"limit_up"}

def build_json(date: str, sfx: str = "") -> dict:
    meta = json.load(open(SIG/f"{date}{sfx}_meta.json"))
    csv = SIG/f"{date}{sfx}.csv"
    picks = []
    if csv.exists() and csv.stat().st_size > 0:
        df = pd.read_csv(csv, dtype={"stock_id": str}, encoding="utf-8-sig")
        if len(df):
            df = df.rename(columns=COLS)[list(COLS.values())]
            for c in ["prev_close","open","force_cover","limit_up"]: df[c] = df[c].round(2)
            for c in ["gap_pct","atr20_pct","dv20_yi","pred_pct"]: df[c] = df[c].round(2)
            df["lots"] = df.lots.astype(int); df["amount"] = df.amount.round(0).astype(int); df["rank"] = df["rank"].astype(int)
            picks = df.to_dict(orient="records")
    pool_n = None
    pcsv = SIG/f"{date}{sfx}_pool.csv"
    if pcsv.exists():
        pool_n = int(len(pd.read_csv(pcsv, usecols=["stock_id"])))
    tz8 = timezone(timedelta(hours=8))
    return {"date": meta["date"], "variant": meta.get("variant", "base"), "no_trade": meta.get("no_trade", []), "mkt_gap": meta.get("mkt_gap"),
            "n_picks": len(picks), "pool_n": pool_n, "total_amount": int(sum(p["amount"] for p in picks)),
            "source": meta.get("source"), "model": meta.get("model"),
            "published_at": datetime.now(tz8).strftime("%Y-%m-%d %H:%M:%S +0800"), "picks": picks}

def main():
    token = os.environ.get("HF_TOKEN")
    if not token: sys.exit("錯誤: 環境變數 HF_TOKEN 未設定 (需要 HF 寫入 token)")
    if len(sys.argv) > 1: date = sys.argv[1]
    else:
        metas = sorted(glob.glob(str(SIG/"*_meta.json")))
        if not metas: sys.exit("signals/ 沒有任何 *_meta.json")
        date = Path(metas[-1]).name.replace("_meta.json", "")
    files = [SIG/f"{date}{s}" for s in (".csv", ".txt", "_pool.csv", "_meta.json")]
    missing = [f.name for f in files if not f.exists()]
    if missing: sys.exit(f"缺少檔案: {missing}")
    payload = build_json(date)
    js = json.dumps(payload, ensure_ascii=False, indent=1).encode("utf-8")

    from huggingface_hub import HfApi, CommitOperationAdd
    api = HfApi(token=token)
    ops = [CommitOperationAdd(path_in_repo=f"{PREFIX}/{f.name}", path_or_fileobj=str(f)) for f in files]
    ops += [CommitOperationAdd(path_in_repo=f"{PREFIX}/{date}.json", path_or_fileobj=js),
            CommitOperationAdd(path_in_repo=f"{PREFIX}/latest.json", path_or_fileobj=js)]
    msg = f"signals {date}: {payload['n_picks']} picks" + (" (no trade)" if payload["no_trade"] else "")
    # 影子模型 (seq) 名單: 有就一起上傳, 寫 <date>_seq.json 與 latest_seq.json
    for v in ("sr", "seq"):
        vf = [SIG/f"{date}_{v}{s}" for s in (".csv", ".txt", "_pool.csv", "_meta.json")]
        if all(f.exists() for f in vf):
            pj = build_json(date, f"_{v}"); sjs = json.dumps(pj, ensure_ascii=False, indent=1).encode("utf-8")
            ops += [CommitOperationAdd(path_in_repo=f"{PREFIX}/{f.name}", path_or_fileobj=str(f)) for f in vf]
            ops += [CommitOperationAdd(path_in_repo=f"{PREFIX}/{date}_{v}.json", path_or_fileobj=sjs), CommitOperationAdd(path_in_repo=f"{PREFIX}/latest_{v}.json", path_or_fileobj=sjs)]
            msg += f"; {v} {pj['n_picks']} picks"
    info = api.create_commit(repo_id=REPO, repo_type="dataset", operations=ops, commit_message=msg)
    print(f"已上傳 {len(ops)} 個檔案到 {REPO}/{PREFIX} ({msg})")
    print("commit:", getattr(info, "oid", "") or getattr(info, "commit_url", ""))

if __name__ == "__main__":
    main()
