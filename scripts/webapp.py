"""本機網頁: 每日候選名單操作台.  python3 scripts/webapp.py  → http://127.0.0.1:8787"""
import json, glob, subprocess, sys, threading, uuid, time
from pathlib import Path
import pandas as pd
from flask import Flask, jsonify, request, send_from_directory

ROOT = Path(__file__).resolve().parent.parent
SIG = ROOT/"signals"; WEB = ROOT/"web"
app = Flask(__name__, static_folder=str(WEB), static_url_path="/static")
JOBS = {}

def run_job(job_id, args):
    JOBS[job_id] = {"status":"running", "log":"", "started":time.time()}
    p = subprocess.Popen([sys.executable, "-u", str(ROOT/"scripts/daily_candidates.py"), *args], cwd=str(ROOT),
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in p.stdout:
        if "Warning" in line: continue
        JOBS[job_id]["log"] += line
    p.wait(); JOBS[job_id]["status"] = "done" if p.returncode==0 else "error"

@app.get("/")
def index(): return send_from_directory(str(WEB), "index.html")

@app.get("/api/status")
def status():
    d1 = pd.read_parquet(ROOT/"data/processed/d1.parquet", columns=["date"])
    models = sorted(glob.glob(str(ROOT/"models/lgbm_B_pool_*.txt")))
    dates = sorted({Path(f).stem for f in glob.glob(str(SIG/"????-??-??.csv"))}, reverse=True)
    today = pd.Timestamp.today().normalize()
    return jsonify({"d1_last": str(d1.date.max().date()), "model": Path(models[-1]).name if models else None,
                    "today": str(today.date()), "dow": int(today.dayofweek), "dates": dates})

@app.get("/api/signal/<date>")
def signal(date):
    f = SIG/f"{date}.csv"
    if not f.exists(): return jsonify({"error":"no signal"}), 404
    picks = pd.read_csv(f, dtype={"stock_id":str}) if f.stat().st_size>5 else pd.DataFrame()
    pool = pd.read_csv(SIG/f"{date}_pool.csv", dtype={"stock_id":str}).fillna("")
    meta = json.load(open(SIG/f"{date}_meta.json")) if (SIG/f"{date}_meta.json").exists() else {}
    txt = (SIG/f"{date}.txt").read_text() if (SIG/f"{date}.txt").exists() else ""
    return jsonify({"picks": picks.fillna("").to_dict("records"), "pool": pool.to_dict("records"), "meta": meta, "txt": txt})

@app.get("/api/auto")
def auto():
    import subprocess as sp, os, time as _t
    try: lst = sp.run(["launchctl","list"], capture_output=True, text=True).stdout
    except Exception: lst = ""
    jobs = {}
    for n, label in [("com.quant.refresh-d1","08:30 更新 D1"),("com.quant.gen-signals","09:01 產生名單")]:
        lg = ROOT/"logs"/f"{n}.log"
        tail = lg.read_text().strip().splitlines()[-4:] if lg.exists() else []
        jobs[n] = {"label": label, "installed": n in lst, "last_log_time": _t.strftime("%Y-%m-%d %H:%M", _t.localtime(lg.stat().st_mtime)) if lg.exists() else None, "tail": tail}
    return jsonify(jobs)

@app.post("/api/run")
def run():
    body = request.get_json(force=True) or {}
    args = []
    if body.get("refresh"): args.append("--refresh")
    if body.get("date"): args += ["--date", body["date"]]
    for k in ["capital","max_per_stock","max_price","max_n","min_amount"]:
        if body.get(k) not in (None, ""): args += [f"--{k.replace('_','-')}", str(body[k])]
    job_id = uuid.uuid4().hex[:8]
    threading.Thread(target=run_job, args=(job_id, args), daemon=True).start()
    return jsonify({"job": job_id})

@app.get("/api/job/<job_id>")
def job(job_id): return jsonify(JOBS.get(job_id, {"status":"unknown","log":""}))

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8787, debug=False)
