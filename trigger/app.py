"""手機頁的「補跑某天」代理: 收到 {date, pin} → 用 GH_TOKEN 打 GitHub workflow_dispatch (daily_signals.yml, inputs.date).
環境變數: GH_TOKEN (GitHub fine-grained token, 只需 quant_trading repo 的 Actions: Read and write), API_PIN (網頁輸入的 PIN).
部署: Render Web Service (render.yaml), 免費方案; 閒置會休眠, 第一次喚醒約 30~50 秒.
"""
import os, re, time, requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

OWNER, REPO = "raywu918daytrade", "quant_trading"
WORKFLOWS = {"signals": "daily_signals.yml", "paper": "paper_trade.yml"}
app = FastAPI(title="quant trigger")
app.add_middleware(CORSMiddleware, allow_origins=["https://quant.just1stock.com", "https://raywu918daytrade.github.io", "http://localhost:8790"], allow_methods=["GET", "POST"], allow_headers=["*"])
_last = {}   # 簡單節流: 同一 workflow 60 秒內只觸發一次

class Req(BaseModel):
    pin: str
    date: str = ""            # YYYY-MM-DD, 空 = 今天即時
    workflow: str = "signals"

def gh():
    tok = os.environ.get("GH_TOKEN")
    if not tok: raise HTTPException(500, "GH_TOKEN not set")
    return {"Authorization": f"Bearer {tok}", "Accept": "application/vnd.github+json"}

@app.get("/")
def health(): return {"ok": True, "repo": f"{OWNER}/{REPO}"}

@app.post("/replay")
def replay(r: Req):
    if not os.environ.get("API_PIN") or r.pin != os.environ["API_PIN"]: raise HTTPException(401, "PIN 錯誤")
    if r.date and not re.fullmatch(r"\d{4}-\d{2}-\d{2}", r.date): raise HTTPException(400, "date 格式要 YYYY-MM-DD")
    wf = WORKFLOWS.get(r.workflow)
    if not wf: raise HTTPException(400, "workflow 只能是 signals / paper")
    if time.time() - _last.get(wf, 0) < 60: raise HTTPException(429, "60 秒內已觸發過，請稍候")
    resp = requests.post(f"https://api.github.com/repos/{OWNER}/{REPO}/actions/workflows/{wf}/dispatches", headers=gh(), json={"ref": "main", "inputs": {"date": r.date}}, timeout=20)
    if resp.status_code != 204: raise HTTPException(502, f"GitHub {resp.status_code}: {resp.text[:200]}")
    _last[wf] = time.time()
    return {"ok": True, "workflow": wf, "date": r.date or "today", "runs_url": f"https://github.com/{OWNER}/{REPO}/actions/workflows/{wf}"}

@app.get("/runs")
def runs(workflow: str = "signals", n: int = 5):
    wf = WORKFLOWS.get(workflow, "daily_signals.yml")
    resp = requests.get(f"https://api.github.com/repos/{OWNER}/{REPO}/actions/workflows/{wf}/runs", headers=gh(), params={"per_page": n}, timeout=20)
    if resp.status_code != 200: raise HTTPException(502, f"GitHub {resp.status_code}")
    return [{"created_at": x["created_at"], "status": x["status"], "conclusion": x["conclusion"], "event": x["event"], "url": x["html_url"]} for x in resp.json().get("workflow_runs", [])]
