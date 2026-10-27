"""補正某日模擬帳本: 券賣有成交、但買回單沒成交 (例: 2026-10-27 13:26 市價單在集合競價被拒 op_code 88) 的部位,
以當日最後 tick (13:30 收盤競價價) 當買回價重算, exit_reason 記為 close_est, 重算 summary 並重新上傳 HF。
用法: python3 scripts/paper_fix_unfilled.py --date 2026-10-27 --variant sr [--publish]   (上傳需 HF_TOKEN)"""
import argparse, csv, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import paper_trade as P
from huggingface_hub import hf_hub_download

def num(v):
    try: return float(v) if v not in ("", None, "None") else None
    except ValueError: return v

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--date", required=True); ap.add_argument("--variant", default=P.CFG.MAIN_VARIANT, choices=P.CFG.VARIANTS)
    ap.add_argument("--publish", action="store_true"); a = ap.parse_args()
    P.set_variant(a.variant); P.PAPER.mkdir(exist_ok=True); d = a.date
    get = lambda name: hf_hub_download(P.REPO, f"{P.HF_PREFIX}/{name}", repo_type="dataset", force_download=True)
    rows = list(csv.DictReader(open(get(f"{d}_trades.csv"), encoding="utf-8")))
    old = json.load(open(get(f"{d}_summary.json"), encoding="utf-8"))
    latest = json.load(open(get("latest.json"), encoding="utf-8"))
    if a.publish and latest.get("date") != d: sys.exit(f"latest.json 是 {latest.get('date')}, 不是 {d}: publish() 會把 latest 蓋回舊日期, 停止")
    for r in rows:
        for k, v in r.items():
            if k not in ("date", "stock_id", "name", "day_trade", "skipped", "exit_reason"): r[k] = num(v)
        r["touched_force"] = r.get("touched_force") in (True, "True", "true", 1.0)
        if r["sim_sell_px"] and not r["sim_buy_px"] and r["tick_last"]:
            r["sim_buy_px"] = r["tick_last"]; r["sim_buy_lots"] = r["sim_sell_lots"]; r["exit_reason"] = "close_est"
            ret = r["sim_sell_px"] / r["sim_buy_px"] - 1
            r["sim_ret_pct"] = round(ret*100, 3); r["sim_net_pct"] = r["nostop_net_pct"] = round((ret - P.COST_RT)*100, 3)
            P.log(f"  {r['stock_id']} {r['name']}: 券賣 {r['sim_sell_px']} → 收盤估 {r['sim_buy_px']} 淨 {r['sim_net_pct']}%")
    summ = P.summarize(rows, d, old.get("no_trade") or [], (old.get("mode") or "simulation") + " · 補正: 未成交買回以收盤價估")
    summ["finished_at"] = old.get("finished_at", summ["finished_at"]); summ["corrected_at"] = P.now().strftime("%Y-%m-%d %H:%M:%S")
    P.log(f"舊: 淨損益 {old.get('sim_pnl_net')} → 新: {summ['sim_pnl_net']} (平均淨 {summ['sim_avg_net_pct']}%, 勝率 {summ['sim_win_rate']})")
    with open(P.PAPER/f"{d}_trades.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    json.dump(summ, open(P.PAPER/f"{d}_summary.json", "w"), ensure_ascii=False, indent=1)
    if a.publish: P.publish(d, rows, summ)

if __name__ == "__main__":
    main()
