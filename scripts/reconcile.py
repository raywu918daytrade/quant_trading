"""對帳與上線門檻: 把每一天的「名單 → D1 近似結算 → 永豐模擬成交 → 真實 tick」擺在一起逐日、逐筆比對, 並依 GATE 判斷是否達到真錢上線條件.

資料全部從 HF 讀 (不需本機 data/); 有 data/processed/d1.parquet 時用它當交易日曆 (偵測「交易日卻沒有名單」):
  db/signals/<date>.json              名單 (source "TWSE MIS" = 09:01 即時產生; "歷史 D1 open" = 事後回放; model 決定樣本外起點)
  db/signals/history*.csv             D1 近似結算 (record_results.py; 與回測同一套假設: 開盤 +0.03% 進, 觸 +8% 強制價出, 否則收盤)
  db/paper/summary.csv, <date>_trades.csv   永豐模擬成交 (sim) 與真實 tick (09:01~09:03 VWAP 進、依策略規則出) — paper_trade.py
三種日子 (kind):
  即時          名單 09:01 盤中產生 → 唯一算得上實盤證據的樣本, 門檻只看這些
  回放·樣本外   事後回放, 但日期在模型訓練期之後
  回放·訓練期   事後回放且在模型訓練期內 (例: 2026Q4 模型的 9 月) → in-sample, 只供參考
門檻 (GATE, 全部通過才建議小部位真錢):
  G1 樣本量     即時且有交易的日子 ≥ 40
  G2 操作       最近 20 個即時交易日零操作故障 (無名單/名單太晚/模擬未進場/部分成交/買回未成交/現沖資格未過濾/模擬缺紀錄)
  G3 方向       tick 累計淨報酬 > 0
  G4 尾部       tick 最差日 ≥ −8%、MDD ≥ −18% (回測 −5.9% / −15%, 留 1.2~1.4 倍餘裕)
  G5 進場假設   tick VWAP 相對名單開盤價的平均滑價 ≤ +0.20% (回測假設 −0.03%, 即成交略高於開盤)
  G6 結算一致   出場價 |tick − D1| 每筆中位數 ≤ 0.10% 且出場方式 (收盤/強制) 不一致的比例 ≤ 10% (D1 近似結算的出場模型要和分鐘資料一致, 回測才可信)
  (進場端的差異 = 開盤價 vs 09:01~09:03 VWAP, 由 G5 看平均滑價; 每筆差的離散是高 ATR 股一分鐘內的正常波動, 只列不計)
  模擬成交 (sim) 只列不計: 永豐模擬撮合與真實 tick 差距可達 ±3%, 不可靠.
輸出 paper/reconcile_daily.csv, reconcile_trades.csv, reconcile.json, reconcile.md; --publish 上傳 HF db/paper/ (手機頁「模擬」分頁讀 reconcile.json).
用法: python3 scripts/reconcile.py [--variant base] [--publish] [--days 60]
"""
import os, sys, io, re, csv, json, argparse, datetime as dt
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np, pandas as pd, requests
sys.path.insert(0, str(Path(__file__).resolve().parent)); import config as CFG
ROOT = Path(__file__).resolve().parent.parent; REPO = "raywu918daytrade/daytrade"
RES = f"https://huggingface.co/datasets/{REPO}/resolve/main/"
CAP, COST, K, SLIP = 3_000_000, 0.0023, 0.08, 0.001                       # 與 record_results.py 相同
REF = {"daily_pct": 0.63, "per_trade_pct": 0.51, "win_rate": 0.57, "worst_day_pct": -5.9, "mdd_pct": -15.0, "entry_adj_pct": -0.03}   # README 300 萬配置保守回測
GATE = {"min_live_days": 40, "ops_window": 20, "worst_day_pct": -8.0, "mdd_pct": -18.0, "entry_slip_pct": 0.20, "exit_diff_pct": 0.10, "reason_mismatch": 0.10}
LATE_LIST = "09:02:30"   # 即時名單晚於此 = 備援才算出來, 記一次操作事件
KIND_LIVE, KIND_OOS, KIND_IS, KIND_NONE = "即時", "回放·樣本外", "回放·訓練期", "無名單"

def get(path, timeout=30):
    return requests.get(RES + path, params={"t": int(dt.datetime.now().timestamp())}, timeout=timeout)
def read_csv(path):
    r = get(path)
    return pd.read_csv(io.StringIO(r.text), dtype={"stock_id": str}) if r.ok and r.text.strip() else pd.DataFrame()
def list_signal_dates(sfx):
    tree = requests.get(f"https://huggingface.co/api/datasets/{REPO}/tree/main/db/signals", timeout=60).json()
    pat = re.compile(rf"^(\d{{4}}-\d{{2}}-\d{{2}}){re.escape(sfx)}\.json$")
    return sorted(m.group(1) for x in tree if (m := pat.match(x["path"].split("/")[-1])))
def fetch_many(paths, parse):
    def one(p):
        try:
            r = get(p); return p, (parse(r) if r.ok else None)
        except Exception: return p, None
    with ThreadPoolExecutor(8) as ex: return dict(ex.map(one, paths))
def model_oos_start(model):
    m = re.search(r"(\d{4})Q([1-4])", model or "")
    return pd.Period(f"{m.group(1)}Q{m.group(2)}", freq="Q").start_time if m else None
def trading_calendar(first, last):
    """D1 有的日期 = 交易日 (只到 D1 最後一天). 沒有 d1.parquet 回 None (不做漏名單檢查)."""
    p = ROOT/"data/processed/d1.parquet"
    if not p.exists(): return None
    d = pd.to_datetime(pd.read_parquet(p, columns=["date"]).date.drop_duplicates())
    return sorted(x.strftime("%Y-%m-%d") for x in d if first <= x.strftime("%Y-%m-%d") <= last)
def f(x, d=2, sign=False):
    if x is None or (isinstance(x, float) and not np.isfinite(x)): return "–"
    return f"{x:+.{d}f}" if sign else f"{x:.{d}f}"
def md_table(rows, cols):
    if not rows: return "(無)\n"
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in rows: out.append("| " + " | ".join("" if r.get(c) is None else str(r.get(c)) for c in cols) + " |")
    return "\n".join(out) + "\n"

def series_stats(s):
    """逐日報酬 (% 對 300 萬) → 累計/平均/勝率/最差/MDD (加總制, 當沖每日獨立)/t 值."""
    s = pd.Series(s, dtype=float).dropna()
    if not len(s): return {"n": 0}
    cum = s.cumsum(); mdd = float((cum - cum.cummax()).min())
    return {"n": int(len(s)), "cum": float(s.sum()), "mean": float(s.mean()), "std": float(s.std(ddof=1)) if len(s) > 1 else None,
            "win": float((s > 0).mean()), "worst": float(s.min()), "best": float(s.max()), "mdd": mdd,
            "t": float(s.mean() / s.std(ddof=1) * np.sqrt(len(s))) if len(s) > 2 and s.std(ddof=1) > 0 else None}

def build(variant):
    sfx = CFG.suffix(variant); pp = "db/paper" + sfx
    dates = list_signal_dates(sfx)
    if not dates: sys.exit("HF 上沒有任何名單")
    sigs = {p.split("/")[-1][:10]: j for p, j in fetch_many([f"db/signals/{d}{sfx}.json" for d in dates], lambda r: r.json()).items()}
    hist = read_csv(f"db/signals/history{sfx}.csv"); daily = read_csv(f"db/signals/history_daily{sfx}.csv")
    summ = read_csv(f"{pp}/summary.csv")
    summ_by = {r["date"]: r for r in summ.to_dict("records")} if len(summ) else {}
    trades = {p.split("/")[-1][:10]: df for p, df in fetch_many([f"{pp}/{d}_trades.csv" for d in summ_by], lambda r: pd.read_csv(io.StringIO(r.text), dtype={"stock_id": str})).items() if df is not None and "stock_id" in df.columns}
    d1_day = {r["date"]: r for r in daily.to_dict("records")} if len(daily) else {}
    d1_tr = {(r["date"], r["stock_id"]): r for r in hist.to_dict("records")} if len(hist) else {}
    today = dt.date.today().isoformat()
    first_live = min((d for d, j in sigs.items() if j and j.get("source") == "TWSE MIS"), default=None)
    first_paper = min(summ_by) if summ_by else None
    cal = trading_calendar(min(dates), today)
    all_dates = sorted(set(dates) | set(summ_by) | {d for d in (cal or []) if first_live and d >= first_live})

    drows, trows = [], []
    for d in all_dates:
        j = sigs.get(d); ops, notes = [], []
        row = {"date": d, "kind": KIND_NONE, "source": "", "published_at": "", "model": "", "no_trade": "", "n_picks": 0, "amount": 0}
        if j:
            live = j.get("source") == "TWSE MIS"; oos0 = model_oos_start(j.get("model"))
            row.update(kind=KIND_LIVE if live else (KIND_OOS if (oos0 is not None and pd.Timestamp(d) >= oos0) else KIND_IS),
                       source=j.get("source", ""), published_at=(j.get("published_at") or "")[11:19], model=j.get("model", ""),
                       no_trade="|".join(j.get("no_trade") or []), n_picks=int(j.get("n_picks") or 0), amount=int(j.get("total_amount") or 0))
            if live and row["published_at"] > LATE_LIST: ops.append(f"名單 {row['published_at']} 晚於 {LATE_LIST} (備援)")
            if "取得失敗" in (j.get("daytrade_filter") or ""): ops.append("現沖資格未過濾")
        elif cal and d in cal and first_live and d >= first_live and d < today: ops.append("交易日無名單 (故障或假日未排除)")
        trade_day = bool(j) and not row["no_trade"] and row["n_picks"] > 0
        h = d1_day.get(d)
        row.update(d1_ret_pct=float(h["ret_pct"]) if h and trade_day else None, d1_pnl=int(h["pnl"]) if h and trade_day else None,
                   d1_n=int(h["n"]) if h else None, d1_wins=int(h["wins"]) if h else None)
        s = summ_by.get(d)
        row.update(sim_ret_pct=None, sim_pnl=None, n_filled=None, n_skipped=None, n_forced=None, n_unclosed=None)
        if s is not None:
            def num(k, default=0):
                v = s.get(k); return default if v is None or v == "" or (isinstance(v, float) and np.isnan(v)) else float(v)
            n_filled, n_skipped, n_unclosed = int(num("n_filled")), int(num("n_skipped")), int(num("n_unclosed"))
            corrected = "補正" in str(s.get("mode", ""))
            if trade_day and (n_filled or corrected or n_unclosed or num("sim_pnl_net") != 0):
                pnl = num("sim_pnl_net") + num("unclosed_est_pnl_net")
                row.update(sim_ret_pct=pnl / CAP * 100, sim_pnl=int(round(pnl)))
            row.update(n_filled=n_filled, n_skipped=n_skipped, n_forced=int(num("n_forced")), n_unclosed=n_unclosed)
            if trade_day:
                if n_filled == 0: ops.append("模擬未進場")
                elif n_filled < row["n_picks"] - n_skipped: ops.append(f"部分成交 {n_filled}/{row['n_picks']}")
                if n_unclosed or corrected: ops.append("買回未成交 (補正估值)")
                if n_skipped: notes.append(f"略過 {n_skipped} 檔 (不可現沖)")
        elif trade_day and first_paper and d >= first_paper and d < today: ops.append("模擬缺紀錄")
        # 真實 tick, 依策略規則出場 (與 record_results 同: 觸 +8% → 強制價 +0.3%; 否則最後成交價)
        tdf = trades.get(d); tick_pnl = 0.0; tick_amt = 0.0; slips = []
        if tdf is not None and trade_day:
            for r in tdf.to_dict("records"):
                vw, last, hi = r.get("tick_vwap_0901_0903"), r.get("tick_last"), r.get("tick_high")
                opn, lots, pc, fc = r.get("open_signal"), r.get("lots"), r.get("prev_close"), r.get("force_cover")
                ok = all(x is not None and not (isinstance(x, float) and np.isnan(x)) for x in (vw, last, opn, lots, pc, fc)) and vw and last and opn
                dr = d1_tr.get((d, r["stock_id"]))
                tr = {"date": d, "kind": row["kind"], "stock_id": r["stock_id"], "name": r.get("name", ""), "lots": int(float(lots)) if lots == lots else None, "open_signal": opn,
                      "d1_entry": dr and dr.get("entry"), "d1_exit": dr and dr.get("exit"), "d1_reason": dr and dr.get("exit_reason"), "d1_net_pct": float(dr["net_pct"]) if dr else None,
                      "tick_vwap": vw if ok else None, "tick_exit": None, "tick_reason": None, "tick_net_pct": None,
                      "sim_sell": r.get("sim_sell_px"), "sim_buy": r.get("sim_buy_px"), "sim_reason": r.get("exit_reason"), "sim_net_pct": r.get("sim_net_pct"),
                      "entry_slip_tick_pct": None, "entry_slip_sim_pct": None, "diff_tick_d1_pct": None, "diff_sim_tick_pct": None, "diff_exit_pct": None, "reason_match": None}
                if ok:
                    hi = hi if hi == hi else last
                    if hi >= fc: ex, why = fc * (1 + SLIP * 3), "forced"
                    elif last >= pc * 1.0975: ex, why = last, "lock?"       # 收盤在漲停附近: 真實情況是隔日才回補, 這裡先用最後價, 另標記
                    else: ex, why = last, "close"
                    net = vw / ex - 1 - COST; amt = float(opn) * 1000 * float(lots)
                    tick_pnl += amt * net; tick_amt += amt
                    tr.update(tick_exit=round(ex, 2), tick_reason=why, tick_net_pct=round(net * 100, 3), entry_slip_tick_pct=round((opn - vw) / opn * 100, 3))
                    slips.append(tr["entry_slip_tick_pct"])
                    if tr["d1_net_pct"] is not None: tr["diff_tick_d1_pct"] = round(tr["tick_net_pct"] - tr["d1_net_pct"], 3)
                    if dr and dr.get("exit"):   # 出場模型一致性: 收盤價 vs 最後 tick、強制回補是否同樣觸發
                        tr["diff_exit_pct"] = round((ex / float(dr["exit"]) - 1) * 100, 3)
                        tr["reason_match"] = why.rstrip("?") == str(dr.get("exit_reason", "")).replace("lock_pending", "lock")
                    if why == "lock?": notes.append(f"{r['stock_id']} 收在漲停附近")
                sp, sn = r.get("sim_sell_px"), r.get("sim_net_pct")
                if sp == sp and sp and opn: tr["entry_slip_sim_pct"] = round((opn - sp) / opn * 100, 3)
                if sn == sn and sn is not None and tr["tick_net_pct"] is not None: tr["diff_sim_tick_pct"] = round(float(sn) - tr["tick_net_pct"], 3)
                for k in ("sim_sell", "sim_buy", "sim_net_pct", "open_signal"):
                    v = tr[k]; tr[k] = None if (v is None or (isinstance(v, float) and np.isnan(v))) else v
                trows.append(tr)
        row.update(tick_ret_pct=(tick_pnl / CAP * 100) if tick_amt else None, tick_pnl=int(round(tick_pnl)) if tick_amt else None,
                   entry_slip_tick_pct=float(np.mean(slips)) if slips else None,
                   ops="; ".join(ops), notes="; ".join(notes), n_ops=len(ops))
        drows.append(row)
    D = pd.DataFrame(drows); T = pd.DataFrame(trows)

    # ---- 門檻: 只看「即時」且應交易的日子 ----
    L = D[(D.kind == KIND_LIVE) & (D.no_trade == "") & (D.n_picks > 0)]
    L_all = D[D.kind == KIND_LIVE]
    tick, d1, sim = series_stats(L.tick_ret_pct), series_stats(L.d1_ret_pct), series_stats(L.sim_ret_pct)
    recent = L_all[L_all.date < today].tail(GATE["ops_window"]); ops_recent = int(recent.n_ops.sum()); ops_total = int(L_all.n_ops.sum())
    TL = T[T.kind == KIND_LIVE] if len(T) else T
    slip_mean = float(TL.entry_slip_tick_pct.dropna().mean()) if len(TL) and TL.entry_slip_tick_pct.notna().any() else None
    slip_sim = float(TL.entry_slip_sim_pct.dropna().mean()) if len(TL) and TL.entry_slip_sim_pct.notna().any() else None
    settle_diff = float(TL.diff_tick_d1_pct.dropna().abs().median()) if len(TL) and TL.diff_tick_d1_pct.notna().any() else None
    settle_bias = float(TL.diff_tick_d1_pct.dropna().mean()) if len(TL) and TL.diff_tick_d1_pct.notna().any() else None
    sim_diff = float(TL.diff_sim_tick_pct.dropna().abs().median()) if len(TL) and TL.diff_sim_tick_pct.notna().any() else None
    exit_diff = float(TL.diff_exit_pct.dropna().abs().median()) if len(TL) and TL.diff_exit_pct.notna().any() else None
    mism = float((~TL.reason_match.dropna().astype(bool)).mean()) if len(TL) and TL.reason_match.notna().any() else None
    live_days = int(len(L))
    gate = [
        {"id": "G1", "name": "樣本量", "rule": f"即時且有交易的日子 ≥ {GATE['min_live_days']}", "actual": f"{live_days} 天", "pass": live_days >= GATE["min_live_days"]},
        {"id": "G2", "name": "操作", "rule": f"最近 {GATE['ops_window']} 個即時日零操作故障", "actual": f"{ops_recent} 件 (全部 {ops_total})", "pass": len(recent) > 0 and ops_recent == 0},
        {"id": "G3", "name": "方向", "rule": "tick 累計淨報酬 > 0", "actual": f"{f(tick.get('cum'), 2, True)}% ({tick['n']} 天)", "pass": tick["n"] > 0 and tick["cum"] > 0},
        {"id": "G4", "name": "尾部", "rule": f"最差日 ≥ {GATE['worst_day_pct']}%、MDD ≥ {GATE['mdd_pct']}%", "actual": f"最差 {f(tick.get('worst'), 2, True)}%, MDD {f(tick.get('mdd'), 2)}%",
         "pass": tick["n"] > 0 and tick["worst"] >= GATE["worst_day_pct"] and tick["mdd"] >= GATE["mdd_pct"]},
        {"id": "G5", "name": "進場假設", "rule": f"tick VWAP 相對名單開盤價平均滑價 ≤ +{GATE['entry_slip_pct']}%", "actual": f"{f(slip_mean, 3, True)}% (回測假設 {REF['entry_adj_pct']:+.2f}%)", "pass": slip_mean is not None and slip_mean <= GATE["entry_slip_pct"]},
        {"id": "G6", "name": "結算一致", "rule": f"出場價 |tick − D1| 每筆中位數 ≤ {GATE['exit_diff_pct']}%, 出場方式不一致 ≤ {GATE['reason_mismatch']*100:.0f}%", "actual": f"{f(exit_diff, 3)}%, 不一致 {f(None if mism is None else mism*100, 0)}%",
         "pass": exit_diff is not None and exit_diff <= GATE["exit_diff_pct"] and mism is not None and mism <= GATE["reason_mismatch"]},
    ]
    n_pass = sum(g["pass"] for g in gate)
    by_kind = {}
    for k in (KIND_LIVE, KIND_OOS, KIND_IS):
        X = D[(D.kind == k)]; Xt = X[(X.no_trade == "") & (X.n_picks > 0)]
        by_kind[k] = {"days": int(len(X)), "trade_days": int(len(Xt)), "no_trade_days": int((X.no_trade != "").sum()),
                      "d1": series_stats(Xt.d1_ret_pct), "tick": series_stats(Xt.tick_ret_pct), "sim": series_stats(Xt.sim_ret_pct),
                      "first": X.date.min() if len(X) else None, "last": X.date.max() if len(X) else None}
    out = {"as_of": dt.datetime.now().strftime("%Y-%m-%d %H:%M"), "variant": variant, "cap": CAP, "ref": REF, "gate_cfg": GATE,
           "gate": {"pass": n_pass == len(gate), "n_pass": n_pass, "n_total": len(gate), "items": gate,
                    "verdict": "六項全部通過: 可進入小部位真錢 (每檔 1 張) 再觀察" if n_pass == len(gate) else f"{n_pass}/{len(gate)} 通過, 繼續模擬"},
           "live": {"first_date": first_live, "days": live_days, "no_trade_days": int((L_all.no_trade != "").sum()), "tick": tick, "d1": d1, "sim": sim,
                    "entry_slip_tick_mean": slip_mean, "entry_slip_sim_mean": slip_sim, "settle_diff_median": settle_diff, "settle_bias_mean": settle_bias,
                    "sim_tick_diff_median": sim_diff, "exit_diff_median": exit_diff, "reason_mismatch": mism, "ops_recent": ops_recent, "ops_total": ops_total, "trades": int(len(TL))},
           "by_kind": by_kind, "calendar": "D1" if cal else None,
           "ops_log": [{"date": r.date, "kind": r.kind, "events": r.ops} for r in D.itertuples() if r.ops],
           "daily": [{k: (None if (isinstance(v, float) and np.isnan(v)) else v) for k, v in r.items()} for r in D.to_dict("records")]}
    return D, T, out, sfx, pp

def write_md(D, T, out, path):
    L = []; P = L.append
    g = out["gate"]; lv = out["live"]
    P(f"# 對帳與上線門檻 ({out['variant']}) — 更新 {out['as_of']}\n")
    P(f"**判定: {'✅' if g['pass'] else '⏳'} {g['verdict']}**  (即時交易日 {lv['days']}, 首個即時日 {lv['first_date']})\n")
    P("## 1. 上線門檻 (只用「即時」日子: 名單 09:01 盤中產生、模型訓練期之後)")
    P(md_table([{"": "✅" if x["pass"] else "❌", "項目": f"{x['id']} {x['name']}", "條件": x["rule"], "目前": x["actual"]} for x in g["items"]], ["", "項目", "條件", "目前"]))
    P("## 2. 即時日子的三套數字 (對 300 萬, 每日加總)")
    rows = []
    for nm, s in (("D1 近似結算 (回測假設)", lv["d1"]), ("真實 tick (VWAP 進, 依規則出)", lv["tick"]), ("永豐模擬成交 (僅參考)", lv["sim"])):
        rows.append({"來源": nm, "天": s.get("n", 0), "累計%": f(s.get("cum"), 2, True), "日均%": f(s.get("mean"), 3, True), "日勝率": f((s.get("win") or 0) * 100, 0) + "%" if s.get("n") else "–",
                     "最差日%": f(s.get("worst"), 2, True), "MDD%": f(s.get("mdd"), 2), "t值": f(s.get("t"), 2)})
    rows.append({"來源": "回測參考 (README 300 萬保守)", "天": "", "累計%": "", "日均%": f"+{REF['daily_pct']:.2f}", "日勝率": f"{REF['win_rate']*100:.0f}%", "最差日%": f"{REF['worst_day_pct']:.1f}", "MDD%": f"{REF['mdd_pct']:.0f}", "t值": ""})
    P(md_table(rows, ["來源", "天", "累計%", "日均%", "日勝率", "最差日%", "MDD%", "t值"]))
    P(f"- 進場滑價 (名單開盤價 − 成交價, 正 = 賣得比訊號低 = 不利): tick VWAP 平均 {f(lv['entry_slip_tick_mean'], 3, True)}%, 模擬成交平均 {f(lv['entry_slip_sim_mean'], 3, True)}%; 回測假設 {REF['entry_adj_pct']:+.2f}%")
    P(f"- 每筆淨報酬 tick − D1: 平均 {f(lv['settle_bias_mean'], 3, True)}% (= 進場滑價造成的偏差), 中位絕對差 {f(lv['settle_diff_median'], 3)}% (高 ATR 股一分鐘內的正常波動); 出場價 |tick − D1| 中位 {f(lv['exit_diff_median'], 3)}%, 出場方式不一致 {f(None if lv['reason_mismatch'] is None else lv['reason_mismatch']*100, 0)}%")
    P(f"- 永豐模擬成交 vs tick: 每筆 |模擬 − tick| 中位數 {f(lv['sim_tick_diff_median'], 3)}% ({lv['trades']} 筆) — 模擬撮合不可靠, 只列不計")
    P(f"- 操作事件: 最近 {GATE['ops_window']} 個即時日 {lv['ops_recent']} 件, 全部 {lv['ops_total']} 件\n")
    P("## 3. 依日子類型 (D1 近似結算)")
    rows = []
    for k, s in out["by_kind"].items():
        d = s["d1"]; t = s["tick"]
        rows.append({"類型": k, "期間": f"{s['first']} ~ {s['last']}" if s["first"] else "–", "天 (交易/不交易)": f"{s['trade_days']} / {s['no_trade_days']}", "D1 累計%": f(d.get("cum"), 2, True), "D1 日均%": f(d.get("mean"), 3, True),
                     "D1 日勝率": f((d.get("win") or 0) * 100, 0) + "%" if d.get("n") else "–", "D1 最差日%": f(d.get("worst"), 2, True), "tick 累計%": f(t.get("cum"), 2, True) if t.get("n") else "–"})
    P(md_table(rows, ["類型", "期間", "天 (交易/不交易)", "D1 累計%", "D1 日均%", "D1 日勝率", "D1 最差日%", "tick 累計%"]))
    P("「回放·訓練期」是模型訓練資料內的日子 (in-sample), 只能當 sanity check, 不是績效證據。\n")
    if out["ops_log"]:
        P("## 4. 操作事件"); P(md_table(out["ops_log"], ["date", "kind", "events"]))
    P(f"## 5. 逐日 (最近 {min(len(D), 60)} 天)")
    cols = ["date", "kind", "published_at", "no_trade", "n_picks", "d1_ret_pct", "tick_ret_pct", "sim_ret_pct", "entry_slip_tick_pct", "n_forced", "ops", "notes"]
    rows = []
    for r in D.tail(60).to_dict("records"):
        def cell(c):
            v = r[c]
            if v is None or (isinstance(v, float) and np.isnan(v)): return ""
            if c in ("d1_ret_pct", "tick_ret_pct", "sim_ret_pct"): return f(v, 2, True)
            if c == "entry_slip_tick_pct": return f(v, 3, True)
            if c in ("n_picks", "n_forced"): return int(v)
            return v
        rows.append({c: cell(c) for c in cols})
    P(md_table(rows, cols))
    P("\n欄位: d1 = 開盤 +0.03% 進、觸 +8% 強制價出、否則收盤 (record_results); tick = 09:01~09:03 VWAP 進、同樣規則出 (paper_trade 真實 tick); sim = 永豐模擬成交. 全部扣 0.23%, 對 300 萬.")
    Path(path).write_text("\n".join(L), encoding="utf-8")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default=CFG.MAIN_VARIANT, choices=CFG.VARIANTS); ap.add_argument("--publish", action="store_true"); ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()
    D, T, out, sfx, pp = build(a.variant)
    PAPER = ROOT / ("paper" + sfx); PAPER.mkdir(exist_ok=True)
    D.to_csv(PAPER/"reconcile_daily.csv", index=False, encoding="utf-8"); T.to_csv(PAPER/"reconcile_trades.csv", index=False, encoding="utf-8")
    json.dump(out, open(PAPER/"reconcile.json", "w"), ensure_ascii=False, indent=1, default=str)
    write_md(D, T, out, PAPER/"reconcile.md")
    if not a.quiet: print((PAPER/"reconcile.md").read_text(encoding="utf-8"))
    print(f"\n已寫入 {PAPER.name}/reconcile_daily.csv, reconcile_trades.csv, reconcile.json, reconcile.md")
    if a.publish:
        token = os.environ.get("HF_TOKEN")
        if not token: sys.exit("缺 HF_TOKEN")
        from huggingface_hub import HfApi, CommitOperationAdd
        ops = [CommitOperationAdd(f"{pp}/{n}", str(PAPER/n)) for n in ("reconcile.json", "reconcile.md", "reconcile_daily.csv", "reconcile_trades.csv")]
        HfApi(token=token).create_commit(repo_id=REPO, repo_type="dataset", operations=ops, commit_message=f"reconcile{sfx} {out['as_of'][:10]}: gate {out['gate']['n_pass']}/{out['gate']['n_total']}, live {out['live']['days']} days")
        print(f"已上傳 {pp}/reconcile.*")

if __name__ == "__main__":
    main()
