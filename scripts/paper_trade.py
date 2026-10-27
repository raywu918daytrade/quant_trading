"""永豐 Shioaji 模擬環境自動下單 (paper trading)，照定案策略 S9'' 執行當日名單並統計。

流程 (台灣時間):
  08:55~09:10  等 HF db/signals/latest.json 的 date == 今天 (或 --signals 指定本機 csv)
  登入模擬環境 (simulation=True, 永遠不會真的下單), 檢查每檔 contract.day_trade 是否可現沖券賣
  09:01:00     每檔市價「現股當沖先賣」(daytrade_short=True), 同時訂閱 tick
  盤中         tick.close >= 強制回補價 (昨收+8%) → 立即市價買回, 標記 forced
  盤中         (選用, 預設關) 組合未實現損益 <= -PORTFOLIO_STOP → 全部市價買回 (portfolio_stop); 13:24 現價 >= 昨收+6% 先買回 (pre_auction)
  13:26        剩餘部位以漲停價限價買回 (13:25 起集合競價不收市價單; 參與 13:30 收盤競價)
  13:35        結算: 模擬成交價 vs 真實 tick 算的 09:01~09:03 VWAP / 收盤價, 寫 paper/<date>_trades.csv, _summary.json
               --publish 時上傳 HF db/paper/ 並更新 db/paper/latest.json + summary.csv (累計)

憑證: 環境變數 SINOPAC_API_KEY / SINOPAC_SECRET_KEY (或 --env-file .env), 絕不印出。模擬環境不需要憑證 CA。
用法:
  python3 scripts/paper_trade.py --env-file ~/.../.env              # 盤中正式跑 (會等到 09:01)
  python3 scripts/paper_trade.py --env-file ~/.../.env --dry        # 非盤中測試: 登入、讀名單、查合約/快照, 不下單
  python3 scripts/paper_trade.py --env-file ~/.../.env --publish    # 跑完上傳 HF (需 HF_TOKEN)
"""
import argparse, os, sys, json, time, threading, csv, warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from pathlib import Path
from collections import defaultdict
import requests

ROOT = Path(__file__).resolve().parent.parent
TZ = ZoneInfo("Asia/Taipei")
REPO = "raywu918daytrade/daytrade"
sys.path.insert(0, str(ROOT/"scripts")); import config as CFG
VARIANT = CFG.MAIN_VARIANT            # main() 依 --variant 設定; 正式變體不帶後綴, 影子變體帳本與 HF 路徑帶 _<variant>
PAPER = ROOT / "paper"; HF_PREFIX = "db/paper"
SIGNALS_URL = f"https://huggingface.co/datasets/{REPO}/resolve/main/db/signals/latest.json"
def set_variant(v):
    global VARIANT, PAPER, HF_PREFIX, SIGNALS_URL
    VARIANT = v; sfx = CFG.suffix(v)
    PAPER = ROOT / ("paper" + sfx); HF_PREFIX = "db/paper" + sfx
    SIGNALS_URL = f"https://huggingface.co/datasets/{REPO}/resolve/main/db/signals/latest{sfx}.json"
COST_RT = 0.0023          # 來回成本 (券賣手續費+稅+買回手續費), 與回測相同
ENTRY_T = (9, 1, 0)       # 進場時間
VWAP_WIN = ((9, 1, 0), (9, 3, 0))   # 真實 tick 基準: 09:01~09:03 VWAP
PRE_AUCTION_T = (13, 24, 0)   # 逐筆撮合最後一分鐘: 現價 >= 昨收*(1+PRE_AUCTION_K) 的先市價買回 (13:25~13:30 沒有 tick, +8% 規則失效)
PRE_AUCTION_K = 0.06
PORTFOLIO_STOP = None     # 組合停損 (對已投入金額). 2026-10-23 決定不用 (None); 要用填 0.04 等 (stats/portfolio_stop_test*.md / stop_on_rally_days.md: 期望值幾乎不變, 只砍最差日)
EXIT_T = (13, 26, 0)      # 其餘部位回補 (參與 13:30 收盤競價)
AUCTION_T = (13, 25, 0)   # 13:25 起收盤集合競價不收市價單 (op_code 88), 改用漲停價限價 ROD 買回, 以收盤價成交
END_T = (13, 35, 0)
LATE_ABORT_T = (9, 4, 0)
LIVE_EVERY = 300          # 盤中每 5 分鐘寫一次 live.json (持倉與未實現損益) 到 HF  # 超過這時間還沒進場 (例如備援排程太晚啟動) 就放棄, 不在盤中亂下單

def now(): return datetime.now(TZ)
def at(h, m, s=0): return now().replace(hour=h, minute=m, second=s, microsecond=0)
def log(*a):
    print(now().strftime("%H:%M:%S"), *a, flush=True)

def load_env_file(path):
    for line in open(path, encoding="utf-8", errors="ignore"):
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line: continue
        if line.startswith("export "): line = line[7:]
        k, v = line.split("=", 1); k = k.strip(); v = v.strip()
        if len(v) >= 2 and v[0] == v[-1] and v[0] in "\"'": v = v[1:-1]
        os.environ.setdefault(k, v)

SIMULATION = True   # 三道保險之一: 永遠是模擬環境。改成 False 程式會拒絕執行。

def connect_simulation_only(sj):
    """登入永豐『模擬環境』並做三道查核, 任何一道不過就直接退出, 絕不碰正式帳號:
    1. 程式常數 SIMULATION 必須為 True, 且 Shioaji(simulation=True) 寫死。
    2. 本程式從不呼叫 activate_ca(): 正式環境的股票委託必須憑證簽章, 沒簽章連送都送不出去。
    3. 登入後查 account_balance(): 模擬環境回傳 date='' 與餘額 0 (永豐的 simulation default);
       正式帳號一定帶日期與真實餘額 → 一旦看到就視為「不是模擬環境」立即停止。"""
    if SIMULATION is not True: sys.exit("安全停止: SIMULATION 不是 True")
    api = sj.Shioaji(simulation=True)
    api.login(api_key=os.environ["SINOPAC_API_KEY"], secret_key=os.environ["SINOPAC_SECRET_KEY"])
    try:
        bal = api.account_balance()
        date_ = str(getattr(bal, "date", "") or ""); amt = float(getattr(bal, "acc_balance", 0) or 0)
    except Exception as e:
        try: api.logout()
        except Exception: pass
        sys.exit(f"安全停止: 無法查核 account_balance ({e})")
    if date_ != "" or amt != 0:
        try: api.logout()
        except Exception: pass
        sys.exit(f"安全停止: account_balance 回傳真實資料 (date={date_!r}, 餘額非 0)，看起來不是模擬環境，拒絕執行")
    log(f"✓ 模擬環境確認: Shioaji(simulation=True)、未啟用 CA、account_balance 為模擬預設值 (date='', 餘額 0)。帳號 signed={getattr(api.stock_account, 'signed', None)}")
    return api

def fetch_signals(date_str, wait_until=None, local_csv=None):
    """回傳 (date, no_trade, picks[list of dict])。picks 鍵: stock_id,name,prev_close,open,lots,amount,force_cover,limit_up"""
    if local_csv:
        import pandas as pd
        from publish_signals import COLS
        df = pd.read_csv(local_csv, dtype={"stock_id": str}, encoding="utf-8-sig").rename(columns=COLS)
        for c in ["prev_close","open","force_cover","limit_up"]:
            if c in df: df[c] = df[c].astype(float).round(2)   # 與 publish_signals.build_json 一致, 去掉 float32 尾巴
        meta_p = Path(local_csv).with_name(Path(local_csv).stem + "_meta.json")
        meta = json.load(open(meta_p)) if meta_p.exists() else {}
        return meta.get("date", date_str), meta.get("no_trade", []), df.to_dict(orient="records")
    while True:
        try:
            j = requests.get(SIGNALS_URL, params={"t": int(time.time())}, timeout=15).json()
            if j.get("date") == date_str: return j["date"], j.get("no_trade", []), j.get("picks", [])
            log(f"HF latest.json 日期 {j.get('date')} != {date_str}, 等待名單 ...")
        except Exception as e:
            log("讀 latest.json 失敗:", e)
        if wait_until is None or now() >= wait_until: return None, None, None
        time.sleep(20)

def sleep_until(t):
    while (d := (t - now()).total_seconds()) > 0: time.sleep(min(d, 5))

class Paper:
    def __init__(self, api, picks, date_str):
        self.api, self.picks, self.date = api, picks, date_str
        self.by_code = {p["stock_id"]: p for p in picks}
        self.contracts = {}
        self.ticks = defaultdict(list)      # code -> [(dt, price, vol)]
        self.high = defaultdict(float)
        self.deals = defaultdict(list)      # (code, action) -> [(price, qty, ts)]
        self.orders = {}                    # (code, action) -> trade
        self.forced = set(); self.covered = set(); self.skipped = {}
        self.lock = threading.Lock()
        self.seen_events = set()

    # ---- callbacks ----
    def on_order(self, stat, msg):
        try:
            eid = msg.get("event_id")
            if eid in self.seen_events: return
            self.seen_events.add(eid)
            s = str(stat)
            cf = msg.get("custom_field") if "Deal" in s else (msg.get("order", {}) or {}).get("custom_field")
            if cf not in (None, "", VARIANT[:6]): return      # 另一個變體 (同帳戶) 的委託/成交, 略過
            if "Deal" in s:
                code, action = msg.get("code"), str(msg.get("action"))
                act = "Sell" if "Sell" in action else "Buy"
                with self.lock: self.deals[(code, act)].append((float(msg["price"]), int(msg["quantity"]), msg.get("ts")))
                log(f"成交 {code} {act} {msg['quantity']}張 @ {msg['price']}")
            else:
                op = msg.get("operation", {}); o = msg.get("order", {}); c = msg.get("contract", {})
                if op.get("op_code") not in ("00", None):
                    log(f"委託異常 {c.get('code')} {o.get('action')} op_code={op.get('op_code')} {op.get('op_msg')}")
        except Exception as e:
            log("order callback error:", e)

    def on_tick(self, *args):
        tick = args[-1]   # 新版 1 參數 (tick), 舊版 2 參數 (exchange, tick)
        try:
            code = tick.code
            if code not in self.by_code or getattr(tick, "simtrade", False): return
            price = float(tick.close); vol = int(tick.volume)
            dt = datetime.combine(now().date(), datetime.strptime(str(tick.time)[:8], "%H:%M:%S").time(), TZ) if hasattr(tick, "time") else now()
            with self.lock:
                self.ticks[code].append((dt, price, vol))
                if price > self.high[code]: self.high[code] = price
            p = self.by_code[code]
            if code not in self.covered and (code, "Sell") in self.orders and price >= p["force_cover"] and code not in self.forced:
                self.forced.add(code)
                log(f"⚠ {code} {p.get('name','')} 觸及強制回補價 {p['force_cover']} (成交 {price}) → 市價買回")
                if not self.deals[(code, "Sell")]: self.refresh_deals_from_status()
                self.cover(code, "forced")
                threading.Thread(target=self.push_live, args=("running",), daemon=True).start()
        except Exception as e:
            log("tick callback error:", e)

    # ---- orders ----
    def order(self, code, action, qty, daytrade_short=False, limit_px=None):
        import shioaji as sj
        c = self.contracts[code]
        kw = dict(price=limit_px or 0, quantity=qty, action=action, price_type=sj.StockPriceType.LMT if limit_px else sj.StockPriceType.MKT, order_type=sj.OrderType.ROD,
                  order_lot=sj.StockOrderLot.Common, account=self.api.stock_account, custom_field=VARIANT[:6])   # 兩個變體共用同一模擬帳戶, 用 custom_field 區分回報
        if daytrade_short: kw["daytrade_short"] = True
        else: kw["order_cond"] = sj.StockOrderCond.Cash
        trade = self.api.place_order(c, self.api.Order(**kw))
        st = getattr(trade.status, "status", None)
        log(f"委託 {code} {'券賣' if daytrade_short else '買回'} {qty}張 {f'限價 {limit_px}' if limit_px else '市價'} → {st} id={trade.order.id}")
        return trade

    def enter(self):
        import shioaji as sj
        for p in self.picks:
            code = p["stock_id"]
            if code in self.skipped: continue
            try:
                self.orders[(code, "Sell")] = self.order(code, sj.Action.Sell, int(p["lots"]), daytrade_short=True)
            except Exception as e:
                self.skipped[code] = f"券賣委託失敗: {e}"; log("✗", code, self.skipped[code])

    def cover(self, code, reason):
        """買回張數 = 券賣『實際成交』張數 (部分成交只買回成交的部分; 完全沒成交就不買, 並取消還掛著的券賣單)."""
        import shioaji as sj
        if code in self.covered: return
        p = self.by_code[code]
        sell_trade = self.orders.get((code, "Sell"))
        if sell_trade is not None:
            try:   # 先取消還沒成交的券賣單, 避免回補後它才成交變成裸空
                st = str(getattr(sell_trade.status, "status", ""))
                if "Filled" not in st or "PartFilled" in st: self.api.cancel_order(sell_trade); log(f"取消未成交券賣單 {code} ({st})")
            except Exception as e: log(f"取消券賣單 {code} 失敗 (可能已成交): {e}")
        _, filled = self.avg((code, "Sell"))
        if filled <= 0:
            self.covered.add(code); p["_exit_reason"] = "no_fill"; log(f"{code} 券賣未成交, 不買回"); return
        try:
            lim = float(self.contracts[code].limit_up) if now() >= at(*AUCTION_T) else None   # 集合競價只收限價: 掛漲停價 = 必以收盤價成交 (除非收在漲停且買不到)
            self.orders[(code, "Buy")] = self.order(code, sj.Action.Buy, int(filled), limit_px=lim)
            self.covered.add(code); p["_exit_reason"] = reason
        except Exception as e:
            log("✗ 買回委託失敗", code, e)

    def cover_all(self, reason):
        for p in self.picks:
            if (p["stock_id"], "Sell") in self.orders and p["stock_id"] not in self.covered: self.cover(p["stock_id"], reason)

    # ---- 盤中狀態 ----
    def live_snapshot(self, status):
        rows = []; pnl = 0.0; amt = 0.0
        with self.lock:
            for p in self.picks:
                code = p["stock_id"]
                sell_px, sell_q = self.avg((code, "Sell")); buy_px, buy_q = self.avg((code, "Buy"))
                last = self.ticks[code][-1][1] if self.ticks[code] else None
                if code in self.skipped: st = "略過"
                elif (code, "Sell") not in self.orders: st = "未進場"
                elif not sell_px: st = "券賣未成交"
                elif p.get("_exit_reason") == "no_fill": st = "券賣未成交"
                elif buy_px: st = {"forced": "已強制回補", "pre_auction": "已尾盤回補", "portfolio_stop": "已組合停損"}.get(p.get("_exit_reason"), "已收盤回補")
                elif code in self.covered: st = "回補中"
                else: st = "持有中"
                ref = buy_px or last
                upnl = (sell_px / ref - 1) * 100 if (sell_px and ref) else None
                if sell_px and ref: pnl += (sell_px - ref) * 1000 * sell_q; amt += sell_px * 1000 * sell_q
                rows.append({"stock_id": code, "name": p.get("name"), "lots": p.get("lots"), "status": st, "sell_px": sell_px, "last": last,
                             "buy_px": buy_px, "pnl_pct": None if upnl is None else round(upnl, 2), "high": self.high.get(code) or None,
                             "force_cover": p.get("force_cover"), "skipped": self.skipped.get(code, "")})
        return {"date": self.date, "status": status, "updated_at": now().strftime("%Y-%m-%d %H:%M:%S"), "n_open": sum(1 for r in rows if r["status"] == "持有中"),
                "pnl_gross": round(pnl), "pnl_pct": round(pnl / amt * 100, 3) if amt else None, "amount": round(amt), "positions": rows}

    def portfolio_pnl(self):
        """已成交空單的未實現損益 (對券賣金額), 用最後 tick; 已回補的用買回價. 回傳 (pct, amount) 或 (None, 0)."""
        pnl = 0.0; amt = 0.0
        with self.lock:
            for p in self.picks:
                code = p["stock_id"]; sell_px, q = self.avg((code, "Sell"))
                if not sell_px: continue
                buy_px, _ = self.avg((code, "Buy")); ref = buy_px or (self.ticks[code][-1][1] if self.ticks[code] else None)
                if not ref: continue
                pnl += (sell_px - ref)*1000*q; amt += sell_px*1000*q
        return (pnl/amt if amt else None), amt

    def push_live(self, status):
        snap = self.live_snapshot(status)
        try:
            json.dump(snap, open(PAPER/"live.json", "w"), ensure_ascii=False, indent=1)
            if os.environ.get("HF_TOKEN"):
                from huggingface_hub import HfApi
                HfApi(token=os.environ["HF_TOKEN"]).upload_file(path_or_fileobj=str(PAPER/"live.json"), path_in_repo=f"{HF_PREFIX}/live.json", repo_id=REPO, repo_type="dataset",
                                                               commit_message=f"paper{'' if VARIANT==CFG.MAIN_VARIANT else ' '+VARIANT} live {snap['date']} {snap['updated_at'][11:]} {status}")
        except Exception as e:
            log("live.json 上傳失敗:", e)
        return snap

    # ---- settle ----
    def refresh_deals_from_status(self):
        """callback 可能漏, 用 update_status 補 trade.status.deals"""
        try: self.api.update_status(self.api.stock_account)
        except Exception as e: log("update_status 失敗:", e)
        for (code, act), tr in self.orders.items():
            try:
                deals = [(float(d.price), int(d.quantity), getattr(d, "ts", None)) for d in (tr.status.deals or [])]
            except Exception: deals = []
            if deals and not self.deals[(code, act)]: self.deals[(code, act)] = deals

    def vwap(self, code, t0, t1):
        xs = [(p, v) for dt, p, v in self.ticks[code] if t0 <= dt < t1 and v > 0]
        return (sum(p*v for p, v in xs) / sum(v for _, v in xs)) if xs else None

    def avg(self, key):
        ds = self.deals[key]; q = sum(q for _, q, _ in ds)
        return (sum(p*q for p, q, _ in ds) / q, q) if q else (None, 0)

    def settle(self):
        self.refresh_deals_from_status()
        d0 = now().date(); rows = []
        t0 = datetime.combine(d0, datetime.min.time(), TZ).replace(hour=VWAP_WIN[0][0], minute=VWAP_WIN[0][1])
        t1 = t0.replace(hour=VWAP_WIN[1][0], minute=VWAP_WIN[1][1])
        for p in self.picks:
            code = p["stock_id"]
            sell_px, sell_q = self.avg((code, "Sell")); buy_px, buy_q = self.avg((code, "Buy"))
            last = self.ticks[code][-1][1] if self.ticks[code] else None
            vw = self.vwap(code, t0, t1)
            sim_ret = (sell_px / buy_px - 1) if (sell_px and buy_px) else None
            tick_ret = (vw / last - 1) if (vw and last) else None
            # 反事實: 若沒有組合停損 → 依其餘規則用 tick 估 (觸 +8% 以強制價, 否則最後 tick 當收盤)
            if p.get("_exit_reason") == "portfolio_stop" and sell_px:
                hi = self.high.get(code, 0); fc = p["force_cover"]
                nostop_exit = (fc*(1+0.003)) if hi >= fc else last
                nostop_ret = (sell_px/nostop_exit - 1) if nostop_exit else None
            else: nostop_ret = sim_ret
            rows.append({"date": self.date, "stock_id": code, "name": p.get("name"), "lots": p.get("lots"), "prev_close": p.get("prev_close"),
                         "open_signal": p.get("open"), "force_cover": p.get("force_cover"), "day_trade": str(getattr(self.contracts.get(code), "day_trade", "")),
                         "skipped": self.skipped.get(code, ""), "sim_sell_px": sell_px, "sim_sell_lots": sell_q, "sim_buy_px": buy_px, "sim_buy_lots": buy_q,
                         "exit_reason": p.get("_exit_reason", ""), "tick_vwap_0901_0903": vw, "tick_high": self.high.get(code) or None, "tick_last": last,
                         "n_ticks": len(self.ticks[code]), "sim_ret_pct": None if sim_ret is None else round(sim_ret*100, 3),
                         "sim_net_pct": None if sim_ret is None else round((sim_ret - COST_RT)*100, 3),
                         "tick_ret_pct": None if tick_ret is None else round(tick_ret*100, 3),
                         "tick_net_pct": None if tick_ret is None else round((tick_ret - COST_RT)*100, 3),
                         "touched_force": bool(self.high.get(code, 0) >= p["force_cover"]),
                         "nostop_net_pct": None if nostop_ret is None else round((nostop_ret - COST_RT)*100, 3)})
        return rows

def summarize(rows, date_str, no_trade, mode):
    def mean(k):
        xs = [r[k] for r in rows if r.get(k) is not None]; return round(sum(xs)/len(xs), 3) if xs else None
    amt = sum((r["sim_sell_px"] or 0) * 1000 * (r["sim_sell_lots"] or 0) for r in rows)
    closed = [r for r in rows if r["sim_sell_px"] and r["sim_buy_px"]]
    unclosed = [r for r in rows if r["sim_sell_px"] and not r["sim_buy_px"]]   # 券賣成交但買回沒成交 (例: 買回單被拒), 不能當 0 損益
    amt_closed = sum(r["sim_sell_px"] * 1000 * (r["sim_buy_lots"] or 0) for r in closed)
    est_unclosed = sum((r["sim_sell_px"] * (1 - COST_RT) - r["tick_last"]) * 1000 * (r["sim_sell_lots"] or 0) for r in unclosed if r["tick_last"])
    pnl_nostop = sum((r["sim_sell_px"] or 0) * 1000 * (r["sim_sell_lots"] or 0) * ((r.get("nostop_net_pct") or 0)/100 + COST_RT) for r in rows if r["sim_sell_px"])
    pnl = sum((r["sim_sell_px"] - r["sim_buy_px"]) * 1000 * (r["sim_buy_lots"] or 0) for r in closed)
    return {"date": date_str, "mode": mode, "no_trade": no_trade, "n_picks": len(rows), "n_filled": sum(1 for r in rows if r["sim_sell_px"]),
            "n_skipped": sum(1 for r in rows if r["skipped"]), "n_forced": sum(1 for r in rows if r["exit_reason"] == "forced"),
            "portfolio_stop": any(r["exit_reason"] == "portfolio_stop" for r in rows), "n_pre_auction": sum(1 for r in rows if r["exit_reason"] == "pre_auction"),
            "n_touched_force": sum(1 for r in rows if r["touched_force"]), "sim_avg_net_pct": mean("sim_net_pct"), "tick_avg_net_pct": mean("tick_net_pct"),
            "sim_win_rate": (round(sum(1 for r in rows if (r["sim_net_pct"] or 0) > 0) / max(1, sum(1 for r in rows if r["sim_net_pct"] is not None)), 3)
                             if any(r["sim_net_pct"] is not None for r in rows) else None),
            "sim_amount": round(amt), "sim_pnl_gross": round(pnl) if closed else None, "sim_pnl_net": round(pnl - amt_closed * COST_RT) if closed else None,
            "n_unclosed": len(unclosed), "unclosed_est_pnl_net": round(est_unclosed) if unclosed else None,
            "nostop_pnl_net": round(pnl_nostop - amt * COST_RT) if closed else None, "nostop_avg_net_pct": mean("nostop_net_pct"), "finished_at": now().strftime("%Y-%m-%d %H:%M:%S")}

def publish(date_str, rows, summ):
    token = os.environ.get("HF_TOKEN")
    if not token: log("沒有 HF_TOKEN, 跳過上傳"); return
    from huggingface_hub import HfApi, CommitOperationAdd, hf_hub_download
    api = HfApi(token=token)
    # 累計 summary.csv
    try:
        p = hf_hub_download(REPO, f"{HF_PREFIX}/summary.csv", repo_type="dataset", force_download=True); hist = list(csv.DictReader(open(p, encoding="utf-8")))
    except Exception: hist = []
    hist = [h for h in hist if h.get("date") != date_str] + [{k: ("" if v is None else v) for k, v in summ.items() if k != "no_trade"}]
    keys = list(hist[-1].keys()); buf = []
    import io; sio = io.StringIO(); w = csv.DictWriter(sio, fieldnames=keys, extrasaction="ignore"); w.writeheader(); [w.writerow(h) for h in hist]
    ops = [CommitOperationAdd(f"{HF_PREFIX}/{date_str}_trades.csv", str(PAPER/f"{date_str}_trades.csv")),
           CommitOperationAdd(f"{HF_PREFIX}/{date_str}_summary.json", str(PAPER/f"{date_str}_summary.json")),
           CommitOperationAdd(f"{HF_PREFIX}/latest.json", str(PAPER/f"{date_str}_summary.json")),
           CommitOperationAdd(f"{HF_PREFIX}/summary.csv", sio.getvalue().encode("utf-8"))]
    api.create_commit(repo_id=REPO, repo_type="dataset", operations=ops, commit_message=f"paper{'' if VARIANT==CFG.MAIN_VARIANT else ' '+VARIANT} {date_str}: {summ['n_filled']} filled, net {summ['sim_avg_net_pct']}%")
    log(f"已上傳 {HF_PREFIX}/ ({date_str}), 累計 {len(hist)} 天")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env-file", action="append", default=[]); ap.add_argument("--dry", action="store_true", help="登入+讀名單+查合約, 不下單不等待")
    ap.add_argument("--signals", default=None, help="本機 signals/<date>.csv (預設抓 HF latest.json)"); ap.add_argument("--publish", action="store_true")
    ap.add_argument("--date", default=None, help="覆寫日期 (測試用)"); ap.add_argument("--force-trade", action="store_true", help="忽略 no_trade 仍下單 (測試用)")
    ap.add_argument("--variant", default=CFG.MAIN_VARIANT, choices=CFG.VARIANTS, help=f"正式={CFG.MAIN_VARIANT}: latest.json/paper/; 其他: latest_<variant>.json, paper_<variant>/, HF db/paper_<variant>/")
    a = ap.parse_args(); set_variant(a.variant)
    for f in a.env_file: load_env_file(f)
    if not os.environ.get("SINOPAC_API_KEY") or not os.environ.get("SINOPAC_SECRET_KEY"): sys.exit("缺少 SINOPAC_API_KEY / SINOPAC_SECRET_KEY")
    PAPER.mkdir(exist_ok=True); log(f"變體: {VARIANT}, 帳本 {PAPER.name}/, 名單 {SIGNALS_URL.split('/')[-1]}")
    date_str = a.date or now().strftime("%Y-%m-%d")
    if now().weekday() >= 5 and not a.dry: sys.exit("週末不跑")

    sys.path.insert(0, str(ROOT/"scripts"))
    d, no_trade, picks = fetch_signals(date_str, wait_until=None if a.dry else at(9, 10), local_csv=a.signals)
    if picks is None: sys.exit(f"09:10 前沒有等到 {date_str} 的名單, 放棄")
    log(f"名單 {d}: {len(picks)} 檔, no_trade={no_trade}")
    if no_trade and not a.force_trade:
        summ = {"date": date_str, "mode": "simulation", "no_trade": no_trade, "n_picks": len(picks), "n_filled": 0, "n_skipped": 0, "n_forced": 0, "n_touched_force": 0,
                "sim_avg_net_pct": None, "tick_avg_net_pct": None, "sim_win_rate": None, "sim_amount": 0, "sim_pnl_gross": 0, "sim_pnl_net": 0, "finished_at": now().strftime("%Y-%m-%d %H:%M:%S")}
        json.dump(summ, open(PAPER/f"{date_str}_summary.json", "w"), ensure_ascii=False, indent=1)
        with open(PAPER/f"{date_str}_trades.csv", "w", newline="", encoding="utf-8") as f: f.write("date,stock_id\n")
        log("今日不交易，只記錄");
        if a.publish: publish(date_str, [], summ)
        return

    import shioaji as sj
    api = connect_simulation_only(sj)
    pt = Paper(api, picks, d)
    api.set_order_callback(pt.on_order); api.set_on_tick_stk_v1_callback(pt.on_tick)
    for p in picks:
        code = p["stock_id"]
        try:
            c = api.Contracts.Stocks[code]   # contracts.get() 回 BaseContract 沒有 day_trade 等欄位
            if c is None: raise KeyError(code)
            pt.contracts[code] = c
            if str(c.day_trade) != "DayTrade.Yes": pt.skipped[code] = f"不可現沖券賣 ({c.day_trade})"
            ref = float(c.reference or 0)
            if ref and abs(ref - float(p["prev_close"])) > 1e-6: log(f"注意 {code} 合約參考價 {ref} != 名單昨收 {p['prev_close']} (除權息?)")
            log(f"{code} {c.name} day_trade={c.day_trade} ref={c.reference} limit_up={c.limit_up} 名單: open={p['open']} lots={p['lots']} force={p['force_cover']}")
        except Exception as e:
            pt.skipped[code] = f"合約查詢失敗: {e}"; log("✗", code, pt.skipped[code])
    if a.dry:
        cs = [pt.contracts[p["stock_id"]] for p in picks if p["stock_id"] in pt.contracts]
        if cs:
            for s in api.snapshots(cs): log(f"快照 {s.code}: open={s.open} high={s.high} low={s.low} close={s.close} vol={s.total_volume}")
        log("dry run 結束 (未下單)"); api.logout(); return

    for p in picks:
        if p["stock_id"] in pt.contracts: api.subscribe(pt.contracts[p["stock_id"]], quote_type=sj.QuoteType.Tick)
    log(f"已訂閱 {len(pt.contracts)} 檔 tick, 等到 {ENTRY_T[0]:02d}:{ENTRY_T[1]:02d}:{ENTRY_T[2]:02d} 進場 ...")
    sleep_until(at(*ENTRY_T))
    if now() > at(*LATE_ABORT_T):
        log(f"現在 {now().strftime('%H:%M:%S')} 已超過 {LATE_ABORT_T[0]:02d}:{LATE_ABORT_T[1]:02d}，太晚進場，放棄 (今天不做模擬)"); api.logout(); sys.exit(2)
    pt.enter()
    time.sleep(5); pt.refresh_deals_from_status(); snap = pt.push_live("running")
    log(f"進場完成 (持有 {snap['n_open']} 檔), 盯盤到 13:26, 每 {LIVE_EVERY//60} 分鐘更新 live.json ...")
    t_exit = at(*EXIT_T); t_pre = at(*PRE_AUCTION_T); last_hb = now(); pre_done = False; pstop_done = False; last_pchk = now()
    while now() < t_exit:
        time.sleep(1)
        if PORTFOLIO_STOP and not pstop_done and (now() - last_pchk).total_seconds() >= 5:   # 組合停損: 每 5 秒檢查 (PORTFOLIO_STOP=None 時關閉)
            last_pchk = now(); pct, amt = pt.portfolio_pnl()
            if pct is not None and pct <= -PORTFOLIO_STOP:
                pstop_done = True; log(f"⚠ 組合停損: 未實現 {pct*100:.2f}% <= -{PORTFOLIO_STOP*100:.0f}% (投入 {amt:,.0f}) → 全部市價回補")
                pt.refresh_deals_from_status(); pt.cover_all("portfolio_stop"); pt.push_live("running")
        if not pre_done and now() >= t_pre:
            pre_done = True; pt.refresh_deals_from_status(); n = 0
            for p in picks:
                code = p["stock_id"]
                if code in pt.covered or (code, "Sell") not in pt.orders or not pt.ticks[code]: continue
                last = pt.ticks[code][-1][1]; thr = float(p["prev_close"]) * (1 + PRE_AUCTION_K)
                if last >= thr:
                    log(f"13:24 預先回補 {code} {p.get('name','')}: 現價 {last} >= 昨收+{PRE_AUCTION_K*100:.0f}% ({thr:.2f})，不等收盤競價"); pt.cover(code, "pre_auction"); n += 1
            if n: pt.push_live("running")
            log(f"13:24 檢查完成, 預先回補 {n} 檔")
        if (now() - last_hb).total_seconds() >= LIVE_EVERY:
            last_hb = now(); pt.refresh_deals_from_status(); snap = pt.push_live("running")
            log("心跳", f"持有 {snap['n_open']} 檔 未實現 {snap['pnl_pct']}% ({snap['pnl_gross']:,} 元)", " ".join(f"{r['stock_id']}:{r['status']}@{r['last']}" for r in snap["positions"]))
    pt.refresh_deals_from_status(); pt.cover_all("close")
    time.sleep(3); pt.push_live("closing")
    log("收盤前回補單已送出, 等成交到 13:35 ...")
    sleep_until(at(*END_T))
    rows = pt.settle(); summ = summarize(rows, d, no_trade, f"simulation ({VARIANT})")
    with open(PAPER/f"{d}_trades.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["date"]); w.writeheader(); w.writerows(rows)
    json.dump(summ, open(PAPER/f"{d}_summary.json", "w"), ensure_ascii=False, indent=1)
    log("結算:", json.dumps(summ, ensure_ascii=False))
    for r in rows: log(f"  {r['stock_id']} {r['name']}: 券賣 {r['sim_sell_px']} 買回 {r['sim_buy_px']} ({r['exit_reason']}) 模擬淨 {r['sim_net_pct']}% | tick VWAP {r['tick_vwap_0901_0903']} 收 {r['tick_last']} 淨 {r['tick_net_pct']}% 高 {r['tick_high']} {r['skipped']}")
    pt.push_live("done")
    try: api.logout()
    except Exception: pass
    if a.publish: publish(d, rows, summ)

if __name__ == "__main__":
    main()
