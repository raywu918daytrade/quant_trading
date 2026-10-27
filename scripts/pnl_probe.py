"""查永豐模擬環境的損益/部位/交割 API 會回傳什麼 (只查詢, 不下單)。
用法: python3 scripts/pnl_probe.py [--date YYYY-MM-DD]   (需 SINOPAC_API_KEY / SINOPAC_SECRET_KEY)"""
import argparse, sys, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import shioaji as sj
from paper_trade import connect_simulation_only, now, log

def show(name, fn):
    try:
        res = fn()
        items = list(res) if isinstance(res, (list, tuple)) else [res]
        log(f"== {name}: {len(items)} 筆")
        for x in items[:50]: print("   ", x, flush=True)
        return items
    except Exception as e:
        log(f"== {name}: 失敗 {type(e).__name__}: {e}"); return []

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--date", default=now().strftime("%Y-%m-%d")); a = ap.parse_args()
    api = connect_simulation_only(sj); acct = api.stock_account; d = a.date
    log(f"shioaji {sj.__version__}, 查詢日 {d}")
    show("update_status / list_trades", lambda: (api.update_status(acct), api.list_trades())[1])
    show("list_positions (股)", lambda: api.list_positions(acct))
    show("list_positions (張)", lambda: api.list_positions(acct, unit=sj.constant.Unit.Common))
    pls = show(f"list_profit_loss {d}", lambda: api.list_profit_loss(acct, begin_date=d, end_date=d))
    for pl in pls[:20]: show(f"list_profit_loss_detail id={getattr(pl, 'id', '?')}", lambda pl=pl: api.list_profit_loss_detail(acct, detail_id=pl.id))
    show("list_profit_loss_summary", lambda: api.list_profit_loss_summary(acct, begin_date=d, end_date=d))
    show("list_settlements", lambda: api.list_settlements(acct))
    show("account_balance", lambda: api.account_balance())
    show("margin", lambda: api.margin(api.futopt_account) if api.futopt_account else "無期貨帳戶")
    try: api.logout()
    except Exception: pass

if __name__ == "__main__":
    main()
