"""每日候選名單 (做空當沖, 定案策略 S9'').

用法:
  python3 scripts/daily_candidates.py --refresh                 # 盤前: 從 HF 更新最新兩個月 D1, 重建本地 d1.parquet
  python3 scripts/daily_candidates.py                           # 09:01 執行: 從證交所即時行情抓開盤價 → 產生今日名單
  python3 scripts/daily_candidates.py --open-csv opens.csv      # 自備開盤價 (欄位 stock_id,open)
  python3 scripts/daily_candidates.py --date 2026-10-22         # 回放: 用歷史 D1 的開盤價 (驗證用)
"""
import argparse, sys, time, json, glob
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import features as F

ROOT = Path(__file__).resolve().parent.parent
REPO = "raywu918daytrade/daytrade"

# ---- 策略參數 (定案) ----
POOL_DV, POOL_ATR = 5e8, 0.03
MKT_GAP_MIN, GAP_MAX, MAX_PRICE = -0.01, 0.04, 300.0
FORCE_COVER_K = 0.08
NO_TRADE_DOW = {0}   # 週一

def refresh_d1(months_back=2):
    from huggingface_hub import hf_hub_download
    today = pd.Timestamp.today().normalize()
    months = sorted({(today - pd.DateOffset(months=i)).strftime("%Y_%m") for i in range(months_back)})
    for d in ("data/raw/d1", "data/raw/meta", "data/processed"): (ROOT/d).mkdir(parents=True, exist_ok=True)
    for ym in months:
        p = hf_hub_download(REPO, f"db/d1/{ym}.parquet", repo_type="dataset", force_download=True)
        (ROOT/"data/raw/d1"/f"{ym}.parquet").write_bytes(Path(p).read_bytes()); print("updated d1", ym)
    p = hf_hub_download(REPO, "db/info/info.parquet", repo_type="dataset", force_download=True)
    (ROOT/"data/raw/meta/info.parquet").write_bytes(Path(p).read_bytes())
    fs = sorted(glob.glob(str(ROOT/"data/raw/d1/*.parquet")))
    df = pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)
    df["date"] = pd.to_datetime(df["date"])
    df = df.drop_duplicates(["stock_id","date"]).sort_values(["stock_id","date"]).reset_index(drop=True)
    info = pd.read_parquet(ROOT/"data/raw/meta/info.parquet")
    df = df.merge(info[["stock_id","name","market_type","group"]], on="stock_id", how="left")
    df["is_stock"] = df.stock_id.str.fullmatch(r"\d{4}"); df["turnover"] = df.close*df.volume
    df.to_parquet(ROOT/"data/processed/d1.parquet", index=False)
    print(f"d1.parquet rebuilt: {len(df)} rows, last date {df.date.max().date()}")

def fetch_twse_opens(stock_ids, markets, sleep=0.4):
    """證交所即時行情 (mis.twse.com.tw). 回傳 DataFrame(stock_id, open, prev_close_mis, name)."""
    import requests
    s = requests.Session(); s.headers.update({"User-Agent":"Mozilla/5.0", "Accept":"application/json"})
    s.get("https://mis.twse.com.tw/stock/index.jsp", timeout=10)
    codes = [f"{'tse' if m=='twse' else 'otc'}_{sid}.tw" for sid, m in zip(stock_ids, markets)]
    out = []
    for i in range(0, len(codes), 40):
        url = "https://mis.twse.com.tw/stock/api/getStockInfo.jsp?ex_ch=" + "|".join(codes[i:i+40]) + "&json=1&delay=0&_=" + str(int(time.time()*1000))
        for attempt in range(3):
            try:
                r = s.get(url, timeout=10); js = r.json(); break
            except Exception as ex:
                if attempt == 2: print("MIS fetch failed:", ex); js = {"msgArray": []}
                time.sleep(1)
        for m in js.get("msgArray", []):
            o = m.get("o","-"); y = m.get("y","-")
            out.append({"stock_id": m.get("c"), "open": float(o) if o not in ("-","") else np.nan,
                        "prev_close_mis": float(y) if y not in ("-","") else np.nan, "mis_name": m.get("n"), "mis_time": m.get("t"), "mis_date": m.get("d")})
        time.sleep(sleep)
    return pd.DataFrame(out)

def latest_model(variant="base"):
    tag = "B" if variant == "base" else "B" + variant
    fs = sorted(glob.glob(str(ROOT/f"models/lgbm_{tag}_pool_*.txt")))
    if not fs: sys.exit(f"找不到 models/lgbm_{tag}_pool_*.txt" + ("，先跑 07_train.py B pool" if variant == "base" else "，先跑 22_train_seq_model.py"))
    return fs[-1]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true"); ap.add_argument("--months", type=int, default=2, help="--refresh 抓最近幾個月 D1 (雲端無歷史檔時用 6)")
    ap.add_argument("--date", default=None)
    ap.add_argument("--open-csv", default=None); ap.add_argument("--capital", type=float, default=3_000_000)
    ap.add_argument("--max-per-stock", type=float, default=600_000); ap.add_argument("--max-price", type=float, default=MAX_PRICE)
    ap.add_argument("--max-n", type=int, default=10); ap.add_argument("--min-amount", type=float, default=100_000, help="單檔金額低於此不做 (避免零頭部位)")
    ap.add_argument("--no-daytrade-filter", action="store_true", help="不過濾現股當沖先賣後買資格 (對照用)")
    ap.add_argument("--variant", default="base", choices=["base","seq","sr"], help="影子模型: sr = 基準+支撐壓力 (models/lgbm_Bsr_pool_*), seq = 基準+K線序列; 輸出檔名加 _<variant>")
    a = ap.parse_args()
    if a.refresh: refresh_d1(a.months)

    d1 = pd.read_parquet(ROOT/"data/processed/d1.parquet")
    last = d1.date.max()
    tdate = pd.Timestamp(a.date).normalize() if a.date else pd.Timestamp.today().normalize()
    replay = tdate <= last
    print(f"本地 D1 最後日期 {last.date()}；目標日 {tdate.date()} ({'回放模式: 用歷史開盤價' if replay else '即時模式'})")

    if replay:
        d1 = d1[d1.date <= tdate].copy()
        # 回放時把目標日的 close/high/low/volume 遮掉, 確保只用 09:00 可得資訊
        m = d1.date == tdate
        d1.loc[m, ["high","low","close","volume"]] = np.nan
        source = "歷史 D1 open"
    else:
        if (tdate - last).days > 7: print("警告: D1 超過 7 天沒更新，請先 --refresh")
        base_last = d1[d1.date == last][["stock_id","market_type","name","close","volume"]]
        # 需要開盤價的範圍: 流動股 (供 mkt_* 計算) — 用上一日粗估 dv20
        tmp = F.build_B(F.d1_base(d1)); tmp = tmp[tmp.date == last]
        need = base_last.merge(tmp[["stock_id","dv20"]], on="stock_id")
        need = need[(need.dv20 >= 0.8e8) & need.market_type.isin(["twse","tpex"])]
        if a.open_csv:
            op = pd.read_csv(a.open_csv, dtype={"stock_id":str}); source = a.open_csv
        else:
            print(f"向證交所抓 {len(need)} 檔開盤價 ..."); op = fetch_twse_opens(need.stock_id.tolist(), need.market_type.tolist()); source = "TWSE MIS"
            (ROOT/"signals").mkdir(exist_ok=True); op.to_csv(ROOT/"signals"/f"{tdate.date()}_opens.csv", index=False)   # 存下來供 --variant seq 用同一份開盤價
            if op.mis_time.notna().any(): print("行情時間:", op.mis_date.dropna().iloc[0], op.mis_time.dropna().iloc[0])
            # 安全檢查: 行情日期必須等於目標日 (收盤後/盤前抓到的是前一交易日資料)
            mis_dates = set(op.mis_date.dropna())
            if not mis_dates or mis_dates != {tdate.strftime("%Y%m%d")}:
                sys.exit(f"錯誤: 證交所行情日期 {sorted(mis_dates)} 不是目標日 {tdate.strftime('%Y%m%d')}。現在不是盤中，不產生名單。"
                         f" (回放請用 --date，自備開盤價請用 --open-csv)")
            t_now = pd.Timestamp.now()
            if t_now.hour*60+t_now.minute < 9*60+1: print("警告: 09:01 之前很多股票尚未開出，大盤平均跳空可能失真")
        op = op[op.open.notna() & (op.open > 0)]
        print(f"取得 {len(op)} 檔開盤價")
        today = base_last[["stock_id","market_type","name"]].merge(op[["stock_id","open"]], on="stock_id")
        today["date"] = tdate
        for c in ["high","low","close"]: today[c] = np.nan
        today["volume"] = np.nan
        d1 = pd.concat([d1, today], ignore_index=True)

    base_df = F.d1_base(d1); f = F.build_B(base_df)
    FEATS = F.D1_FEATS
    if a.variant == "seq": f = f.merge(F.build_seq(base_df), on=["stock_id","date"], how="left")
    FEATS = F.D1_FEATS + F.shadow_feats(a.variant)
    sfx = "" if a.variant == "base" else "_" + a.variant
    t = f[f.date == tdate].copy()
    if len(t) == 0: sys.exit("目標日沒有任何資料")
    t = t.merge(d1[d1.date == tdate][["stock_id","name"]].drop_duplicates("stock_id"), on="stock_id", how="left")
    mkt_gap = t.mkt_gap.iloc[0]
    dow = tdate.dayofweek

    # ---- 池 + 濾網 ----
    pool = t[F.universe_mask(t) & (t.dv20_prev >= POOL_DV) & (t.atr20 >= POOL_ATR) & t.open.notna()].copy()
    if a.variant == "sr":   # 支撐壓力特徵只對池內股票算 (回看 60 天), 省時間
        sr = F.build_sr(base_df, pool.stock_id.unique()); sr = sr[sr.date == tdate]
        pool = pool.merge(sr, on=["stock_id","date"], how="left")
    import lightgbm as lgb
    mpath = latest_model(a.variant); model = lgb.Booster(model_file=mpath)
    pool["pred"] = model.predict(pool[FEATS])
    pool = pool.sort_values("pred")
    pool["rank_all"] = np.arange(1, len(pool)+1)
    pool["flag"] = ""
    # 現股當沖先賣後買資格 (證交所/櫃買每日公布; 不在名單 = 不可當沖, 註記 = 暫停先賣後買)
    dt_note = ""
    if not a.no_daytrade_filter:
        try:
            import daytrade_list as DL
            elig = DL.fetch_daytrade_eligible(tdate)
            can_short = pool.stock_id.map(lambda s: elig.get(s))
            pool.loc[can_short.isna(), "flag"] += "不可當沖 "
            pool.loc[can_short == False, "flag"] += "暫停先賣 "
            dt_note = f"現沖券賣資格: 已過濾 (池內不可當沖 {int(can_short.isna().sum())} 檔, 暫停先賣 {int((can_short == False).sum())} 檔)"
        except Exception as ex:
            dt_note = f"現沖券賣資格: 取得失敗, 未過濾 ({ex})"; print("警告:", dt_note)
    pool.loc[pool.gap > GAP_MAX, "flag"] += "跳空>4% "
    pool.loc[pool.open <= pool.prev_close*0.905, "flag"] += "跌停附近 "
    pool.loc[pool.open > a.max_price, "flag"] += f"股價>{a.max_price:.0f} "
    elig = pool[pool.flag == ""].copy()

    header = [f"日期 {tdate.date()}  開盤價來源: {source}  模型: {Path(mpath).name}" + (f"  [影子模型 {a.variant}]" if a.variant != "base" else ""),
              f"池 {len(pool)} 檔 (dv20≥5億, ATR20≥3%)；大盤平均跳空 {mkt_gap*100:+.2f}%；上漲比例 {t[t.dv20>=1e8].gap.gt(0).mean()*100:.0f}%"]
    if dt_note: header.append(dt_note)
    no_trade = []
    if dow in NO_TRADE_DOW: no_trade.append("週一不交易")
    if mkt_gap < MKT_GAP_MIN: no_trade.append(f"大盤平均跳空 {mkt_gap*100:.2f}% < -1%")

    # ---- 配置 ----
    cash = a.capital; picks = []
    for _, r in elig.iterrows():
        if len(picks) >= a.max_n: break
        lot = r.open*1000
        if cash < lot: continue
        lots = int(min(a.max_per_stock, cash)//lot)
        if lots < 1 or lots*lot < a.min_amount: continue
        amt = lots*lot; cash -= amt
        picks.append({"stock_id": r.stock_id, "name": r["name"], "昨收": r.prev_close, "開盤": r.open, "跳空%": r.gap*100, "ATR20%": r.atr20*100,
                      "20日均額(億)": r.dv20_prev/1e8, "模型分數%": r.pred*100, "池內排名": r.rank_all, "張數": lots, "金額": amt,
                      "強制回補價(昨收+8%)": F.round_up_tick(r.prev_close*(1+FORCE_COVER_K)), "漲停價": F.limit_up(r.prev_close)})
    out = pd.DataFrame(picks)
    lines = header + [""]
    if no_trade:
        lines.append("※ 今日不交易: " + "；".join(no_trade)); lines.append("  (以下名單僅供參考)")
    if len(out):
        lines.append(out.round(2).to_string(index=False))
        lines.append(f"\n合計 {len(out)} 檔，投入 {out['金額'].sum():,.0f} 元 ({out['金額'].sum()/a.capital*100:.1f}%)，剩餘 {cash:,.0f}")
    else:
        lines.append("沒有符合條件的候選")
    lines.append("\n被濾掉的前 10 名 (供對照):")
    lines.append(pool[pool.flag != ""].head(10)[["stock_id","name","prev_close","open","gap","pred","flag"]].assign(gap=lambda x: x.gap*100, pred=lambda x: x.pred*100).round(2).to_string(index=False))
    lines.append("\n出場規則: 13:30 收盤競價回補；盤中成交價 ≥ 強制回補價 立即市價回補；無其他停利停損。")
    txt = "\n".join(lines); print(txt)
    sig = ROOT/"signals"; sig.mkdir(exist_ok=True)
    out.to_csv(sig/f"{tdate.date()}{sfx}.csv", index=False, encoding="utf-8-sig")
    pool.assign(elig=pool.flag=="")[["stock_id","name","prev_close","open","gap","atr20","dv20_prev","pred","rank_all","flag","elig"]].to_csv(sig/f"{tdate.date()}{sfx}_pool.csv", index=False, encoding="utf-8-sig")
    (sig/f"{tdate.date()}{sfx}.txt").write_text(txt)
    json.dump({"date": str(tdate.date()), "no_trade": no_trade, "mkt_gap": float(mkt_gap), "n_picks": len(out), "source": source, "model": Path(mpath).name, "daytrade_filter": dt_note, "variant": a.variant},
              open(sig/f"{tdate.date()}{sfx}_meta.json","w"), ensure_ascii=False)
    print(f"\n已寫入 signals/{tdate.date()}{sfx}.csv / .txt / _pool.csv")

if __name__ == "__main__":
    main()
