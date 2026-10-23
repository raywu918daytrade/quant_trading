"""型態當出場條件: 09:01 照常進場; 到 cutoff 還沒出現「紅轉綠且綠>紅」的 5 分 K 型態 → 在 cutoff 那根開盤市價回補 (+0.1% 滑價);
有出現的照現行抱到收盤 (+8% 強制 / 13:24 / 收盤). cutoff ∈ {09:30, 10:00, 10:30, 11:00}. 另測「出現過型態但之後又出現反向 (綠轉紅且紅>綠) → 回補」.
樣本與 28 相同. 輸出 stats/m5_pattern_exit.md"""
import sys, glob, numpy as np, pandas as pd
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
ROOT = Path(__file__).resolve().parent.parent; OUT = ROOT/"stats"
BASE, SLIP, K, PEN, CAP, PER_STOCK, MAX_N, PRE_K = 0.0023, 0.001, 0.08, 0.07, 3_000_000, 600_000, 10, 0.06
CUTOFFS = {"09:30": 30, "10:00": 60, "10:30": 90, "11:00": 120}

def simulate_month(args):
    f, trades = args
    df = pd.read_parquet(f); ts = pd.to_datetime(df["date"]); df["day"] = ts.dt.normalize(); df["t"] = ts.dt.hour*60 + ts.dt.minute - 540
    df = df[(df.t>=0)&(df.t<=270)]
    key = set(zip(trades.stock_id, trades.date)); df = df[[(s, d) in key for s, d in zip(df.stock_id, df.day)]]
    out = []
    for (sid, day), g in df.groupby(["stock_id","day"], sort=False):
        tr = trades[(trades.stock_id==sid)&(trades.date==day)].iloc[0]; pc = tr.prev_close; entry = tr.entry
        g = g.sort_values("t"); t = g.t.values; o = g.open.values; h = g.high.values; c = g.close.values
        auction = c[t==270][-1] if (t==270).any() else c[-1]
        k = pd.DataFrame({"b": t//5, "o": o, "h": h, "c": c, "t": t}).groupby("b").agg(o=("o","first"), c=("c","last"), t=("t","first")).reset_index(); k = k[k.t < 265]
        ko, kc, kt = k.o.values, k.c.values, k.t.values
        pat_t = None; rev_t = None
        for i in range(1, len(k)):
            if pat_t is None and kc[i-1] > ko[i-1] and kc[i] < ko[i] and (ko[i]-kc[i]) > (kc[i-1]-ko[i-1]): pat_t = kt[i] + 5
            elif pat_t is not None and rev_t is None and kt[i] >= pat_t and kc[i-1] < ko[i-1] and kc[i] > ko[i] and (kc[i]-ko[i]) > (ko[i-1]-kc[i-1]): rev_t = kt[i] + 5
        # 基準出場 (進場後)
        def base_exit():
            if (h[(t >= 4) & (t < 264)] >= pc*(1+K)).any(): return pc*(1+K)*(1+SLIP*3), "forced", int(t[(t>=4)&(h>=pc*(1+K))][0])
            i264 = np.where(t == 264)[0]
            if len(i264) and c[i264[0]] >= pc*(1+PRE_K): return c[i264[0]]*(1+SLIP), "pre", 264
            if auction >= pc*1.0975: return (tr.next_open*(1+PEN) if np.isfinite(tr.next_open) else auction), "lock", 270
            return auction, "close", 270
        bx, br, bt = base_exit()
        res = {"stock_id": sid, "date": day, "rank": tr["rank"], "entry": entry, "g|現行": entry/bx - 1, "pat_t": pat_t if pat_t is not None else -1}
        def exit_at(tm):   # 在 tm 分鐘開盤回補, 但若之前已被強制回補則用強制
            if bt < tm and br in ("forced",): return bx
            idx = np.where(t >= tm)[0]
            return o[idx[0]]*(1+SLIP) if len(idx) else bx
        for lab, cut in CUTOFFS.items():
            if pat_t is not None and pat_t <= cut: res[f"g|沒型態@{lab}回補"] = entry/bx - 1
            else: res[f"g|沒型態@{lab}回補"] = entry/exit_at(cut) - 1
        # 反向型態回補
        if rev_t is not None and rev_t < bt: res["g|反向型態回補"] = entry/exit_at(rev_t) - 1
        else: res["g|反向型態回補"] = entry/bx - 1
        # 兩者合併 (10:00)
        if pat_t is not None and pat_t <= 60: res["g|沒型態@10:00回補+反向回補"] = res["g|反向型態回補"]
        else: res["g|沒型態@10:00回補+反向回補"] = res["g|沒型態@10:00回補"]
        out.append(res)
    return pd.DataFrame(out)

if __name__ == "__main__":
    R = pd.read_parquet(ROOT/"data/processed/trailing_stop_trades_sr.parquet")[["stock_id","date","entry","prev_close","next_open","rank"]]; R["ym"] = R.date.dt.strftime("%Y_%m")
    jobs = [(str(ROOT/"data/raw/m1"/f"{ym}.parquet"), g) for ym, g in R.groupby("ym") if (ROOT/"data/raw/m1"/f"{ym}.parquet").exists()]
    with ProcessPoolExecutor(6) as ex: parts = list(ex.map(simulate_month, jobs))
    S = pd.concat(parts, ignore_index=True); S.to_parquet(ROOT/"data/processed/m5_pattern_exit_trades.parquet", index=False); print("simulated", len(S), flush=True)
    all_days = pd.Series(sorted(S.date.unique())); names = [c[2:] for c in S.columns if c.startswith("g|")]
    def alloc(col):
        recs = []
        for dt, gg in S.sort_values(["date","rank"]).groupby("date", sort=False):
            cash = CAP; amts=[]; nets=[]
            for _, r in gg.iterrows():
                lc = r.entry*1000
                if cash < lc: continue
                lots = int(min(PER_STOCK, cash)//lc)
                if lots < 1: continue
                a = lots*lc; cash -= a; amts.append(a); nets.append(r[f"g|{col}"]-BASE)
                if len(amts) >= MAX_N: break
            recs.append((dt, sum(a*n for a, n in zip(amts, nets))/CAP))
        return pd.DataFrame(recs, columns=["date","ret"]).set_index("date").ret.reindex(all_days, fill_value=0.0)
    L = []; P = lambda s="": L.append(str(s))
    P("# 型態當出場條件 (2020~2026, sr 模型, M1 逐分鐘, 300 萬)\n")
    P("- 09:01 照常進場. 『沒型態@T回補』: T 前沒出現紅轉綠(綠>紅) 5 分 K 型態 → T 那根開盤市價回補 (+0.1%); 有出現的照現行抱到收盤. 『反向型態回補』: 出現型態之後若再出現綠轉紅(紅>綠) → 回補.\n")
    rows = []
    for nm in names:
        s = alloc(nm); g = S[f"g|{nm}"] - BASE; mo = s.groupby(s.index.to_period("M")).apply(lambda x: ((1+x).prod()-1)*100)
        rows.append({"規則": nm, "每筆淨%": g.mean()*100, "勝率": (g>0).mean(), "日淨%": s.mean()*100, "Sharpe": s.mean()/s.std()*np.sqrt(245), "MDD%": ((1+s).cumprod()/(1+s).cumprod().cummax()-1).min()*100, "最差月%": mo.min(), "負月比例": (mo<0).mean(), "年化(萬)": s.mean()*245*CAP/1e4})
    P("## 1. 全期間"); P(pd.DataFrame(rows).set_index("規則").round(3).to_markdown()); P("")
    yr = pd.DataFrame({nm: (S[f"g|{nm}"]-BASE).groupby(S.date.dt.year).mean()*100 for nm in names}); P("## 2. 各年每筆淨%"); P(yr.round(2).to_markdown()); P("")
    m = (S.pat_t > 0) & (S.pat_t <= 60)
    P("## 3. 10:00 前沒型態的交易: 抱到收盤 vs 10:00 回補")
    P(f"- 沒型態 {int((~m).sum())} 筆: 抱到收盤 {(S.loc[~m,'g|現行']-BASE).mean()*100:.3f}% / 10:00 回補 {(S.loc[~m,'g|沒型態@10:00回補']-BASE).mean()*100:.3f}%；有型態 {int(m.sum())} 筆 抱到收盤 {(S.loc[m,'g|現行']-BASE).mean()*100:.3f}%")
    (OUT/"m5_pattern_exit.md").write_text("\n".join(L)); print("\n".join(L))
