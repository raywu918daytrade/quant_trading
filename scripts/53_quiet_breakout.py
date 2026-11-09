"""10:00 之後「平淡一段時間 → 突然動起來」再進場 的統計. 全市場 universe, 2020-01 ~, M1 逐分鐘.
Universe (都用前一日以前的資料): 上市/上櫃、4 碼、非 00 開頭 ETF、昨收 ≥ 10、前 20 日均成交額 ≥ 1 億.
平淡期: 訊號那根之前連續 Q 分鐘 (Q = 30 / 60) 的 (最高 − 最低) / 昨收 ≤ R (R = 0.8% / 1.5%), 而且平淡期全部在 10:00 之後.
動起來: 訊號那根 (10:00+Q ~ 12:30) 收盤 > 平淡期最高 (向上) 或 < 平淡期最低 (向下), 且該分鐘量 ≥ V × 平淡期每分鐘均量 (V = 3 / 5).
每檔每天每個方向只取第一次. 進場 = 下一分鐘開盤, 再扣 0.1% 滑價.
做法: 順勢 = 往上動就多、往下動就空; 反做 = 反過來. 出場: 30 分鐘後收盤 / 收盤競價 (收盤版加 ±8% 強制出場 +0.2% 滑價、鎖漲跌停 −7%).
成本 0.23% (現股當沖). t 用每日平均算, |t|>2 才算不是運氣. IS 2020~2023 / OOS 2024~.
-> stats/quiet_breakout.md"""
import os, numpy as np, pandas as pd
from pathlib import Path
from numpy.lib.stride_tricks import sliding_window_view as swv
from concurrent.futures import ProcessPoolExecutor
ROOT = Path(os.environ.get("QT_ROOT", Path(__file__).resolve().parent.parent)); OUT = ROOT / "stats"; RAW = ROOT / "data/raw"
COST, SLIP, FSLIP, FORCE, LOCK, SPLIT = 0.0023, 0.001, 0.002, 0.08, 0.07, 2024
NT, T10, TLAST, HOLD = 271, 60, 210, 30
QS, RS, VS = (30, 60), (0.008, 0.015), (3, 5)

def universe():
    d = pd.concat([pd.read_parquet(f) for f in sorted((RAW / "d1").glob("*.parquet"))], ignore_index=True)
    d["date"] = pd.to_datetime(d.date); d = d.sort_values(["stock_id", "date"])
    info = pd.read_parquet(RAW / "meta/info.parquet")[["stock_id", "market_type"]]
    d = d.merge(info, on="stock_id", how="left")
    g = d.groupby("stock_id", sort=False)
    d["pc"] = g.close.shift(1); d["dv20"] = g.close.shift(1).mul(g.volume.shift(1)).groupby(d.stock_id).transform(lambda s: s.rolling(20, min_periods=15).mean())
    ok = d.stock_id.str.fullmatch(r"\d{4}") & ~d.stock_id.str.startswith("00") & d.market_type.isin(["twse", "tpex"]) & (d.pc >= 10) & (d.dv20 >= 1e8) & (d.date >= "2020-01-01")
    return d.loc[ok, ["stock_id", "date", "pc"]]

def grid(df, col, keys, fill=None):
    a = df.pivot_table(index=["stock_id", "day"], columns="t", values=col, aggfunc="last").reindex(keys).reindex(columns=range(NT))
    return a

def process(args):
    f, U = args
    df = pd.read_parquet(f); ts = pd.to_datetime(df.date); df["day"] = ts.dt.normalize(); df["t"] = ts.dt.hour * 60 + ts.dt.minute - 540
    df = df[(df.t >= 0) & (df.t < NT)].merge(U.rename(columns={"date": "day"}), on=["stock_id", "day"], how="inner")
    if df.empty: return None
    keys = pd.MultiIndex.from_frame(df[["stock_id", "day"]].drop_duplicates())
    C = grid(df, "close", keys).ffill(axis=1); O = grid(df, "open", keys); H = grid(df, "high", keys); L = grid(df, "low", keys); V = grid(df, "volume", keys).fillna(0).values
    O = O.fillna(C).values; H = H.fillna(C).values; L = L.fillna(C).values; C = C.values
    pc = U.set_index(["stock_id", "date"]).pc.reindex(keys).values
    n = len(keys); rows = np.arange(n)
    auc = C[:, -1]; sufH = np.flip(np.fmax.accumulate(np.flip(np.nan_to_num(H, nan=-1), 1), 1), 1); sufL = np.flip(np.fmin.accumulate(np.flip(np.nan_to_num(L, nan=1e18), 1), 1), 1)
    out = []
    for Q in QS:
        # 視窗 [s−Q, s) 的高 / 低 / 均量, 對所有 s (index s 對應 window 起點 s−Q)
        wh = swv(np.nan_to_num(H, nan=-1), Q, axis=1).max(-1); wl = swv(np.nan_to_num(L, nan=1e18), Q, axis=1).min(-1); wv = swv(V, Q, axis=1).mean(-1)
        S = np.arange(T10 + Q, TLAST + 1)                        # 訊號那根 t
        hh = wh[:, S - Q]; ll = wl[:, S - Q]; vv = wv[:, S - Q]; cs = C[:, S]; vs = V[:, S]; rng = (hh - ll) / pc[:, None]
        for R in RS:
            for VM in VS:
                base = (rng <= R) & (vv > 0) & (vs >= VM * vv) & np.isfinite(cs)
                for dirn, cond in (("上", cs > hh), ("下", cs < ll)):
                    m = base & cond; has = m.any(1); j = m.argmax(1); i = rows[has]; s = S[j[has]]
                    e = s + 1; ent = O[i, e]; ok = np.isfinite(ent) & (e < NT)
                    i, s, e, ent = i[ok], s[ok], e[ok], ent[ok]
                    x30 = C[i, np.minimum(e + HOLD, NT - 1)]; xa = auc[i]; p = pc[i]
                    sd = 1 if dirn == "上" else -1                       # 順勢方向
                    for kind, side in (("順勢", sd), ("反做", -sd)):
                        en = ent * (1 + SLIP * side)                     # 多單買貴、空單賣便宜
                        r30 = side * (x30 / en - 1) - COST
                        if side > 0:
                            forced = sufL[i, e] <= p * (1 - FORCE); ex = np.where(forced, p * (1 - FORCE) * (1 - FSLIP), xa); lock = (~forced) & (ex <= p * 0.9 * 1.001)
                        else:
                            forced = sufH[i, e] >= p * (1 + FORCE); ex = np.where(forced, p * (1 + FORCE) * (1 + FSLIP), xa); lock = (~forced) & (ex >= p * 1.1 * 0.999)
                        rc = side * (ex / en - 1) - COST - np.where(lock, LOCK, 0)
                        out.append(pd.DataFrame({"stock_id": keys.get_level_values(0)[i], "date": keys.get_level_values(1)[i], "t": s, "Q": Q, "R": R, "V": VM, "dir": dirn, "kind": kind,
                                                 "r30": r30 * 100, "rc": rc * 100, "forced": forced}))
    return pd.concat(out, ignore_index=True) if out else None

def tstat(x):
    d = x.groupby(x.columns[0])[x.columns[1]].mean(); return d.mean() / d.std() * np.sqrt(len(d)) if len(d) > 2 and d.std() > 0 else np.nan

if __name__ == "__main__":
    U = universe(); U["ym"] = U.date.dt.strftime("%Y_%m")
    jobs = [(str(RAW / "m1" / f"{ym}.parquet"), g[["stock_id", "date", "pc"]]) for ym, g in U.groupby("ym") if (RAW / "m1" / f"{ym}.parquet").exists()]
    with ProcessPoolExecutor(6) as ex: X = pd.concat([b for b in ex.map(process, jobs) if b is not None], ignore_index=True)
    X["oos"] = X.date.dt.year >= SPLIT; ndays = U.date.nunique()
    L = []; P = lambda s="": L.append(str(s))
    P(f"# 10:00 後平淡 → 突然動起來 再進場 (全市場 universe, {U.date.min():%Y-%m-%d}~{U.date.max():%Y-%m-%d}, {len(U)} 檔日 / {ndays} 天)"); P()
    P("- universe: 上市櫃 4 碼非 ETF、昨收 ≥10、前 20 日均成交額 ≥1 億 (約每天 " + f"{len(U) / ndays:.0f} 檔)")
    P("- 平淡期 Q 分鐘: 10:00 之後連續 Q 分鐘 (高−低)/昨收 ≤ R; 動起來: 下一根收盤突破平淡期高/低, 且那一分鐘量 ≥ V × 平淡期每分鐘均量; 訊號最晚 12:30")
    P("- 進場 = 訊號下一分鐘開盤 ±0.1% 滑價; 出場 30 分後收盤 或 收盤競價 (含 ±8% 強制出場、鎖漲跌停 −7%); 成本 0.23%")
    P("- 順勢 = 往上動做多、往下動做空; 反做 = 反過來. 每筆淨% 已扣成本; t 用每日平均算, |t|>2 才算不是運氣; IS 2020~23 / OOS 2024~"); P()
    rows = []
    for (Q, R, VM, dirn, kind), x in X.groupby(["Q", "R", "V", "dir", "kind"], sort=True):
        i = x[~x.oos]; o = x[x.oos]
        rows.append({"平淡分鐘": Q, "平淡幅度≤%": R * 100, "量倍數≥": VM, "動的方向": dirn, "做法": kind + ("多" if (dirn == "上") == (kind == "順勢") else "空"),
                     "筆數": len(x), "日均筆": len(x) / ndays,
                     "30分 每筆淨%": x.r30.mean(), "30分 t": tstat(x[["date", "r30"]]),
                     "收盤 每筆淨%": x.rc.mean(), "收盤 勝率%": (x.rc > 0).mean() * 100, "收盤 t": tstat(x[["date", "rc"]]),
                     "收盤 IS": i.rc.mean(), "t IS": tstat(i[["date", "rc"]]), "收盤 OOS": o.rc.mean(), "t OOS": tstat(o[["date", "rc"]])})
    T = pd.DataFrame(rows)
    P("## 全部組合"); P(T.round(3).to_markdown(index=False)); P()
    best = T.sort_values("收盤 t", ascending=False).head(5)
    P("## 收盤出場 t 最高的 5 個"); P(best.round(3).to_markdown(index=False)); P()
    k = best.iloc[0]; x = X[(X.Q == k["平淡分鐘"]) & np.isclose(X.R, k["平淡幅度≤%"] / 100) & (X.V == k["量倍數≥"]) & (X.dir == k["動的方向"]) & (X.kind == k["做法"][:2])]
    P(f"## 最佳那組各年 (收盤出場, 每筆淨% / 筆數)")
    P(x.assign(year=x.date.dt.year).groupby("year").rc.agg(["mean", "count"]).round(3).T.to_markdown()); P()
    P("## 訊號時間分布 (全部組合合併, 筆數占比%)")
    tb = pd.cut(X.t, [0, 120, 150, 180, 210], labels=["10:00~11:00", "11:00~11:30", "11:30~12:00", "12:00~12:30"]).value_counts(normalize=True).sort_index() * 100
    P(tb.round(1).to_frame("占比%").T.to_markdown()); P()
    (OUT / "quiet_breakout.md").write_text("\n".join(L), encoding="utf-8"); print("\n".join(L))
