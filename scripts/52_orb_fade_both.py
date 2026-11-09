"""開盤 5 分區間「反著做」(多空都做) vs 順勢. 樣本 = trailing_stop_trades_sr.parquet (sr 模型每日前 10 檔, 2020~2026), M1 逐分鐘.
開盤 5 分 = 09:00~09:04 的高 rh / 低 rl, 09:00 開 o0, 09:04 收 c4. 訊號都在 09:05~12:00 (t 5~180) 第一次出現.
事件 (每檔每天每種事件取第一次):
  UP   向上突破 : 第一根 high > rh, 進場價 = max(rh, 該根開盤)            → 反做 = 空, 順勢 = 多
  DN   向下跌破 : 第一根 low  < rl, 進場價 = min(rl, 該根開盤)            → 反做 = 多 (接刀), 順勢 = 空
  DNR  破底翻   : DN 之後第一根 close > rl (收回區間) → 下一根開盤          → 反做 = 多, 順勢 = 空
  UPR  假突破   : UP 之後第一根 close < rh (跌回區間) → 下一根開盤          → 反做 = 空, 順勢 = 多
  K5   開盤 5 分 K 方向: c4 > o0 (紅 K) / c4 < o0 (黑 K), 09:05 開盤進     → 反做 = 紅空 / 黑多
出場: 收盤競價. 現實結算: 空單到昨收 +8% 強制回補 (+0.2% 滑價)、收盤鎖漲停再 −7%;
      多單對稱 (−8% 強制賣出, 收盤鎖跌停 −7%). 進場前 (開盤區間內) 已到 ±8% 或摸漲跌停的那一邊不做. 成本 0.23%.
對照: 同一檔同一天現行 09:01 進場的每筆淨 (09:01 單%).
分組: 全部 / 開盤跳空 ≥2% / ≤−2% / 平開 |gap|<1% (gap = 09:00 開盤 / 昨收 − 1).
組合: 每天最多 10 檔, 先出訊號先做, 等權, 沒交易日計 0.
-> stats/orb_fade_both.md"""
import os, numpy as np, pandas as pd
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
ROOT = Path(os.environ.get("QT_ROOT", Path(__file__).resolve().parent.parent)); OUT = ROOT / "stats"
COST, SPLIT = 0.0023, 2024
W, DEADLINE, FORCE, SLIP, LOCK, NMAX = 5, 180, 0.08, 0.002, 0.07, 10
KEY = ["stock_id", "date"]

def first_bar(x, mask):
    return x[mask].groupby(KEY, sort=False).head(1).set_index(KEY)

def after_extremes(x, et, name):
    y = x.join(et.rename("et"), on=KEY); y = y[y.et.notna() & (y.t >= y.et)]
    g = y.groupby(KEY, sort=False)
    return pd.DataFrame({f"{name}_hi": g.high.max(), f"{name}_lo": g.low.min()})

def process(args):
    f, keys = args
    df = pd.read_parquet(f, columns=["stock_id", "date", "open", "high", "low", "close", "volume"])
    df["ts"] = pd.to_datetime(df.date); df["date"] = df.ts.dt.normalize()
    df = df.merge(keys, on=KEY, how="inner")
    if df.empty: return None
    df["t"] = (df.ts.dt.hour * 60 + df.ts.dt.minute - 540).astype(np.int16)
    df = df[(df.t >= 0) & (df.t <= 270)].drop_duplicates(KEY + ["t"], keep="last").sort_values(KEY + ["t"]).reset_index(drop=True)
    g = df.groupby(KEY, sort=False); w = df[df.t < W].groupby(KEY, sort=False)
    base = pd.DataFrame({"auc_c": g.close.last(), "rh": w.high.max(), "rl": w.low.min(), "o0": w.open.first(), "c4": w.close.last()})
    base = base[base.rh.notna()]
    x = df.join(base[["rh", "rl"]], on=KEY); x = x[x.rh.notna()]
    win = (x.t >= W) & (x.t <= DEADLINE)
    up = first_bar(x, win & (x.high > x.rh)); dn = first_bar(x, win & (x.low < x.rl))
    base["UP_t"] = up.t; base["UP_px"] = np.maximum(up.rh, up.open)
    base["DN_t"] = dn.t; base["DN_px"] = np.minimum(dn.rl, dn.open)
    for ev, src, cond in [("DNR", "DN_t", lambda y: y.close > y.rl), ("UPR", "UP_t", lambda y: y.close < y.rh)]:
        y = x.join(base[src].rename("bt"), on=KEY); y = y[y.bt.notna() & (y.t > y.bt) & (y.t <= DEADLINE)]
        b = first_bar(y, cond(y))
        nx = x.join(b.t.rename("st"), on=KEY); nx = nx[nx.st.notna() & (nx.t > nx.st)]
        nb = nx.groupby(KEY, sort=False).head(1).set_index(KEY)
        base[f"{ev}_t"] = nb.t; base[f"{ev}_px"] = nb.open
    k5 = first_bar(x, x.t >= W); base["K5_t"] = k5.t; base["K5_px"] = k5.open
    base.loc[base.K5_t > DEADLINE, ["K5_t", "K5_px"]] = np.nan
    for ev in ["UP", "DN", "DNR", "UPR", "K5"]:
        base = base.join(after_extremes(x, base[f"{ev}_t"], ev))
    return base.reset_index()

def settle(s, ev, side):
    """side = +1 多 / −1 空. 回傳 (r %, forced); 進場不合格的回 NaN."""
    ent = s[f"{ev}_px"]; hi = s[f"{ev}_hi"]; lo = s[f"{ev}_lo"]; pc = s.pc
    if side < 0:
        ok = (s.rh < pc * (1 + FORCE)) & (s.rh < pc * 1.1 * 0.999) & (ent < pc * (1 + FORCE))
        forced = hi >= pc * (1 + FORCE); ex = np.where(forced, pc * (1 + FORCE) * (1 + SLIP), s.auc_c)
        locked = (~forced) & (ex >= pc * 1.1 * 0.999); r = -(ex / ent - 1) - COST - np.where(locked, LOCK, 0)
    else:
        ok = (s.rl > pc * (1 - FORCE)) & (s.rl > pc * 0.9 * 1.001) & (ent > pc * (1 - FORCE))
        forced = lo <= pc * (1 - FORCE); ex = np.where(forced, pc * (1 - FORCE) * (1 - SLIP), s.auc_c)
        locked = (~forced) & (ex <= pc * 0.9 * 1.001); r = (ex / ent - 1) - COST - np.where(locked, LOCK, 0)
    return np.where(ok & ent.notna(), r * 100, np.nan), np.asarray(forced)

def tstat(day):
    return day.mean() / day.std() * np.sqrt(len(day)) if len(day) > 2 and day.std() > 0 else np.nan

if __name__ == "__main__":
    R = pd.read_parquet(ROOT / "data/processed/trailing_stop_trades_sr.parquet")
    R = R[["stock_id", "date", "prev_close", "gross|現行 (無移動停利)"]].rename(columns={"prev_close": "pc", "gross|現行 (無移動停利)": "g0"})
    R["date"] = pd.to_datetime(R.date).astype("datetime64[ns]"); R["ym"] = R.date.dt.strftime("%Y_%m")
    jobs = [(str(ROOT / "data/raw/m1" / f"{ym}.parquet"), g[KEY]) for ym, g in R.groupby("ym") if (ROOT / "data/raw/m1" / f"{ym}.parquet").exists()]
    with ProcessPoolExecutor(6) as ex:
        M = pd.concat([b for b in ex.map(process, jobs) if b is not None], ignore_index=True)
    S = R.merge(M, on=KEY, how="inner"); S = S[S.auc_c.notna()].reset_index(drop=True)
    S["gap"] = S.o0 / S.pc - 1; S["oos"] = S.date.dt.year >= SPLIT; S["base"] = (S.g0 - COST) * 100
    ALLDAYS = np.sort(S.date.unique())
    SPECS = [("UP", "向上突破", -1, "空"), ("DN", "向下跌破", +1, "多 (接刀)"), ("DNR", "破底翻 (跌破後收回)", +1, "多"), ("UPR", "假突破 (突破後跌回)", -1, "空")]
    T = []
    def add(ev, kind, side, lab, mask=None):
        r, forced = settle(S, ev, side)
        if mask is not None: r = np.where(mask, r, np.nan)
        T.append(pd.DataFrame({"date": S.date, "stock_id": S.stock_id, "gap": S.gap, "oos": S.oos, "t": S[f"{ev}_t"], "r": r, "forced": forced, "base": S.base, "kind": kind, "label": lab}))
    for ev, lab, fade_side, fade_lab in SPECS:
        add(ev, "反做", fade_side, f"{lab} → {fade_lab}")
        add(ev, "順勢", -fade_side, f"{lab} → {'多' if fade_side < 0 else '空'}")
    red = (S.c4 > S.o0).values; blk = (S.c4 < S.o0).values
    add("K5", "反做", -1, "開盤 5 分紅 K → 空", red); add("K5", "反做", +1, "開盤 5 分黑 K → 多", blk)
    add("K5", "順勢", +1, "開盤 5 分紅 K → 多", red); add("K5", "順勢", -1, "開盤 5 分黑 K → 空", blk)
    T = pd.concat(T, ignore_index=True); T = T[T.r.notna()].reset_index(drop=True)

    GROUPS = [("全部", lambda d: d.gap.notna()), ("開盤跳空 ≥2%", lambda d: d.gap >= 0.02), ("開盤跳空 ≤−2%", lambda d: d.gap <= -0.02), ("平開 |gap|<1%", lambda d: d.gap.abs() < 0.01)]
    def summ(x):
        if len(x) < 50: return None
        i = x[~x.oos]; o = x[x.oos]
        top = x.sort_values(["date", "t"]).groupby("date").head(NMAX).groupby("date").r.mean()
        p = pd.Series(0.0, index=ALLDAYS); p.loc[top.index] = top.values
        return {"筆數": len(x), "占檔日%": len(x) / len(S) * 100, "每筆淨%": x.r.mean(), "勝率%": (x.r > 0).mean() * 100, "強制%": x.forced.mean() * 100,
                "t": tstat(x.groupby("date").r.mean()), "每筆 IS": i.r.mean(), "t IS": tstat(i.groupby("date").r.mean()), "每筆 OOS": o.r.mean(), "t OOS": tstat(o.groupby("date").r.mean()),
                "09:01 單%": x.base.mean(), "組合 日淨%": p.mean(), "Sharpe": p.mean() / p.std() * np.sqrt(250) if p.std() > 0 else np.nan}
    L = []; P = lambda s="": L.append(str(s))
    P(f"# 開盤 5 分區間 反著做 (多空都做) vs 順勢 (sr 每日前 10 檔, {S.date.min():%Y-%m-%d}~{S.date.max():%Y-%m-%d}, {len(S)} 檔日 / {len(ALLDAYS)} 天)"); P()
    P("- 開盤 5 分 = 09:00~09:04 高低; 訊號 09:05~12:00 第一次出現; 收盤競價出場; 成本 0.23%; ±8% 強制出場 (+0.2% 滑價), 收盤鎖漲跌停 −7%")
    P("- 反做 = 漲上去就空、跌下來就多; 順勢 = 反過來")
    P("- 破底翻 = 跌破區間低之後第一根收盤收回區間 → 下一根開盤進; 假突破 = 突破區間高之後第一根收盤跌回區間 → 下一根開盤進")
    P("- t 用每日平均報酬算, |t|>2 才算不是運氣; IS = 2020~2023, OOS = 2024~")
    P("- 09:01 單% = 同一檔同一天現行 09:01 放空的每筆淨 (看訊號是不是挑到本來就好/壞的單); 組合 = 每天最多 10 筆、先出訊號先做、等權、沒交易日計 0")
    P(f"- 參考: 全部候選現行 09:01 放空每筆淨 {S.base.mean():.3f}%"); P()
    order = list(dict.fromkeys(T.label))
    for gname, gf in GROUPS:
        P(f"## {gname}"); rows = []
        for lab in order:
            x = T[T.label == lab]; x = x[gf(x)]; s = summ(x)
            if s: rows.append({"規則": lab, "類": x.kind.iloc[0], **s})
        P(pd.DataFrame(rows).round(3).to_markdown(index=False)); P()
    P("## 各年 每筆淨% (全部)")
    P(T.assign(year=T.date.dt.year).groupby(["label", "year"]).r.mean().unstack().reindex(order).round(2).to_markdown()); P()
    (OUT / "orb_fade_both.md").write_text("\n".join(L), encoding="utf-8"); print("\n".join(L))
