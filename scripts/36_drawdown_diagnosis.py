"""回撤來源診斷 + 兩個降回撤/提勝率的候選: (1) 大盤狀態濾網 (2) 等風險配置 (每檔金額 ∝ 1/ATR20).
日報酬序列 = 現行 (sr, 10 檔 × 60 萬, 300 萬). 大盤狀態用流動股 (dv20 ≥ 1 億) 的 D1 平均. 輸出 stats/drawdown_diagnosis.md"""
import glob, numpy as np, pandas as pd
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent; OUT = ROOT/"stats"; BASE, CAP, PER_STOCK, MAX_N = 0.0023, 3_000_000, 600_000, 10
R = pd.read_parquet(ROOT/"data/processed/trailing_stop_trades_sr.parquet"); R = R[["stock_id","date","entry","prev_close","rank","gross|現行 (無移動停利)"]].rename(columns={"gross|現行 (無移動停利)": "g"}); R["net"] = R.g - BASE
d1 = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(str(ROOT/"data/raw/d1/*.parquet")))], ignore_index=True); d1["date"] = pd.to_datetime(d1.date); d1 = d1.drop_duplicates(["stock_id","date"]).sort_values(["stock_id","date"]).reset_index(drop=True)
g = d1.groupby("stock_id"); pc = g.close.shift(1)
tr = pd.concat([d1.high-d1.low, (d1.high-pc).abs(), (d1.low-pc).abs()], axis=1).max(axis=1); d1["atr20"] = tr.groupby(d1.stock_id).transform(lambda s: s.shift(1).rolling(20).mean())/pc
d1["ret"] = d1.close/pc - 1; d1["gap"] = d1.open/pc - 1; d1["intra"] = d1.close/d1.open - 1; d1["dv20"] = (d1.close*d1.volume).groupby(d1.stock_id).transform(lambda s: s.shift(1).rolling(20).mean())
liq = d1[d1.dv20 >= 1e8]; mk = liq.groupby("date").agg(mkt_ret=("ret","mean"), mkt_gap=("gap","mean"), mkt_intra=("intra","mean"), breadth=("ret", lambda s: (s > 0).mean())).sort_index()
mk["mkt_pret"] = mk.mkt_ret.shift(1); mk["mkt_ret5"] = mk.mkt_ret.rolling(5).sum().shift(1); mk["mkt_ret20"] = mk.mkt_ret.rolling(20).sum().shift(1); mk["p_breadth"] = mk.breadth.shift(1)
mk["mkt_vol20"] = mk.mkt_ret.rolling(20).std().shift(1); up = (mk.mkt_ret > 0).astype(int); mk["up_streak"] = up.groupby((up != up.shift()).cumsum()).cumsum().shift(1).where(up.shift(1) == 1, 0)
S = R.merge(d1[["stock_id","date","atr20"]], on=["stock_id","date"], how="left")
all_days = pd.Series(sorted(S.date.unique()))
def alloc(S, weight=None, mask=None):
    recs = []
    for dt, gg in (S if mask is None else S[mask]).sort_values(["date","rank"]).groupby("date", sort=False):
        cash = CAP; amts = []; nets = []
        if weight is not None:
            w = weight(gg); w = w/w.sum()*CAP*1.25   # 放大 25% 讓零股湊整後仍能用滿 300 萬
        for i, (_, r) in enumerate(gg.iterrows()):
            lc = r.entry*1000; cap_i = PER_STOCK if weight is None else w.iloc[i]
            if cash < lc: continue
            lots = int(min(cap_i, cash)//lc)
            if lots < 1: continue
            a = lots*lc; cash -= a; amts.append(a); nets.append(r.net)
            if len(amts) >= MAX_N: break
        recs.append((dt, len(amts), sum(amts)/CAP, sum(a*n for a, n in zip(amts, nets))/CAP))
    return pd.DataFrame(recs, columns=["date","n","util","ret"]).set_index("date").reindex(all_days).fillna({"n": 0, "util": 0, "ret": 0.0})
def st(name, s, D=None):
    mo = s.groupby(s.index.to_period("M")).apply(lambda x: ((1+x).prod()-1)*100); eq = (1+s).cumprod(); dd = eq/eq.cummax()-1
    return {"規則": name, "日淨%": s.mean()*100, "日勝率": (s[s != 0] > 0).mean(), "日淨標準差%": s.std()*100, "Sharpe": s.mean()/s.std()*np.sqrt(245), "MDD%": dd.min()*100, "最差日%": s.min()*100, "最差月%": mo.min(), "負月比例": (mo < 0).mean(), "交易日占比%": (s != 0).mean()*100, "資金使用率%": (D.util.mean()*100 if D is not None else np.nan), "年化(萬)": s.mean()*245*CAP/1e4}
L = []; P = lambda s="": L.append(str(s))
D0 = alloc(S); s0 = D0.ret; eq = (1+s0).cumprod(); dd = eq/eq.cummax()-1
P("# 回撤來源診斷 (2020~2026, 現行 sr 10 檔 × 60 萬)\n")
P(f"- 每筆勝率 {(S.net > 0).mean():.3f}；日勝率 {(s0[s0 != 0] > 0).mean():.3f}；月勝率 {1-(s0.groupby(s0.index.to_period('M')).apply(lambda x: (1+x).prod()-1) < 0).mean():.3f}；MDD {dd.min()*100:.1f}%")
P("\n## 1. 最深的 6 段回撤")
eps = []; in_dd = False
for dt, v in dd.items():
    if v < 0 and not in_dd: in_dd = True; start = dt; trough = dt; tv = v
    elif v < 0 and v < tv: trough = dt; tv = v
    if v == 0 and in_dd: in_dd = False; eps.append((start, trough, dt, tv))
if in_dd: eps.append((start, trough, dd.index[-1], tv))
E = pd.DataFrame(eps, columns=["開始","谷底","回復","深度"]).sort_values("深度").head(6)
rows = []
for _, e in E.iterrows():
    seg = s0[(s0.index >= e.開始) & (s0.index <= e.谷底)]; m = mk.reindex(seg.index)
    rows.append({"開始": e.開始.date(), "谷底": e.谷底.date(), "回復": e.回復.date(), "深度%": e.深度*100, "下跌天數": len(seg), "期間最差日%": seg.min()*100, "負日數": int((seg < 0).sum()), "大盤期間累計%": m.mkt_ret.sum()*100, "大盤日內平均%": m.mkt_intra.mean()*100, "大盤前20日累計(起點)%": mk.mkt_ret20.get(e.開始, np.nan)*100})
P(pd.DataFrame(rows).round(2).to_markdown(index=False))
worst = s0.sort_values().head(10); m = mk.reindex(worst.index)
P("\n## 2. 最差的 10 天跟當天大盤"); P(pd.DataFrame({"日淨%": worst*100, "大盤跳空%": m.mkt_gap*100, "大盤日內%": m.mkt_intra*100, "大盤前1日%": m.mkt_pret*100, "大盤前5日%": m.mkt_ret5*100, "前日上漲家數比": m.p_breadth, "連漲天數": m.up_streak}).round(2).to_markdown())
P("\n## 3. 日淨 vs 前一日可知的大盤狀態 (找有沒有該休息的日子)")
M = mk.reindex(s0.index); df = pd.DataFrame({"ret": s0, "trade": s0 != 0}).join(M)
for col, bins, labs, title in [("mkt_pret", [-1, -.02, -.01, 0, .01, .02, 1], ["<−2%","−2~−1%","−1~0%","0~1%","1~2%",">2%"], "大盤前 1 日"), ("mkt_ret5", [-1, -.04, -.02, 0, .02, .04, 1], ["<−4%","−4~−2%","−2~0%","0~2%","2~4%",">4%"], "大盤前 5 日累計"),
                               ("mkt_ret20", [-1, -.08, -.04, 0, .04, .08, 1], ["<−8%","−8~−4%","−4~0%","0~4%","4~8%",">8%"], "大盤前 20 日累計"), ("mkt_gap", [-1, -.01, -.005, 0, .005, .01, .02, 1], ["<−1%","−1~−0.5%","−0.5~0%","0~0.5%","0.5~1%","1~2%",">2%"], "大盤今日跳空 (09:00 可知)"),
                               ("p_breadth", [0, .3, .4, .5, .6, .7, 1], ["<30%","30~40%","40~50%","50~60%","60~70%",">70%"], "前日上漲家數比"), ("up_streak", [-1, 0, 1, 2, 3, 99], ["0","1","2","3","≥4"], "大盤連漲天數"), ("mkt_vol20", [0, .006, .008, .010, .013, 1], ["<0.6%","0.6~0.8%","0.8~1.0%","1.0~1.3%",">1.3%"], "大盤 20 日波動")]:
    b = pd.cut(df[col], bins, labels=labs); t = df[df.trade]
    T = pd.DataFrame({"交易日數": t.groupby(b[df.trade]).size(), "日淨平均%": t.ret.groupby(b[df.trade]).mean()*100, "日勝率": (t.ret > 0).groupby(b[df.trade]).mean(), "最差日%": t.ret.groupby(b[df.trade]).min()*100, "合計(萬)": t.ret.groupby(b[df.trade]).sum()*CAP/1e4})
    P(f"\n### {title}"); P(T.round(3).to_markdown())
P("\n## 4. 候選規則的整體效果")
rows = [st("現行", s0, D0)]
for name, cond in [("大盤前 1 日 > +1.5% 不做", M.mkt_pret > .015), ("大盤前 1 日 > +2% 不做", M.mkt_pret > .02), ("大盤前 5 日 > +4% 不做", M.mkt_ret5 > .04), ("大盤連漲 ≥ 3 天不做", M.up_streak >= 3), ("前日上漲家數比 > 70% 不做", M.p_breadth > .7), ("大盤今日跳空 > +1% 不做", M.mkt_gap > .01), ("大盤今日跳空 > +1.5% 不做", M.mkt_gap > .015), ("大盤 20 日波動 > 1.3% 不做", M.mkt_vol20 > .013)]:
    s = s0.where(~cond.fillna(False), 0.0); rows.append(st(name, s))
Dw = alloc(S, weight=lambda gg: 1/gg.atr20.fillna(gg.atr20.median()).clip(lower=.02)); rows.append(st("等風險配置 (金額 ∝ 1/ATR20)", Dw.ret, Dw))
Dw2 = alloc(S, weight=lambda gg: (1/gg.atr20.fillna(gg.atr20.median()).clip(lower=.02)).clip(upper=(1/gg.atr20.fillna(gg.atr20.median()).clip(lower=.02)).median()*1.5)); rows.append(st("等風險配置 (上限 1.5× 中位)", Dw2.ret, Dw2))
P(pd.DataFrame(rows).set_index("規則").round(3).to_markdown())
(OUT/"drawdown_diagnosis.md").write_text("\n".join(L)); print("\n".join(L))
