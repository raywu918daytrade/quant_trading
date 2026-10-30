"""類股分散 vs 現行策略 (sr, 前 10 名, 300 萬, 2020~2026).
每類股限 k 檔 (同類股依 rank 留前 k), 兩種資金處理: 名額閒置 (每檔仍 60 萬) / 資金攤平 (300 萬平均攤給留下的檔數);
類股等權: 每個類股分到 300 萬 / 類股數, 類股內再平均. 每檔仍以整張計. 類股定義同 41_industry. 輸出 stats/industry_diversify.md"""
import numpy as np, pandas as pd
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent; OUT = ROOT/"stats"; BASE, CAP, PER_STOCK = 0.0023, 3_000_000, 600_000
ALIAS = {"數位雲端類": "數位雲端", "綠能環保類": "綠能環保", "運動休閒類": "運動休閒", "居家生活類": "居家生活", "其他電子類": "其他電子", "其他電子業": "其他電子", "金融業": "金融保險", "農業科技業": "農業科技", "觀光事業": "觀光餐旅"}
info = pd.read_parquet(ROOT/"data/raw/meta/info.parquet")[["stock_id","group"]]; info["group"] = info.group.replace(ALIAS)
R = pd.read_parquet(ROOT/"data/processed/trailing_stop_trades_sr.parquet"); R = R[["stock_id","date","entry","rank","gross|現行 (無移動停利)"]].rename(columns={"gross|現行 (無移動停利)": "g"}); R["net"] = R.g - BASE
R["date"] = R.date.astype("datetime64[ns]"); R = R.merge(info, on="stock_id", how="left"); R["group"] = R.group.fillna("未知")
R = R.sort_values(["date","rank"]).reset_index(drop=True); R["k_in_grp"] = R.groupby(["date","group"]).cumcount()
DAYS = pd.DatetimeIndex(sorted(R.date.unique()))

def run(S, target):
    """target(gg) -> 每檔目標金額 (Series, 與 gg 同 index); 依 rank 下單, 整張, 現金不足就跳過."""
    recs = []
    for dt, gg in S.groupby("date", sort=False):
        tg = target(gg); cash = CAP; amt = 0.0; pnl = 0.0; n = 0
        for j, (i, r) in enumerate(gg.iterrows()):
            t = cash/(len(gg)-j) if tg is None else tg[i]   # None = 剩餘現金平均分給剩下的檔數 (買不起的額度往後遞延)
            lc = r.entry*1000; lots = int(min(t, cash)//lc)
            if lots < 1: continue
            a = lots*lc; cash -= a; amt += a; pnl += a*r.net; n += 1
        recs.append((dt, n, amt/CAP, pnl/CAP))
    return pd.DataFrame(recs, columns=["date","n","used","ret"]).set_index("date").reindex(DAYS).fillna({"n": 0, "used": 0, "ret": 0.0})
fixed = lambda gg: pd.Series(PER_STOCK, index=gg.index)
spread = lambda gg: None
def by_grp(gg):
    ng = gg.group.nunique(); return CAP/ng/gg.groupby("group").stock_id.transform("size")
def wt(S, how):
    """不計整張的理想權重 (資金 100% 用完): eq = 留下的檔數等權; grp = 類股等權, 類股內平均."""
    w = 1/S.groupby("date").stock_id.transform("size") if how == "eq" else 1/S.groupby("date").group.transform("nunique")/S.groupby(["date","group"]).stock_id.transform("size")
    return (w*S.net).groupby(S.date).sum().reindex(DAYS, fill_value=0.0)

def st(name, D):
    s = D.ret; mo = s.groupby(s.index.to_period("M")).apply(lambda x: ((1+x).prod()-1)*100); eq = (1+s).cumprod(); dd = eq/eq.cummax()-1; s4 = s[s.index.year >= 2024]
    return {"方案": name, "平均檔數": D.n.mean(), "資金使用%": D.used.mean()*100, "日淨%": s.mean()*100, "日標準差%": s.std()*100, "日勝率": (s[s != 0] > 0).mean(), "Sharpe": s.mean()/s.std()*np.sqrt(245),
            "MDD%": dd.min()*100, "最差日%": s.min()*100, "最差月%": mo.min(), "負月比例": (mo < 0).mean(), "年化(萬)": s.mean()*245*CAP/1e4, "2024~ 日淨%": s4.mean()*100, "2024~ Sharpe": s4.mean()/s4.std()*np.sqrt(245)}

rows = [st("現行 (每檔 60 萬)", run(R, fixed))]
for k in [1, 2, 3]:
    S = R[R.k_in_grp < k]; rows.append(st(f"每類股最多 {k} 檔 · 名額閒置", run(S, fixed))); rows.append(st(f"每類股最多 {k} 檔 · 資金攤平", run(S, spread)))
rows.append(st("類股等權 (每類股同額, 類股內平均)", run(R, by_grp)))
T = pd.DataFrame(rows).set_index("方案")
W = [("前 10 名等權", wt(R, "eq"))] + [(f"每類股最多 {k} 檔 · 等權", wt(R[R.k_in_grp < k], "eq")) for k in [1, 2, 3]] + [("類股等權", wt(R, "grp"))]
TW = pd.DataFrame([st(n, pd.DataFrame({"ret": s, "n": np.nan, "used": 1.0})) for n, s in W]).set_index("方案").drop(columns=["平均檔數","資金使用%"])

ng = R.groupby("date").group.nunique(); mx = R.groupby("date").k_in_grp.max()+1; base = run(R, fixed).ret
L = []; P = lambda s="": L.append(str(s))
P("# 類股分散投資 vs 現行策略 (2020~2026, sr 前 10 名, 300 萬, 做空當沖)\n")
P(f"- {len(DAYS)} 個交易日。每天前 10 名平均橫跨 {ng.mean():.1f} 個類股，當天最多同類股平均 {mx.mean():.1f} 檔；最多同類股 ≥ 4 檔的天數占 {(mx >= 4).mean()*100:.0f}%")
P("- 回測只存每天前 10 名，所以限檔後沒有第 11 名以後遞補。「名額閒置」= 砍掉的 60 萬不用；「資金攤平」= 剩餘現金平均分給剩下的檔數 (單檔可超過 60 萬)；1b 用不計整張的理想權重排除整張誤差")
P("- 每檔以整張計；日淨 = 當天損益 / 300 萬\n")
P("## 1. 各方案整體績效"); P(T.round(3).to_markdown())
P("\n## 1b. 理想權重 (不計整張、資金全用完, 純看分散效果)"); P(TW.round(3).to_markdown())
P("\n## 2. 依當天類股集中度分日 (現行)")
g = base.groupby(pd.cut(mx.reindex(DAYS), [0, 2, 3, 4, 10], labels=["最多 ≤2 檔同類","3 檔","4 檔","5 檔以上"]), observed=False)
P(pd.DataFrame({"天數": g.size(), "日淨平均%": g.mean()*100, "日標準差%": g.std()*100, "日勝率": g.apply(lambda s: (s > 0).mean()), "最差日%": g.min()*100}).round(3).to_markdown())
P("\n## 3. 分年日淨% (現行 vs 每類股最多 2 檔·資金攤平 vs 類股等權)")
Y = pd.DataFrame({"現行": base, "最多 2 檔·攤平": run(R[R.k_in_grp < 2], spread).ret, "類股等權": run(R, by_grp).ret})
P((Y.groupby(Y.index.year).mean()*100).round(3).to_markdown())
P("\n分年 Sharpe:"); P(Y.groupby(Y.index.year).apply(lambda d: d.mean()/d.std()*np.sqrt(245)).round(2).to_markdown())
(OUT/"industry_diversify.md").write_text("\n".join(L)); print("\n".join(L))
