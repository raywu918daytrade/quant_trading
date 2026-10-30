"""類股 (產業別) vs 現行策略 (sr, 10 檔 × 60 萬, 2020~2026) 的每筆/日淨/勝率.
類股: data/raw/meta/info.parquet 的 group (證交所/櫃買產業別), 上市櫃同義名稱合併, 交易數 < 100 的併成「小類合計」.
類股強弱: 用 data/raw/d1 全市場流動股 (前 20 日均成交額 >= 1 億) 算每日類股平均, 前一日漲跌 / 今日跳空 (09:00 可知) / 今日日內.
交易樣本同 36_drawdown_diagnosis (trailing_stop_trades_sr.parquet, 現行出場, 每日只存前 10 名, 所以排除後空出的額度不遞補). 輸出 stats/industry.md"""
import glob, numpy as np, pandas as pd
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent; OUT = ROOT/"stats"; BASE, CAP, PER_STOCK, MAX_N, MIN_N = 0.0023, 3_000_000, 600_000, 10, 100
ALIAS = {"數位雲端類": "數位雲端", "綠能環保類": "綠能環保", "運動休閒類": "運動休閒", "居家生活類": "居家生活", "其他電子類": "其他電子", "其他電子業": "其他電子", "金融業": "金融保險", "農業科技業": "農業科技", "觀光事業": "觀光餐旅"}
ELEC = {"電子工業", "半導體業", "電子零組件業", "電腦及週邊設備業", "通信網路業", "光電業", "其他電子", "電子通路業", "資訊服務業", "數位雲端"}

info = pd.read_parquet(ROOT/"data/raw/meta/info.parquet")[["stock_id","group"]]; info["group"] = info.group.replace(ALIAS)
R = pd.read_parquet(ROOT/"data/processed/trailing_stop_trades_sr.parquet"); R = R[["stock_id","date","entry","prev_close","rank","gross|現行 (無移動停利)"]].rename(columns={"gross|現行 (無移動停利)": "g"}); R["net"] = R.g - BASE
R["date"] = R.date.astype("datetime64[ns]"); R = R.merge(info, on="stock_id", how="left"); R["group"] = R.group.fillna("未知")
cnt = R.group.value_counts(); R["grp"] = np.where(R.group.map(cnt) >= MIN_N, R.group, "小類合計"); R["elec"] = np.where(R.group.isin(ELEC), "電子", "非電子")

d1 = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(str(ROOT/"data/raw/d1/*.parquet")))], ignore_index=True); d1["date"] = pd.to_datetime(d1.date).astype("datetime64[ns]")
d1 = d1.drop_duplicates(["stock_id","date"]).sort_values(["stock_id","date"]).reset_index(drop=True).merge(info, on="stock_id", how="left")
pc = d1.groupby("stock_id").close.shift(1); d1["gap"] = d1.open/pc - 1; d1["intra"] = d1.close/d1.open - 1; d1["ret"] = d1.close/pc - 1
d1["dv20"] = (d1.close*d1.volume).groupby(d1.stock_id).transform(lambda s: s.shift(1).rolling(20).mean())
liq = d1[(d1.dv20 >= 1e8) & d1.group.notna()]
ig = liq.groupby(["group","date"]).agg(ind_gap=("gap","mean"), ind_intra=("intra","mean"), ind_ret=("ret","mean"), ind_n=("gap","size")).reset_index().sort_values(["group","date"])
ig["ind_prev"] = ig.groupby("group").ind_ret.shift(1); ig["ind_prev5"] = ig.groupby("group").ind_ret.transform(lambda s: (1+s).rolling(5).apply(np.prod, raw=True).shift(1) - 1)
mk = liq.groupby("date").agg(mkt_gap=("gap","mean"), mkt_ret=("ret","mean")).reset_index(); mk["mkt_prev"] = mk.mkt_ret.shift(1)
T = R.merge(ig, on=["group","date"], how="left").merge(mk, on="date", how="left")
T["rel_ind_gap"] = T.ind_gap - T.mkt_gap; T["rel_ind_prev"] = T.ind_prev - T.mkt_prev
T["stk_gap"] = T.entry/T.prev_close - 1; T["stk_vs_ind"] = T.stk_gap - T.ind_gap
T["n_same"] = T.groupby(["date","group"]).stock_id.transform("size")

def alloc(S):
    recs = []
    for dt, gg in S.sort_values(["date","rank"]).groupby("date", sort=False):
        cash = CAP; amts = []; nets = []
        for _, r in gg.iterrows():
            lc = r.entry*1000
            if cash < lc: continue
            lots = int(min(PER_STOCK, cash)//lc)
            if lots < 1: continue
            a = lots*lc; cash -= a; amts.append(a); nets.append(r.net)
            if len(amts) >= MAX_N: break
        recs.append((dt, len(amts), sum(a*n for a, n in zip(amts, nets))/CAP))
    return pd.DataFrame(recs, columns=["date","n","ret"]).set_index("date")
DAYS = pd.DatetimeIndex(sorted(R.date.unique()))
def st(name, S, days=DAYS):
    s = alloc(S).ret.reindex(days, fill_value=0.0); mo = s.groupby(s.index.to_period("M")).apply(lambda x: ((1+x).prod()-1)*100); eq = (1+s).cumprod(); dd = eq/eq.cummax()-1
    return {"規則": name, "筆數": len(S), "每筆淨%": S.net.mean()*100, "日淨%": s.mean()*100, "日勝率": (s[s != 0] > 0).mean(), "Sharpe": s.mean()/s.std()*np.sqrt(245), "MDD%": dd.min()*100, "最差月%": mo.min(), "年化(萬)": s.mean()*245*CAP/1e4}
def grp(by, X=T):
    g = X.groupby(by, observed=False)
    return pd.DataFrame({"筆數": g.size(), "占比%": g.size()/len(X)*100, "每筆勝率": g.net.apply(lambda s: (s > 0).mean()), "每筆淨%": g.net.mean()*100, "中位數%": g.net.median()*100, "標準差%": g.net.std()*100,
                         "最差筆%": g.net.min()*100, "合計淨(單位%)": g.net.sum()*100, "t值": g.net.apply(lambda s: s.mean()/s.std()*np.sqrt(len(s)) if len(s) > 1 else np.nan)})

L = []; P = lambda s="": L.append(str(s))
P("# 類股 (產業別) vs 現行策略 (2020~2026, sr 10 檔 × 60 萬, 做空當沖)\n")
P(f"- 交易 {len(T)} 筆 / {T.date.nunique()} 天，整體每筆勝率 {(T.net > 0).mean():.3f}、每筆淨 {T.net.mean()*100:.3f}%")
P(f"- 類股來源: info.parquet 的產業別 (上市櫃同義名稱合併)，交易數 < {MIN_N} 筆的類股併成「小類合計」；對不到類股 {(T.group == '未知').sum()} 筆")
P(f"- 類股強弱指標以流動股 (20 日均額 ≥ 1 億) 平均計算，對得到的交易 {T.ind_gap.notna().mean()*100:.1f}%")
P("- 每筆淨 = 做空毛報酬 − 0.23% 成本；t值 = 每筆淨平均 / 標準誤，|t| < 2 代表跟整體差不多是雜訊")

P("\n## 1. 各類股的每筆績效 (依筆數排序)")
G = grp("grp"); P(G.sort_values("筆數", ascending=False).round(3).to_markdown())
P("\n### 電子 vs 非電子"); P(grp("elec").round(3).to_markdown())
P("\n### 上市 vs 上櫃 (參考)")
mt = pd.read_parquet(ROOT/"data/raw/meta/info.parquet")[["stock_id","market_type"]]; P(grp("market_type", T.merge(mt, on="stock_id", how="left")).round(3).to_markdown())

P("\n## 2. 分年穩定度: 主要類股每筆淨平均% (括號 = 筆數)")
top = G[G.index != "小類合計"].sort_values("筆數", ascending=False).index[:12]
Y = T[T.grp.isin(top)].groupby(["grp", T.date.dt.year]).net.agg(["mean","size"]); Yt = Y.apply(lambda r: f"{r['mean']*100:.2f} ({int(r['size'])})", axis=1).unstack()
P(Yt.reindex(top).to_markdown())

P("\n## 3. 同一天同類股的檔數 (類股集中)")
P(grp(pd.cut(T.n_same, [0, 1, 2, 3, 5, 10], labels=["1 檔","2 檔","3 檔","4~5 檔","6 檔以上"])).round(3).to_markdown())
dd = T.groupby("date").agg(maxsame=("n_same","max")); s0 = alloc(T).ret.reindex(DAYS, fill_value=0.0); dd = dd.join(s0.rename("ret"))
P("\n以「當天最多幾檔同類股」分日:"); g = dd.groupby(pd.cut(dd.maxsame, [0, 2, 3, 4, 6, 10], labels=["≤2","3","4","5~6","7+"]), observed=False)
P(pd.DataFrame({"天數": g.size(), "日淨平均%": g.ret.mean()*100, "日勝率": g.ret.apply(lambda s: (s > 0).mean()), "最差日%": g.ret.min()*100}).round(3).to_markdown())

P("\n## 4. 類股強弱 (做空的這檔所屬類股) 分組")
for col, bins, labs, title in [("ind_prev", [-1, -.02, -.01, 0, .01, .02, .04, 1], ["<−2%","−2~−1%","−1~0%","0~1%","1~2%","2~4%",">4%"], "類股前一日漲跌"),
                               ("ind_prev5", [-1, -.05, 0, .05, .10, 1], ["<−5%","−5~0%","0~5%","5~10%",">10%"], "類股前 5 日累計"),
                               ("ind_gap", [-1, -.01, 0, .005, .01, .02, 1], ["<−1%","−1~0%","0~0.5%","0.5~1%","1~2%",">2%"], "類股今日跳空 (09:00 可知)"),
                               ("rel_ind_gap", [-1, -.01, -.003, .003, .01, 1], ["弱於大盤 >1%","弱 0.3~1%","±0.3%","強 0.3~1%","強於大盤 >1%"], "類股跳空 − 大盤跳空"),
                               ("stk_vs_ind", [-1, 0, .01, .02, .04, 1], ["<0","0~1%","1~2%","2~4%",">4%"], "個股跳空 − 類股跳空"),
                               ("ind_intra", [-1, -.02, -.01, 0, .01, 1], ["<−2%","−2~−1%","−1~0%","0~1%",">1%"], "類股今日日內 (事後, 僅解釋用)")]:
    P(f"\n### {title}"); P(grp(pd.cut(T[col], bins, labels=labs)).round(3).to_markdown())

P("\n## 5. 候選規則的整體效果 (排除的名額不遞補, 資金閒置)")
TR, TE = T[T.date.dt.year <= 2023], T[T.date.dt.year >= 2024]
gtr = TR.groupby("grp").net.agg(["mean","size"]); bad = gtr[(gtr["size"] >= 50) & (gtr["mean"] < TR.net.mean() - .002)].sort_values("mean").index.tolist()
P(f"- 用 2020~2023 選出「每筆淨比整體低 0.2% 以上」的類股: {bad or '無'}；下表 2024~ 欄是樣本外結果")
rows = []
for name, keep in [("現行", T.net.notna()), ("排除航運業", T.group != "航運業"), ("排除生技醫療 (含化學生技)", ~T.group.isin(["生技醫療業","化學生技醫療"])), ("只做電子", T.elec == "電子"), ("只做非電子", T.elec == "非電子"),
                   (f"排除 2020~23 較差類股", ~T.grp.isin(bad)), ("同類股最多 3 檔", None),
                   ("類股跳空 > 2% 不做", ~(T.ind_gap > .02)), ("類股前日 > +4% 不做", ~(T.ind_prev > .04)), ("個股跳空 − 類股跳空 < 0 不做", ~(T.stk_vs_ind < 0))]:
    if name == "同類股最多 3 檔":
        o = T.sort_values(["date","rank"]); keep = (o.groupby(["date","group"]).cumcount() < 3).reindex(T.index)
    r = st(name, T[keep]); r["2024~ 日淨%"] = st(name, TE[keep.loc[TE.index]], DAYS[DAYS.year >= 2024])["日淨%"]; r["2020~23 日淨%"] = st(name, TR[keep.loc[TR.index]], DAYS[DAYS.year <= 2023])["日淨%"]; rows.append(r)
P(pd.DataFrame(rows).set_index("規則").round(3).to_markdown())
(OUT/"industry.md").write_text("\n".join(L)); print("\n".join(L))
