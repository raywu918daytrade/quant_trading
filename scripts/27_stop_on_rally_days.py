"""現行 vs 組合停損 −3%, 在「大盤出大利多 / 日內大漲」的日子的對比. 讀 25 存的 portfolio_stop_daily{,_sr}.csv.
大盤日內 = 流動股 (dv20≥1億, 上市櫃) 當天 開→收 的平均. 輸出 stats/stop_on_rally_days.md"""
import sys, numpy as np, pandas as pd
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent; OUT = ROOT/"stats"; CAP = 3_000_000
L = []; P = lambda s="": L.append(str(s))
P("# 現行 vs 組合停損 −3%: 大盤日內大漲 (大利多) 的日子\n")
P("- 大盤日內 = 流動股開盤→收盤平均報酬 (D1)。空單在這種日子最痛。日淨 = 對 300 萬。")
for tag, label in [("", "影子 = 基準模型"), ("_sr", "正式 = 基準+支撐壓力")]:
    f = ROOT/"data/processed"/f"portfolio_stop_daily{tag}.csv"
    if not f.exists(): P(f"\n(缺 {f.name})"); continue
    D = pd.read_csv(f, index_col=0, parse_dates=True)
    cur, stp = D["現行"], D["stop3"]
    P(f"\n## {label}")
    rows = []
    P("### 依『組合盤中最深浮虧』分組 (= 持股同時在盤中上漲的程度; 停損只在 ≤ −3% 的日子會動作)")
    for th in [0.01, 0.02, 0.03, 0.04, 0.05]:
        m = D.intra_mdd <= -th
        rows.append({"組合盤中曾浮虧 ≤": f"−{th*100:.0f}%", "天數": int(m.sum()), "占比%": m.mean()*100, "現行 日淨平均%": cur[m].mean()*100, "停損 日淨平均%": stp[m].mean()*100, "差 (停損−現行)%": (stp[m]-cur[m]).mean()*100,
                     "現行 最差日%": cur[m].min()*100, "停損 最差日%": stp[m].min()*100, "現行 總損益(萬)": cur[m].sum()*CAP/1e4, "停損 總損益(萬)": stp[m].sum()*CAP/1e4, "這些天大盤日內平均%": D.mkt_intra[m].mean()*100})
    P(pd.DataFrame(rows).set_index("組合盤中曾浮虧 ≤").round(3).to_markdown()); P("")
    rows = []
    for th in [0.005, 0.01, 0.015, 0.02, 0.03]:
        m = D.mkt_intra >= th
        rows.append({"大盤日內 ≥": f"+{th*100:.1f}%", "天數": int(m.sum()), "現行 日淨平均%": cur[m].mean()*100, "停損 日淨平均%": stp[m].mean()*100, "差 (停損−現行)%": (stp[m]-cur[m]).mean()*100,
                     "現行 最差日%": cur[m].min()*100, "停損 最差日%": stp[m].min()*100, "現行 總損益(萬)": cur[m].sum()*CAP/1e4, "停損 總損益(萬)": stp[m].sum()*CAP/1e4})
    P("### 輔助: 依大盤日內漲幅 (流動股開→收平均) 分組"); P(pd.DataFrame(rows).set_index("大盤日內 ≥").round(3).to_markdown()); P("")
    m = D.mkt_intra < 0
    P(f"- 對照: 大盤日內下跌的 {int(m.sum())} 天, 現行 {cur[m].mean()*100:.3f}% / 停損 {stp[m].mean()*100:.3f}% (差 {(stp[m]-cur[m]).mean()*100:+.3f}%)；全部 {len(D)} 天 現行 {cur.mean()*100:.3f}% / 停損 {stp.mean()*100:.3f}%")
    top = D.sort_values("intra_mdd").head(15)
    P("\n### 組合盤中被軋最深的 15 天 (持股同時上漲最嚴重)");
    t = pd.DataFrame({"大盤日內%": top.mkt_intra*100, "大盤跳空%": top.mkt_gap*100, "現行 日淨%": top["現行"]*100, "停損 日淨%": top["stop3"]*100, "盤中組合最深浮虧%": top.intra_mdd*100})
    P(t.round(2).to_markdown()); P("")
    worst = D.sort_values("現行").head(15)
    P("### 現行最差的 15 天 (停損版同日表現)")
    t = pd.DataFrame({"現行 日淨%": worst["現行"]*100, "停損 日淨%": worst["stop3"]*100, "大盤日內%": worst.mkt_intra*100, "大盤跳空%": worst.mkt_gap*100})
    P(t.round(2).to_markdown()); P("")
    P(f"- 這 15 天合計: 現行 {worst['現行'].sum()*CAP/1e4:.0f} 萬 / 停損 {worst['stop3'].sum()*CAP/1e4:.0f} 萬")
(OUT/"stop_on_rally_days.md").write_text("\n".join(L)); print("\n".join(L))
