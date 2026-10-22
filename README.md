# quant_trading — 台股當沖量化分析 (D1 + M1)

資料來源: HF dataset `raywu918daytrade/daytrade` 的 `db/d1`、`db/m1`，抓 2023-10 ~ 2026-10 (最近 3 年)。

## 目錄
```
data/raw/{d1,m1,meta}/        原始 parquet (月檔; m1 共 1.2GB)
data/processed/
  d1.parquet                  D1 合併 (160 萬列, 2336 檔)
  d1_features.parquet         D1 + 報酬/跳空/日內/振幅/滾動統計  (注意: dv20/vol20 含當日, 建模時另行 shift)
  m1_daily/YYYY_MM.parquet    M1 彙總成每檔每日: 開盤 5/15/30/60 分 OHLCV、VWAP、realized var、日高低時間、13:30 競價...
  m1_profile/YYYY_MM.parquet  每分鐘聚合 (量、|報酬|、振幅)
  m1d_features_{all,liquid}.parquet   M1 日特徵 + D1 合併
  dataset_A.parquet / dataset_B.parquet   訓練資料集
  preds_A.parquet / preds_B.parquet       walk-forward out-of-sample 預測
scripts/01~08                 依序執行即可重現
stats/                        統計摘要 (md) + 圖 (png)
models/                       各季 LightGBM 模型
```

## 執行順序
```bash
python3 scripts/01_download.py      # 下載 (已完成, 會跳過已存在檔案)
python3 scripts/02_build_d1.py
python3 scripts/03_stats_d1.py      # -> stats/d1_summary.md
python3 scripts/04_build_m1.py      # 4 process, ~1 分鐘
python3 scripts/05_stats_m1.py      # -> stats/m1_summary.md
python3 scripts/06_build_dataset.py
python3 scripts/07_train.py A       # -> stats/model_A_report.md
python3 scripts/07_train.py B       # -> stats/model_B_report.md
python3 scripts/08_baseline_compare.py   # -> stats/model_vs_baseline.md
```

## 資料備註
- D1 volume 單位 = 股；M1 volume 單位 = 張 (×1000 ≈ D1，D1 略多約 1%，應含盤後/零股)。
- M1 只有有成交的分鐘才有 bar；09:00 bar 含開盤競價、13:30 bar 為收盤競價；13:30 之後偶有零星 bar (1.6% 的股票日)，已排除。
- market_type: twse / tpex / emerging(興櫃, 無漲跌幅限制, 統計與建模均排除)。
- 流動性 universe: 前 20 日均成交金額 ≥ 1 億元, 上市+上櫃, 非 ETF, 股價 ≥ 5 元 → 約 1200 檔。

## 主要統計發現 (詳見 stats/*.md)
1. **隔夜溢酬 / 盤中回落**: 跳空平均 +0.22%，開→收平均 -0.16% (流動性股 -0.26%)。3 年每一年、每個時段 (09:00-09:30 … 13:00-13:30) 都是負的。
2. **跳空回補**: corr(跳空, 日內報酬) = -0.14。跳空 >+3% 的股票當日日內平均 -0.9%~-1.5%；跳空 <-3% 則 +0.8%~+1.8%。
3. **漲停續航**: 漲停隔日平均 +2.1%，幾乎全在隔日跳空 (+2.4%)，日內反而 -0.3%。
4. **開盤區間**: 前 30 分鐘走完全日振幅的 2/3 (中位數)，59% 的日高、45% 的日低落在前 30 分；24% 的日子開盤價就是最高價。
5. **量能 U 型**: 09:00-09:05 占全日量 9%，13:30 收盤競價占 2.9% (單一分鐘最大)。
6. **開盤動能是反轉不是延續**: 開盤 30 分報酬與其後報酬 rank-IC ≈ -0.02；開盤 30 分放量 (>30% 全日量) 者之後平均 -0.55%。
7. **波動可預測但方向難**: vol20 與隔日 |報酬| corr 0.35；量比與隔日 |報酬| corr 0.14；日報酬 lag-1 自相關僅 +0.05。

## 模型 (LightGBM, 按季 walk-forward 2024Q4~2026Q4, 499 個測試日)
| | 模型 A: 09:30 進→13:30 出 (D1+開盤30分 M1) | 模型 B: 09:00 進→13:30 出 (D1+跳空) |
|---|---|---|
| 日均 rank-IC | 0.109 (IR 0.61, P>0 73%) | 0.120 (IR 0.83, P>0 79%) |
| 十分位 1→10 平均報酬 | -0.60% → -0.01% | -0.73% → +0.15% |
| Short 最看空 10% 日淨報酬 (扣 0.23%) | +0.35% (Sharpe 3.7) | +0.48% (Sharpe 4.4) |
| L/S 10% 日淨報酬 | +0.06% | +0.20% (Sharpe 4.2) |
| 純多 top5% 日淨報酬 | -0.22% | -0.01% |
| 與 -vol60 排序的相關 | 0.72 (大部分是波動排序) | 0.34 (有波動以外的資訊, 同波動分組內 IC 仍 0.10) |

**解讀與警告**
- 幾乎所有十分位的日內報酬都是負的 → 這個市場 (這 3 年) 的 alpha 在**空方**。純多當沖扣成本後平均賠錢。
- 模型 A 大部分只是在做「波動排序」(-vol60 單變數 IC 就有 0.105)；模型 B 因為有跳空回補效果，比波動排序多出明顯資訊。
- 回測假設: 每日等權、09:30 (或 09:00 開盤價) 成交、13:30 收盤價出、來回成本 0.23%、不考慮滑價/券源/處置股/漲跌停無法成交/先賣後買資格。空方策略在台股受借券與現沖券賣資格限制, 實際可執行性要另外驗證。
- 時間範圍只有 3 年且 2024-2026 為同一市場環境, 樣本外有 regime 風險。

## 下一步建議
- 加入 `tick_universe` 的當沖資格 (day_trade_tier) 過濾, 只回測可現沖券賣的股票。
- 加滑價模型 (用 M1 的 09:30 後幾根 bar 的 VWAP 當成交價而非最後一筆)。
- 嘗試 rank / 分類目標、或直接以 Sharpe-aware 的權重; 並做 2020-2023 的 M1 擴展以增加樣本。

## 當沖池 (2026-10-23 新增)
每日動態選池 (只用前一日止的資料): 20 日均成交額 ≥ 5 億 且 ATR20 ≥ 3%，上市櫃非 ETF，股價 ≥ 5 → 每天約 150 檔 (p10 109 / p90 246)。
`scripts/09_daytrade_pool.py` 產生 `dataset_{A,B}_pool.parquet` 與 `stats/daytrade_pool.md`；`07_train.py A pool`、`07_train.py B pool`、`08_baseline_compare.py pool` 產生池版模型報告。

| 池內 (09:00→收) | 全體流動股 | 當沖池 |
|---|---|---|
| 跳空平均 | +0.36% | +0.48% |
| 日內平均 / 中位數 | -0.27% / -0.37% | -0.33% / -0.50% |
| 日內 std | 2.96% | 3.47% |

- 效應隨 ATR 單調放大: ATR 3-4% → -0.23%，>7% → -0.49%；成交額大小影響不大。
- 池內跳空 >+7% 日內 -1.1%；跳空 <-5% 日內 +1.4% (但中位數 0, 靠少數大反彈)。
- 今日大盤平均跳空 <-1% 時，池內日內 +0.36%（唯一穩定為正的條件）；>+1% 時 -0.47%。
- 開盤 30 分相對量 >50% 日均量者，之後 -0.54%；開盤 30 分漲 >5% 且放量者 -0.76%。
- 池版模型: A_pool IC 0.072 (≈波動排序, 與 -vol60 相關 0.46)；B_pool IC 0.077，與 -vol60 相關僅 0.15，L/S 10% 日淨 +0.15% (Sharpe 2.7)，short bottom 10% 日淨 +0.44% (Sharpe 3.3)。池內純多仍為負。

## 做空策略回測 (2026-10-23)
`scripts/10_build_entry_prices.py` 從 raw M1 算 09:00/09:30/10:00 三分鐘 VWAP 進場價與進場後極值；`scripts/11_short_strategy.py` → `stats/short_strategy.md`。
假設: 池內、進場 VWAP、13:30 收盤競價出場、成本 0.23%、排除跌停價放空；期間切成探索/驗證/鎖定三段 (模型預測本身已是 walk-forward 樣本外)。

| 策略 (每日) | 日淨% | Sharpe | MDD | 三段期間日淨% | 觸漲停% |
|---|---|---|---|---|---|
| S1 模型B 最看空 10 檔 09:00 | 0.75 | 5.1 | -13% | 0.75 / 0.72 / 0.78 | 21 |
| S8 模型B 底 20% ∩ 跳空>0 ∩ 大盤跳空>-0.5% | 0.62 | 5.4 | -8% | 0.63 / 0.53 / 0.73 | 19 |
| S9 模型B 最看空 20 檔, 大盤跳空<-1% 不做 | 0.66 | 5.3 | -11% | 0.62 / 0.63 / 0.76 | 17 |
| S7 基準: ATR 最高 20 檔 | 0.61 | 4.0 | -16% | 0.72 / 0.47 / 0.67 | 13 |
| S5 規則: 跳空>+3% | 0.76 | 4.2 | -13% | 1.10 / 0.54 / 0.65 | 38 |

結論與風險:
- 三段期間 (含鎖定期) 全部為正、NW-t 皆 >3；成本提高到 0.40% 仍有 0.44~0.58%/日。
- 但 ATR 基準 (S7) 就有 0.61%/日 → 大部分報酬是「放空高波動股的日內風險溢酬」，模型加的是穩定度 (Sharpe 4→5.4、MDD 16%→8%)，不是魔法。
- 進場後觸及漲停的交易占 16~20%，這些交易平均 -2.9%；回測假設能以收盤價回補，實際漲停鎖死可能回補不到，這是最大未量化風險。個股跳空 >3% 的交易觸漲停率 50%。
- 固定 % 停損會毀掉策略 (2% 停損: 日淨 0.61→0.19)：高波動股盤中來回很大，停損被掃的機率太高。要控風險應靠分散檔數與大盤濾網，而非個股停損。
- 平盤以上限制反而提高每筆淨 (0.61→0.73)，因為剔除的是跳空開低那群；tier=both 過濾略降但仍 0.55。
- 該避開: 大盤平均跳空 <-1% 的日子 (每筆 ≈0)、週一 (0.32)。最肥: ATR >7% (1.17)、大盤跳空 >1% (1.05, 但觸漲停 33%)。

## 漲停鎖死真實代價 (2026-10-23) — `scripts/12_limit_up_risk.py` → `stats/limit_up_risk.md`
- 做空交易中 16~21% 進場後曾觸漲停、10~13% **收盤仍鎖在漲停**。鎖住的交易隔日開盤平均再漲 2.8%，12% 隔日再漲停。
- 保守情境 (收盤鎖漲停 → 隔日 09:00 VWAP 回補 + 0.2% 額外成本): S1/S2/S8/S9 的日淨從 0.6~0.75% 砍半到 0.33~0.38%，Sharpe 5 → 2.3~2.8，MDD 翻倍。鎖定期只剩 0.19~0.27%。
- 鎖漲停幾乎全集中在「個股跳空 >6%」(鎖死率 47~58%) 與「跳空 3~6% 且 ATR 高」(17~22%)。
- **定案候選 S9'**: 模型B 最看空 20 檔、大盤平均跳空 ≥ -1%、個股跳空 ≤ 6%、09:00~09:02 VWAP 進、13:30 收盤競價出、不設個股停損。
  保守情境: 全期間日淨 0.53%、Sharpe 3.4、每筆 0.43%、勝率 58%、MDD -14%、收盤鎖漲停 6.6%、每天約 17 檔；三段期間 0.62 / 0.41 / 0.59。
  注意「跳空 ≤6%」這條是看過全期間後才加的，鎖定期對 S9' 已不是乾淨樣本外。

## 目前策略規格 (S9')
1. 每日收盤後: 算池 (20日均成交額 ≥5億, ATR20 ≥3%, 上市櫃非ETF, 價 ≥5)，更新 D1 特徵。
2. 09:00 開盤後: 取當日 open 算 gap / mkt_gap / mkt_breadth 等，用最近一季的 `models/lgbm_B_pool_*.txt` 預測池內每檔 y_intra。
3. 濾: 大盤平均跳空 ≥ -1% 才交易；個股跳空 ≤ 6%；進場價不在跌停附近。
4. 取預測最低 20 檔，09:00~09:02 分批賣出 (現沖券賣)，等權。
5. 13:30 收盤競價回補；不設個股停損；鎖漲停者隔日開盤回補。

## 進場延遲敏感度 (2026-10-23) — `scripts/13_entry_delay.py` → `stats/entry_delay.md`
09:00 開盤價出來才能跑模型，實際成交落在 09:01 之後。S9' 以不同進場視窗 VWAP 重算 (保守情境: 鎖漲停隔日回補):

| 進場視窗 | 保守每筆淨% | 保守日淨% | 保守 Sharpe |
|---|---|---|---|
| 09:00~09:02 (原, 偏樂觀) | 0.46 | 0.56 | 3.6 |
| 09:01~09:03 (實際可行) | 0.39 | 0.50 | 3.2 |
| 09:02~09:04 | 0.37 | 0.47 | 3.1 |
| 09:03~09:05 | 0.34 | 0.45 | 3.0 |
| 09:05~09:07 | 0.31 | 0.40 | 2.7 |
| 09:10~09:12 | 0.24 | 0.32 | 2.2 |

候選股開盤後平均每分鐘往下約 0.013%，每晚 1 分鐘進場大約少 0.02% 每筆。**可執行的基準數字: 09:01~09:03 進場, 每筆 0.39%, 日淨 0.50%, Sharpe 3.2。**

## 漲停前強制回補 (2026-10-23) — `scripts/14_limit_up_avoid.py` → `stats/limit_up_avoid.md`
台灣現沖券賣若收盤鎖漲停無法回補，需標借 (費率最高 7%) 或違約交割，代價遠大於先前假設的 0.2%。
- 罰金 7% 情境下 S9' 日淨從 0.56% 掉到 0.10%，MDD -43% → 鎖死是必須用規則消滅的風險，不能靠分散吸收。
- 規則「盤中價格一到昨收 +8% 就市價回補」: 觸發率 16%，殘餘鎖死 0.25%，日淨 0.46%、Sharpe 3.55。再把個股跳空上限降到 4%: 殘餘鎖死 0.013% (8335 筆中 1 筆)，日淨 0.47%、Sharpe 3.57、MDD -11%。
- 這條是對「昨收」的門檻，不是對進場價的停損；它只砍掉會鎖死的尾巴，效果和之前測的停損完全不同。

## 目前策略規格 (S9'' = 定案)
1. 池: 前一日止 20日均成交額 ≥5億, ATR20 ≥3%, 上市櫃非ETF, 價 ≥5。
2. 09:00 用最近一季 `models/lgbm_B_pool_*.txt` 預測池內 y_intra。
3. 濾: 週一不交易；大盤平均跳空 ≥ -1%；個股跳空 ≤ 4%；進場價不在跌停附近。
4. 股價 ≤300 的候選依預測由低到高往下填，每檔上限 60 萬，填到 300 萬用完 (約 6~7 檔，最多 10 檔)，09:00~09:02 分批現沖券賣。
5. 出場: 13:30 收盤競價回補；**盤中價格觸及昨收 +8% 立即市價回補**；不設其他停損停利。
   保守回測 (含標借 7% 罰金): 每筆淨 0.41%、勝率 57%、日淨 0.47%、Sharpe 3.6、MDD -11%、每日約 17 檔。

## 集中度 (2026-10-23) — `scripts/15_concentration.py` → `stats/concentration.md`
定案規則下每日做 N 檔: 20 檔每筆 0.43% / 日淨 0.43% / MDD -9%；10 檔每筆 0.51% / 日淨 0.50% / MDD -12%，Sharpe 都 3.6；5 檔以下每筆不再提高但 MDD 爆到 -24~-32%。
分數排名第 1~5 與 6~10 的每筆相同 (0.51 / 0.50)，11~20 掉到 0.36 → 10 檔是自然切點。「只做開高」無幫助；用分數絕對門檻讓檔數浮動在驗證期崩掉，不採用。
**定案改為每日 10 檔，每檔 10% 資金。** 保守回測: 每筆淨 0.51%、勝率 57%、日淨 0.50%、日 std 2.2%、Sharpe 3.6、MDD -12%、最差單日 -5.4%。

## 300 萬資金的真實配置 (2026-10-23) — `scripts/16_capital_alloc.py` → `stats/capital_alloc.md`
- 前 10 名候選中 22% 股價 >300 (一張 >30 萬)、14% >600。股價 ≤300 的上限不傷績效 (反而略好)；≤150 小傷；≤100 明顯傷 (日淨 0.50→0.39)，因為 100~150 元這段每筆 0.72% 是最好的。
- 「10 檔每檔 30 萬」因張數捨去只用到 84% 資金，對 300 萬日淨 0.42%。
- **定案配置: 股價 ≤300，依模型排名往下填，每檔上限 60 萬，填到資金用完 (平均 6.9 檔)**: 資金使用率 99.6%，對 300 萬日淨 0.63%、Sharpe 3.9、MDD -15%、最差日 -5.9%，日均獲利約 1.9 萬元。保守替代: 6 檔每檔 50 萬 → 0.58%、Sharpe 4.0、MDD -11%。
- 流動性無虞: 單檔部位占 09:00~09:02 三分鐘成交額中位 1.6%，>10% 的天數 4%。

## 過濾提高勝率 (2026-10-23) — `scripts/17_filters_winrate.py` → `stats/filters_winrate.md`
基底 (跳空≤4%, 大盤跳空≥-1%, 股價≤300, 排名前10) 勝率 57.6%。測 36 個單一條件: 勝率天花板約 60%，多數條件是拿容量換 1 個百分點。
- 唯一三段期間都一致且全面變好的: **不做週一** → 勝率 58.7%、每筆 0.58%、對 300 萬日淨 0.73%、Sharpe 4.5、MDD -13%，容量不變。週一本身每筆只有 0.22%，驗證/鎖定期接近 0。
- 勝率最高的「ATR 3~5%」(59.5%) 在鎖定期掉到 49%，不採用。「大盤昨日跌」探索期弱，不採用。
- 若要壓回撤: 非週一 ∩ 大盤跳空≥0 ∩ 前一日非漲停 ∩ 股價≥50 → 勝率 60.8%、MDD -10%，但日淨降到 0.63%、資金使用率 90%。
- **定案加入「週一不交易」**。其餘條件不加，避免過度擬合。

## 每日候選名單腳本 (2026-10-23) — `scripts/daily_candidates.py` + `scripts/features.py`
`features.py` 是訓練 (06) 與每日腳本共用的特徵計算，已驗證與 dataset_B_pool 的 33 個特徵完全一致；回放 2026-10-22 的池 (221 檔) 與預測值和回測相同。

每日流程:
```bash
# 1) 盤前 (08:30 前, 或前一天 16:00 後): 從 HF 更新最新兩個月 D1 並重建本地檔
python3 scripts/daily_candidates.py --refresh
# 2) 09:01~09:02: 從證交所即時行情抓約 1200 檔流動股開盤價, 產生名單
python3 scripts/daily_candidates.py
# 替代: 自備開盤價 CSV (stock_id,open) / 回放歷史某日驗證
python3 scripts/daily_candidates.py --open-csv opens.csv
python3 scripts/daily_candidates.py --date 2026-10-22
```
參數: `--capital 3000000 --max-per-stock 600000 --max-price 300 --max-n 10 --min-amount 100000`。
輸出到 `signals/<date>.csv` (名單: 張數、金額、強制回補價、漲停價)、`<date>_pool.csv` (整個池的分數與被濾原因)、`<date>.txt`、`<date>_meta.json`。
週一或大盤平均跳空 < -1% 時會印「今日不交易」，名單仍列出供參考。
注意:
- 開盤價要在 09:01 之後抓 (大部分股票已開出)，太早抓會有很多檔沒有開盤價，大盤平均跳空會失真。
- 模型用 `models/` 裡最新一季的 `lgbm_B_pool_*.txt`；每季初重訓: 依序跑 02 → 03 → 06 → 09 → `07_train.py B pool`，並重新確認 15~17 的結論沒變。
- 興櫃股票不在 MIS 抓取範圍，對大盤平均跳空的影響 <2%，可忽略。

## 網頁操作台 (2026-10-23) — `scripts/webapp.py` + `web/index.html`
```bash
python3 scripts/webapp.py        # 開 http://127.0.0.1:8787
```
- 按鈕: ① 更新 D1 (從 HF 抓最新兩個月並重建) → ② 產生今日名單 (09:01 後，抓證交所開盤價) → 回放所選日期 (驗證用)。按鈕只是呼叫 `daily_candidates.py`，邏輯單一來源。
- 畫面: 名單日期 / D1 最後日 / 大盤平均跳空 / 池·合格·選入 / 投入金額 / 模型；綠色橫幅=可交易，紅色=今日不交易 (週一或大盤跳空 < -1%)；今日名單表 (含張數、金額、強制回補價、漲停價)；整個池可排序、可搜尋，被濾掉的有旗標；執行紀錄即時顯示。
- 歷史名單從 `signals/` 讀，下拉切換日期。只綁 127.0.0.1，不對外。

## 自動排程 (2026-10-23) — `scripts/install_schedule.sh`
```bash
zsh scripts/install_schedule.sh install     # 平日 08:30 更新 D1、09:01 產生名單 (launchd, ~/Library/LaunchAgents/com.quant.*)
zsh scripts/install_schedule.sh status
zsh scripts/install_schedule.sh uninstall
```
log 在 `logs/com.quant.*.log`。網頁改為被動顯示 (每分鐘自動更新)，手動按鈕收在「手動執行（備援）」。
注意: Mac 必須在 09:01 醒著且有網路；若睡眠中錯過，launchd 會在喚醒後補跑一次，因為開盤價不會變，補跑的名單仍正確，只是晚到。

## 雲端自動產生名單 + 手機查看 (2026-10-23) — GitHub Actions + HF `db/signals/`
不依賴 Mac 的流程:
```
GitHub Actions (平日 UTC 01:01 = 台灣 09:01, .github/workflows/daily_signals.yml)
  → 從 HF 抓最近 6 個月 db/d1 + db/info/info.parquet, 重建 data/processed/d1.parquet  (daily_candidates.refresh_d1(6))
  → scripts/daily_candidates.py (向證交所 MIS 抓約 560 檔開盤價, 跑模型, 寫 signals/<date>.*)
  → scripts/publish_signals.py (上傳 signals/<date>.* 到 HF db/signals/, 並寫 <date>.json 與 latest.json)
手機: 開 https://raywu918daytrade.github.io/quant_trading/web/mobile.html (GitHub Pages, 加到主畫面即可), 它直接 fetch
  https://huggingface.co/datasets/raywu918daytrade/daytrade/resolve/main/db/signals/latest.json
  (HF 的 CORS 允許瀏覽器直接讀; ?date=YYYY-MM-DD 可看歷史名單)
```
- GitHub repo: https://github.com/raywu918daytrade/quant_trading (公開; data/ logs/ signals/ 不進版控)。改了 scripts/ 或 models/ 要 push 才會生效。push 含 workflow 的 commit 需要 gh token 有 `workflow` scope (目前只有 raywu918python 有, 它是協作者)。
- 已驗證 (2026-10-23): GitHub runner 可連 mis.twse.com.tw (mis_probe)；用 date=2026-10-22 回放跑完整 workflow，結果與本機一致，HF 上出現 db/signals/latest.json。
- 需要 60 個交易日以上歷史 (ret60/ma60)，所以雲端抓 6 個月 D1；`--refresh --months N` 可調。
- `HF_TOKEN` 放 GitHub repo Settings → Secrets → Actions；必須是對 `raywu918daytrade/daytrade` 有寫入權的 token。workflow 不會印出 token。
- `.github/workflows/mis_probe.yml`: 手動觸發，只抓 5 檔開盤價，用來驗證 GitHub runner 能連到 `mis.twse.com.tw`。
- 準時觸發: cron-job.org (job 「quant_trading daily signals 09:01」) 每個平日台灣 09:01 打 GitHub `workflow_dispatch`，秒級準時；`scripts/setup_cronjob.py --env-file .env --test` 可重建/更新該 job (需要 cron-job.org API key 與 raywu918daytrade 的 GitHub token, 只給 Actions 寫入權)。
- GitHub 自己的 cron 改為台灣 09:15 當備援 (可能延遲 3~15 分鐘)；同一天跑兩次結果相同，只是覆寫同一份檔案。
- 假日/非盤中: MIS 回前一交易日的日期，`daily_candidates.py` 會拒絕產生名單 (exit 非 0)，workflow 顯示失敗、`latest.json` 維持上一次名單，屬預期行為。
- 手動補跑: GitHub → Actions → Daily signals → Run workflow (可填 `date` 回放歷史日)。

**Mac 備援**: 2026-10-23 起以雲端為主，本機 launchd 排程已移除 (`zsh scripts/install_schedule.sh install` 可裝回)；`scripts/webapp.py` 仍可手動用。若 GitHub Actions 失敗 (例如 MIS 擋海外 IP)，在 Mac 09:01 跑 `python3 scripts/daily_candidates.py` 後執行
`HF_TOKEN=... python3 scripts/publish_signals.py` 即可把名單推到 HF，手機頁照樣看得到。
若 MIS 完全擋 GitHub IP 的替代方案: Mac 於 09:01 只抓開盤價上傳 HF `db/opens/<date>.csv` (stock_id,open)，雲端 workflow 改用 `--open-csv` 讀它。
