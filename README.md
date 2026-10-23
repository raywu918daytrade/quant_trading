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
2. 09:00 用最近一季 `models/lgbm_Bsr_pool_*.txt` (基準 33 + 支撐壓力 13 特徵; 2026-10-23 起) 預測池內 y_intra。
3. 濾: 週一不交易；大盤平均跳空 ≥ -1%；個股跳空 ≤ 4%；進場價不在跌停附近。
4. 股價 ≤300 的候選依預測由低到高往下填，每檔上限 60 萬，填到 300 萬用完 (約 6~7 檔，最多 10 檔)，09:00~09:02 分批現沖券賣。
5. 出場: 13:30 收盤競價回補；**盤中價格觸及昨收 +8% 立即市價回補**；13:24 現價 ≥ 昨收 +6% 先回補；不設單檔停損停利，也不設組合停損 (2026-10-23 測過 −3%/−4%: 期望值幾乎不變、只砍最差日，決定不用；程式保留 `PORTFOLIO_STOP` 開關)。
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

## 現沖券賣資格過濾 (2026-10-23) — `scripts/daytrade_list.py`
`daily_candidates.py` 選池後先查證交所 (`TWTB4U`) 與櫃買 (`intraday/list`) 每日公布的「當日可現股當沖」名單 (公開 API, 不需登入, 盤前就有, 支援歷史日期供回放; 快取 `data/raw/meta/daytrade_<日期>.json`)。
池內「不在名單」(不可當沖) 或「註記暫停現股賣出後現款買進」(只能先買後賣) 的股票標記為不合格，排名往下補。`--no-daytrade-filter` 可關閉 (對照用)。取得失敗時不過濾、在 txt 與 meta 標註警告。
10/22 回放對照: 池 221 檔中 6 檔被標 (5 不可當沖、1 暫停先賣)，原本合格 120 → 117，被剔除的排名最高是第 26 名，前 10 名單與投入金額完全不變。
另外 `paper_trade.py` 下單前再用永豐合約的 `day_trade` 欄位 (Yes / OnlyBuy / No) 查核一次，不是 Yes 的略過並記錄，兩層保險。

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
手機: 開 https://quant.just1stock.com/ (自訂網域, 根目錄自動跳到 web/mobile.html; 備用網址 https://raywu918daytrade.github.io/quant_trading/web/mobile.html), 它直接 fetch
  https://huggingface.co/datasets/raywu918daytrade/daytrade/resolve/main/db/signals/latest.json
  (HF 的 CORS 允許瀏覽器直接讀; ?date=YYYY-MM-DD 可看歷史名單)
```
- 自訂網域: GoDaddy 的 just1stock.com 加 CNAME `quant` → `raywu918daytrade.github.io`；repo 根目錄 `CNAME` 檔寫 `quant.just1stock.com`，Pages 設定 custom domain + Enforce HTTPS。
- GitHub repo: https://github.com/raywu918daytrade/quant_trading (公開; data/ logs/ signals/ 不進版控)。改了 scripts/ 或 models/ 要 push 才會生效。push 含 workflow 的 commit 需要 gh token 有 `workflow` scope (目前只有 raywu918python 有, 它是協作者)。
- 已驗證 (2026-10-23): GitHub runner 可連 mis.twse.com.tw (mis_probe)；用 date=2026-10-22 回放跑完整 workflow，結果與本機一致，HF 上出現 db/signals/latest.json。
- 需要 60 個交易日以上歷史 (ret60/ma60)，所以雲端抓 6 個月 D1；`--refresh --months N` 可調。
- `HF_TOKEN` 放 GitHub repo Settings → Secrets → Actions；必須是對 `raywu918daytrade/daytrade` 有寫入權的 token。workflow 不會印出 token。
- `.github/workflows/mis_probe.yml`: 手動觸發，只抓 5 檔開盤價，用來驗證 GitHub runner 能連到 `mis.twse.com.tw`。
- 準時觸發: cron-job.org (job 「quant_trading daily signals 09:01」) 每個平日台灣 09:01 打 GitHub `workflow_dispatch`，秒級準時；`scripts/setup_cronjob.py --env-file .env --test` 可重建/更新該 job (需要 cron-job.org API key 與 raywu918daytrade 的 GitHub token, 只給 Actions 寫入權)。
- 兩條 workflow 都不用 GitHub 自己的 schedule (會延遲、且 60 天沒 commit 會被停用)，只由 cron-job.org 觸發，不會跑兩次。cron-job.org 掛掉就不會有名單，此時手動到 Actions 按 Run workflow，或用 Mac 跑 `daily_candidates.py` + `publish_signals.py`。
- 假日/非盤中: MIS 回前一交易日的日期，`daily_candidates.py` 會拒絕產生名單 (exit 非 0)，workflow 顯示失敗、`latest.json` 維持上一次名單，屬預期行為。
- 手動補跑: GitHub → Actions → Daily signals → Run workflow (可填 `date` 回放歷史日)。

**Mac 備援**: 2026-10-23 起以雲端為主，本機 launchd 排程已移除 (`zsh scripts/install_schedule.sh install` 可裝回)；`scripts/webapp.py` 仍可手動用。若 GitHub Actions 失敗 (例如 MIS 擋海外 IP)，在 Mac 09:01 跑 `python3 scripts/daily_candidates.py` 後執行
`HF_TOKEN=... python3 scripts/publish_signals.py` 即可把名單推到 HF，手機頁照樣看得到。
若 MIS 完全擋 GitHub IP 的替代方案: Mac 於 09:01 只抓開盤價上傳 HF `db/opens/<date>.csv` (stock_id,open)，雲端 workflow 改用 `--open-csv` 讀它。

## 模擬下單統計 (2026-10-23) — `scripts/paper_trade.py` + `.github/workflows/paper_trade.yml`
用永豐 Shioaji **模擬環境** (`sj.Shioaji(simulation=True)`, 程式寫死, 永遠不會真的下單) 照定案策略自動執行當日名單並記錄:
- 08:50 cron-job.org 打 `workflow_dispatch` 啟動 GitHub Actions (沒有 GitHub 自己的排程；超過 09:04 才進場會自動放棄) → 等 HF `db/signals/latest.json` 變成今天 (最晚 09:10) → 模擬登入 → 檢查每檔 `contract.day_trade == Yes` (不可現沖券賣的直接略過並記錄)。
- 09:01:00 每檔市價「現股當沖先賣」(`daytrade_short=True`)，同時訂閱 tick。
- 盤中任一 tick ≥ 強制回補價 (昨收 +8%) → 立即市價買回 (`exit_reason=forced`)。
- 13:24 (逐筆撮合最後一分鐘) 現價 ≥ 昨收 +6% 的先市價買回 (`exit_reason=pre_auction`)，因為 13:25~13:30 沒有成交、+8% 規則在那五分鐘失效，避免收盤競價跳到漲停鎖死；門檻在 `PRE_AUCTION_K`，未回測，只影響尾盤漲 6% 以上的少數交易。
- 13:26 剩餘部位市價買回 (參與 13:30 收盤競價)；13:35 結算。
- 每檔同時記兩組數字: **模擬成交價** (券賣/買回) 與 **真實 tick** 算的 09:01~09:03 VWAP 與最後成交價 (回測的假設)。兩者差距就是模擬撮合的偏差；前幾天先看 `sim_net_pct` 與 `tick_net_pct` 是否接近，再決定信哪個。
- 輸出 `paper/<date>_trades.csv`、`<date>_summary.json`，`--publish` 上傳 HF `db/paper/` 並累計 `db/paper/summary.csv`；手機頁「模擬下單」區直接顯示。
- 盤中狀態: 進場後、每 5 分鐘、每次強制回補時寫 `db/paper/live.json` (各檔狀態、券賣價、現價、未實現損益)，手機頁盤中顯示「持有 N 檔 / 未實現損益」表，13:35 結算後自動換成最終結果。
- 需要 GitHub Secrets `SINOPAC_API_KEY`、`SINOPAC_SECRET_KEY` (模擬環境不需要 CA 憑證)。`sinopac_probe.yml` 手動觸發可驗證 GitHub runner 能否登入永豐。
- 本機: `python3.12 -m venv .venv && .venv/bin/pip install shioaji pandas requests huggingface_hub`，然後
  `.venv/bin/python scripts/paper_trade.py --env-file <有 SINOPAC_* 的 .env> --dry` (非盤中測試) 或盤中不加 `--dry` 正式跑。
- 限制: 模擬環境不支援興櫃與零股；模擬撮合規則永豐未公開；成本假設 0.23% 來回與回測相同。

## 延伸回測 2020~2026 (2026-10-23) — `scripts/18_extended_backtest.py` + `19_build_m1_entry_ext.py` → `stats/extended_backtest.md`
D1 抓到 2018 (2018~2019 只當訓練)、M1 抓到 2020，池版模型 B 按季 walk-forward 新訓 2020Q1~2024Q3 (`models/ext/`)，2024Q4 起沿用正式模型；規則與定案完全相同 (跳空≤4%、大盤跳空≥−1%、股價≤300、非週一、+8% 強制回補、鎖漲停隔日回補 −7%、300 萬每檔 60 萬填滿、成本 0.23%)。
成交假設兩套並列: D1 近似 (開盤價 +0.03% 校準) 與 M1 精確 (09:01~09:03 VWAP 進、13:30 收盤競價出、進場後分鐘最高價判強制回補)。無交易日 (週一/大盤跳空<−1%) 報酬計 0 併入日均。

| M1 精確, 300 萬 | 每筆淨% | 勝率 | 日淨% | Sharpe | MDD | 年化(萬) |
|---|---|---|---|---|---|---|
| 2020 | 0.47 | 60% | 0.46 | 4.2 | −8% | 337 |
| 2021 | 1.06 | 63% | 0.87 | 5.3 | −9% | 637 |
| 2022 | 0.38 | 56% | 0.35 | 3.0 | −9% | 256 |
| 2023 | 0.35 | 56% | 0.37 | 2.8 | −8% | 271 |
| 2024 | 0.44 | 59% | 0.39 | 2.9 | −9% | 288 |
| 2025 | 0.39 | 59% | 0.43 | 3.8 | −6% | 318 |
| 2026 至 10/22 | 0.60 | 56% | 0.60 | 3.7 | −16% | 438 |
| **2020~2024Q3 (新樣本外)** | **0.53** | **59%** | **0.48** | **3.6** | **−10%** | **356** |

- 七年、28 季全部為正；最弱 2023Q1 (每筆 0.13%)。負報酬月份 10% (8/82)，最差單月 −5.1% (2026-09)。成本 0.40% 時每筆仍 0.36%。
- **模型 vs 純波動排序 (ATR 前 10) 每筆淨%**: 2020 0.47/0.33、2021 1.06/0.64、2026 0.60/0.42 模型贏；**2022 0.38/0.53、2023 0.35/0.47、2024 0.44/0.57、2025 0.39/0.44 ATR 基準反而較好**。rank-IC 每年皆正 (0.05~0.11) 但 2022~2023 偏低。結論: 報酬主體是「放空高波動股的日內溢酬」，模型在高波動多頭年 (2020-21、2026) 加分明顯，在平淡年份不比 ATR 排序好；D1 近似版模型每年都贏 ATR，差異來自 M1 進場價，顯示高 ATR 股開盤後幾分鐘的滑價較小。
- 限制: 2020~2023 無現沖券賣資格名單；2020 上半年模型只用兩年資料訓練。
- 選股規則比較 (`scripts/20_selection_variants.py` → `stats/selection_variants.md`, 2020~2026 M1 精確): 純模型 / 純 ATR / 純 vol60 / 混合 / ATR 前 20~30 內用模型 / 模型前 30 內用 ATR。每筆淨 0.49~0.57% 全部接近且全為正；但**純模型的 Sharpe 最高 (3.67 vs 2.8~3.5)、MDD 最小 (−15.6% vs −18~−28%)、最差月最小 (−5.1% vs −6.6~−11.3%)、負月比例最低 (10% vs 12~23%)**。模型的價值不在每筆多賺，在於挑出來的組合波動與回撤小。**結論: 維持純模型，不改。**
- 模型 × ATR 交集 (`stats/selection_overlap.md`): 每天模型前 10 與 ATR 前 10 只重疊 3.1 檔 (31%)；交集每筆淨 0.67%、只有模型 0.46%、只有 ATR 0.41%，七年中六年交集最好。但交集每天只有 3 檔填不滿 300 萬 (使用率 59%, 日淨反而低)；把交集股加碼 (90~150 萬/檔) 年化多 5~14%，代價是 MDD 從 −15.6% 惡化到 −20~−26%、Sharpe 由 4.2 降到 3.7。**結論: 維持純模型等權 (每檔 60 萬)，不加碼交集。**
- K 線序列特徵對照 (`scripts/21_kline_seq_test.py` → `stats/kline_seq_test.md`): 基準 33 特徵 vs 加 45 個最近 10 天 K 棒序列特徵，2020Q1~2026Q4 同 pipeline 各訓 28 季。序列版每筆淨 0.52→0.59%，七年每一年每筆都較高；日淨 0.47→0.52%，Sharpe 3.44→4.00；但 MDD −13.9%→−18.1%，每日 IC 差 t-stat 僅 1.54。序列特徵 gain 占 46% (上下影線、各日跳空最重要)。尚未採用，待決定。

## 正式 = 基準+支撐壓力，影子 = 基準 (2026-10-23 定案) — 角色在 `scripts/config.py`
- `MAIN_VARIANT = "sr"` (基準 33 + `features.SR_FEATS` 13 個支撐壓力特徵, `models/lgbm_Bsr_pool_<季>.txt`, D1 從 2018 起訓練)；`SHADOW_VARIANT = "base"` (原本的 33 特徵模型 `lgbm_B_pool_<季>.txt`)。要對調只改這兩行、workflow 矩陣與 `web/mobile.html` 的 `SHADOW` 常數。
- 正式產出不帶後綴 (`signals/<date>.*`, `db/signals/latest.json`, `paper/`, `db/paper/`)；影子帶 `_base` (`latest_base.json`, `paper_base/`, `db/paper_base/`)。
- 名單: `daily_candidates.py --variant all` 一次算正式+影子 (特徵只算一次、開盤價只抓一次, 約 10 秒)；`publish_signals.py` 兩份一起上傳。單獨跑: `--variant sr` / `--variant base`。模擬: `paper_trade.yml` 矩陣 `[sr, base]`，同一模擬帳戶、委託帶 `custom_field` 區分。
- 手機頁三段切換: 正式：支撐壓力 / 影子：基準 / 對照。
- 每季初重訓: `07_train.py B pool` (基準) 與 `22_train_seq_model.py sr <季>` (正式) 都要跑。K 線序列變體 `seq` 程式保留 (`22_train_seq_model.py seq`) 但不排程。
- 依據 `stats/sr_feature_test.md`: 加支撐壓力後每筆 0.52→0.57% (七年皆升)、Sharpe 3.44→3.84、MDD 不變、IC 差 t=3.07。注意正式模型訓練資料較長 (2018 起) 而影子基準只用 2023-10 起三年，對照含此差異。一個月後若基準反而較好就對調回來。
- D1 支撐壓力特徵對照 (`scripts/23_sr_feature_test.py` → `stats/sr_feature_test.md`): 回看 60 天轉折高/低點 (前後 3 天極值)、±1% 帶觸碰次數，13 個特徵 (開盤離最近壓力/支撐距離與強度、60 日高低距離等)。基準+SR: 每筆 0.52→0.57% (七年每年都升)、Sharpe 3.44→3.84、**MDD 不變 (−14.0%)、每日 IC 差 t=3.07 顯著**。基準+seq+SR 每筆最高 0.62% 但 MDD −22%。「觸碰次數」本身重要度低，有用的是距離 (sr_lo60_dist / sr_hi60_dist / sr_sup_dist / sr_res_dist)。
- 移動停利測試 (`scripts/24_trailing_stop_test.py` → `stats/trailing_stop_test.md`, 2020~2026 M1 逐分鐘, 12,528 筆): 百分比追蹤 (1~3%, 含「浮盈 1~2% 才啟動」) 與分鐘均線 (MA10/20/30) 共 13 種，**全部輸給現行的抱到收盤**。最好的「trail 3% after +2%」每筆 0.43% vs 現行 0.53%、Sharpe 3.95 vs 4.29；無啟動門檻的 1~2% 追蹤與所有 MA 版本直接變負。被停利的單若抱到收盤平均多賺 0.56~0.66%/筆。結論: 不加移動停利，維持收盤出場。
- 組合層級盤中停損測試 (`scripts/25_portfolio_stop_test.py` → `stats/portfolio_stop_test.md`, 2020~2026 M1 逐分鐘): 現行最差日 −6.4% (2024-08-27)，盤中組合浮虧達 −3% 的日子 (6%) 收盤轉正的只有 5%、平均收在 −3.0%；達 −4% 的 28 天沒有一天轉正。組合浮虧 −3% 全部回補: 日淨 0.654→0.628% (−4%)、Sharpe 4.29→4.09、但最差日 −6.4→−4.2%、最差月 −5.2→−4.2%、MDD −14.1→−13.3%；−4%: 日淨 −1%、最差日 −4.8%。結論: 組合停損是便宜的尾端保險 (和單檔移動停利完全不同)，待決定是否加 (建議 −3% 或 −4%)。
- 組合停損「加 vs 不加」在持股同時被軋的日子 (`scripts/27_stop_on_rally_days.py` → `stats/stop_on_rally_days.md`, 基準與 sr 各一份): 觸發日 (組合浮虧 ≤ −3%, 約 6%) 平均反而多賠 0.4% (多數收在 −3% 附近, 砍在 −3.4%)；跌破 −4% 的日子少賠 0.6~0.9%、跌破 −5% 少賠 1.5%。最差 15 天合計: 基準 −217 → −156 萬、sr −226 → −160 萬。sr 模型: −3% 年化少 4.5%、−4% 幾乎免費 (0.688→0.687%, 最差日 −6.1→−5.0%)。

## 交易紀錄 (每日自動, 2026-10-23) — `scripts/record_results.py` → HF `db/signals/history.csv` / `history_daily.csv` (影子 `history_base.csv`)
每天 09:01 名單 workflow 在 D1 更新、名單發布後執行: 把「已發布但還沒結算」的日子用 D1 逐檔結算 (進場開盤 +0.03%、high ≥ 昨收×1.08 → 強制價、否則收盤價；收盤鎖漲停 → 隔日開盤 −7%；成本 0.23%)，每檔一列追加到 `history.csv`、每天一行到 `history_daily.csv`。只記當天真的發布過的名單，換模型也接續記錄 (每行有 `model` 欄位可區分)，純追加、沒有刪除功能。手機頁「交易紀錄」區顯示累積損益、每日列表、點開看各檔；正式與影子各自一份。

## 手機頁「補跑這天」代理 (2026-10-23) — `trigger/worker.js` (Cloudflare Workers, 免費、不休眠)
一次設定: Cloudflare Dashboard → Workers & Pages → Create → Hello World → 名稱 `quant-trigger` → Edit code → 貼 `trigger/worker.js` → Deploy → Settings → Variables and Secrets 加 `GH_TOKEN` (GitHub fine-grained token，只給 quant_trading 的 Actions: Read and write，設 Secret) 與 `API_PIN` (自訂)。網址形如 `https://quant-trigger.<子網域>.workers.dev`。
網頁: 選日期按「載入」，平日沒有名單時自動補跑「該月整月」(一次補齊缺的日子, 已發布的跳過; 第一次問 PIN) 並每 10 秒輪詢；每個日期只自動補跑一次；5 分鐘內或 GitHub 執行中不重複觸發，週末/未來日期只顯示沒名單。workflow 的 `date` 可填 YYYY-MM-DD / YYYY-MM / A..B；多日回放特徵只算一次 (10 月 5 天實測 85 秒)。

- 分K (2026-10-23): `scripts/build_bars.py` 從 HF `db/m1/<月>.parquet` 取出已發布名單 (正式+影子聯集) 當天的 5 分 K → `db/signals/bars/<date>.json`，每次名單 workflow 結尾補缺的日子 (當天的分 K 要等 HF m1 更新後、隔天才有)。手機頁卡片「分K ▾」展開用 canvas 畫，標昨收/開盤/+8% 線。
- 5 分 K 紅轉綠進場時機 (`scripts/28_m5_pattern_entry.py` → `stats/m5_pattern_entry.md`): 型態 (紅 K 後接更大的綠 K) 10:00 前出現的交易，現行進場每筆 +1.04% (勝率 65%)，未出現 −0.007% (52%) → 型態很有資訊；但等型態出現才進場每筆只剩 0.40% (少 0.65%)，所有等型態版本都輸現行。結論: 不當進場條件；改測「沒型態就提早回補」(29)。
- 5 分 K 型態當出場條件 (`scripts/29_m5_pattern_exit.py` → `stats/m5_pattern_exit.md`): 09:01 照常進場、到 T 還沒出現型態就回補 (09:30~11:00) 或出現反向型態就回補 → 全部輸現行 (最好的 11:00 回補每筆 0.498% vs 0.573%)。沒型態的交易 10:00 回補 −0.417%、抱到收盤 −0.013%: 早盤還在漲的空單，到收盤多半會回來。結論: 5 分 K 紅轉綠型態不加入進場也不加入出場。
- 開盤爆量 (`scripts/30_open_volume_test.py` 比值版 → `stats/open_volume_test.md`；`scripts/31_open_volume_pr.py` PR 版 → `stats/open_volume_pr.md`): 第一根 M1/M5 量相對前 1/5/20/60 日 (比值或百分位)。爆量組確實更常被強制回補 (PR 95+ 約 20% vs 低量組 2~6%)、盤中最高漲幅也較大，但每筆淨反而最高 (PR 95+ 約 0.74~0.87% vs 全體 0.57%)：漲得多也跌得多，模型挑的爆量股是更好的空單。任何「爆量不做」濾網都輸現行；M5 版還要延到 09:05 進場，再少 0.12%/筆。結論: 不加。
- 開盤量 PR 當模型特徵 (`scripts/32_open_volume_feature.py` → `stats/open_volume_feature.md`): 正式模型加第一根 M1 量的 PR/比值特徵，全期間跟現行幾乎相同 (Sharpe 4.61 vs 4.45) 但 MDD −15.9% vs −14.0%、最差月 −8% vs −5%，近 3 年全輸；M5 版連樂觀進場都輸。結論: 不採用。
- 大賠交易的 D1 特徵 (`scripts/33_big_loss_profile.py` → `stats/big_loss_profile.md`): 大賠 (≤ −7%) 354 筆 = 2.8%，合計 −1728 萬 (總獲利 4309 萬)。「盤整後突破」組大賠率 2.6% 反而低於全體 2.8%，大賠集中在前 10 日區間 > 22% 的高波動股 (3.8%)；昨日漲停組強制回補 21% 但每筆淨最高。任何排除濾網都沒改善。結論: 不加。
- D1 均線斜率/加速度 (`scripts/34_ma_slope_test.py` → `stats/ma_slope_test.md`): MA5/10/20 斜率越陡、加速越大，大賠率與強制回補率越高 (MA20 五日斜率 >10% 組大賠 4.5%)，但每筆淨也越高 (同組 1.31%)。「MA10 向上且加速」占 45%，每筆 0.53% vs 向上但減速 0.81%；排掉後每筆升到 0.61% 但檔數不足、日淨 0.54% vs 0.69%。排 MA5 斜率 >10% 年化持平 (510 vs 506)、MDD 更深 (−17% vs −14%)。結論: 不加。
- 排斜率加速 + 資金集中 3~5 檔 (`scripts/35_slope_filter_concentrate.py` → `stats/slope_filter_concentrate.md`): 排「MA10 向上且加速」後每檔 100 萬 (日均 3.6 檔) 日淨 0.684% ≈ 現行 0.688%，但 MDD −18.6% vs −14.0%；150 萬/檔 日淨 0.78% 但 MDD −24%。同集中度下不濾網反而更高 (無濾網 100 萬/檔 0.78%)，集中帶來的增益來自集中本身不是濾網，而且 MDD 隨集中度惡化 (60→100→150 萬: −14 → −20 → −34%)。結論: 不改。
- 回撤診斷與勝率候選 (`scripts/36_drawdown_diagnosis.py` → `stats/drawdown_diagnosis.md`；`scripts/37_rank_variants.py` → `stats/rank_variants.md`): 每筆勝率 58.9%、日 59.3%、月 86.6%。最深回撤都是 8~31 天的大盤上漲段，最差日全是大盤日內 +1~4%，前一日大盤狀態預測不了 (大盤前日跌 >2% 的日子反而 +0.82%)。大盤狀態濾網、等風險配置 (∝1/ATR)、分類目標排序 (勝率反降到 58.0%)、迴歸÷ATR 排序、名次混合全部輸現行。結論: 維持現行。
