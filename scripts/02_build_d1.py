"""Merge monthly D1 parquet -> data/processed/d1.parquet (sorted, typed, with meta)."""
import glob, pandas as pd, numpy as np
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
fs = sorted(glob.glob(str(ROOT/"data/raw/d1/*.parquet")))
df = pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)
df["date"] = pd.to_datetime(df["date"])
df = df.drop_duplicates(["stock_id","date"]).sort_values(["stock_id","date"]).reset_index(drop=True)
info = pd.read_parquet(ROOT/"data/raw/meta/info.parquet")
df = df.merge(info[["stock_id","name","market_type","group"]], on="stock_id", how="left")
# ETF / ETN / 權證等 : 非 4 碼純數字代號
df["is_stock"] = df.stock_id.str.fullmatch(r"\d{4}")
df["turnover"] = df.close * df.volume  # 成交金額(元) 近似
df.to_parquet(ROOT/"data/processed/d1.parquet", index=False)
print(df.shape, df.date.min().date(), df.date.max().date(), "stocks:", df.stock_id.nunique(),
      "pure-4digit:", df[df.is_stock].stock_id.nunique(), "open NaN:", df.open.isna().sum())
print(df.market_type.value_counts(dropna=False).to_string())
