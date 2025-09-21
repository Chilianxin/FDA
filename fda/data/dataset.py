from __future__ import annotations

import pandas as pd
import numpy as np
from typing import List, Dict, Optional


REQUIRED_COLS = [
    'ts_code','trade_date','open','high','low','close','pct_chg','vol','amount',
    'turnover_rate','volume_ratio','pe','pb','circ_mv'
]


def validate_columns(df: pd.DataFrame, extra: Optional[List[str]] = None) -> None:
    cols = set(REQUIRED_COLS + (extra or []))
    missing = cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")


def load_panel(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, dtype={'ts_code': str, 'trade_date': str})
    validate_columns(df)
    df = df.sort_values(['trade_date','ts_code']).reset_index(drop=True)
    # log return from pct_chg
    df['r'] = np.log1p(df['pct_chg'] / 100.0)
    # Parkinson volatility proxy
    hl = (df['high'] / df['low']).clip(lower=1.0)
    df['hl_vol'] = (1.0 / (4.0 * np.log(2.0))) * (np.log(hl) ** 2)
    # ADV and Amihud
    df['ADV'] = df.groupby('ts_code')['amount'].transform(lambda s: s.rolling(60, min_periods=1).mean())
    df['amihud'] = df.groupby('ts_code').apply(lambda g: (g['r'].abs() / g['amount'].replace(0,np.nan)).rolling(60, min_periods=1).mean()).reset_index(level=0, drop=True)
    # static
    df['log_circ_mv'] = np.log(df['circ_mv'].replace(0, np.nan))
    # cross-sectional z
    for col in ['turnover_rate','volume_ratio','amihud','log_circ_mv']:
        df[f'{col}_xz'] = df.groupby('trade_date')[col].transform(lambda s: (s - s.mean())/(s.std(ddof=0)+1e-8))
    for col in ['pe','pb']:
        df[f'{col}_z'] = df.groupby('trade_date')[col].transform(lambda s: (s - s.median())/(s.mad()+1e-6))
    return df


def pivot_returns(df: pd.DataFrame) -> pd.DataFrame:
    R = df.pivot(index='trade_date', columns='ts_code', values='r').sort_index()
    return R

