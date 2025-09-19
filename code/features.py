import numpy as np
import pandas as pd


def compute_returns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # pct_chg is in percentage; convert to decimal log return
    df['r'] = np.log1p(df['pct_chg'] / 100.0)
    return df


def compute_volatility(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Parkinson volatility proxy using OHLC
    # Ensure positive highs/lows
    hl = (df['high'] / df['low']).clip(lower=1.0)
    df['hl_vol'] = (1.0 / (4.0 * np.log(2.0))) * (np.log(hl) ** 2)
    return df


def compute_liquidity_features(df: pd.DataFrame, adv_window: int = 60) -> pd.DataFrame:
    df = df.copy()
    # ADV from amount (assumed in currency units)
    df['ADV'] = df.groupby('ts_code')['amount'].transform(lambda s: s.rolling(adv_window, min_periods=1).mean())
    # Amihud illiquidity proxy: E(|r| / amount)
    df['amihud_raw'] = (df['r'].abs() / (df['amount'].replace(0, np.nan)))
    df['amihud'] = df.groupby('ts_code')['amihud_raw'].transform(lambda s: s.rolling(adv_window, min_periods=1).mean())
    df.drop(columns=['amihud_raw'], inplace=True)
    return df


def add_static_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Scale market cap
    if 'circ_mv' in df.columns:
        df['log_circ_mv'] = np.log(df['circ_mv'].replace(0, np.nan))
    # Simple z-scores per day for pe/pb
    for col in ['pe', 'pb']:
        if col in df.columns:
            df[f'{col}_z'] = df.groupby('trade_date')[col].transform(lambda s: (s - s.median()) / (s.mad() + 1e-6))
    return df


def compute_cross_sectional_zscores(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    df = df.copy()
    for col in cols:
        df[f'{col}_xz'] = df.groupby('trade_date')[col].transform(lambda s: (s - s.mean()) / (s.std(ddof=0) + 1e-8))
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Run the default feature pipeline on the raw dataframe.

    Expected columns: ts_code, trade_date, open, high, low, close, pct_chg, vol, amount,
    turnover_rate, volume_ratio, pe, pb, circ_mv
    """
    out = compute_returns(df)
    out = compute_volatility(out)
    out = compute_liquidity_features(out)
    out = add_static_features(out)
    out = compute_cross_sectional_zscores(out, ['turnover_rate', 'volume_ratio', 'amihud', 'log_circ_mv'])
    return out

