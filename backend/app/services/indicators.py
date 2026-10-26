"""
Technical Indicator Service
----------------------------
Pure functions over a price DataFrame. Kept separate from stock_data.py so
the ML service and the API layer can both reuse this without re-fetching.
"""

import pandas as pd
import ta


def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Given a DataFrame with columns [date, open, high, low, close, volume],
    return a copy with technical indicator columns appended.
    """
    out = df.copy().sort_values("date").reset_index(drop=True)

    close = out["close"]
    high = out["high"]
    low = out["low"]
    volume = out["volume"]

    # Trend
    out["sma_20"] = ta.trend.sma_indicator(close, window=20)
    out["sma_50"] = ta.trend.sma_indicator(close, window=50)
    out["ema_12"] = ta.trend.ema_indicator(close, window=12)
    out["ema_26"] = ta.trend.ema_indicator(close, window=26)

    # MACD
    macd = ta.trend.MACD(close, window_slow=26, window_fast=12, window_sign=9)
    out["macd"] = macd.macd()
    out["macd_signal"] = macd.macd_signal()
    out["macd_diff"] = macd.macd_diff()

    # Momentum
    out["rsi_14"] = ta.momentum.RSIIndicator(close, window=14).rsi()
    out["stoch_k"] = ta.momentum.StochasticOscillator(high, low, close, window=14).stoch()

    # Volatility / Bollinger Bands
    bb = ta.volatility.BollingerBands(close, window=20, window_dev=2)
    out["bb_upper"] = bb.bollinger_hband()
    out["bb_lower"] = bb.bollinger_lband()
    out["bb_width"] = bb.bollinger_wband()
    out["atr_14"] = ta.volatility.AverageTrueRange(high, low, close, window=14).average_true_range()

    # Volume
    out["obv"] = ta.volume.OnBalanceVolumeIndicator(close, volume).on_balance_volume()

    return out


def to_indicator_response(df: pd.DataFrame, ticker: str) -> dict:
    """Shape an indicator DataFrame into the API response schema (JSON-safe)."""
    d = df.copy()
    d["date"] = d["date"].dt.strftime("%Y-%m-%d")

    def col(name):
        return [None if pd.isna(v) else round(float(v), 4) for v in d[name]]

    return {
        "ticker": ticker.upper(),
        "dates": d["date"].tolist(),
        "close": col("close"),
        "sma_20": col("sma_20"),
        "sma_50": col("sma_50"),
        "ema_12": col("ema_12"),
        "ema_26": col("ema_26"),
        "rsi_14": col("rsi_14"),
        "macd": col("macd"),
        "macd_signal": col("macd_signal"),
        "bb_upper": col("bb_upper"),
        "bb_lower": col("bb_lower"),
    }
