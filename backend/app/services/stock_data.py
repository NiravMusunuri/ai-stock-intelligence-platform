"""
Stock Data Service
-------------------
Wraps yfinance so the rest of the app never talks to a data provider directly.
This is deliberate: swapping in Alpha Vantage / Polygon later means changing
only this file, not every caller.
"""



import pandas as pd
import yfinance as yf



class StockDataError(Exception):
    pass




import time

_cache: dict[str, tuple[float, pd.DataFrame]] = {}
_CACHE_TTL_SECONDS = 300  # 5 minutes


def get_history(ticker: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
    ticker = ticker.upper().strip()
    cache_key = f"{ticker}:{period}:{interval}"

    cached = _cache.get(cache_key)
    if cached is not None:
        cached_at, cached_df = cached
        if time.time() - cached_at < _CACHE_TTL_SECONDS:
            return cached_df.copy()

    stock = yf.Ticker(ticker)
    df = stock.history(period=period, interval=interval, auto_adjust=True)

    if df.empty:
        raise StockDataError(f"No price data found for ticker '{ticker}'. Check the symbol.")

    df = df.reset_index()
    date_col = "Date" if "Date" in df.columns else "Datetime"
    df = df.rename(columns={
        date_col: "date", "Open": "open", "High": "high",
        "Low": "low", "Close": "close", "Volume": "volume",
    })
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    df = df[["date", "open", "high", "low", "close", "volume"]]

    _cache[cache_key] = (time.time(), df)
    return df.copy()

def get_quote(ticker: str) -> dict:
    """Current price + key fundamentals for the Stock Lookup Tool."""
    ticker = ticker.upper().strip()
    stock = yf.Ticker(ticker)

    # fast_info hits a lighter Yahoo endpoint than .info and is rate-limited
    # far less often. Get the core numbers from here first.
    try:
        fi = stock.fast_info
        price = fi.get("lastPrice")
        market_cap = fi.get("marketCap")
        fifty_two_high = fi.get("yearHigh")
        fifty_two_low = fi.get("yearLow")
        currency = fi.get("currency", "USD")
    except Exception:
        price = market_cap = fifty_two_high = fifty_two_low = None
        currency = "USD"

    # Fall back to history if fast_info didn't give us a price
    if price is None:
        try:
            hist = get_history(ticker, period="5d")
            price = float(hist.iloc[-1]["close"])
        except StockDataError:
            raise StockDataError(f"No quote data found for ticker '{ticker}'.")

    # P/E ratio only lives in the heavier .info endpoint. Try it, but don't
    # fail the whole request if Yahoo rate-limits this specific call.
    pe_ratio = None
    try:
        info = stock.info
        pe_ratio = info.get("trailingPE")
    except Exception:
        pass  # P/E just comes back null; the rest of the quote still works

    return {
        "ticker": ticker,
        "price": price,
        "market_cap": market_cap,
        "pe_ratio": pe_ratio,
        "fifty_two_week_high": fifty_two_high,
        "fifty_two_week_low": fifty_two_low,
        "currency": currency,
    }


def get_beta(ticker: str) -> float | None:
    """Beta from yfinance's info dict (relative to its benchmark, usually S&P 500)."""
    ticker = ticker.upper().strip()
    info = yf.Ticker(ticker).info
    return info.get("beta") if info else None
