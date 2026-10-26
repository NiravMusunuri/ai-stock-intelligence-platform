"""
Screener Universe
------------------
A fixed, curated list of large-cap tickers across sectors.

Why fixed and small (not "all 500 S&P stocks"): each ticker requires its own
Yahoo Finance fetch. Yahoo rate-limits aggressively (as you've already hit).
Fetching all 500 live, on every screener request, would fail constantly.

Production upgrade path: replace this with a scheduled batch job that
fetches the full S&P 500 once a day into Postgres, and have the screener
read from the database instead of calling Yahoo live.
"""

UNIVERSE = [
    ("AAPL", "Technology"), ("MSFT", "Technology"), ("GOOGL", "Technology"),
    ("NVDA", "Technology"), ("AMD", "Technology"), ("CRM", "Technology"),
    ("ORCL", "Technology"), ("ADBE", "Technology"), ("INTC", "Technology"),
    ("CSCO", "Technology"),

    ("AMZN", "Consumer Discretionary"), ("TSLA", "Consumer Discretionary"),
    ("HD", "Consumer Discretionary"), ("NKE", "Consumer Discretionary"),
    ("MCD", "Consumer Discretionary"), ("SBUX", "Consumer Discretionary"),

    ("JPM", "Financials"), ("BAC", "Financials"), ("GS", "Financials"),
    ("V", "Financials"), ("MA", "Financials"),

    ("JNJ", "Healthcare"), ("UNH", "Healthcare"), ("PFE", "Healthcare"),
    ("LLY", "Healthcare"), ("ABBV", "Healthcare"),

    ("XOM", "Energy"), ("CVX", "Energy"),

    ("BA", "Industrials"), ("CAT", "Industrials"), ("GE", "Industrials"),

    ("META", "Communication Services"), ("NFLX", "Communication Services"),
    ("DIS", "Communication Services"),

    ("PG", "Consumer Staples"), ("KO", "Consumer Staples"), ("PEP", "Consumer Staples"),

    ("NEE", "Utilities"), ("AMT", "Real Estate"),
]

TICKERS = [t for t, _ in UNIVERSE]
SECTOR_MAP = {t: sector for t, sector in UNIVERSE}