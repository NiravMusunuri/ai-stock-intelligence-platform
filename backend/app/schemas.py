from datetime import date as date_type
from typing import Optional

from pydantic import BaseModel


class PricePoint(BaseModel):
    date: date_type
    open: float
    high: float
    low: float
    close: float
    volume: int


class QuoteResponse(BaseModel):
    ticker: str
    price: Optional[float] = None
    market_cap: Optional[float] = None
    pe_ratio: Optional[float] = None
    fifty_two_week_high: Optional[float] = None
    fifty_two_week_low: Optional[float] = None
    currency: Optional[str] = None


class IndicatorsResponse(BaseModel):
    ticker: str
    dates: list[str]
    close: list[float]
    sma_20: list[Optional[float]]
    sma_50: list[Optional[float]]
    ema_12: list[Optional[float]]
    ema_26: list[Optional[float]]
    rsi_14: list[Optional[float]]
    macd: list[Optional[float]]
    macd_signal: list[Optional[float]]
    bb_upper: list[Optional[float]]
    bb_lower: list[Optional[float]]


class PredictionResponse(BaseModel):
    ticker: str
    direction: str  # "UP" or "DOWN"
    confidence: float  # 0-1
    horizon: str  # "next_day"
    model_used: str
    backtest_accuracy: float
    as_of_date: str
    disclaimer: str = (
        "This is a statistical estimate based on historical technical patterns, "
        "not financial advice. Past performance does not guarantee future results."
    )


class RiskResponse(BaseModel):
    ticker: str
    annualized_volatility: float
    beta: Optional[float]
    sharpe_ratio: Optional[float]
    max_drawdown: float

class RankedStockOut(BaseModel):
    model_config = {"from_attributes": True}

    ticker: str
    sector: str
    score: float
    momentum_1mo: float
    rsi_14: float


class ScreenerResponse(BaseModel):
    picks: list[RankedStockOut]
    backtest_accuracy: float
    universe_size: int
    horizon: str
    disclaimer: str = (
        "Ranking is based on historical technical and momentum patterns across "
        "a curated stock universe, not financial advice. A low backtest accuracy "
        "means the ranking has limited historical reliability."
    )

class NewsArticleOut(BaseModel):
    model_config = {"from_attributes": True}

    title: str
    publisher: str
    link: str
    published_at: Optional[str] = None
    sentiment_label: str
    sentiment_score: float


class NewsSentimentResponse(BaseModel):
    ticker: str
    articles: list[NewsArticleOut]
    overall_score: float
    overall_label: str
    article_count: int
    disclaimer: str = (
        "Sentiment is scored from recent headlines using FinBERT, a finance-tuned "
        "language model. This reflects tone of coverage, not verified facts or "
        "investment advice."
    )


class WatchlistItemCreate(BaseModel):
    ticker: str
    note: Optional[str] = None


class WatchlistItemOut(BaseModel):
    id: int
    ticker: str
    note: Optional[str] = None

    class Config:
        from_attributes = True
