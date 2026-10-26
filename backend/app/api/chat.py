from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import WatchlistItem
from app.schemas import (
    QuoteResponse, IndicatorsResponse, PredictionResponse, RiskResponse,
    WatchlistItemCreate, WatchlistItemOut, ScreenerResponse, NewsSentimentResponse,
)
from app.services import stock_data, indicators as ind_service, ml_service, ranking_service, news_services
from app.services.universe import TICKERS
from yfinance.exceptions import YFRateLimitError
import pandas as pd


router = APIRouter()


# ---------- Stock Lookup Tool ----------

@router.get("/stocks/{ticker}/quote", response_model=QuoteResponse)
def get_quote(ticker: str):
    try:
        return stock_data.get_quote(ticker)
    except stock_data.StockDataError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except YFRateLimitError:
        raise HTTPException(
            status_code=429,
            detail="Yahoo Finance is rate-limiting requests right now. Wait a minute and try again."
        )


@router.get("/stocks/{ticker}/history")
def get_history(ticker: str, period: str = "1y", interval: str = "1d"):
    try:
        df = stock_data.get_history(ticker, period=period, interval=interval)
    except stock_data.StockDataError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except YFRateLimitError:
        raise HTTPException(
            status_code=429,
            detail="Yahoo Finance is rate-limiting requests right now. Wait a minute and try again."
        )

    df = df.copy()
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")
    df = df.where(pd.notnull(df), None)
    return {"ticker": ticker.upper(), "data": df.to_dict(orient="records")}

# ---------- Technical Indicator Tool ----------

@router.get("/stocks/{ticker}/indicators", response_model=IndicatorsResponse)
def get_indicators(ticker: str, period: str = "1y"):
    try:
        df = stock_data.get_history(ticker, period=period)
    except stock_data.StockDataError as e:
        raise HTTPException(status_code=404, detail=str(e))
    enriched = ind_service.add_all_indicators(df)
    return ind_service.to_indicator_response(enriched, ticker)


# ---------- Prediction Tool ----------

@router.get("/stocks/{ticker}/predict", response_model=PredictionResponse)
def predict(ticker: str, period: str = "2y"):
    try:
        df = stock_data.get_history(ticker, period=period)
    except stock_data.StockDataError as e:
        raise HTTPException(status_code=404, detail=str(e))

    try:
        result = ml_service.predict_next_day_direction(df, ticker)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    return PredictionResponse(
        ticker=ticker.upper(),
        direction=result.direction,
        confidence=result.confidence,
        horizon="next_day",
        model_used=result.model_used,
        backtest_accuracy=result.backtest_accuracy,
        as_of_date=result.as_of_date,
    )


# ---------- Risk Analysis Tool ----------

@router.get("/stocks/{ticker}/risk", response_model=RiskResponse)
def get_risk(ticker: str, period: str = "2y", benchmark: str = "SPY"):
    try:
        df = stock_data.get_history(ticker, period=period)
    except stock_data.StockDataError as e:
        raise HTTPException(status_code=404, detail=str(e))

    benchmark_df = None
    try:
        benchmark_df = stock_data.get_history(benchmark, period=period)
    except stock_data.StockDataError:
        pass  # beta will just come back null if benchmark fetch fails

    metrics = ml_service.compute_risk_metrics(df, benchmark_df)
    return RiskResponse(ticker=ticker.upper(), **metrics)

# ---------- News Sentiment Tool ----------

@router.get("/stocks/{ticker}/news", response_model=NewsSentimentResponse)
def get_news(ticker: str, limit: int = 10):
    try:
        articles, overall_score, overall_label = news_services.get_news_sentiment(ticker, limit=limit)
    except news_services.NewsServiceError as e:
        raise HTTPException(status_code=404, detail=str(e))

    return NewsSentimentResponse(
        ticker=ticker.upper(),
        articles=articles,
        overall_score=overall_score,
        overall_label=overall_label,
        article_count=len(articles),
    )

# ---------- Screener / Multi-Stock Ranking Tool ----------

@router.get("/screen/top-picks", response_model=ScreenerResponse)
def get_top_picks(n: int = 10, period: str = "1y"):
    try:
        picks, backtest_acc = ranking_service.get_top_picks(n=n, period=period)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    return ScreenerResponse(
        picks=picks,
        backtest_accuracy=backtest_acc,
        universe_size=len(TICKERS),
        horizon="1_week",
    )


# ---------- Watchlist ----------

@router.get("/watchlist", response_model=list[WatchlistItemOut])
def list_watchlist(db: Session = Depends(get_db)):
    return db.query(WatchlistItem).all()


@router.post("/watchlist", response_model=WatchlistItemOut)
def add_watchlist_item(item: WatchlistItemCreate, db: Session = Depends(get_db)):
    ticker = item.ticker.upper().strip()
    existing = db.query(WatchlistItem).filter(WatchlistItem.ticker == ticker).first()
    if existing:
        raise HTTPException(status_code=400, detail=f"{ticker} is already on your watchlist.")

    # Validate the ticker actually exists before saving it
    try:
        stock_data.get_quote(ticker)
    except stock_data.StockDataError as e:
        raise HTTPException(status_code=404, detail=str(e))

    db_item = WatchlistItem(ticker=ticker, note=item.note)
    db.add(db_item)
    db.commit()
    db.refresh(db_item)
    return db_item


@router.delete("/watchlist/{item_id}")
def remove_watchlist_item(item_id: int, db: Session = Depends(get_db)):
    item = db.query(WatchlistItem).filter(WatchlistItem.id == item_id).first()
    if not item:
        raise HTTPException(status_code=404, detail="Watchlist item not found.")
    db.delete(item)
    db.commit()
    return {"status": "deleted", "id": item_id}
