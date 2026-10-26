"""
News Sentiment Service
------------------------
1. Fetch recent news headlines for a ticker via yfinance's built-in news
   (no separate News API key required).
2. Score each headline's sentiment using FinBERT, a BERT model fine-tuned
   specifically on financial text (ProsusAI/finbert).

FinBERT outputs one of three labels per headline: positive, negative, neutral,
each with a confidence score. We turn that into a single signed score per
headline (positive confidence minus negative confidence) so we can average
across headlines into one aggregate "bullish / bearish / neutral" read.

Note on cost: the FinBERT model (~400MB) downloads on first use and is then
cached in memory for the life of the container process. The first request
after a fresh container start will be noticeably slower than subsequent ones.
"""

from dataclasses import dataclass
from typing import Optional

import yfinance as yf

_sentiment_pipeline = None


def _get_pipeline():
    global _sentiment_pipeline
    if _sentiment_pipeline is None:
        from transformers import pipeline
        _sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model="ProsusAI/finbert",
            tokenizer="ProsusAI/finbert",
        )
    return _sentiment_pipeline


@dataclass
class NewsArticle:
    title: str
    publisher: str
    link: str
    published_at: Optional[str]
    sentiment_label: str
    sentiment_score: float


class NewsServiceError(Exception):
    pass


def _extract_articles_raw(ticker: str, limit: int) -> list[dict]:
    """
    yfinance's .news schema has changed across versions -- handle both the
    flat shape and the newer nested 'content' shape defensively.
    """
    stock = yf.Ticker(ticker.upper().strip())
    raw_items = stock.news or []

    articles = []
    for item in raw_items[:limit]:
        content = item.get("content", item)

        title = content.get("title") or item.get("title")
        if not title:
            continue

        publisher = (
            content.get("provider", {}).get("displayName")
            if isinstance(content.get("provider"), dict)
            else item.get("publisher", "Unknown")
        )
        link = (
            content.get("canonicalUrl", {}).get("url")
            if isinstance(content.get("canonicalUrl"), dict)
            else item.get("link", "")
        )
        published_at = content.get("pubDate") or item.get("providerPublishTime")

        articles.append({
            "title": title,
            "publisher": publisher or "Unknown",
            "link": link or "",
            "published_at": str(published_at) if published_at else None,
        })

    return articles


def _score_sentiment(text: str) -> tuple[str, float]:
    result = _get_pipeline()(text[:512])[0]
    label = result["label"].lower()
    confidence = float(result["score"])

    if label == "positive":
        signed_score = confidence
    elif label == "negative":
        signed_score = -confidence
    else:
        signed_score = 0.0

    return label, round(signed_score, 4)


def get_news_sentiment(ticker: str, limit: int = 10) -> tuple[list[NewsArticle], float, str]:
    raw_articles = _extract_articles_raw(ticker, limit)

    if not raw_articles:
        raise NewsServiceError(f"No recent news found for '{ticker}'.")

    scored_articles = []
    for a in raw_articles:
        label, score = _score_sentiment(a["title"])
        scored_articles.append(NewsArticle(
            title=a["title"],
            publisher=a["publisher"],
            link=a["link"],
            published_at=a["published_at"],
            sentiment_label=label,
            sentiment_score=score,
        ))

    overall_score = round(sum(a.sentiment_score for a in scored_articles) / len(scored_articles), 4)

    if overall_score > 0.15:
        overall_label = "bullish"
    elif overall_score < -0.15:
        overall_label = "bearish"
    else:
        overall_label = "neutral"

    return scored_articles, overall_score, overall_label