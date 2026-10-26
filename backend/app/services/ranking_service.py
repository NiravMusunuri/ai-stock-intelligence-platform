"""
Screener / Ranking Service
----------------------------
Answers a different question than ml_service.py.

ml_service.py:       "Will THIS stock go up tomorrow?"      (single stock, direction)
ranking_service.py:  "Which of THESE stocks looks strongest right now?"  (many
                      stocks, relative ranking)

Design simplification worth being upfront about: this uses a binary
classifier (top-quintile membership vs. not) rather than a true pairwise/
listwise ranking objective (like XGBoost's rank:pairwise). A binary
top-quintile classifier is simpler to reason about and its predicted
probabilities still sort into a usable ranking, but a real learning-to-rank
objective is the more "correct" tool for this problem and is a natural
upgrade later.
"""

from dataclasses import dataclass
import time
import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from app.services import stock_data
from app.services.indicators import add_all_indicators
from app.services.universe import TICKERS, SECTOR_MAP

FEATURE_COLUMNS = [
    "rsi_14", "macd_diff", "dist_from_sma20", "dist_from_sma50",
    "momentum_1mo", "momentum_3mo", "volatility_20d",
    "rel_momentum_1mo", "rel_momentum_3mo",
]

FORWARD_WINDOW_DAYS = 5
TOP_QUINTILE = 0.20


@dataclass
class RankedStock:
    ticker: str
    sector: str
    score: float
    momentum_1mo: float
    rsi_14: float


def _fetch_universe(period: str = "1y") -> dict[str, pd.DataFrame]:
    data = {}
    for ticker in TICKERS:
        try:
            data[ticker] = stock_data.get_history(ticker, period=period)
        except Exception:
            continue
        time.sleep(0.3)  # small delay between requests to avoid bursting the rate limit
    return data


def _build_stock_features(ticker: str, df: pd.DataFrame) -> pd.DataFrame:
    enriched = add_all_indicators(df).sort_values("date").reset_index(drop=True)

    enriched["dist_from_sma20"] = (enriched["close"] - enriched["sma_20"]) / enriched["sma_20"]
    enriched["dist_from_sma50"] = (enriched["close"] - enriched["sma_50"]) / enriched["sma_50"]
    enriched["momentum_1mo"] = enriched["close"].pct_change(21)
    enriched["momentum_3mo"] = enriched["close"].pct_change(63)
    enriched["volatility_20d"] = enriched["close"].pct_change().rolling(20).std()

    enriched["ticker"] = ticker
    enriched["sector"] = SECTOR_MAP.get(ticker, "Unknown")

    enriched["forward_return"] = enriched["close"].shift(-FORWARD_WINDOW_DAYS) / enriched["close"] - 1

    return enriched[["date", "ticker", "sector", "close", "forward_return"] + FEATURE_COLUMNS[:7]]


def _add_relative_features(panel: pd.DataFrame) -> pd.DataFrame:
    panel = panel.copy()
    daily_avg_1mo = panel.groupby("date")["momentum_1mo"].transform("mean")
    daily_avg_3mo = panel.groupby("date")["momentum_3mo"].transform("mean")
    panel["rel_momentum_1mo"] = panel["momentum_1mo"] - daily_avg_1mo
    panel["rel_momentum_3mo"] = panel["momentum_3mo"] - daily_avg_3mo
    return panel


def _build_panel(universe_history: dict[str, pd.DataFrame]) -> pd.DataFrame:
    frames = [_build_stock_features(t, df) for t, df in universe_history.items()]
    panel = pd.concat(frames, ignore_index=True)
    panel = _add_relative_features(panel)
    return panel


def _label_top_quintile(panel: pd.DataFrame) -> pd.DataFrame:
    panel = panel.copy()
    panel["label"] = (
        panel.groupby("date")["forward_return"]
        .rank(pct=True, ascending=True) >= (1 - TOP_QUINTILE)
    ).astype(int)
    return panel


def _walk_forward_backtest(panel: pd.DataFrame, min_train_days: int = 120, step_days: int = 10) -> float:
    clean = panel.dropna(subset=FEATURE_COLUMNS + ["label"]).sort_values("date")
    unique_dates = clean["date"].unique()

    if len(unique_dates) < min_train_days + step_days:
        return float("nan")

    correct, total = 0, 0
    i = min_train_days
    while i + step_days <= len(unique_dates):
        train_dates = unique_dates[:i]
        test_dates = unique_dates[i:i + step_days]

        train = clean[clean["date"].isin(train_dates)]
        test = clean[clean["date"].isin(test_dates)]

        if train["label"].nunique() < 2 or len(test) == 0:
            i += step_days
            continue

        model = XGBClassifier(
            n_estimators=100, max_depth=3, learning_rate=0.1,
            eval_metric="logloss", use_label_encoder=False,
        )
        model.fit(train[FEATURE_COLUMNS], train["label"])
        preds = model.predict(test[FEATURE_COLUMNS])

        correct += int((preds == test["label"]).sum())
        total += len(test)
        i += step_days

    return correct / total if total > 0 else float("nan")


def get_top_picks(n: int = 10, period: str = "1y") -> tuple[list[RankedStock], float]:
    universe_history = _fetch_universe(period=period)
    if len(universe_history) < 10:
        raise ValueError(
            f"Only fetched {len(universe_history)} of {len(TICKERS)} tickers "
            "(likely Yahoo Finance rate limiting). Try again shortly."
        )

    panel = _build_panel(universe_history)
    panel = _label_top_quintile(panel)

    backtest_acc = _walk_forward_backtest(panel)

    trainable = panel.dropna(subset=FEATURE_COLUMNS + ["label"])
    model = XGBClassifier(
        n_estimators=150, max_depth=3, learning_rate=0.1,
        eval_metric="logloss", use_label_encoder=False,
    )
    model.fit(trainable[FEATURE_COLUMNS], trainable["label"])

    latest = (
        panel.dropna(subset=FEATURE_COLUMNS)
        .sort_values("date")
        .groupby("ticker")
        .tail(1)
        .copy()
    )
    latest["score"] = model.predict_proba(latest[FEATURE_COLUMNS])[:, 1]

    ranked = latest.sort_values("score", ascending=False).head(n)

    results = [
        RankedStock(
            ticker=row["ticker"],
            sector=row["sector"],
            score=round(float(row["score"]), 4),
            momentum_1mo=round(float(row["momentum_1mo"]), 4) if pd.notna(row["momentum_1mo"]) else 0.0,
            rsi_14=round(float(row["rsi_14"]), 2) if pd.notna(row["rsi_14"]) else 0.0,
        )
        for _, row in ranked.iterrows()
    ]

    return results, round(backtest_acc, 4) if not np.isnan(backtest_acc) else 0.0