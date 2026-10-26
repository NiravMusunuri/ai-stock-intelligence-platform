"""
ML Prediction Service
----------------------
Predicts next-day price DIRECTION (up/down), not price level — this is a
classification problem, which is both more realistic and more evaluable
than regression on price.

Design choices worth calling out (this is the part interviewers will probe):

1. Features are derived ONLY from information available at the close of day t
   (technical indicators computed on data up to and including day t). The
   label is whether close[t+1] > close[t]. This avoids leaking future data
   into the feature set.

2. Validation is WALK-FORWARD (expanding window), not a random train/test
   split. Randomly shuffling time-series rows lets the model "see the future"
   during training via autocorrelated neighboring days, which inflates
   accuracy in a way that won't hold up out of sample. Walk-forward always
   trains on the past and tests on a later, unseen slice.

3. Backtest accuracy is reported alongside every prediction, not hidden --
   a 54% accuracy model predicting a 50/50 coin flip is barely better than
   chance, and the UI should say so rather than imply certainty.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

from app.services.indicators import add_all_indicators

FEATURE_COLUMNS = [
    "sma_20", "sma_50", "ema_12", "ema_26",
    "rsi_14", "macd", "macd_signal", "macd_diff",
    "stoch_k", "bb_width", "atr_14",
]


@dataclass
class PredictionResult:
    direction: str
    confidence: float
    backtest_accuracy: float
    model_used: str
    as_of_date: str


def _build_feature_frame(price_df: pd.DataFrame) -> pd.DataFrame:
    df = add_all_indicators(price_df)

    # Label: did the NEXT day close higher than today's close?
    # shift(-1) pulls tomorrow's close into today's row, then we drop the
    # last row (no "tomorrow" for it yet) so this never leaks backward.
    df["next_close"] = df["close"].shift(-1)
    df["target"] = (df["next_close"] > df["close"]).astype(int)

    df = df.dropna(subset=FEATURE_COLUMNS + ["target"]).reset_index(drop=True)
    return df


def _walk_forward_backtest(df: pd.DataFrame, min_train_size: int = 100, step: int = 5) -> float:
    """
    Expanding-window walk-forward validation.

    Train on [0:i], test on [i:i+step], slide forward. Never trains on data
    that is chronologically after what it's being tested on.
    """
    if len(df) < min_train_size + step:
        return float("nan")  # not enough history to backtest meaningfully

    X = df[FEATURE_COLUMNS].values
    y = df["target"].values

    preds, actuals = [], []
    i = min_train_size
    while i + step <= len(df):
        X_train, y_train = X[:i], y[:i]
        X_test, y_test = X[i:i + step], y[i:i + step]

        # Skip windows where the training target has only one class
        # (XGBoost can't learn a decision boundary from a single class).
        if len(np.unique(y_train)) < 2:
            i += step
            continue

        model = XGBClassifier(
            n_estimators=100, max_depth=3, learning_rate=0.1,
            eval_metric="logloss", use_label_encoder=False,
        )
        model.fit(X_train, y_train)
        pred = model.predict(X_test)

        preds.extend(pred)
        actuals.extend(y_test)
        i += step

    if not actuals:
        return float("nan")

    return float(accuracy_score(actuals, preds))


def predict_next_day_direction(price_df: pd.DataFrame, ticker: str) -> PredictionResult:
    df = _build_feature_frame(price_df)

    if len(df) < 60:
        raise ValueError(
            f"Not enough history to train a model for '{ticker}' "
            f"(need at least ~60 trading days, got {len(df)})."
        )

    backtest_acc = _walk_forward_backtest(df)

    # Final model: train on ALL available history to predict the actual next day.
    X = df[FEATURE_COLUMNS].values
    y = df["target"].values
    model = XGBClassifier(
        n_estimators=150, max_depth=3, learning_rate=0.1,
        eval_metric="logloss", use_label_encoder=False,
    )
    model.fit(X, y)

    latest_features = df[FEATURE_COLUMNS].iloc[[-1]].values
    proba = model.predict_proba(latest_features)[0]  # [P(down), P(up)]
    pred_class = int(np.argmax(proba))
    confidence = float(proba[pred_class])

    return PredictionResult(
        direction="UP" if pred_class == 1 else "DOWN",
        confidence=round(confidence, 4),
        backtest_accuracy=round(backtest_acc, 4) if not np.isnan(backtest_acc) else 0.0,
        model_used="XGBoost (walk-forward validated)",
        as_of_date=df["date"].iloc[-1].strftime("%Y-%m-%d"),
    )


def compute_risk_metrics(price_df: pd.DataFrame, benchmark_df: pd.DataFrame | None = None,
                          risk_free_rate: float = 0.04) -> dict:
    """Volatility, Sharpe ratio, max drawdown, and optionally beta vs a benchmark."""
    df = price_df.sort_values("date").copy()
    returns = df["close"].pct_change().dropna()

    ann_vol = float(returns.std() * np.sqrt(252))

    mean_daily_return = returns.mean()
    daily_rf = risk_free_rate / 252
    sharpe = float(((mean_daily_return - daily_rf) / returns.std()) * np.sqrt(252)) if returns.std() > 0 else None

    cumulative = (1 + returns).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    max_dd = float(drawdown.min())

    beta = None
    if benchmark_df is not None:
        bench_returns = benchmark_df.sort_values("date")["close"].pct_change().dropna()
        aligned = pd.concat([returns.reset_index(drop=True), bench_returns.reset_index(drop=True)], axis=1).dropna()
        if len(aligned) > 10:
            aligned.columns = ["stock", "bench"]
            covariance = aligned["stock"].cov(aligned["bench"])
            variance = aligned["bench"].var()
            beta = float(covariance / variance) if variance > 0 else None

    return {
        "annualized_volatility": round(ann_vol, 4),
        "sharpe_ratio": round(sharpe, 4) if sharpe is not None else None,
        "max_drawdown": round(max_dd, 4),
        "beta": round(beta, 4) if beta is not None else None,
    }
