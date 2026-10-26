# AI Stock Intelligence Platform — Phase 1

Phase 1 scope: historical data, technical indicators, next-day direction
prediction (with walk-forward backtesting), risk metrics, and a watchlist —
end to end, running in Docker.

Not yet built (later phases): news sentiment, the LLM agent/chatbot,
portfolio analyzer, multi-provider data sources.

## Stack

- **Backend**: FastAPI + SQLAlchemy + PostgreSQL
- **ML**: XGBoost classifier, walk-forward validated
- **Data**: yfinance (no API key required)
- **Frontend**: Streamlit + Plotly
- **Infra**: Docker Compose

## Quickstart

```bash
cp .env.example .env
docker compose up --build
```

- Frontend: http://localhost:8501
- Backend API docs (Swagger): http://localhost:8000/docs
- Postgres: localhost:5432 (user/pass/db: `stockintel`)

No API keys are required for Phase 1 — yfinance is free. The `.env` file has
placeholders for Alpha Vantage / Polygon / News API / Anthropic for later
phases.

## Project layout

```
stock-intel/
├── docker-compose.yml
├── .env.example
├── backend/
│   ├── Dockerfile
│   ├── requirements.txt
│   └── app/
│       ├── main.py              # FastAPI app entrypoint
│       ├── config.py            # env-based settings
│       ├── database.py          # SQLAlchemy engine/session
│       ├── models.py            # Watchlist / Portfolio tables
│       ├── schemas.py           # Pydantic request/response models
│       ├── services/
│       │   ├── stock_data.py    # yfinance wrapper (swap providers here)
│       │   ├── indicators.py    # RSI, MACD, SMA, EMA, Bollinger, ATR, OBV
│       │   └── ml_service.py    # XGBoost direction classifier + backtest
│       └── api/
│           └── routes.py        # all REST endpoints
└── frontend/
    ├── Dockerfile
    ├── requirements.txt
    └── app.py                   # Streamlit UI
```

## API endpoints

| Endpoint | Purpose |
|---|---|
| `GET /api/v1/stocks/{ticker}/quote` | Price, market cap, P/E, 52w range |
| `GET /api/v1/stocks/{ticker}/history?period=1y` | OHLCV history |
| `GET /api/v1/stocks/{ticker}/indicators?period=1y` | RSI, MACD, SMA, EMA, Bollinger Bands |
| `GET /api/v1/stocks/{ticker}/predict?period=2y` | Next-day direction + backtest accuracy |
| `GET /api/v1/stocks/{ticker}/risk?period=2y&benchmark=SPY` | Volatility, beta, Sharpe, max drawdown |
| `GET/POST/DELETE /api/v1/watchlist` | Watchlist CRUD |

## ML methodology (read this before trusting the numbers)

The model predicts **direction** (up/down), not price — direction accuracy
is measurable and honest; a point price forecast mostly isn't.

**Why walk-forward validation, not a random train/test split:** stock
returns are autocorrelated day to day. A random 80/20 split puts days from
the middle of the backtest window into training, so the model effectively
"sees" data from just before and after each test point — that inflates
accuracy in a way that won't survive contact with real, live data. Instead,
`ml_service.py` trains on an expanding window of *only past* data and tests
on the next unseen slice, sliding forward through history. That's a closer
proxy to how the model would actually perform deployed.

**What "confidence" means:** the model's predicted probability for its
chosen class, not a guarantee. The backtest accuracy is surfaced next to it
so you can judge whether the model has real signal or is close to a coin
flip (many day-ahead direction models hover in the 52-58% range — that's
normal and worth stating plainly rather than dressing up).

**Known limitations to be upfront about (good interview material):**
- Trained per-ticker on that ticker's own history only — no cross-sectional
  or market-wide features yet.
- No transaction costs / slippage modeled — a real trading strategy built on
  this would need that.
- `beta` and market-relative metrics depend on `SPY` history also being
  fetchable; if that fetch fails, beta comes back `null` rather than the
  request failing outright.

## Suggested next phases

1. **News sentiment** — News API + FinBERT, cached in Postgres so you're not
   re-scoring the same articles every request.
2. **AI agent/chatbot** — Anthropic API with tool calling over the existing
   endpoints (quote, indicators, predict, risk) as tools; this is mostly
   prompt + orchestration work since the tools already exist.
3. **Portfolio analyzer** — sector exposure and diversification scoring on
   top of `PortfolioHolding`.
4. **Deployment** — this is already Dockerized; a next step would be a
   docker-compose override for prod (gunicorn workers, no `--reload`, secrets
   via a real secret manager instead of `.env`).
