import os

import httpx
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8000")

st.set_page_config(page_title="AI Stock Intelligence Platform", page_icon="📈", layout="wide")


@st.cache_data(ttl=300)
def api_get(path: str, params: dict | None = None, timeout: float = 30):
    with httpx.Client(timeout=timeout) as client:
        resp = client.get(f"{BACKEND_URL}{path}", params=params or {})
        resp.raise_for_status()
        return resp.json()


def api_post(path: str, json: dict):
    with httpx.Client(timeout=30) as client:
        resp = client.post(f"{BACKEND_URL}{path}", json=json)
        return resp


def api_delete(path: str):
    with httpx.Client(timeout=30) as client:
        return client.delete(f"{BACKEND_URL}{path}")


st.sidebar.title("📈 Stock Intelligence")
page = st.sidebar.radio("Navigate", ["Stock Analysis", "Screener", "Watchlist"])

# ---------------- Stock Analysis ----------------
if page == "Stock Analysis":
    ticker = st.sidebar.text_input("Ticker symbol", value="AAPL").upper().strip()
    period = st.sidebar.selectbox("History period", ["3mo", "6mo", "1y", "2y", "5y"], index=2)

    st.title(f"{ticker} — Stock Intelligence")

    if not ticker:
        st.info("Enter a ticker to get started.")
        st.stop()

    try:
        quote = api_get(f"/api/v1/stocks/{ticker}/quote")
    except httpx.HTTPStatusError as e:
        try:
            detail = e.response.json().get("detail", str(e))
        except Exception:
            detail = e.response.text[:300] or str(e)
        st.error(f"Couldn't fetch data for '{ticker}': {detail}")
        st.stop()
    except httpx.ConnectError:
        st.error(f"Can't reach the backend at {BACKEND_URL}. Is it running?")
        st.stop()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Price", f"${quote['price']:.2f}" if quote.get("price") else "—")
    col2.metric("Market Cap", f"${quote['market_cap']/1e9:.1f}B" if quote.get("market_cap") else "—")
    col3.metric("P/E Ratio", f"{quote['pe_ratio']:.1f}" if quote.get("pe_ratio") else "—")
    col4.metric("52W Range",
                f"${quote['fifty_two_week_low']:.0f}–${quote['fifty_two_week_high']:.0f}"
                if quote.get("fifty_two_week_low") else "—")

    tabs = st.tabs(["📊 Chart", "📐 Indicators", "🤖 Prediction", "⚠️ Risk", "📰 News Sentiment"])

    # --- Chart tab ---
    with tabs[0]:
        try:
            hist = api_get(f"/api/v1/stocks/{ticker}/history", params={"period": period})
        except httpx.HTTPStatusError as e:
            try:
                detail = e.response.json().get("detail", str(e))
            except Exception:
                detail = e.response.text[:300] or str(e)
            st.error(f"Couldn't load chart data: {detail}")
            st.stop()
        df = pd.DataFrame(hist["data"])
        df["date"] = pd.to_datetime(df["date"])


        fig = go.Figure(data=[go.Candlestick(
            x=df["date"], open=df["open"], high=df["high"], low=df["low"], close=df["close"],
            name=ticker,
        )])
        fig.update_layout(title=f"{ticker} Price ({period})", xaxis_rangeslider_visible=False, height=500)
        st.plotly_chart(fig, use_container_width=True)

        vol_fig = go.Figure(data=[go.Bar(x=df["date"], y=df["volume"], name="Volume")])
        vol_fig.update_layout(title="Volume", height=200)
        st.plotly_chart(vol_fig, use_container_width=True)

    # --- Indicators tab ---
    with tabs[1]:
        ind = api_get(f"/api/v1/stocks/{ticker}/indicators", params={"period": period})
        dates = pd.to_datetime(ind["dates"])

        price_fig = go.Figure()
        price_fig.add_trace(go.Scatter(x=dates, y=ind["close"], name="Close", line=dict(color="white")))
        price_fig.add_trace(go.Scatter(x=dates, y=ind["sma_20"], name="SMA 20"))
        price_fig.add_trace(go.Scatter(x=dates, y=ind["sma_50"], name="SMA 50"))
        price_fig.add_trace(go.Scatter(x=dates, y=ind["bb_upper"], name="BB Upper",
                                        line=dict(dash="dot", color="gray")))
        price_fig.add_trace(go.Scatter(x=dates, y=ind["bb_lower"], name="BB Lower",
                                        line=dict(dash="dot", color="gray")))
        price_fig.update_layout(title="Price with SMA & Bollinger Bands", height=400)
        st.plotly_chart(price_fig, use_container_width=True)

        c1, c2 = st.columns(2)
        with c1:
            rsi_fig = go.Figure()
            rsi_fig.add_trace(go.Scatter(x=dates, y=ind["rsi_14"], name="RSI 14"))
            rsi_fig.add_hline(y=70, line_dash="dash", line_color="red")
            rsi_fig.add_hline(y=30, line_dash="dash", line_color="green")
            rsi_fig.update_layout(title="RSI (14)", height=300)
            st.plotly_chart(rsi_fig, use_container_width=True)

        with c2:
            macd_fig = go.Figure()
            macd_fig.add_trace(go.Scatter(x=dates, y=ind["macd"], name="MACD"))
            macd_fig.add_trace(go.Scatter(x=dates, y=ind["macd_signal"], name="Signal"))
            macd_fig.update_layout(title="MACD", height=300)
            st.plotly_chart(macd_fig, use_container_width=True)

    # --- Prediction tab ---
    with tabs[2]:
        st.caption("Predicts next-day direction using an XGBoost classifier, validated with "
                   "walk-forward backtesting (never trained on future data relative to its test window).")
        if st.button("Run prediction", type="primary"):
            try:
                with st.spinner("Training model and backtesting..."):
                    pred = api_get(f"/api/v1/stocks/{ticker}/predict", params={"period": "2y"})
            except httpx.HTTPStatusError as e:
                st.error(e.response.json().get("detail", str(e)))
            else:
                direction_color = "green" if pred["direction"] == "UP" else "red"
                arrow = "⬆️" if pred["direction"] == "UP" else "⬇️"
                st.markdown(f"### {arrow} Predicted next-day direction: "
                            f":{direction_color}[{pred['direction']}]")
                c1, c2, c3 = st.columns(3)
                c1.metric("Model confidence", f"{pred['confidence']*100:.1f}%")
                c2.metric("Backtest accuracy", f"{pred['backtest_accuracy']*100:.1f}%")
                c3.metric("As of", pred["as_of_date"])
                st.caption(pred["disclaimer"])
                if pred["backtest_accuracy"] < 0.55:
                    st.warning("Backtest accuracy is close to a coin flip (50%). Treat this "
                               "prediction as low-signal.")

    # --- Risk tab ---
    with tabs[3]:
        try:
            risk = api_get(f"/api/v1/stocks/{ticker}/risk", params={"period": "2y"})
        except httpx.HTTPStatusError as e:
            st.error(e.response.json().get("detail", str(e)))
        else:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Annualized Volatility", f"{risk['annualized_volatility']*100:.1f}%")
            c2.metric("Beta (vs SPY)", f"{risk['beta']:.2f}" if risk.get("beta") else "—")
            c3.metric("Sharpe Ratio", f"{risk['sharpe_ratio']:.2f}" if risk.get("sharpe_ratio") else "—")
            c4.metric("Max Drawdown", f"{risk['max_drawdown']*100:.1f}%")
        # --- News Sentiment tab ---
        with tabs[4]:
            st.caption("Recent headlines scored with FinBERT, a language model fine-tuned on "
                       "financial text. First load after a backend restart can take a while "
                       "while the model downloads.")
            if st.button("Load news sentiment", type="primary"):
                try:
                    with st.spinner("Fetching headlines and scoring sentiment..."):
                        news = api_get(f"/api/v1/stocks/{ticker}/news", params={"limit": 10})
                except httpx.HTTPStatusError as e:
                    try:
                        detail = e.response.json().get("detail", str(e))
                    except Exception:
                        detail = e.response.text[:300] or str(e)
                    st.error(detail)
                except httpx.ConnectError:
                    st.error(f"Can't reach the backend at {BACKEND_URL}.")
                else:
                    label_emoji = {"bullish": "🟢", "bearish": "🔴", "neutral": "⚪"}
                    emoji = label_emoji.get(news["overall_label"], "⚪")
                    c1, c2 = st.columns(2)
                    c1.metric("Overall sentiment", f"{emoji} {news['overall_label'].capitalize()}")
                    c2.metric("Score", f"{news['overall_score']:+.2f}")
                    st.caption(news["disclaimer"])

                    for article in news["articles"]:
                        sentiment_emoji = {"positive": "🟢", "negative": "🔴", "neutral": "⚪"}.get(
                            article["sentiment_label"], "⚪"
                        )
                        st.markdown(
                            f"{sentiment_emoji} **[{article['title']}]({article['link']})**  \n"
                            f"*{article['publisher']}* — {article['sentiment_label'].capitalize()} "
                            f"({article['sentiment_score']:+.2f})"
                        )
                        st.divider()

    if st.sidebar.button(f"⭐ Add {ticker} to watchlist"):
        resp = api_post("/api/v1/watchlist", {"ticker": ticker})
        if resp.status_code == 200:
            st.sidebar.success(f"Added {ticker} to watchlist.")
        else:
            st.sidebar.warning(resp.json().get("detail", "Couldn't add to watchlist."))

# ---------------- Screener ----------------
elif page == "Screener":
    st.title("🔍 Stock Screener")
    st.caption(
        "Ranks a curated universe of large-cap stocks by a model-estimated probability "
        "of outperforming the group over the next week, using cross-sectional momentum "
        "and technical features. Validated with walk-forward backtesting, same "
        "methodology as the single-stock prediction tool."
    )

    n_picks = st.slider("How many top picks to show", min_value=5, max_value=20, value=10)

    if st.button("Run screener", type="primary"):
        try:
            with st.spinner("Fetching universe, building features, backtesting, and ranking... "
                             "this can take a minute for ~40 tickers."):
                result = api_get("/api/v1/screen/top-picks", params={"n": n_picks}, timeout=180)
        except httpx.HTTPStatusError as e:
            st.error(e.response.json().get("detail", str(e)))
        except httpx.ConnectError:
            st.error(f"Can't reach the backend at {BACKEND_URL}.")
        else:
            c1, c2 = st.columns(2)
            c1.metric("Backtest accuracy", f"{result['backtest_accuracy']*100:.1f}%")
            c2.metric("Universe size", result["universe_size"])
            st.caption(result["disclaimer"])
            if result["backtest_accuracy"] < 0.55:
                st.warning("Backtest accuracy is close to chance-level for a top-quintile "
                           "classifier. Treat this ranking as low-signal.")

            picks_df = pd.DataFrame(result["picks"])
            picks_df.index = picks_df.index + 1
            picks_df = picks_df.rename(columns={
                "ticker": "Ticker", "sector": "Sector", "score": "Score",
                "momentum_1mo": "1mo Momentum", "rsi_14": "RSI (14)",
            })
            picks_df["Score"] = (picks_df["Score"] * 100).round(1).astype(str) + "%"
            picks_df["1mo Momentum"] = (picks_df["1mo Momentum"] * 100).round(1).astype(str) + "%"

            st.dataframe(picks_df, use_container_width=True)

            fig = go.Figure(data=[go.Bar(
                x=[p["ticker"] for p in result["picks"]],
                y=[p["score"] for p in result["picks"]],
            )])
            fig.update_layout(title="Ranking score by ticker", yaxis_title="Model score", height=350)
            st.plotly_chart(fig, use_container_width=True)



# ---------------- Watchlist ----------------
elif page == "Watchlist":
    st.title("⭐ Watchlist")
    try:
        items = api_get("/api/v1/watchlist")
    except httpx.ConnectError:
        st.error(f"Can't reach the backend at {BACKEND_URL}.")
        st.stop()

    if not items:
        st.info("Your watchlist is empty. Add tickers from the Stock Analysis page.")
    else:
        for item in items:
            c1, c2, c3 = st.columns([2, 5, 1])
            c1.write(f"**{item['ticker']}**")
            c2.write(item.get("note") or "")
            if c3.button("Remove", key=f"remove_{item['id']}"):
                api_delete(f"/api/v1/watchlist/{item['id']}")
                st.rerun()
