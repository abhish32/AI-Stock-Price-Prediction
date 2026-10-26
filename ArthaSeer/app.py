"""ArthaSeer dashboard.  Run:  streamlit run app.py"""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from arthaseer.backtest import backtest
from arthaseer.data import load_prices
from arthaseer.model import ArthaSeerModel

st.set_page_config(page_title="ArthaSeer", page_icon="📈", layout="wide")
st.title("📈 ArthaSeer — AI Stock Price Predictor")
st.caption("Educational project. Predictions are not financial advice.")

with st.sidebar:
    ticker = st.text_input("Ticker", "AAPL", help="Examples: AAPL, MSFT, TCS.NS, RELIANCE.NS, ^NSEI")
    period = st.selectbox("History", ["2y", "5y", "10y"], index=1)
    horizon = st.slider("Predict N days ahead", 1, 5, 1)
    days = st.slider("Forecast path (days)", 3, 30, 10)
    run = st.button("Train & Predict", type="primary")


@st.cache_data(show_spinner=False)
def _load(t, p):
    return load_prices(t.strip(), p)


@st.cache_resource(show_spinner=False)
def _train(t, p, h):
    df, _ = _load(t, p)
    return ArthaSeerModel(t, h).fit(df)


if run:
    with st.spinner("Downloading data and training models..."):
        df, src = _load(ticker, period)
        model = _train(ticker, period, horizon)
    if src == "synthetic":
        st.warning("Live data unavailable — showing synthetic demo data.")

    res = model.predict_next(df)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Last close", f"{res['last_close']:.2f}")
    c2.metric(f"Predicted ({horizon}d)", f"{res['predicted_price']:.2f}", f"{res['predicted_return']*100:+.2f}%")
    c3.metric("95% range", f"{res['low_price']:.1f} – {res['high_price']:.1f}")
    c4.metric("Signal", res["signal"])

    fc = model.forecast_path(df, days)
    hist = df["Close"].tail(120)
    fig = go.Figure()
    fig.add_scatter(x=hist.index, y=hist, name="Close")
    fig.add_scatter(x=fc.index, y=fc["forecast"], name="Forecast", line=dict(dash="dash"))
    fig.add_scatter(x=fc.index, y=fc["high"], line=dict(width=0), showlegend=False)
    fig.add_scatter(x=fc.index, y=fc["low"], fill="tonexty", line=dict(width=0), name="Uncertainty")
    fig.update_layout(height=450, margin=dict(l=0, r=0, t=20, b=0))
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Honest validation (walk-forward, no look-ahead)")
    st.dataframe(pd.DataFrame(model.metrics).T.round(4))

    bt = backtest(model.cv_predictions)
    b1, b2 = st.columns(2)
    b1.write("**Strategy**"); b1.json({k: round(v, 3) for k, v in bt["strategy"].items()})
    b2.write("**Buy & hold**"); b2.json({k: round(v, 3) for k, v in bt["buy_and_hold"].items()})
    eq = pd.DataFrame({"Strategy": bt["equity_strategy"], "Buy & hold": bt["equity_buyhold"]})
    st.line_chart(eq)
else:
    st.info("Choose a ticker in the sidebar and press **Train & Predict**.")
