import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import date

st.set_page_config(page_title="NSE Stock Analyzer", page_icon="📈", layout="wide", initial_sidebar_state="collapsed")
st.title("📈 NSE Stock Analyzer")
st.caption("Educational technical-analysis tool. Levels are estimates, not guaranteed prices or investment advice.")

@st.cache_data(ttl=900)
def load_data(ticker, period, interval):
    df = yf.download(ticker, period=period, interval=interval, auto_adjust=False, progress=False)
    if df.empty:
        return df
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.rename(columns=str.title)
    required = ["Open", "High", "Low", "Close", "Volume"]
    return df.dropna(subset=[c for c in required if c in df.columns])

def pivots(df):
    prev = df.iloc[-2] if len(df) > 1 else df.iloc[-1]
    h, l, c = float(prev["High"]), float(prev["Low"]), float(prev["Close"])
    p = (h + l + c) / 3
    return {"Pivot": p, "R1": 2*p-l, "R2": p+(h-l), "R3": h+2*(p-l),
            "S1": 2*p-h, "S2": p-(h-l), "S3": l-2*(h-p)}

def swing_levels(df, window=5, tolerance_pct=0.8):
    highs, lows = [], []
    h = df["High"].astype(float).to_numpy()
    l = df["Low"].astype(float).to_numpy()
    for i in range(window, len(df)-window):
        if h[i] == np.max(h[i-window:i+window+1]):
            highs.append(h[i])
        if l[i] == np.min(l[i-window:i+window+1]):
            lows.append(l[i])
    price = float(df["Close"].iloc[-1])
    def cluster(vals):
        vals = sorted(vals)
        zones = []
        for v in vals:
            if not zones or abs(v-np.mean(zones[-1]))/max(abs(np.mean(zones[-1])), 1e-9)*100 > tolerance_pct:
                zones.append([v])
            else:
                zones[-1].append(v)
        return [float(np.mean(z)) for z in zones]
    supports = [x for x in cluster(lows) if x < price]
    resistances = [x for x in cluster(highs) if x > price]
    supports = sorted(supports, key=lambda x: price-x)[:3]
    resistances = sorted(resistances, key=lambda x: x-price)[:3]
    return supports, resistances

def atr(df, period=14):
    high, low, close = df["High"].astype(float), df["Low"].astype(float), df["Close"].astype(float)
    tr = pd.concat([high-low, (high-close.shift()).abs(), (low-close.shift()).abs()], axis=1).max(axis=1)
    return float(tr.rolling(period).mean().iloc[-1])

def rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = -delta.clip(upper=0).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return float((100 - 100/(1+rs)).iloc[-1])

with st.sidebar:
    st.header("Analysis settings")
    symbol = st.text_input("NSE symbol", "KAYNES").strip().upper()
    interval_label = st.selectbox("Chart timeframe", ["Daily", "Weekly", "Monthly"])
    period = st.selectbox("Historical data", ["6mo", "1y", "2y", "5y"], index=1)
    lookback = st.slider("Swing detection sensitivity", 2, 12, 5)
    st.caption("Enter the NSE symbol only, e.g. RELIANCE, TCS, KAYNES.")
    st.caption("Data source: Yahoo Finance (may be delayed or incomplete).")

interval = {"Daily": "1d", "Weekly": "1wk", "Monthly": "1mo"}[interval_label]
ticker = symbol + ".NS"

if not symbol:
    st.info("Enter an NSE symbol in the sidebar.")
    st.stop()

try:
    df = load_data(ticker, period, interval)
    if df.empty or len(df) < 25:
        st.error("Data not found or insufficient. Check the NSE symbol and try again.")
        st.stop()

    current = float(df["Close"].iloc[-1])
    levels = pivots(df)
    supports, resistances = swing_levels(df, window=lookback)
    current_atr = atr(df)
    current_rsi = rsi(df["Close"].astype(float))
    prev_close = float(df["Close"].iloc[-2])
    breakout = "Bullish breakout" if current > max(resistances, default=np.inf) else (
        "Bearish breakdown" if current < min(supports, default=-np.inf) else "No confirmed swing-level breakout"
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Latest close", f"₹{current:,.2f}", f"{current-prev_close:+.2f}")
    c2.metric("RSI (14)", f"{current_rsi:.1f}" if np.isfinite(current_rsi) else "N/A")
    c3.metric("ATR (14)", f"₹{current_atr:,.2f}" if np.isfinite(current_atr) else "N/A")
    c4.metric("Signal check", breakout)

    st.subheader("Estimated support and resistance")
    left, right = st.columns(2)
    with left:
        st.markdown("**Swing support zones**")
        if supports:
            for i, value in enumerate(supports, 1):
                st.write(f"S{i}: ₹{value:,.2f}  ({(value/current-1)*100:+.2f}%)")
        else:
            st.write("No clear swing support found in selected history.")
    with right:
        st.markdown("**Swing resistance zones**")
        if resistances:
            for i, value in enumerate(resistances, 1):
                st.write(f"R{i}: ₹{value:,.2f}  ({(value/current-1)*100:+.2f}%)")
        else:
            st.write("No clear swing resistance found in selected history.")

    st.markdown("**Classic pivot levels (based on previous candle)**")
    pivot_df = pd.DataFrame([{"Level": k, "Price": v} for k, v in levels.items()])
    st.dataframe(pivot_df.style.format({"Price": "₹{:,.2f}"}), use_container_width=True, hide_index=True)

    st.subheader("Price chart with swing zones")
    fig = go.Figure(data=[go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"], name="Price"
    )])
    for i, value in enumerate(supports, 1):
        fig.add_hline(y=value, line_dash="dash", annotation_text=f"Support {i} ₹{value:.2f}")
    for i, value in enumerate(resistances, 1):
        fig.add_hline(y=value, line_dash="dash", annotation_text=f"Resistance {i} ₹{value:.2f}")
    fig.update_layout(height=620, xaxis_rangeslider_visible=False, yaxis_title="Price (₹)")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Illustrative risk levels")
    st.write("These are ATR-based reference levels, not personalized trade instructions.")
    col1, col2, col3 = st.columns(3)
    col1.metric("ATR reference stop (long)", f"₹{current-1.5*current_atr:,.2f}")
    col2.metric("ATR reference target 1", f"₹{current+2*current_atr:,.2f}")
    col3.metric("ATR reference target 2", f"₹{current+3*current_atr:,.2f}")

    st.subheader("Recent data")
    st.dataframe(df.tail(10).sort_index(ascending=False), use_container_width=True)

    st.warning("Important: Support/resistance are zones, not exact prices. A breakout should be checked with candle close, volume, broader trend, liquidity, and company news. This tool does not account for corporate actions perfectly and is not a buy/sell recommendation.")
except Exception as e:
    st.error(f"Could not complete analysis: {e}")
    st.info("Check your internet connection, symbol, and Yahoo Finance data availability.")
