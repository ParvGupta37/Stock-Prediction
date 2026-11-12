"""A small Streamlit interface for the stock forecasting pipeline."""

import streamlit as st

from bayesian_stock_forecasting import run_forecast


st.set_page_config(page_title="Bayesian Stock Forecasting", layout="wide")
st.title("Bayesian Linear Regression for Stock Forecasting")
st.write(
    "This application uses Bayesian Linear Regression to forecast stock prices "
    "while quantifying prediction uncertainty."
)

with st.sidebar:
    st.header("Forecast settings")
    ticker = st.text_input("Stock ticker", value="AAPL").strip().upper()
    years = st.number_input("Years of historical data", min_value=1, max_value=20, value=5, step=1)
    run_button = st.button("Run Forecast", type="primary")

