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

if run_button:
    if not ticker:
        st.error("Enter a stock ticker to continue.")
    else:
        try:
            with st.spinner(f"Loading data and forecasting {ticker}..."):
                result = run_forecast(ticker, int(years))
            st.subheader(f"Selected stock: {result['ticker']}")
            st.metric("Latest closing price", f"${result['latest_close']:.2f}")
            col1, col2, col3 = st.columns(3)
            col1.metric("Predicted next-day price", f"${result['next_price']:.2f}")
            col2.metric("Lower 95% bound", f"${result['next_lower']:.2f}")
            col3.metric("Upper 95% bound", f"${result['next_upper']:.2f}")

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("MAE", f"{result['mae']:.2f}")
            col2.metric("RMSE", f"{result['rmse']:.2f}")
            col3.metric("R²", f"{result['r2']:.2f}")
            col4.metric("95% interval coverage", f"{result['coverage']:.1%}")

            st.subheader("Model comparison on the test set")
            comparison = st.columns(3)
            comparison[0].metric("Linear Regression MAE", f"{result['linear_mae']:.2f}")
            comparison[1].metric("Linear Regression RMSE", f"{result['linear_rmse']:.2f}")
            comparison[2].metric("Linear Regression R²", f"{result['linear_r2']:.2f}")
            st.caption(
                f"Next-day basic linear regression estimate: ${result['linear_next_price']:.2f}. "
                f"Bayesian alpha selected on validation data: {result['selected_alpha']:g}. "
                "The Bayesian model also reports a prediction interval."
            )

            st.subheader("Historical stock price")
            st.line_chart(result["prices"].set_index("Date")["Close"])

            st.subheader("Actual vs predicted")
            actual_predicted = result["predictions"].set_index("Date")[[
                "Actual_Price", "Predicted_Price", "Linear_Regression_Predicted_Price"
            ]]
            st.line_chart(actual_predicted)

            st.subheader("Prediction uncertainty")
            uncertainty = result["predictions"].set_index("Date")[["Actual_Price", "Predicted_Price", "Lower_95", "Upper_95"]]
            st.line_chart(uncertainty)
        except Exception as error:
            st.error(f"Forecast failed: {error}")

