# Bayesian Linear Regression for Stock Forecasting

## Problem Statement

Stock prices vary over time and are difficult to forecast. A normal linear regression model returns one estimate, which does not describe how uncertain that estimate is.

## Objective

This B.Tech project uses a simple Bayesian Linear Regression model to estimate the next trading day's closing price and provide a 95% prediction interval. It demonstrates probabilistic forecasting; it does not claim to predict the market reliably.

## Technologies Used

Python, NumPy, pandas, Matplotlib, scikit-learn, yfinance, and Streamlit.

## Dataset

Daily historical prices are downloaded from Yahoo Finance with `yfinance`. The default ticker is AAPL and the default history is five years. Other examples include `MSFT`, `GOOGL`, `RELIANCE.NS`, `TCS.NS`, and `INFY.NS`. The downloaded prices are cached in `data/stock_data.csv`; a different ticker or history length refreshes the cache.

## Features Used

- Previous day's closing price
- 5-day moving average
- 10-day moving average
- 20-day moving average
- Daily return

The target is the following trading day's closing price. The first rows without a complete moving average and the last row without a known next-day target are omitted from model training.

