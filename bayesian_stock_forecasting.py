"""Train and evaluate a basic Bayesian stock price forecasting model."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

from models.bayesian_model import BayesianLinearRegression


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "stock_data.csv"
PREDICTIONS_PATH = ROOT / "outputs" / "predictions.csv"
PLOTS_DIR = ROOT / "outputs" / "plots"
FEATURES = ["Previous_Close", "MA_5", "MA_10", "MA_20", "Daily_Return"]
ALPHA_CANDIDATES = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0]


def load_stock_data(ticker="AAPL", years=5):
    """Load a matching local CSV, or download and cache historical data."""
    ticker = ticker.strip().upper()
    if DATA_PATH.exists() and DATA_PATH.stat().st_size > 0:
        try:
            cached = pd.read_csv(DATA_PATH, parse_dates=["Date"])
            if (
                not cached.empty
                and "Ticker" in cached.columns
                and "Years" in cached.columns
                and cached["Ticker"].iloc[0] == ticker
                and int(cached["Years"].iloc[0]) == int(years)
            ):
                return _prepare_prices(cached)
        except (ValueError, KeyError, pd.errors.ParserError):
            pass

    downloaded = yf.download(
        ticker, period=f"{int(years)}y", interval="1d", auto_adjust=False,
        progress=False, threads=False
    )
    if downloaded.empty:
        raise RuntimeError(f"No historical data was returned for {ticker}.")

    # Recent yfinance versions may return a two-level column index for one ticker.
    if isinstance(downloaded.columns, pd.MultiIndex):
        downloaded.columns = downloaded.columns.get_level_values(0)
    downloaded = downloaded.reset_index()
    downloaded.columns = [str(column) for column in downloaded.columns]
    if "Date" not in downloaded.columns:
        downloaded = downloaded.rename(columns={downloaded.columns[0]: "Date"})
    if "Close" not in downloaded.columns:
        raise RuntimeError("Downloaded data does not contain a Close column.")

    downloaded["Date"] = pd.to_datetime(downloaded["Date"]).dt.tz_localize(None)
    downloaded["Ticker"] = ticker
    downloaded["Years"] = int(years)
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    downloaded.to_csv(DATA_PATH, index=False)
    return _prepare_prices(downloaded)


def _prepare_prices(data):
    data = data.copy()
    data["Date"] = pd.to_datetime(data["Date"])
    data = data.sort_values("Date").drop_duplicates("Date").reset_index(drop=True)
    data["Close"] = pd.to_numeric(data["Close"], errors="coerce")
    return data.dropna(subset=["Close"])

