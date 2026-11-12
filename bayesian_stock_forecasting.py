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


def create_features(prices):
    """Build five simple features and the following day's closing-price target."""
    frame = prices[["Date", "Close"]].copy()
    frame["Previous_Close"] = frame["Close"]
    frame["MA_5"] = frame["Close"].rolling(5).mean()
    frame["MA_10"] = frame["Close"].rolling(10).mean()
    frame["MA_20"] = frame["Close"].rolling(20).mean()
    frame["Daily_Return"] = frame["Close"].pct_change()
    frame["Target"] = frame["Close"].shift(-1)
    return frame


def _make_plots(prices, result, ticker):
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    dates = result["test_dates"]
    actual = result["actual"]
    predicted = result["predicted"]
    linear_predicted = result["linear_predicted"]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(dates, actual, label="Actual", linewidth=1.6)
    ax.plot(dates, predicted, label="Bayesian Linear Regression", linewidth=1.6)
    ax.plot(dates, linear_predicted, label="Basic Linear Regression", linewidth=1.4, alpha=0.85)
    ax.set(title=f"Model Prediction Comparison: {ticker}", xlabel="Date", ylabel="Price")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "actual_vs_predicted.png", dpi=160)
    fig.savefig(PLOTS_DIR / "tuned_model_comparison.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(dates, actual, label="Actual", color="black", linewidth=1.3)
    ax.plot(dates, predicted, label="Predicted mean", color="tab:blue", linewidth=1.5)
    ax.fill_between(dates, result["lower"], result["upper"], color="tab:blue", alpha=0.2, label="95% prediction interval")
    ax.set(title=f"Predictions with 95% Uncertainty Interval: {ticker}", xlabel="Date", ylabel="Price")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "prediction_uncertainty.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(prices["Date"], prices["Close"], color="tab:blue", linewidth=1.2)
    ax.set(title=f"Stock Price History: {ticker}", xlabel="Date", ylabel="Closing price")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "stock_history.png", dpi=160)
    plt.close(fig)

    # Separate metric panels keep error metrics and R² on readable scales.
    metric_names = ["MAE", "RMSE", "R²"]
    bayesian_metrics = [result["mae"], result["rmse"], result["r2"]]
    linear_metrics = [result["linear_mae"], result["linear_rmse"], result["linear_r2"]]
    fig, axes = plt.subplots(1, 3, figsize=(10, 4))
    positions = np.arange(2)
    for ax, name, bayes_value, linear_value in zip(
        axes, metric_names, bayesian_metrics, linear_metrics
    ):
        ax.bar(positions, [bayes_value, linear_value], color=["tab:blue", "tab:orange"])
        ax.set_xticks(positions, ["Bayesian", "Basic Linear"])
        ax.set_title(name)
        ax.grid(axis="y", alpha=0.25)
        for position, value in zip(positions, [bayes_value, linear_value]):
            ax.text(position, value, f"{value:.2f}", ha="center", va="bottom", fontsize=9)
    fig.suptitle(f"Test-set Model Metrics: {ticker}")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "model_metrics_comparison.png", dpi=160)
    plt.close(fig)

