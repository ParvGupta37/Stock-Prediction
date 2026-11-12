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


def run_forecast(ticker="AAPL", years=5):
    prices = load_stock_data(ticker, years)
    featured = create_features(prices)
    usable = featured.dropna(subset=FEATURES + ["Target"]).reset_index(drop=True)
    if len(usable) < 25:
        raise RuntimeError("Not enough valid observations; at least 25 are needed.")

    X = usable[FEATURES].to_numpy(dtype=float)
    y = usable["Target"].to_numpy(dtype=float)
    split = int(len(usable) * 0.8)
    train_end = int(split * 0.9)
    validation_start = int(train_end * 0.89)
    if validation_start < 2 or validation_start >= train_end or train_end >= split:
        raise RuntimeError("Could not create a chronological 80/20 train/test split.")

    # Keep separate chronological sections for model fitting, alpha selection,
    # interval calibration, and final testing. Test data is never used for tuning.
    validation_scaler = StandardScaler().fit(X[:validation_start])
    validation_train = validation_scaler.transform(X[:validation_start])
    validation_features = validation_scaler.transform(X[validation_start:train_end])
    alpha_scores = {}
    for alpha in ALPHA_CANDIDATES:
        validation_model = BayesianLinearRegression(
            alpha=alpha, fit_intercept=True
        ).fit(validation_train, y[:validation_start])
        validation_prediction = validation_model.predict(validation_features)[0]
        alpha_scores[alpha] = mean_absolute_error(
            y[validation_start:train_end], validation_prediction
        )
    selected_alpha = min(alpha_scores, key=alpha_scores.get)

    # Refit through the tuning period, leaving a later calibration period unseen.
    scaler = StandardScaler().fit(X[:train_end])
    X_train = scaler.transform(X[:train_end])
    X_calibration = scaler.transform(X[train_end:split])
    X_test = scaler.transform(X[split:])
    test_model = BayesianLinearRegression(
        alpha=selected_alpha, fit_intercept=True
    ).fit(X_train, y[:train_end])
    predicted, std, _, _ = test_model.predict_with_uncertainty(X_test)
    # The ordinary linear regression baseline uses exactly the same training rows
    # and standardized feature matrix, with a fitted intercept.
    linear_model = LinearRegression().fit(X_train, y[:train_end])
    linear_predicted = linear_model.predict(X_test)
    actual = y[split:]

    # Calibrate interval width from a held-out chronological calibration segment.
    # This replaces the normal-theory 1.96 multiplier with an empirical quantile.
    calibration_mean, calibration_std, _, _ = test_model.predict_with_uncertainty(
        X_calibration
    )
    calibration_scores = np.abs(y[train_end:split] - calibration_mean) / np.maximum(
        calibration_std, 1e-12
    )
    quantile_rank = min(
        int(np.ceil(0.95 * (len(calibration_scores) + 1))), len(calibration_scores)
    )
    interval_multiplier = float(np.sort(calibration_scores)[quantile_rank - 1])
    lower = predicted - interval_multiplier * std
    upper = predicted + interval_multiplier * std
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    r2 = r2_score(actual, predicted) if len(actual) > 1 else float("nan")
    coverage = np.mean((actual >= lower) & (actual <= upper))
    linear_mae = mean_absolute_error(actual, linear_predicted)
    linear_rmse = np.sqrt(mean_squared_error(actual, linear_predicted))
    linear_r2 = r2_score(actual, linear_predicted) if len(actual) > 1 else float("nan")

    prediction_frame = pd.DataFrame({
        "Date": usable["Date"].iloc[split:].dt.strftime("%Y-%m-%d").to_numpy(),
        "Actual_Price": actual,
        "Predicted_Price": predicted,
        "Linear_Regression_Predicted_Price": linear_predicted,
        "Prediction_Std": std,
        "Lower_95": lower,
        "Upper_95": upper,
    })
    PREDICTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    prediction_frame.to_csv(PREDICTIONS_PATH, index=False)

    # Refit on all labeled history, then forecast one trading day after latest data.
    full_scaler = StandardScaler().fit(X)
    final_model = BayesianLinearRegression(
        alpha=selected_alpha, fit_intercept=True
    ).fit(full_scaler.transform(X), y)
    final_linear_model = LinearRegression().fit(full_scaler.transform(X), y)
    latest_features = featured.dropna(subset=FEATURES).iloc[[-1]][FEATURES].to_numpy(dtype=float)
    latest_features = full_scaler.transform(latest_features)
    next_price, next_std, _, _ = final_model.predict_with_uncertainty(latest_features)
    next_lower = next_price - interval_multiplier * next_std
    next_upper = next_price + interval_multiplier * next_std
    next_linear_price = final_linear_model.predict(
        full_scaler.transform(featured.dropna(subset=FEATURES).iloc[[-1]][FEATURES].to_numpy(dtype=float))
    )
    result = {
        "ticker": ticker.strip().upper(), "years": int(years),
        "training_samples": train_end, "calibration_samples": split - train_end,
        "testing_samples": len(actual),
        "mae": mae, "rmse": rmse, "r2": r2, "coverage": coverage,
        "selected_alpha": selected_alpha,
        "validation_alpha_mae": alpha_scores[selected_alpha],
        "interval_multiplier": interval_multiplier,
        "linear_mae": linear_mae, "linear_rmse": linear_rmse,
        "linear_r2": linear_r2, "linear_next_price": float(next_linear_price[0]),
        "latest_close": float(prices["Close"].iloc[-1]),
        "next_price": float(next_price[0]), "next_std": float(next_std[0]),
        "next_lower": float(next_lower[0]), "next_upper": float(next_upper[0]),
        "test_dates": usable["Date"].iloc[split:].to_numpy(), "actual": actual,
        "predicted": predicted, "lower": lower, "upper": upper,
        "linear_predicted": linear_predicted,
        "predictions": prediction_frame, "prices": prices,
    }
    _make_plots(prices, result, result["ticker"])
    return result


def main():
    result = run_forecast()
    print("=" * 50)
    print("BAYESIAN STOCK FORECASTING RESULTS")
    print("=" * 50)
    print(f"Stock: {result['ticker']}")
    print(f"Training Samples: {result['training_samples']}")
    print(f"Testing Samples: {result['testing_samples']}")
    print(f"Calibration Samples: {result['calibration_samples']}")
    print(f"MAE: {result['mae']:.2f}")
    print(f"RMSE: {result['rmse']:.2f}")
    print(f"R² Score: {result['r2']:.2f}")
    print(f"95% Interval Coverage: {result['coverage']:.2%}")
    print(f"Selected Prior Strength (alpha): {result['selected_alpha']:g}")
    print(f"Validation MAE for selected alpha: {result['validation_alpha_mae']:.2f}")
    print(f"Calibrated interval multiplier: {result['interval_multiplier']:.2f}")
    print("\nMODEL COMPARISON (chronological test set)")
    print(f"Bayesian Linear Regression - MAE: {result['mae']:.2f}, RMSE: {result['rmse']:.2f}, R²: {result['r2']:.2f}")
    print(f"Basic Linear Regression   - MAE: {result['linear_mae']:.2f}, RMSE: {result['linear_rmse']:.2f}, R²: {result['linear_r2']:.2f}")
    print(f"Next Day Predicted Price: ${result['next_price']:.2f}")
    print(f"Basic Linear Regression Next Day Estimate: ${result['linear_next_price']:.2f}")
    print(f"95% Prediction Interval: ${result['next_lower']:.2f} - ${result['next_upper']:.2f}")
    print("=" * 50)
    print(f"Saved predictions to {PREDICTIONS_PATH}")
    print(f"Saved plots to {PLOTS_DIR}")


if __name__ == "__main__":
    main()
