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

## Methodology

1. Load or download historical daily prices.
2. Calculate the five features and next-day target.
3. Split observations chronologically into model-fitting, alpha-validation, interval-calibration, and final test segments.
4. Standardize features using training data only.
5. Select the Bayesian prior strength (`alpha`) using validation data, then compare both regressions on identical test rows.
6. Calibrate Bayesian interval width with held-out chronological calibration residuals.
7. Refit both models on all labeled observations to estimate the next trading day's close.
8. Save predictions and plots under `outputs/`.

## Bayesian Linear Regression

The model assumes a linear relationship between the features and the closing price, with Gaussian observation noise. It uses NumPy to calculate the posterior directly; it does not use sampling or a Bayesian framework.

### Prior and Posterior

The prior on feature weights is `w ~ N(0, alpha^-1 I)`, a Gaussian distribution centered at zero. The intercept is left unregularized so it can represent the average price level. The noise precision `beta` is estimated from least-squares training residuals. Combining this prior with the observed training data gives a Gaussian posterior:

`S_N = (Lambda + beta X^T X)^-1`

`m_N = beta S_N X^T y`

`Lambda` contains zero precision for the intercept and `alpha` on the feature weights.

Here, `m_N` is the posterior mean weight vector and `S_N` is its covariance. The mean describes the estimated weights; the covariance describes uncertainty in those weights.

## Prediction Uncertainty

For a new standardized feature row `x`, the predictive mean is `x m_N`. The predictive variance is `beta^-1 + x S_N x^T`: it includes both observation noise and uncertainty about the weights. The model's initial interval uses 1.96 predictive standard deviations. The pipeline then calibrates its width using a separate chronological calibration segment, and reports empirical coverage on the final test segment. This is an estimate, not a guarantee.

## Evaluation Metrics

- **MAE:** average absolute difference between actual and predicted prices.
- **RMSE:** square root of average squared prediction error; larger errors count more.
- **R²:** proportion of test-set variation explained relative to predicting the test mean.
- **95% interval coverage:** fraction of actual test prices inside their corresponding 95% intervals.

## Project Structure

```text
Bayesian-Stock-Forecasting/
├── data/stock_data.csv
├── models/bayesian_model.py
├── outputs/predictions.csv
├── outputs/plots/
├── app.py
├── bayesian_stock_forecasting.py
├── requirements.txt
└── README.md
```

## Installation

Use Python 3.9 or newer, then install the listed packages:

```bash
python -m pip install -r requirements.txt
```

## How to Run

Run the command-line forecasting script:

```bash
python bayesian_stock_forecasting.py
```

To change the default ticker or history length when using the script, call `run_forecast("MSFT", 5)` from Python. To use the simple interface:

```bash
streamlit run app.py
```

Enter a ticker and history length in the sidebar, then click **Run Forecast**.

## Expected Output

The script prints test metrics for both models, Bayesian interval coverage, the selected `alpha`, and both next-day estimates. Alpha selection and interval calibration use separate earlier chronological segments; the final test period is reserved for comparison. It writes `outputs/predictions.csv`, including `Linear_Regression_Predicted_Price` alongside Bayesian results. Plots are saved in `outputs/plots/`: `actual_vs_predicted.png` and `tuned_model_comparison.png` compare actual prices with both models, `model_metrics_comparison.png` compares MAE, RMSE, and R², while `prediction_uncertainty.png` and `stock_history.png` show the Bayesian interval and price history.

## Limitations

Stock movements can be affected by news, market conditions, and many other factors not represented by these five price-based features. A linear model cannot capture every pattern, and historical test results do not guarantee future performance. The interval depends on the model and Gaussian noise assumptions.

## Future Scope

Possible extensions include comparing against a simple baseline, testing more tickers and time periods, and studying additional features. Any extension should preserve chronological evaluation and clearly report its limitations.
