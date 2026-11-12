"""A small, NumPy-only Bayesian linear regression model."""

import numpy as np


class BayesianLinearRegression:
    """Bayesian linear regression with a Gaussian prior on the weights."""

    def __init__(self, alpha=1.0, fit_intercept=False):
        # Feature-weight prior: w ~ N(0, alpha^-1 I). Larger alpha means
        # stronger shrinkage toward zero. The intercept can be left unregularized.
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.beta = None
        self.posterior_mean = None
        self.posterior_covariance = None
        self.n_features_in_ = None

    def _design_matrix(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if self.fit_intercept:
            X = np.column_stack([np.ones(len(X)), X])
        return X

    def fit(self, X, y):
        self.n_features_in_ = np.asarray(X).shape[-1]
        X = self._design_matrix(X)
        y = np.asarray(y, dtype=float).reshape(-1)
        identity = np.eye(X.shape[1])

        # Estimate observation precision beta from ordinary least-squares
        # residuals. The small floor avoids division by zero on perfect fits.
        
        least_squares_weights = np.linalg.lstsq(X, y, rcond=None)[0]
        residuals = y - X @ least_squares_weights
        residual_variance = max(float(np.mean(residuals**2)), 1e-8)
        self.beta = 1.0 / residual_variance

        # Posterior covariance S_N and posterior mean m_N follow directly
        # from combining the Gaussian prior with the Gaussian likelihood.
        
        prior_precision = self.alpha * identity
        if self.fit_intercept:
            # Give the intercept a broad (effectively unregularized) prior;
            # alpha regularizes the standardized feature coefficients only.
            prior_precision[0, 0] = 0.0
        precision = prior_precision + self.beta * (X.T @ X)
        self.posterior_covariance = np.linalg.inv(precision)
        self.posterior_mean = self.beta * self.posterior_covariance @ X.T @ y
        return self

