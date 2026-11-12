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

