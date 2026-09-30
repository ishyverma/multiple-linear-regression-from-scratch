import numpy as np

class MultipleLinearRegression:
    """
    Multiple Linear Regression implemented from scratch
    using the Ordinary Least Squares (OLS) method.
    """

    def __init__(self):
        self.intercept_ = None
        self.coef_ = None
    
    def fit(self, X, y):
        """
        Fit the model using the OLS closed-form solution.

        Parameters
        ----------
        X : array-like
            Training features.
        y : array-like
            Target values.

        Returns
        -------
        self
            Fitted model.
        """

        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        # Add intercept column to the X matrix
        X_b = np.c_[np.ones(X.shape[0]), X]

        XTX = X_b.T @ X_b
        XTy = X_b.T @ y

        # beta = (X^T X)^(-1) X^T y
        beta = np.linalg.inv(XTX) @ XTy

        self.intercept_ = beta[0]
        self.coef_ = beta[1:]

        return self
    
    def predict(self, X):
        """
        Generate predictions using the fitted model.

        Parameters
        ----------
        X : array-like
            Input features.

        Returns
        -------
        np.ndarray
            Predicted values.
        """

        if self.coef_ is None:
            raise ValueError("Model must be fitted before predictions")

        X = np.asarray(X, dtype=float)

        return X @ self.coef_ + self.intercept_