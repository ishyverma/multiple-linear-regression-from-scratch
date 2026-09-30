import numpy as np


def mae(y_true, y_pred):
    """
    Mean Absolute Error.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    return np.mean(np.abs(y_true - y_pred))


def mse(y_true, y_pred):
    """
    Mean Squared Error.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    return np.mean((y_true - y_pred) ** 2)


def rmse(y_true, y_pred):
    """
    Root Mean Squared Error.
    """
    return np.sqrt(mse(y_true, y_pred))


def r2_score(y_true, y_pred):
    """
    R² coefficient of determination.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)

    return 1 - (ss_res / ss_tot)