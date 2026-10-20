"""Evaluation metrics, all in millimetres of daily rainfall."""
import numpy as np


def rmse(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return float(np.mean(np.abs(y_true - y_pred)))


def r2(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    denom = np.sum((y_true - y_true.mean()) ** 2)
    return float("nan") if denom == 0 else float(1 - np.sum((y_true - y_pred) ** 2) / denom)


def all_metrics(y_true, y_pred):
    return {"MAE": mae(y_true, y_pred), "RMSE": rmse(y_true, y_pred), "R2": r2(y_true, y_pred)}
