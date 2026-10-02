from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score, mean_absolute_error, mean_squared_error, precision_score, r2_score, recall_score


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    true = y_true.reshape(-1)
    pred = y_pred.reshape(-1)
    mse = float(mean_squared_error(true, pred))
    return {
        "MSE": mse,
        "MAE": float(mean_absolute_error(true, pred)),
        "RMSE": float(np.sqrt(mse)),
        "R2": float(r2_score(true, pred)),
    }


def classification_metrics(y_true: np.ndarray, logits: np.ndarray) -> dict[str, float]:
    pred = logits.argmax(axis=1)
    return {
        "F1_macro": float(f1_score(y_true, pred, average="macro", zero_division=0)),
        "precision_macro": float(precision_score(y_true, pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, pred, average="macro", zero_division=0)),
    }


def distribution_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    eps = 1e-12
    true = np.clip(y_true.astype(np.float64), eps, 1.0)
    pred = np.clip(y_prob.astype(np.float64), eps, 1.0)
    true = true / true.sum(axis=1, keepdims=True)
    pred = pred / pred.sum(axis=1, keepdims=True)
    return {
        "KL": float(np.mean(np.sum(true * (np.log(true) - np.log(pred)), axis=1))),
        "L1": float(np.mean(np.sum(np.abs(true - pred), axis=1))),
        "Chebyshev": float(np.mean(np.max(np.abs(true - pred), axis=1))),
    }
