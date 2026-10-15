import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from torch import nn


@torch.no_grad()
def predict(model: nn.Module, X: np.ndarray, batch_size: int = 4096) -> np.ndarray:
    model.eval()
    preds = []
    for i in range(0, len(X), batch_size):
        preds.append(model(torch.from_numpy(X[i:i + batch_size])).argmax(1).numpy())
    return np.concatenate(preds)


def evaluate(model: nn.Module, X: np.ndarray, y: np.ndarray, benign_class: int = 0) -> dict:
    pred = predict(model, X)
    labels = np.arange(int(max(y.max(), pred.max())) + 1)
    # Detection view: anything not benign counts as an alert.
    is_attack, flagged = y != benign_class, pred != benign_class
    benign = ~is_attack
    return {
        "accuracy": accuracy_score(y, pred),
        "macro_f1": f1_score(y, pred, average="macro", labels=labels, zero_division=0),
        "macro_precision": precision_score(y, pred, average="macro", labels=labels, zero_division=0),
        "macro_recall": recall_score(y, pred, average="macro", labels=labels, zero_division=0),
        "detection_rate": float(flagged[is_attack].mean()) if is_attack.any() else float("nan"),
        "false_alarm_rate": float(flagged[benign].mean()) if benign.any() else float("nan"),
        "confusion_matrix": confusion_matrix(y, pred, labels=labels).tolist(),
    }
