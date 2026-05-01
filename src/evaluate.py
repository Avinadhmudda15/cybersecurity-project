from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neural_network import MLPClassifier


@dataclass
class EvalResult:
    name: str
    model: Any
    accuracy: float
    macro_f1: float
    weighted_f1: float


def build_models(random_state: int = 42) -> dict[str, Any]:
    models: dict[str, Any] = {
        "logistic_regression": LogisticRegression(
            max_iter=2000,
            random_state=random_state,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=200,
            random_state=random_state,
            n_jobs=-1,
            class_weight="balanced_subsample",
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            max_depth=12,
            max_iter=250,
            learning_rate=0.05,
            random_state=random_state,
        ),
        "mlp_neural_net": MLPClassifier(
            hidden_layer_sizes=(256, 128),
            max_iter=80,
            early_stopping=True,
            validation_fraction=0.1,
            random_state=random_state,
        ),
    }
    try:
        from xgboost import XGBClassifier  # type: ignore[import-untyped]

        models["xgboost"] = XGBClassifier(
            n_estimators=300,
            max_depth=8,
            learning_rate=0.08,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=random_state,
            tree_method="hist",
            n_jobs=-1,
            verbosity=0,
        )
    except ImportError:
        pass
    return models


def fit_eval(
    name: str,
    model: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    labels: list[int],
) -> EvalResult:
    m = clone(model)
    m.fit(X_train, y_train)
    pred = m.predict(X_test)
    acc = float(accuracy_score(y_test, pred))
    macro = float(f1_score(y_test, pred, average="macro", zero_division=0, labels=labels))
    weighted = float(f1_score(y_test, pred, average="weighted", zero_division=0, labels=labels))
    return EvalResult(name=name, model=m, accuracy=acc, macro_f1=macro, weighted_f1=weighted)
