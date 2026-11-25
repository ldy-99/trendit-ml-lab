"""전처리도 train에서만 fit한다. 검증/시험에서는 transform만 한다."""

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
                             brier_score_loss, confusion_matrix, f1_score, log_loss,
                             precision_score, recall_score, roc_auc_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .data import validate_features

GROWTH_24H = "view_growth_24h_log"


def _pipeline(estimator, numeric, categorical):
    prep = ColumnTransformer([
        ("numeric", StandardScaler(), numeric),
        ("category", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical),
    ], remainder="drop")
    return Pipeline([("preprocess", prep), ("model", estimator)])


def build_forest(config, *, include_24h_growth=False):
    numeric, categorical = config["numeric_features"], config["categorical_features"]
    validate_features(numeric, categorical)
    # The T0 allowlist stays in force. Only the explicit 24h arm adds this feature.
    if include_24h_growth:
        numeric = [*numeric, GROWTH_24H]
    forest = RandomForestClassifier(**config["forest"], random_state=config["seed"])
    return _pipeline(forest, numeric, categorical)


def build_models(config):
    numeric, categorical = config["numeric_features"], config["categorical_features"]
    validate_features(numeric, categorical)

    # 순서가 동률 처리 규칙이다. Dummy도 후보에 포함해 개선이 없으면 그대로 선택한다.
    return {
        "dummy": _pipeline(DummyClassifier(strategy="prior", random_state=config["seed"]),
                           numeric, categorical),
        "logistic": _pipeline(LogisticRegression(**config["logistic"], random_state=config["seed"]),
                              numeric, categorical),
        "random_forest": build_forest(config),
    }


def evaluate(y, probability, threshold):
    prediction = (probability >= threshold).astype(int)
    return {
        "accuracy": float(accuracy_score(y, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(y, prediction)),
        "precision": float(precision_score(y, prediction, zero_division=0)),
        "recall": float(recall_score(y, prediction, zero_division=0)),
        "f1": float(f1_score(y, prediction, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, probability)),
        "average_precision": float(average_precision_score(y, probability)),
        "log_loss": float(log_loss(y, probability, labels=[0, 1])),
        "brier_score": float(brier_score_loss(y, probability)),
        "confusion_matrix": confusion_matrix(y, prediction, labels=[0, 1]).tolist(),
    }


def positive_probability(model, X):
    positive_index = int(np.flatnonzero(model.classes_ == 1)[0])
    return model.predict_proba(X)[:, positive_index]
