import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline

from src.utils.logger import get_logger

logger = get_logger(__name__)

RANDOM_STATE = 42
CV_FOLDS = 5
SCORING = ["accuracy", "f1", "precision", "recall", "roc_auc"]


def get_baselines() -> dict:
    """Retorna dicionário nome → estimador para treino independente do pipeline."""
    return {
        "dummy": DummyClassifier(strategy="stratified", random_state=RANDOM_STATE),
        "logistic_regression": LogisticRegression(
            max_iter=1000, random_state=RANDOM_STATE, class_weight="balanced"
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=100, random_state=RANDOM_STATE, class_weight="balanced", n_jobs=-1
        ),
        "gradient_boosting": GradientBoostingClassifier(
            n_estimators=100, random_state=RANDOM_STATE
        ),
    }


def evaluate_model(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
) -> dict[str, float]:
    """Calcula métricas completas incluindo PR-AUC e componentes da matriz de confusão."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "pr_auc": float(average_precision_score(y_true, y_proba)),
        "f1": float(f1_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred)),
        "recall": float(recall_score(y_true, y_pred)),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
    }


# Alias retrocompatível — mantém interface dos testes existentes
def compute_metrics(y_true, y_pred, y_prob=None) -> dict:
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
    }
    if y_prob is not None:
        metrics["auc_roc"] = roc_auc_score(y_true, y_prob)
        metrics["pr_auc"] = average_precision_score(y_true, y_prob)
    return metrics


def train_baseline(
    pipeline: Pipeline,
    X_train,
    y_train,
    X_test,
    y_test,
    model_name: str,
    params: dict | None = None,
) -> dict:
    import mlflow
    import mlflow.sklearn

    with mlflow.start_run(run_name=model_name, nested=True):
        mlflow.set_tag("model_type", "baseline")
        if params:
            mlflow.log_params(params)

        cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        cv_results = cross_validate(pipeline, X_train, y_train, cv=cv, scoring=SCORING)

        for metric in SCORING:
            mlflow.log_metric(f"cv_{metric}_mean", cv_results[f"test_{metric}"].mean())

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        y_prob = pipeline.predict_proba(X_test)[:, 1] if hasattr(pipeline, "predict_proba") else None

        metrics = compute_metrics(y_test, y_pred, y_prob)
        for name, val in metrics.items():
            mlflow.log_metric(f"test_{name}", val)

        mlflow.sklearn.log_model(pipeline, "model")
        logger.info(
            "{} — Test F1: {:.4f} | AUC: {:.4f}",
            model_name, metrics["f1"], metrics.get("auc_roc", 0),
        )
        logger.info("{}", classification_report(y_test, y_pred))

    return {"pipeline": pipeline, "metrics": metrics}


def build_baselines() -> list:
    """Retorna lista de (nome, pipeline, params) com feature engineering incluso."""
    from src.data.preprocessing import build_full_pipeline

    def _pipeline(classifier) -> Pipeline:
        return Pipeline([("pre", build_full_pipeline()), ("model", classifier)])

    return [
        (
            "dummy_classifier",
            _pipeline(DummyClassifier(strategy="stratified", random_state=RANDOM_STATE)),
            {"strategy": "stratified"},
        ),
        (
            "logistic_regression",
            _pipeline(LogisticRegression(
                random_state=RANDOM_STATE, max_iter=1000, C=1.0, class_weight="balanced"
            )),
            {"C": 1.0, "max_iter": 1000},
        ),
        (
            "random_forest",
            _pipeline(RandomForestClassifier(
                n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1, class_weight="balanced"
            )),
            {"n_estimators": 100},
        ),
        (
            "gradient_boosting",
            _pipeline(GradientBoostingClassifier(
                n_estimators=100, random_state=RANDOM_STATE
            )),
            {"n_estimators": 100},
        ),
    ]
