"""Main training script — trains baselines + MLP, logs everything to MLflow."""

import logging
import os
import random

import joblib
import mlflow
import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, train_test_split

from src.data.preprocessing import build_preprocessing_pipeline, load_data
from src.models.baseline import evaluate_model, get_baselines
from src.models.mlp import ChurnMLP, predict_proba, train_mlp

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger(__name__)

SEED = 42
DATA_PATH = os.getenv("DATA_PATH", "data/Telco_customer_churn.xlsx")
MODEL_DIR = "models"
EXPERIMENT_NAME = "telco-churn"

RF_SEARCH_SPACE = {
    "n_estimators": [100, 200, 300, 500],
    "max_depth": [None, 5, 10, 15, 20],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", "log2", 0.5],
}

MLP_CONFIG = {
    "hidden_dims": [64, 32, 16],
    "lr": 1e-3,
    "batch_size": 64,
    "max_epochs": 200,
    "dropout": 0.3,
    "patience": 15,
}


def _fix_seeds() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)


def _train_baselines(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> dict[str, dict]:
    results = {}
    for name, model in get_baselines().items():
        with mlflow.start_run(run_name=name, nested=True):
            logger.info("Training %s …", name)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]
            metrics = evaluate_model(y_test, y_pred, y_proba)
            mlflow.log_params({"model_type": name})
            mlflow.log_metrics(metrics)
            results[name] = {"model": model, "metrics": metrics}
            logger.info("%s — AUC=%.4f F1=%.4f", name, metrics["roc_auc"], metrics["f1"])
    return results


def _tune_random_forest(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    n_iter: int = 20,
    cv: int = 5,
) -> dict:
    with mlflow.start_run(run_name="random_forest_tuned", nested=True):
        logger.info("Tuning Random Forest (n_iter=%d, cv=%d) …", n_iter, cv)
        base = RandomForestClassifier(random_state=SEED, class_weight="balanced", n_jobs=-1)
        search = RandomizedSearchCV(
            base,
            param_distributions=RF_SEARCH_SPACE,
            n_iter=n_iter,
            cv=cv,
            scoring="roc_auc",
            random_state=SEED,
            n_jobs=-1,
            verbose=0,
        )
        search.fit(X_train, y_train)
        best = search.best_estimator_
        logger.info("Best RF params: %s", search.best_params_)
        logger.info("Best CV AUC: %.4f", search.best_score_)

        y_pred = best.predict(X_test)
        y_proba = best.predict_proba(X_test)[:, 1]
        metrics = evaluate_model(y_test, y_pred, y_proba)

        mlflow.log_params({"model_type": "random_forest_tuned", **search.best_params_})
        mlflow.log_metric("cv_roc_auc", search.best_score_)
        mlflow.log_metrics(metrics)
        logger.info("RF Tuned — AUC=%.4f F1=%.4f", metrics["roc_auc"], metrics["f1"])

    return {"model": best, "metrics": metrics}


def _train_mlp_run(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> tuple["ChurnMLP", dict]:
    with mlflow.start_run(run_name="mlp", nested=True):
        mlflow.log_params({"model_type": "mlp", **MLP_CONFIG})

        model, history = train_mlp(
            X_train,
            y_train.astype(np.float32),
            X_val,
            y_val.astype(np.float32),
            **MLP_CONFIG,
        )

        y_proba = predict_proba(model, X_test)
        y_pred = (y_proba >= 0.5).astype(int)
        metrics = evaluate_model(y_test, y_pred, y_proba)

        mlflow.log_metrics(metrics)
        mlflow.log_metric("epochs_trained", len(history["train_loss"]))
        logger.info("MLP — AUC=%.4f F1=%.4f", metrics["roc_auc"], metrics["f1"])

    return model, metrics


def main() -> None:
    _fix_seeds()
    os.makedirs(MODEL_DIR, exist_ok=True)
    mlflow.set_experiment(EXPERIMENT_NAME)

    X, y = load_data(DATA_PATH)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.15, stratify=y_train, random_state=SEED
    )
    logger.info("Split — train=%d val=%d test=%d", len(X_train), len(X_val), len(X_test))

    pipeline = build_preprocessing_pipeline()
    X_train_t = pipeline.fit_transform(X_train, y_train)
    X_val_t = pipeline.transform(X_val)
    X_test_t = pipeline.transform(X_test)
    logger.info("Feature dim after preprocessing: %d", X_train_t.shape[1])

    pipeline_path = os.path.join(MODEL_DIR, "preprocessing_pipeline.joblib")
    joblib.dump(pipeline, pipeline_path)
    logger.info("Pipeline saved → %s", pipeline_path)

    with mlflow.start_run(run_name="experiment"):
        mlflow.log_params(
            {
                "seed": SEED,
                "train_size": len(X_train),
                "val_size": len(X_val),
                "test_size": len(X_test),
                "n_features": X_train_t.shape[1],
            }
        )

        logger.info("=== Baselines ===")
        baseline_results = _train_baselines(
            X_train_t, y_train.values, X_test_t, y_test.values
        )

        logger.info("=== RF Tuned ===")
        rf_tuned_result = _tune_random_forest(
            X_train_t, y_train.values, X_test_t, y_test.values
        )

        logger.info("=== MLP ===")
        mlp_model, mlp_metrics = _train_mlp_run(
            X_train_t, y_train.values, X_val_t, y_val.values, X_test_t, y_test.values
        )

    # Save MLP checkpoint
    model_path = os.path.join(MODEL_DIR, "mlp_model.pt")
    torch.save(
        {
            "input_dim": X_train_t.shape[1],
            "hidden_dims": MLP_CONFIG["hidden_dims"],
            "state_dict": mlp_model.state_dict(),
        },
        model_path,
    )
    logger.info("MLP saved → %s", model_path)

    # Summary table
    all_metrics = {
        **{k: v["metrics"] for k, v in baseline_results.items()},
        "random_forest_tuned": rf_tuned_result["metrics"],
        "mlp": mlp_metrics,
    }
    print("\n" + "=" * 60)
    print(f"{'Model':<26} {'AUC-ROC':>8} {'F1':>8} {'PR-AUC':>8}")
    print("-" * 60)
    for name, m in sorted(all_metrics.items(), key=lambda x: -x[1]["roc_auc"]):
        print(f"{name:<26} {m['roc_auc']:>8.4f} {m['f1']:>8.4f} {m['pr_auc']:>8.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
