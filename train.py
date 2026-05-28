"""Script principal de treino — baselines + RF tuned + MLP com MLflow tracking."""

import json
import os
import random
from pathlib import Path

import joblib
import mlflow
import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, train_test_split

from src.data.preprocessing import build_preprocessing_pipeline
from src.models.baseline import evaluate_model, get_baselines
from src.models.mlp import ChurnMLP, predict_proba, train_mlp
from src.utils.logger import get_logger

logger = get_logger(__name__)

SEED = 42
DATA_PATH = os.getenv("DATA_PATH", "data/raw/Telco_customer_churn.csv")
MODEL_DIR = Path("models")
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
    torch.cuda.manual_seed_all(SEED)


def _train_baselines(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> dict[str, dict]:
    results = {}
    for name, model in get_baselines().items():
        with mlflow.start_run(run_name=name, nested=True):
            logger.info("Training {} ...", name)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]
            metrics = evaluate_model(y_test, y_pred, y_proba)
            mlflow.log_params({"model_type": name})
            mlflow.log_metrics({k: v for k, v in metrics.items() if isinstance(v, float)})
            results[name] = {"model": model, "metrics": metrics}
            logger.info("{} — AUC={:.4f} F1={:.4f} PR-AUC={:.4f}", name, metrics["roc_auc"], metrics["f1"], metrics["pr_auc"])
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
        logger.info("Tuning Random Forest (n_iter={}, cv={}) ...", n_iter, cv)
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
        logger.info("Best RF params: {}", search.best_params_)
        logger.info("Best CV AUC: {:.4f}", search.best_score_)

        y_pred = best.predict(X_test)
        y_proba = best.predict_proba(X_test)[:, 1]
        metrics = evaluate_model(y_test, y_pred, y_proba)

        mlflow.log_params({"model_type": "random_forest_tuned", **search.best_params_})
        mlflow.log_metric("cv_roc_auc", search.best_score_)
        mlflow.log_metrics({k: v for k, v in metrics.items() if isinstance(v, float)})
        logger.info("RF Tuned — AUC={:.4f} F1={:.4f}", metrics["roc_auc"], metrics["f1"])

    return {"model": best, "metrics": metrics}


def _train_mlp_run(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> tuple[ChurnMLP, dict]:
    with mlflow.start_run(run_name="mlp", nested=True):
        mlflow.log_params({"model_type": "mlp", **{k: str(v) for k, v in MLP_CONFIG.items()}})

        model, history = train_mlp(
            X_train, y_train.astype(np.float32),
            X_val, y_val.astype(np.float32),
            **MLP_CONFIG,
        )

        y_proba = predict_proba(model, X_test)
        y_pred = (y_proba >= 0.5).astype(int)
        metrics = evaluate_model(y_test, y_pred, y_proba)

        mlflow.log_metrics({k: v for k, v in metrics.items() if isinstance(v, float)})
        mlflow.log_metric("epochs_trained", len(history["train_loss"]))
        logger.info("MLP — AUC={:.4f} F1={:.4f} PR-AUC={:.4f}", metrics["roc_auc"], metrics["f1"], metrics["pr_auc"])

    return model, metrics


def main() -> None:
    _fix_seeds()
    MODEL_DIR.mkdir(exist_ok=True)
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
    mlflow.set_experiment(EXPERIMENT_NAME)

    import pandas as pd

    from src.data.preprocessing import clean_data
    raw_df = pd.read_csv(DATA_PATH)
    df = clean_data(raw_df)
    y = df["churn"]
    X = df.drop(columns=["churn"])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.15, stratify=y_train, random_state=SEED
    )
    logger.info("Split — train={} val={} test={}", len(X_train), len(X_val), len(X_test))

    pipeline = build_preprocessing_pipeline()
    X_train_t = pipeline.fit_transform(X_train, y_train)
    X_val_t = pipeline.transform(X_val)
    X_test_t = pipeline.transform(X_test)
    logger.info("Feature dim after preprocessing: {}", X_train_t.shape[1])

    pipeline_path = MODEL_DIR / "preprocessing_pipeline.joblib"
    joblib.dump(pipeline, pipeline_path)
    logger.info("Pipeline saved → {}", pipeline_path)

    with mlflow.start_run(run_name="experiment"):
        mlflow.log_params({
            "seed": SEED,
            "train_size": len(X_train),
            "val_size": len(X_val),
            "test_size": len(X_test),
            "n_features": X_train_t.shape[1],
        })

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
            X_train_t, y_train.values,
            X_val_t, y_val.values,
            X_test_t, y_test.values,
        )

    # Salva checkpoint MLP
    model_path = MODEL_DIR / "mlp_model.pt"
    torch.save(
        {
            "input_dim": X_train_t.shape[1],
            "hidden_dims": MLP_CONFIG["hidden_dims"],
            "state_dict": mlp_model.state_dict(),
        },
        model_path,
    )
    logger.info("MLP saved → {}", model_path)

    # Salva model_config para a API
    cfg_path = MODEL_DIR / "model_config.json"
    with open(cfg_path, "w") as f:
        json.dump({"input_dim": X_train_t.shape[1], "hidden_dims": MLP_CONFIG["hidden_dims"]}, f)

    # Tabela resumo
    all_metrics = {
        **{k: v["metrics"] for k, v in baseline_results.items()},
        "random_forest_tuned": rf_tuned_result["metrics"],
        "mlp": mlp_metrics,
    }
    print("\n" + "=" * 68)
    print(f"{'Model':<26} {'AUC-ROC':>8} {'F1':>8} {'PR-AUC':>8} {'Accuracy':>9}")
    print("-" * 68)
    for name, m in sorted(all_metrics.items(), key=lambda x: -x[1]["roc_auc"]):
        print(f"{name:<26} {m['roc_auc']:>8.4f} {m['f1']:>8.4f} {m['pr_auc']:>8.4f} {m['accuracy']:>9.4f}")
    print("=" * 68)


if __name__ == "__main__":
    main()
