import json
from pathlib import Path

import joblib
import mlflow
import mlflow.pytorch
import mlflow.sklearn
import torch

from src.data.preprocessing import (
    build_full_pipeline,
    clean_data,
    load_data,
    split_data,
)
from src.data.schema import validate_raw
from src.models.baseline import build_baselines, compute_metrics, train_baseline
from src.models.mlp import MLPTrainer
from src.utils.logger import get_logger
from src.utils import settings

logger = get_logger(__name__)

RANDOM_STATE = 42
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

DATA_PATH = settings.data_path
MLFLOW_EXPERIMENT = "churn-prediction"
MLFLOW_TRACKING_URI = settings.mlflow_tracking_uri

MLP_PARAMS = {
    "hidden_dims": [128, 64, 32],
    "dropout_rate": 0.3,
    "lr": 1e-3,
    "batch_size": 64,
    "max_epochs": 100,
    "patience": 10,
    "use_batch_norm": True,
}


def train_mlp(X_train, y_train, X_val, y_val, X_test, y_test, input_dim: int) -> dict:
    with mlflow.start_run(run_name="mlp_pytorch", nested=True):
        mlflow.set_tag("model_type", "neural_network")
        mlflow.set_tag("framework", "pytorch")
        mlflow.log_params(MLP_PARAMS)

        trainer = MLPTrainer(
            input_dim=input_dim,
            **MLP_PARAMS,
            random_state=RANDOM_STATE,
        )
        trainer.fit(X_train, y_train, X_val, y_val)

        y_pred = trainer.predict(X_test)
        y_prob = trainer.predict_proba(X_test)
        metrics = compute_metrics(y_test, y_pred, y_prob)

        for name, val in metrics.items():
            mlflow.log_metric(f"test_{name}", val)

        mlflow.log_dict(
            {
                "train_loss": trainer.history["train_loss"],
                "val_loss": trainer.history["val_loss"],
            },
            "training_history.json",
        )
        mlflow.pytorch.log_model(trainer.model, "model")

        logger.info(
            "MLP — Test F1: {:.4f} | AUC: {:.4f}",
            metrics["f1"], metrics.get("auc_roc", 0),
        )

    return {"trainer": trainer, "metrics": metrics}


def main():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    logger.info("Loading and validating data from {}", DATA_PATH)
    df_raw = load_data(DATA_PATH)
    validate_raw(df_raw)

    df = clean_data(df_raw)
    X_train_df, X_val_df, X_test_df, y_train, y_val, y_test = split_data(df)

    # Pipeline reprodutível: FeatureEngineer → ColumnTransformer
    pipeline = build_full_pipeline()
    X_train_arr = pipeline.fit_transform(X_train_df)
    X_val_arr = pipeline.transform(X_val_df)
    X_test_arr = pipeline.transform(X_test_df)
    joblib.dump(pipeline, MODELS_DIR / "preprocessor.joblib")
    logger.info("Full pipeline fitted. Feature dim: {}", X_train_arr.shape[1])

    results: dict = {}

    with mlflow.start_run(run_name="churn_experiment"):
        mlflow.log_param("dataset", DATA_PATH)
        mlflow.log_param("train_size", len(X_train_df))
        mlflow.log_param("val_size", len(X_val_df))
        mlflow.log_param("test_size", len(X_test_df))
        mlflow.log_param("random_state", RANDOM_STATE)

        logger.info("Training baselines...")
        best_baseline_f1 = 0.0
        best_baseline_pipeline = None
        best_baseline_name = ""

        for name, bl_pipeline, params in build_baselines():
            res = train_baseline(
                bl_pipeline, X_train_df, y_train, X_test_df, y_test, name, params
            )
            results[name] = res["metrics"]
            if res["metrics"]["f1"] > best_baseline_f1:
                best_baseline_f1 = res["metrics"]["f1"]
                best_baseline_pipeline = res["pipeline"]
                best_baseline_name = name

        if best_baseline_pipeline:
            joblib.dump(best_baseline_pipeline, MODELS_DIR / "best_baseline.joblib")
            logger.info(
                "Best baseline: {} (F1={:.4f})", best_baseline_name, best_baseline_f1
            )
            mlflow.log_param("best_baseline", best_baseline_name)
            mlflow.log_metric("best_baseline_f1", best_baseline_f1)

        logger.info("Training MLP PyTorch...")
        mlp_res = train_mlp(
            X_train_arr, y_train.values,
            X_val_arr, y_val.values,
            X_test_arr, y_test.values,
            input_dim=X_train_arr.shape[1],
        )
        results["mlp_pytorch"] = mlp_res["metrics"]

        torch.save(
            {
                "input_dim": X_train_arr.shape[1],
                "hidden_dims": MLP_PARAMS["hidden_dims"],
                "state_dict": mlp_res["trainer"].model.state_dict(),
            },
            MODELS_DIR / "mlp_model.pt",
        )
        with open(MODELS_DIR / "model_config.json", "w") as f:
            json.dump({"input_dim": X_train_arr.shape[1], "hidden_dims": MLP_PARAMS["hidden_dims"]}, f)
        logger.info("Saved mlp_model.pt and model_config.json (input_dim={})", X_train_arr.shape[1])

        mlp_f1 = mlp_res["metrics"]["f1"]
        mlflow.log_metric(
            "mlp_vs_best_baseline_f1_delta", mlp_f1 - best_baseline_f1
        )

    logger.info("\nResults summary:")
    for name, metrics in results.items():
        logger.info(
            "  {:<28} F1={:.4f}  AUC={:.4f}  Precision={:.4f}  Recall={:.4f}",
            name,
            metrics["f1"],
            metrics.get("auc_roc", 0),
            metrics["precision"],
            metrics["recall"],
        )

    with open(MODELS_DIR / "results.json", "w") as f:
        json.dump(
            {k: {m: round(v, 4) for m, v in v.items()} for k, v in results.items()},
            f,
            indent=2,
        )
    logger.info("Results saved to models/results.json")


if __name__ == "__main__":
    main()
