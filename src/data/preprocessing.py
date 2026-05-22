import logging

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

logger = logging.getLogger(__name__)

RANDOM_STATE = 42
TARGET_COL = "Churn"

NUMERIC_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]
BINARY_COLS = ["SeniorCitizen"]
CATEGORICAL_COLS = [
    "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod",
]

# Colunas geradas por src/features/engineering.py
ENGINEERED_NUMERIC_COLS = ["charges_per_tenure", "num_services"]
ENGINEERED_BINARY_COLS = [
    "is_new_customer", "is_long_term", "is_monthly_contract", "is_electronic_check",
    "has_phoneservice", "has_multiplelines", "has_onlinesecurity", "has_onlinebackup",
    "has_deviceprotection", "has_techsupport", "has_streamingtv", "has_streamingmovies",
]


def load_data(path: str) -> pd.DataFrame:
    logger.info("Loading data from %s", path)
    df = pd.read_csv(path)
    logger.info("Loaded %d rows, %d columns", *df.shape)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    if df["TotalCharges"].dtype == object:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    for col in NUMERIC_COLS:
        if col in df.columns:
            n_missing = df[col].isnull().sum()
            if n_missing > 0:
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)
                logger.info(
                    "Imputed %d nulls in '%s' with median %.2f",
                    n_missing, col, median_val,
                )

    for col in CATEGORICAL_COLS:
        if col in df.columns:
            n_missing = df[col].isnull().sum()
            if n_missing > 0:
                mode_val = df[col].mode()[0]
                df[col] = df[col].fillna(mode_val)
                logger.info(
                    "Imputed %d nulls in '%s' with mode '%s'",
                    n_missing, col, mode_val,
                )

    if TARGET_COL in df.columns:
        df[TARGET_COL] = (
            df[TARGET_COL].str.strip().str.lower() == "yes"
        ).astype(int)

    logger.info("Cleaned data: %d rows, %d columns", *df.shape)
    return df


def build_preprocessor() -> ColumnTransformer:
    """ColumnTransformer que processa features originais + engineered."""
    numeric_transformer = Pipeline([("scaler", StandardScaler())])
    categorical_transformer = Pipeline([
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer(
        transformers=[
            (
                "num",
                numeric_transformer,
                NUMERIC_COLS + ENGINEERED_NUMERIC_COLS,
            ),
            (
                "bin",
                "passthrough",
                BINARY_COLS + ENGINEERED_BINARY_COLS,
            ),
            (
                "cat",
                categorical_transformer,
                CATEGORICAL_COLS,
            ),
        ],
        remainder="drop",
    )


def build_full_pipeline() -> Pipeline:
    """Pipeline sklearn completo: FeatureEngineer → ColumnTransformer.

    Salvo em joblib, garante que inferência e treino passem pelo mesmo
    pré-processamento de forma reprodutível.
    """
    from src.features.engineering import FeatureEngineerTransformer

    return Pipeline([
        ("features", FeatureEngineerTransformer()),
        ("transform", build_preprocessor()),
    ])


def split_data(
    df: pd.DataFrame,
    test_size: float = 0.2,
    val_size: float = 0.1,
) -> tuple:
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_STATE, stratify=y
    )
    val_fraction = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train,
        test_size=val_fraction,
        random_state=RANDOM_STATE,
        stratify=y_train,
    )

    logger.info(
        "Train: %d | Val: %d | Test: %d",
        len(X_train), len(X_val), len(X_test),
    )
    return X_train, X_val, X_test, y_train, y_val, y_test
