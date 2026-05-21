import logging

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

logger = logging.getLogger(__name__)

TARGET_COL = "Churn Value"

# Columns that leak post-churn information or are identifiers/geographic
_COLS_TO_DROP = [
    "CustomerID",
    "Count",
    "Country",
    "State",
    "City",
    "Zip Code",
    "Lat Long",
    "Latitude",
    "Longitude",
    "Churn Label",
    "Churn Score",
    "CLTV",
    "Churn Reason",
]

NUMERICAL_FEATURES = ["Tenure Months", "Monthly Charges", "Total Charges"]

BINARY_FEATURES = [
    "Senior Citizen",
    "Partner",
    "Dependents",
    "Phone Service",
    "Paperless Billing",
]

CATEGORICAL_FEATURES = [
    "Gender",
    "Multiple Lines",
    "Internet Service",
    "Online Security",
    "Online Backup",
    "Device Protection",
    "Tech Support",
    "Streaming TV",
    "Streaming Movies",
    "Contract",
    "Payment Method",
]


class OutlierClipper(BaseEstimator, TransformerMixin):
    """Limita outliers nas features numéricas via percentil (fit no treino, apply no resto)."""

    def __init__(self, lower_percentile: float = 1.0, upper_percentile: float = 99.0) -> None:
        self.lower_percentile = lower_percentile
        self.upper_percentile = upper_percentile

    def fit(self, X, y=None):
        self.lower_bounds_ = np.percentile(X, self.lower_percentile, axis=0)
        self.upper_bounds_ = np.percentile(X, self.upper_percentile, axis=0)
        return self

    def transform(self, X, y=None):
        return np.clip(X, self.lower_bounds_, self.upper_bounds_)


class TotalChargesImputer(BaseEstimator, TransformerMixin):
    """Coerces Total Charges to numeric and imputes blanks with the training median."""

    def fit(self, X, y=None):
        self.median_ = pd.to_numeric(X["Total Charges"], errors="coerce").median()
        return self

    def transform(self, X):
        X = X.copy()
        X["Total Charges"] = pd.to_numeric(
            X["Total Charges"], errors="coerce"
        ).fillna(self.median_)
        return X


class BinaryEncoder(BaseEstimator, TransformerMixin):
    """Maps Yes/No (and Male/Female for Gender) columns to 1/0.

    Also handles the case where Senior Citizen is already stored as 0/1 int.
    """

    _MAP = {"Yes": 1, "No": 0, "Male": 1, "Female": 0}

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        for col in BINARY_FEATURES:
            if col not in X.columns:
                continue
            if X[col].dtype == object:
                X[col] = X[col].map(self._MAP).fillna(0).astype(int)
            else:
                X[col] = X[col].fillna(0).astype(int)
        return X


def build_preprocessing_pipeline() -> Pipeline:
    numerical_transformer = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("clipper", OutlierClipper()),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    column_transformer = ColumnTransformer(
        [
            ("num", numerical_transformer, NUMERICAL_FEATURES),
            ("bin", "passthrough", BINARY_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ]
    )

    return Pipeline(
        [
            ("total_charges", TotalChargesImputer()),
            ("binary", BinaryEncoder()),
            ("features", column_transformer),
        ]
    )


def load_data(path: str) -> tuple[pd.DataFrame, pd.Series]:
    logger.info("Loading data from %s", path)
    df = pd.read_excel(path)
    logger.info("Loaded %d rows, %d columns", len(df), len(df.columns))

    cols_to_drop = [c for c in _COLS_TO_DROP if c in df.columns]
    df = df.drop(columns=cols_to_drop)

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL].astype(int)

    logger.info("Class distribution — %s", y.value_counts().to_dict())
    return X, y
