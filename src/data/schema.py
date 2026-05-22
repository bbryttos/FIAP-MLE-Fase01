import logging

import pandas as pd
import pandera as pa
from pandera import Check, Column, DataFrameSchema

logger = logging.getLogger(__name__)

RAW_SCHEMA = DataFrameSchema(
    columns={
        "gender": Column(str, Check.isin(["Male", "Female"])),
        "SeniorCitizen": Column(int, Check.isin([0, 1])),
        "Partner": Column(str, Check.isin(["Yes", "No"])),
        "Dependents": Column(str, Check.isin(["Yes", "No"])),
        "tenure": Column(int, Check.ge(0)),
        "PhoneService": Column(str, Check.isin(["Yes", "No"])),
        "MultipleLines": Column(
            str, Check.isin(["Yes", "No", "No phone service"])
        ),
        "InternetService": Column(
            str, Check.isin(["DSL", "Fiber optic", "No"])
        ),
        "OnlineSecurity": Column(
            str, Check.isin(["Yes", "No", "No internet service"])
        ),
        "OnlineBackup": Column(
            str, Check.isin(["Yes", "No", "No internet service"])
        ),
        "DeviceProtection": Column(
            str, Check.isin(["Yes", "No", "No internet service"])
        ),
        "TechSupport": Column(
            str, Check.isin(["Yes", "No", "No internet service"])
        ),
        "StreamingTV": Column(
            str, Check.isin(["Yes", "No", "No internet service"])
        ),
        "StreamingMovies": Column(
            str, Check.isin(["Yes", "No", "No internet service"])
        ),
        "Contract": Column(
            str, Check.isin(["Month-to-month", "One year", "Two year"])
        ),
        "PaperlessBilling": Column(str, Check.isin(["Yes", "No"])),
        "PaymentMethod": Column(
            str,
            Check.isin([
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)",
            ]),
        ),
        "MonthlyCharges": Column(float, Check.gt(0), coerce=True),
    },
    coerce=True,
    strict=False,  # allow extra columns (customerID, TotalCharges, Churn)
)


def validate_raw(df: pd.DataFrame) -> pd.DataFrame:
    try:
        validated = RAW_SCHEMA.validate(df, lazy=True)
        logger.info("Schema validation passed (%d rows)", len(validated))
        return validated
    except pa.errors.SchemaErrors as exc:
        logger.error("Schema validation failed:\n%s", exc.failure_cases)
        raise
