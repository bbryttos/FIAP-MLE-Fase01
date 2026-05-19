from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ClienteInput(BaseModel):
    model_config = ConfigDict(
        str_strip_whitespace=True,
        json_schema_extra={
            "example": {
                "SeniorCitizen": 0,
                "tenure": 12,
                "MonthlyCharges": 65.5,
                "TotalCharges": 786.0,
                "gender": "Male",
                "Partner": "Yes",
                "Dependents": "No",
                "PhoneService": "Yes",
                "MultipleLines": "No",
                "InternetService": "Fiber optic",
                "OnlineSecurity": "No",
                "OnlineBackup": "Yes",
                "DeviceProtection": "No",
                "TechSupport": "No",
                "StreamingTV": "No",
                "StreamingMovies": "No",
                "Contract": "Month-to-month",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
            }
        },
    )

    SeniorCitizen: int = Field(..., ge=0, le=1, description="0=Nao, 1=Sim")
    tenure: int = Field(..., ge=0, description="Meses como cliente")
    MonthlyCharges: float = Field(..., gt=0, description="Cobranca mensal em USD")
    TotalCharges: float = Field(..., ge=0, description="Cobranca total acumulada")

    gender: Literal["Male", "Female"]
    Partner: Literal["Yes", "No"]
    Dependents: Literal["Yes", "No"]
    PhoneService: Literal["Yes", "No"]
    MultipleLines: Literal["Yes", "No", "No phone service"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: Literal["Yes", "No", "No internet service"]
    OnlineBackup: Literal["Yes", "No", "No internet service"]
    DeviceProtection: Literal["Yes", "No", "No internet service"]
    TechSupport: Literal["Yes", "No", "No internet service"]
    StreamingTV: Literal["Yes", "No", "No internet service"]
    StreamingMovies: Literal["Yes", "No", "No internet service"]
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: Literal["Yes", "No"]
    PaymentMethod: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ]


class PredictionOutput(BaseModel):
    churn_probability: float = Field(..., description="Probabilidade de churn (0 a 1)")
    prediction: int = Field(..., description="0=Nao churn, 1=Churn")
    risk_level: Literal["low", "medium", "high"]


class HealthOutput(BaseModel):
    status: str
    model_loaded: bool
    version: str
