import math
from pydantic import BaseModel, ConfigDict, Field, field_validator


class TransactionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    transaction_id: str = Field(..., min_length=1, description="Unique transaction ID")
    Time: float = Field(..., ge=0, description="Seconds elapsed since first transaction")
    Amount: float | None = Field(default=None, ge=0, le=1e8, description="Transaction amount (optional)")

    V1: float | None = None
    V2: float | None = None
    V3: float | None = None
    V4: float | None = None
    V5: float | None = None
    V6: float | None = None
    V7: float | None = None
    V8: float | None = None
    V9: float | None = None
    V10: float | None = None
    V11: float | None = None
    V12: float | None = None
    V13: float | None = None
    V14: float | None = None
    V15: float | None = None
    V16: float | None = None
    V17: float | None = None
    V18: float | None = None
    V19: float | None = None
    V20: float | None = None
    V21: float | None = None
    V22: float | None = None
    V23: float | None = None
    V24: float | None = None
    V25: float | None = None
    V26: float | None = None
    V27: float | None = None
    V28: float | None = None

    @field_validator("*", mode="before")
    @classmethod
    def reject_nan_or_inf(cls, v):
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            raise ValueError("NaN or Infinity values are not allowed; use null for missing fields")
        return v


class PredictionResponse(BaseModel):
    transaction_id: str
    fraud_probability: float
    is_fraud: bool
    threshold_used: float
    scoring_mode: str
    latency_ms: float