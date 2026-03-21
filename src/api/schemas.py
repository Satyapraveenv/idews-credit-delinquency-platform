"""
IDEWS — FastAPI Request & Response Schemas
Defines the exact shape of data going in and out of the scoring API.
Uses Pydantic v2 for automatic validation — bad data never reaches the model.
"""

from typing import Optional
from pydantic import BaseModel, Field, field_validator


# ----------------------------------------------------------------
# Request Schema
# ----------------------------------------------------------------
class ScoringRequest(BaseModel):
    """
    Input payload for a single account delinquency risk score.
    All fields match the UCI Credit Card Default dataset schema.
    """
    # Account identifier
    account_id: str = Field(..., description="Unique account identifier", example="ACC_001")

    # Credit profile
    LIMIT_BAL: float = Field(..., ge=10_000, le=1_000_000, description="Credit limit in NT dollars", example=150000)
    SEX:       int   = Field(..., ge=1, le=2,               description="1=Male, 2=Female",           example=2)
    EDUCATION: int   = Field(..., ge=0, le=6,               description="1=Graduate, 2=University, 3=High School", example=2)
    MARRIAGE:  int   = Field(..., ge=0, le=3,               description="1=Married, 2=Single, 3=Other", example=1)
    AGE:       int   = Field(..., ge=18, le=100,            description="Age in years",               example=35)

    # Payment delay history (last 6 months)
    # -2=no credit, -1=paid duly, 0=minimum payment, 1=1 month delay, etc.
    PAY_0: int = Field(..., ge=-2, le=8, description="Repayment status September (most recent)", example=0)
    PAY_2: int = Field(..., ge=-2, le=8, description="Repayment status August",                  example=0)
    PAY_3: int = Field(..., ge=-2, le=8, description="Repayment status July",                    example=0)
    PAY_4: int = Field(..., ge=-2, le=8, description="Repayment status June",                    example=0)
    PAY_5: int = Field(..., ge=-2, le=8, description="Repayment status May",                     example=0)
    PAY_6: int = Field(..., ge=-2, le=8, description="Repayment status April",                   example=0)

    # Bill statement amounts (NT dollars)
    BILL_AMT1: float = Field(..., description="Bill amount September", example=85000)
    BILL_AMT2: float = Field(..., description="Bill amount August",    example=80000)
    BILL_AMT3: float = Field(..., description="Bill amount July",      example=76000)
    BILL_AMT4: float = Field(..., description="Bill amount June",      example=72000)
    BILL_AMT5: float = Field(..., description="Bill amount May",       example=68000)
    BILL_AMT6: float = Field(..., description="Bill amount April",     example=65000)

    # Previous payment amounts (NT dollars)
    PAY_AMT1: float = Field(..., ge=0, description="Payment September", example=5000)
    PAY_AMT2: float = Field(..., ge=0, description="Payment August",    example=4500)
    PAY_AMT3: float = Field(..., ge=0, description="Payment July",      example=4000)
    PAY_AMT4: float = Field(..., ge=0, description="Payment June",      example=4200)
    PAY_AMT5: float = Field(..., ge=0, description="Payment May",       example=3800)
    PAY_AMT6: float = Field(..., ge=0, description="Payment April",     example=4100)

    @field_validator("LIMIT_BAL")
    @classmethod
    def validate_credit_limit(cls, v):
        if v <= 0:
            raise ValueError("Credit limit must be positive")
        return v

    class Config:
        json_schema_extra = {
            "example": {
                "account_id": "ACC_001",
                "LIMIT_BAL": 150000, "SEX": 2, "EDUCATION": 2, "MARRIAGE": 1, "AGE": 35,
                "PAY_0": 0, "PAY_2": 0, "PAY_3": 0, "PAY_4": 0, "PAY_5": 0, "PAY_6": 0,
                "BILL_AMT1": 85000, "BILL_AMT2": 80000, "BILL_AMT3": 76000,
                "BILL_AMT4": 72000, "BILL_AMT5": 68000, "BILL_AMT6": 65000,
                "PAY_AMT1": 5000, "PAY_AMT2": 4500, "PAY_AMT3": 4000,
                "PAY_AMT4": 4200, "PAY_AMT5": 3800, "PAY_AMT6": 4100,
            }
        }


# ----------------------------------------------------------------
# Response Schema
# ----------------------------------------------------------------
class RiskFactor(BaseModel):
    """A single contributing factor to the risk score."""
    feature:       str   = Field(..., description="Feature name")
    shap_value:    float = Field(..., description="SHAP contribution (positive = increases risk)")
    feature_value: float = Field(..., description="Actual value of this feature for this account")
    direction:     str   = Field(..., description="'increases' or 'decreases' risk")
    reason_code:   str   = Field(..., description="Plain-English regulatory reason code")


class ScoringResponse(BaseModel):
    """
    Output payload from the IDEWS scoring API.
    Includes the risk score, binary flag, top risk factors, and regulatory codes.
    """
    account_id:             str           = Field(..., description="Echo of input account_id")
    risk_score:             float         = Field(..., ge=0, le=1, description="Probability of default (0=lowest risk, 1=highest risk)")
    risk_flag:              bool          = Field(..., description="True if account is high-risk (score >= operating threshold)")
    risk_band:              str           = Field(..., description="Risk band: LOW / MEDIUM / HIGH / CRITICAL")
    behavioural_risk_score: float         = Field(..., description="Human-interpretable composite risk score (0-100)")
    top_risk_factors:       list[RiskFactor] = Field(..., description="Top contributing risk factors with SHAP explanations")
    adverse_action_codes:   list[str]     = Field(..., description="Regulatory adverse action reason codes (ECOA compliant)")
    model_version:          str           = Field(..., description="Model version that generated this score")
    response_time_ms:       float         = Field(..., description="API response time in milliseconds")


class HealthResponse(BaseModel):
    status:        str  = Field(..., example="healthy")
    model_loaded:  bool = Field(..., description="True if model is loaded and ready")
    model_version: str  = Field(..., description="Currently loaded model version")
    uptime_seconds: float = Field(..., description="API uptime in seconds")


class MetricsResponse(BaseModel):
    total_requests:    int   = Field(..., description="Total scoring requests since startup")
    avg_response_ms:   float = Field(..., description="Average response time in ms")
    high_risk_flagged: int   = Field(..., description="Accounts flagged high risk")
    model_version:     str   = Field(..., description="Active model version")
