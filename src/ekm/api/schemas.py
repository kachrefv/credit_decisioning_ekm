from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, AliasChoices
from datetime import datetime

class BorrowerRequest(BaseModel):
    id: str
    name: str
    credit_score: int
    income: float
    employment_years: float
    debt_to_income_ratio: float
    address: str
    phone: str
    email: str
    employment_history: Optional[List[Dict[str, Any]]] = None
    metadata: Optional[Dict[str, Any]] = Field(None, validation_alias=AliasChoices("extra_metadata", "metadata"))

class LoanApplicationRequest(BaseModel):
    id: str
    borrower_id: str
    loan_amount: float
    loan_purpose: str
    term_months: int
    interest_rate: float
    collateral_value: Optional[float] = None
    down_payment: Optional[float] = None
    property_address: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = Field(None, validation_alias=AliasChoices("extra_metadata", "metadata"))

class CreditDecisionRequest(BaseModel):
    borrower: BorrowerRequest
    application: LoanApplicationRequest

class TrainAgentRequest(BaseModel):
    borrowers: List[BorrowerRequest]
    applications: List[LoanApplicationRequest]
    decisions: List[Dict[str, Any]]

class CreditDecisionResponse(BaseModel):
    id: str
    application_id: str
    borrower_id: str
    decision: str
    risk_score: float
    confidence: float
    reason: str
    similar_cases: Optional[List[str]] = []
    timestamp: datetime
    borrower_risk_profile: Optional[str] = "n/a"
    fraud_indicators: Optional[List[str]] = []
    credit_metrics: Optional[Dict[str, float]] = {}
    expert_notes: Optional[str] = None
    
    class Config:
        from_attributes = True

class TrainAgentResponse(BaseModel):
    success: bool
    message: str
    trained_models: int
    training_duration: float

class AnomalyDetectionResponse(BaseModel):
    anomalies_detected: int
    anomalous_risk_factors: List[Dict[str, Any]]
    timestamp: datetime

class HealthCheckResponse(BaseModel):
    status: str
    timestamp: datetime
    version: str

class LoginRequest(BaseModel):
    email: str
    password: str

class RegisterRequest(BaseModel):
    email: str
    password: str
    name: Optional[str] = None

class TokenResponse(BaseModel):
    access_token: str
    id_token: Optional[str] = None
    token_type: str
    expires_in: int

class BorrowerResponse(BorrowerRequest):
    timestamp: datetime
    
    class Config:
        from_attributes = True

class ListBorrowersResponse(BaseModel):
    borrowers: List[BorrowerResponse]
    total: int

class RiskFactorSchema(BaseModel):
    id: str
    risk_factor: str
    risk_level: str
    source_application_ids: List[str]
    timestamp: datetime

    class Config:
        from_attributes = True

class RiskFactorAnalyticsSchema(BaseModel):
    total_count: int
    level_distribution: Dict[str, int]
    top_descriptions: List[Dict[str, Any]]

class RiskFactorStatusResponse(BaseModel):
    analytics: RiskFactorAnalyticsSchema
    factors: List[RiskFactorSchema]

class UserProfileResponse(BaseModel):
    email: str
    name: Optional[str] = None
    picture: Optional[str] = None
    # Add other fields as needed, e.g. role, created_at
    
class UserUpdateSchema(BaseModel):
    name: Optional[str] = None
    email: Optional[str] = None
    picture: Optional[str] = None
    password: Optional[str] = None

class HumanDecisionRequest(BaseModel):
    application_id: str
    borrower_id: str
    decision: str
    risk_score: float
    reason: str
    expert_notes: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
class ListDecisionsResponse(BaseModel):
    decisions: List[CreditDecisionResponse]
    total: int
