from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field
import numpy as np
import time

@dataclass
class BorrowerProfile:
    """Credit-specific borrower profile with multimodal data."""
    id: str
    name: str
    credit_score: int
    income: float
    employment_years: float
    debt_to_income_ratio: float
    address: str
    phone: str
    email: str
    employment_history: List[Dict[str, Any]] = field(default_factory=list)
    financial_documents_embeddings: Optional[List[np.ndarray]] = None
    embedding: Optional[np.ndarray] = None
    timestamp: float = field(default_factory=time.time)
    metadata: Dict = field(default_factory=dict)

@dataclass
class LoanApplication:
    """Credit-specific loan application with multimodal data."""
    id: str
    borrower_id: str
    loan_amount: float
    loan_purpose: str
    term_months: int
    interest_rate: float
    collateral_value: Optional[float] = None
    down_payment: Optional[float] = None
    property_address: Optional[str] = None
    embedding: Optional[np.ndarray] = None
    timestamp: float = field(default_factory=time.time)
    metadata: Dict = field(default_factory=dict)

@dataclass
class CreditRiskFactor:
    """Atomic Credit Unit - discrete risk factor proposition."""
    id: str
    risk_factor: str 
    risk_level: str  # "low", "medium", "high", "critical"
    source_application_ids: List[str]
    embedding: Optional[np.ndarray] = None
    structural_signature: Optional[np.ndarray] = None
    status: str = "active"  # "active", "archived"
    timestamp: float = field(default_factory=time.time)
    metadata: Dict = field(default_factory=dict)

@dataclass
class RiskPattern:
    """Global Risk Unit - conceptual cluster of similar risk factors."""
    id: str
    label: str
    risk_factor_ids: List[str]
    risk_profile: str  # "safe", "moderate", "risky", "critical"
    centroid: Optional[np.ndarray] = None
    metadata: Dict = field(default_factory=dict)

@dataclass
class CreditDecision:
    """Credit decision record with explanation."""
    id: str
    application_id: str
    borrower_id: str
    decision: str  # "approved", "rejected", "requires_manual_review"
    risk_score: float  # 0.0 to 1.0
    confidence: float  # 0.0 to 1.0
    reason: str
    expert_notes: Optional[str] = None
    similar_cases: List[str] = field(default_factory=list)
    embedding: Optional[np.ndarray] = None
    timestamp: float = field(default_factory=time.time)
    metadata: Dict = field(default_factory=dict)
