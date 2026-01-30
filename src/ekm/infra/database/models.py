from sqlalchemy import Column, String, Integer, Float, JSON, DateTime, Boolean
import uuid

from ekm.infra.database.config import Base
from datetime import datetime

class BorrowerORM(Base):
    __tablename__ = "borrowers"
    __table_args__ = {"extend_existing": True}

    id = Column(String, primary_key=True, index=True)
    name = Column(String)
    credit_score = Column(Integer)
    income = Column(Float)
    employment_years = Column(Float)
    debt_to_income_ratio = Column(Float)
    address = Column(String)
    phone = Column(String)
    email = Column(String)
    employment_history = Column(JSON, default=[])
    is_trained = Column(Boolean, default=False)
    extra_metadata = Column("metadata", JSON, default={})
    timestamp = Column(DateTime, default=datetime.utcnow)

class ApplicationORM(Base):
    __tablename__ = "applications"
    __table_args__ = {"extend_existing": True}

    id = Column(String, primary_key=True, index=True)
    borrower_id = Column(String, index=True)
    loan_amount = Column(Float)
    loan_purpose = Column(String)
    term_months = Column(Integer)
    interest_rate = Column(Float)
    is_trained = Column(Boolean, default=False)
    collateral_value = Column(Float, nullable=True)
    down_payment = Column(Float, nullable=True)
    property_address = Column(String, nullable=True)
    extra_metadata = Column("metadata", JSON, default={})
    timestamp = Column(DateTime, default=datetime.utcnow)

class DecisionORM(Base):
    __tablename__ = "decisions"
    __table_args__ = {"extend_existing": True}

    id = Column(String, primary_key=True, index=True)
    application_id = Column(String, index=True)
    borrower_id = Column(String, index=True)
    decision = Column(String)
    risk_score = Column(Float)
    confidence = Column(Float)
    reason = Column(String)
    expert_notes = Column(String, nullable=True)
    is_trained = Column(Boolean, default=False)
    similar_cases = Column(JSON, default=[])
    extra_metadata = Column("metadata", JSON, default={})
    timestamp = Column(DateTime, default=datetime.utcnow)

class UserORM(Base):
    __tablename__ = "users"
    __table_args__ = {"extend_existing": True}

    id = Column(String, primary_key=True, index=True, default=lambda: str(uuid.uuid4()))
    email = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    name = Column(String, nullable=True)
    picture = Column(String, nullable=True)
    is_active = Column(Boolean, default=True)
    is_superuser = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_login = Column(DateTime, nullable=True)

class RiskFactorORM(Base):
    __tablename__ = "risk_factors"
    __table_args__ = {"extend_existing": True}

    id = Column(String, primary_key=True, index=True)
    risk_factor = Column(String)
    risk_level = Column(String)
    source_application_ids = Column(JSON, default=[])
    status = Column(String, default="active") # "active", "archived"
    embedding = Column(JSON, nullable=True)
    extra_metadata = Column("metadata", JSON, default={})
    timestamp = Column(DateTime, default=datetime.utcnow)
