import asyncio
import csv
import io
import uuid
from fastapi import FastAPI, HTTPException, Depends, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict, Any
import os
import time
from datetime import datetime
import numpy as np
from dotenv import load_dotenv

# Load environment variables at the very beginning
load_dotenv()

from .auth import get_current_user, AuthManager
from .schemas import (
    BorrowerRequest,
    LoanApplicationRequest,
    CreditDecisionRequest,
    TrainAgentRequest,
    CreditDecisionResponse,
    TrainAgentResponse,
    AnomalyDetectionResponse,
    HealthCheckResponse,
    LoginRequest,
    RegisterRequest,
    TokenResponse,
    BorrowerResponse,
    ListBorrowersResponse,
    RiskFactorStatusResponse,
    UserProfileResponse,
    UserUpdateSchema,
    HumanDecisionRequest,
    ListDecisionsResponse
)
from ..domain.credit.memory import CreditDecisionMemory
from ..domain.credit.models import BorrowerProfile, LoanApplication, CreditDecision
from ..services.embedding_service import SentenceTransformerEmbeddingService

from sqlalchemy.orm import Session
from ekm.infra.database.config import SessionLocal, engine, Base, get_db
from ekm.infra.database.models import BorrowerORM, ApplicationORM, DecisionORM
from ekm.infra.database.repository import CreditRepository

# Database initialization will happen in startup event

deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
credit_memory = CreditDecisionMemory(mesh_threshold=50, deepseek_api_key=deepseek_api_key)

app = FastAPI(title="Credit Decision Memory API", version="1.0.0")

@app.on_event("startup")
async def startup_event():
    # Create database tables if they don't exist
    Base.metadata.create_all(bind=engine)
    
    # Load ALL existing data into memory on startup
    db = SessionLocal()
    try:
        await _ingest_untrained_data(db, full_load=True)
    finally:
        db.close()

async def _ingest_untrained_data(db: Session, full_load: bool = False):
    """Helper to fetch from DB and ingest into credit_memory."""
    if full_load:
        credit_memory.clear_memory(clear_storage=True)

    repo = CreditRepository(db)
    
    # If full_load, we fetch EVERYTHING (for startup)
    # Otherwise, we only fetch UNTRAINED records (for incremental /train)
    filter_trained = None if full_load else False
    
    hist_borrowers = repo.get_borrowers(is_trained=filter_trained, limit=2000)
    hist_apps = repo.get_applications(is_trained=filter_trained, limit=2000)
    hist_decs = repo.get_decisions(is_trained=filter_trained, limit=2000)
    
    if not hist_borrowers and not hist_apps and not hist_decs:
        return 0

    borrowers = [
        BorrowerProfile(
            id=b.id, name=b.name, credit_score=b.credit_score, income=b.income,
            employment_years=b.employment_years, debt_to_income_ratio=b.debt_to_income_ratio,
            address=b.address, phone=b.phone, email=b.email, 
            employment_history=b.employment_history or [],
            embedding=np.random.randn(768),
            metadata=b.extra_metadata or {},
            timestamp=b.timestamp.timestamp() if b.timestamp else time.time()
        ) for b in hist_borrowers
    ]
    
    apps = [
        LoanApplication(
            id=a.id, borrower_id=a.borrower_id, loan_amount=a.loan_amount,
            loan_purpose=a.loan_purpose, term_months=a.term_months,
            interest_rate=a.interest_rate, metadata=a.extra_metadata or {},
            timestamp=a.timestamp.timestamp() if a.timestamp else time.time()
        ) for a in hist_apps
    ]
    
    decs = [
        CreditDecision(
            id=d.id, application_id=d.application_id, borrower_id=d.borrower_id,
            decision=d.decision, risk_score=d.risk_score, confidence=d.confidence,
            reason=d.reason, expert_notes=d.expert_notes,
            similar_cases=d.similar_cases or [],
            metadata=d.extra_metadata or {},
            timestamp=d.timestamp.timestamp() if d.timestamp else time.time()
        ) for d in hist_decs
    ]
    
    # Ingest into memory (Append mode)
    credit_memory.ingest_credit_data(borrowers, apps, decs)
    
    # Mark records as trained in DB
    b_ids = [b.id for b in hist_borrowers]
    a_ids = [a.id for a in hist_apps]
    d_ids = [d.id for d in hist_decs]
    repo.mark_records_as_trained(b_ids, a_ids, d_ids)
    
    return len(hist_apps)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def to_borrower_model(request: BorrowerRequest) -> BorrowerProfile:
    return BorrowerProfile(
        id=request.id, name=request.name, credit_score=request.credit_score,
        income=request.income, employment_years=request.employment_years,
        debt_to_income_ratio=request.debt_to_income_ratio, address=request.address,
        phone=request.phone, email=request.email, employment_history=request.employment_history or [],
        embedding=np.random.randn(768), metadata=request.metadata or {}
    )

def to_application_model(request: LoanApplicationRequest) -> LoanApplication:
    return LoanApplication(
        id=request.id, borrower_id=request.borrower_id, loan_amount=request.loan_amount,
        loan_purpose=request.loan_purpose, term_months=request.term_months,
        interest_rate=request.interest_rate, collateral_value=request.collateral_value,
        down_payment=request.down_payment, property_address=request.property_address,
        embedding=np.random.randn(768), metadata=request.metadata or {}
    )

@app.get("/health", response_model=HealthCheckResponse)
async def health():
    return HealthCheckResponse(status="healthy", timestamp=datetime.utcnow(), version="1.0.0")

@app.post("/train", response_model=TrainAgentResponse, dependencies=[Depends(get_current_user)])
async def train(request: TrainAgentRequest, db: Session = Depends(get_db)):
    """Incremental training: only process what hasn't been trained yet."""
    start = time.time()
    repo = CreditRepository(db)
    
    # 1. Persist any incoming data provided in the request body
    for b in request.borrowers:
        repo.save_borrower(b.dict())
    for a in request.applications:
        repo.save_application(a.dict())
    for d in request.decisions:
        # Decisions are dicts in the request schema
        repo.save_decision(d)
    
    # 2. Trigger incremental ingestion of all untrained data (including what we just saved)
    newly_trained_count = await _ingest_untrained_data(db, full_load=False)
    
    duration = time.time() - start
    return TrainAgentResponse(
        success=True,
        message=f"Incremental training complete. Processed {newly_trained_count} new application(s).",
        trained_models=len(credit_memory.risk_factors),
        training_duration=duration
    )

@app.post("/train/bulk", response_model=TrainAgentResponse, dependencies=[Depends(get_current_user)])
async def train_bulk(file: UploadFile = File(...), db: Session = Depends(get_db)):
    """Upload a CSV to bulk ingest historical training data."""
    start_time = time.time()
    
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files are supported")
    
    try:
        content = await file.read()
        stream = io.StringIO(content.decode("utf-8"))
        reader = csv.DictReader(stream)
        
        required_cols = ['name', 'email', 'income', 'credit_score', 'decision', 'reason']
        
        repo = CreditRepository(db)
        count = 0
        
        for row in reader:
            # Validate required columns
            if not all(col in row for col in required_cols):
                 continue
                 
            # Generate IDs
            borrower_id = f"B-{uuid.uuid4().hex[:8].upper()}"
            app_id = f"A-{uuid.uuid4().hex[:8].upper()}"
            dec_id = f"D-{uuid.uuid4().hex[:8].upper()}"
            
            # Save Borrower
            repo.save_borrower({
                "id": borrower_id,
                "name": row['name'],
                "email": row['email'],
                "income": float(row['income']),
                "credit_score": int(row['credit_score']),
                "employment_years": float(row.get('employment_years', 0)),
                "debt_to_income_ratio": float(row.get('debt_to_income_ratio', 0.3)),
                "address": str(row.get('address', 'Unknown')),
                "phone": str(row.get('phone', 'N/A')),
                "is_trained": False
            })
            
            # Save Application
            repo.save_application({
                "id": app_id,
                "borrower_id": borrower_id,
                "loan_amount": float(row.get('loan_amount', 0)),
                "loan_purpose": str(row.get('loan_purpose', 'Personal')),
                "term_months": int(row.get('term_months', 36)),
                "interest_rate": float(row.get('interest_rate', 0.1)),
                "is_trained": False
            })
            
            # Save Decision
            repo.save_decision({
                "id": dec_id,
                "application_id": app_id,
                "borrower_id": borrower_id,
                "decision": str(row['decision']).lower(),
                "reason": str(row['reason']),
                "expert_notes": str(row.get('expert_notes', '')),
                "risk_score": float(0.5 if row['decision'].lower() == 'approved' else 0.8),
                "confidence": 1.0,
                "is_trained": False
            })
            count += 1
            
        # Trigger incremental ingestion
        newly_trained_count = await _ingest_untrained_data(db, full_load=False)
        
        duration = time.time() - start_time
        return TrainAgentResponse(
            success=True,
            message=f"Bulk training complete. Processed {count} rows. Ingested {newly_trained_count} new events.",
            trained_models=len(credit_memory.risk_factors),
            training_duration=duration
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Bulk training failed: {str(e)}")

@app.post("/decide", response_model=CreditDecisionResponse, dependencies=[Depends(get_current_user)])
async def decide(request: CreditDecisionRequest, db: Session = Depends(get_db)):
    repo = CreditRepository(db)
    # Save borrower only if it's a new profile; prevent re-creating/modifying old ones
    if not repo.get_borrower(request.borrower.id):
        repo.save_borrower(request.borrower.dict())
    
    repo.save_application(request.application.dict())
    
    b = to_borrower_model(request.borrower)
    a = to_application_model(request.application)
    decision = await credit_memory.evaluate_credit_application(b, a)
    # Save decision
    repo.save_decision({
        "id": decision.id,
        "application_id": decision.application_id,
        "borrower_id": decision.borrower_id,
        "decision": decision.decision,
        "risk_score": decision.risk_score,
        "confidence": decision.confidence,
        "reason": decision.reason,
        "similar_cases": decision.similar_cases,
        "metadata": decision.metadata
    })
    return CreditDecisionResponse(
        id=decision.id, application_id=decision.application_id, borrower_id=decision.borrower_id,
        decision=decision.decision, risk_score=decision.risk_score, confidence=decision.confidence,
        reason=decision.reason, similar_cases=decision.similar_cases, timestamp=datetime.fromtimestamp(decision.timestamp),
        borrower_risk_profile=decision.metadata.get("borrower_risk_profile", "n/a"),
        fraud_indicators=decision.metadata.get("fraud_indicators", []),
        credit_metrics=decision.metadata.get("credit_metrics", {})
    )

@app.get("/status", dependencies=[Depends(get_current_user)])
async def status():
    return {"mode": credit_memory.mode, "risk_factors": len(credit_memory.risk_factors)}

@app.get("/status/risk-factors", response_model=RiskFactorStatusResponse, dependencies=[Depends(get_current_user)])
async def risk_factors():
    analytics = credit_memory.get_risk_factor_analytics()
    factors = []
    # risk_factors is now a dict
    for rf in credit_memory.risk_factors.values():
        factors.append({
            "id": rf.id,
            "risk_factor": rf.risk_factor,
            "risk_level": rf.risk_level,
            "source_application_ids": rf.source_application_ids,
            "timestamp": datetime.fromtimestamp(rf.metadata.get("timestamp", time.time()))
        })
    return RiskFactorStatusResponse(
        analytics=analytics,
        factors=factors
    )

@app.post("/auth/login", response_model=TokenResponse)
async def login(request: LoginRequest, db: Session = Depends(get_db)):
    auth_manager = AuthManager()
    return auth_manager.login(request.email, request.password, db)

@app.post("/auth/register")
async def register(request: RegisterRequest, db: Session = Depends(get_db)):
    auth_manager = AuthManager()
    return auth_manager.register(request.email, request.password, request.name, db)

@app.post("/decisions/human", response_model=CreditDecisionResponse, dependencies=[Depends(get_current_user)])
async def human_decision(request: HumanDecisionRequest, db: Session = Depends(get_db)):
    """
    Endpoint to receive human decisions and trigger auto-training.
    """
    repo = CreditRepository(db)
    
    # 1. Save the decision to DB
    decision_id = f"dec_human_{request.application_id}"
    decision_data = {
        "id": decision_id,
        "application_id": request.application_id,
        "borrower_id": request.borrower_id,
        "decision": request.decision,
        "risk_score": request.risk_score,
        "confidence": 1.0,
        "reason": f"Human Decision: {request.reason}",
        "expert_notes": request.expert_notes,
        "similar_cases": [],
        "metadata": request.metadata or {}
    }
    repo.save_decision(decision_data)
    
    # 2. Trigger auto-training for this specific case
    # This will load the application, borrower, and decision into EKM memory
    hist_borrower = repo.get_borrower(request.borrower_id)
    hist_app = repo.get_application(request.application_id)
    
    if hist_borrower and hist_app:
        borrower = BorrowerProfile(
            id=hist_borrower.id, name=hist_borrower.name, credit_score=hist_borrower.credit_score,
            income=hist_borrower.income, employment_years=hist_borrower.employment_years,
            debt_to_income_ratio=hist_borrower.debt_to_income_ratio, address=hist_borrower.address,
            phone=hist_borrower.phone, email=hist_borrower.email,
            employment_history=hist_borrower.employment_history or [],
            embedding=np.random.randn(768), metadata=hist_borrower.extra_metadata or {}
        )
        app = LoanApplication(
            id=hist_app.id, borrower_id=hist_app.borrower_id, loan_amount=hist_app.loan_amount,
            loan_purpose=hist_app.loan_purpose, term_months=hist_app.term_months,
            interest_rate=hist_app.interest_rate, metadata=hist_app.extra_metadata or {}
        )
        decision = CreditDecision(
            id=decision_id, application_id=request.application_id, borrower_id=request.borrower_id,
            decision=request.decision, risk_score=request.risk_score, confidence=1.0,
            reason=f"Human Decision: {request.reason}", expert_notes=request.expert_notes,
            similar_cases=[],
            embedding=app.embedding if app.embedding is not None else np.random.randn(768),
            metadata=request.metadata or {}
        )
        credit_memory.ingest_credit_data([borrower], [app], [decision])
        
        # Mark as trained
        repo.mark_records_as_trained([borrower.id], [app.id], [decision.id])

    return CreditDecisionResponse(
        id=decision_id, application_id=request.application_id, borrower_id=request.borrower_id,
        decision=request.decision, risk_score=request.risk_score, confidence=1.0,
        reason=f"Human Decision: {request.reason}", 
        expert_notes=request.expert_notes,
        similar_cases=[], 
        timestamp=datetime.utcnow(),
        borrower_risk_profile="n/a",
        fraud_indicators=[],
        credit_metrics={}
    )

@app.get("/borrowers", response_model=ListBorrowersResponse, dependencies=[Depends(get_current_user)])
async def list_borrowers(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    repo = CreditRepository(db)
    borrowers = repo.get_borrowers(skip=skip, limit=limit)
    return ListBorrowersResponse(
        borrowers=[BorrowerResponse.model_validate(b) for b in borrowers],
        total=len(borrowers)
    )

@app.get("/decisions", response_model=ListDecisionsResponse, dependencies=[Depends(get_current_user)])
async def list_decisions(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    repo = CreditRepository(db)
    decisions = repo.get_decisions(skip=skip, limit=limit)
    return ListDecisionsResponse(
        decisions=[CreditDecisionResponse.model_validate(d) for d in decisions],
        total=len(decisions)
    )

@app.get("/borrowers/{borrower_id}", response_model=BorrowerResponse, dependencies=[Depends(get_current_user)])
async def get_borrower(borrower_id: str, db: Session = Depends(get_db)):
    repo = CreditRepository(db)
    borrower = repo.get_borrower(borrower_id)
    if not borrower:
        raise HTTPException(status_code=404, detail="Borrower not found")
    return BorrowerResponse.model_validate(borrower)

@app.post("/borrowers", response_model=BorrowerResponse, dependencies=[Depends(get_current_user)])
async def create_borrower(request: BorrowerRequest, db: Session = Depends(get_db)):
    repo = CreditRepository(db)
    borrower = repo.save_borrower(request.dict())
    return BorrowerResponse.model_validate(borrower)

@app.put("/borrowers/{borrower_id}", response_model=BorrowerResponse, dependencies=[Depends(get_current_user)])
async def update_borrower(borrower_id: str, request: BorrowerRequest, db: Session = Depends(get_db)):
    repo = CreditRepository(db)
    data = request.dict()
    data['id'] = borrower_id # Ensure ID matches path
    borrower = repo.save_borrower(data)
    return BorrowerResponse.model_validate(borrower)

@app.get("/users/me", response_model=UserProfileResponse, dependencies=[Depends(get_current_user)])
async def read_users_me(current_user: dict = Depends(get_current_user)):
    """
    Get current user profile.
    The current_user dict comes from the decoded JWT token.
    """
    return UserProfileResponse(
        email=current_user.get("email"),
        name=current_user.get("name") or current_user.get("nickname"),
        picture=current_user.get("picture")
    )

@app.put("/users/me", response_model=UserProfileResponse, dependencies=[Depends(get_current_user)])
async def update_user_me(user_update: UserUpdateSchema, current_user: dict = Depends(get_current_user)):
    """
    Update current user profile.
    Note: direct update to Auth0 requires Management API. 
    For this demo, we will simulate the update or return the existing user.
    """
    # In a real implementation with Auth0, we would call the Management API here.
    # For now, we return the user as if updated, or partial update if we had a local DB for user extras.
    
    # Simulating update by returning the requested changes mixed with current data
    updated_name = user_update.name if user_update.name else current_user.get("name")
    updated_email = user_update.email if user_update.email else current_user.get("email")
    updated_picture = user_update.picture if user_update.picture else current_user.get("picture")
    
    # In a real app, we would hash the password and update DB here if user_update.password is set
    
    return UserProfileResponse(
        email=updated_email,
        name=updated_name,
        picture=updated_picture
    )
