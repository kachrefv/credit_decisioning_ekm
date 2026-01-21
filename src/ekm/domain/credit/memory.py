from typing import List, Dict, Tuple, Optional, Any
import numpy as np
from datetime import datetime
from .models import BorrowerProfile, LoanApplication, CreditRiskFactor, RiskPattern, CreditDecision
from .graph import CreditGraphEngine
from .retrieval import CreditDecisionRetriever
from .utils import calculate_credit_metrics, assess_borrower_risk_profile, detect_fraud_indicators
from ...infra.deepseek import DeepSeekCreditAgent

class CreditDecisionMemory:
    def __init__(self, d: int = 768, k: int = 10, mesh_threshold: int = 1000, deepseek_api_key: Optional[str] = None):
        self.d = d
        self.k = k
        self.mesh_threshold = mesh_threshold
        self.borrowers = []
        self.applications = []
        self.decisions = []
        self.risk_factors = {} # Changed to dict for uniqueness
        self.graph_engine = CreditGraphEngine(k=k)
        self.retriever = CreditDecisionRetriever(d=d)
        self.mode = "Cold Start"

        try:
            self.deepseek_agent = DeepSeekCreditAgent(api_key=deepseek_api_key)
        except Exception as e:
            print(f"Warning: Could not initialize DeepSeek agent: {e}")
            self.deepseek_agent = None

    def ingest_credit_data(self, borrowers: List[BorrowerProfile], applications: List[LoanApplication], decisions: List[CreditDecision]):
        # Use dicts to deduplicate incoming domain models by ID
        existing_borrowers = {b.id: b for b in self.borrowers}
        for b in borrowers:
            existing_borrowers[b.id] = b
        self.borrowers = list(existing_borrowers.values())

        existing_apps = {a.id: a for a in self.applications}
        for a in applications:
            existing_apps[a.id] = a
        self.applications = list(existing_apps.values())

        existing_decs = {d.id: d for d in self.decisions}
        for d in decisions:
            existing_decs[d.id] = d
        self.decisions = list(existing_decs.values())
        
        for app in applications:
            for rf in self._extract_risk_factors_from_application(app):
                self.risk_factors[rf.id] = rf
        
        for dec in decisions:
            for rf in self._extract_risk_factors_from_decision(dec):
                self.risk_factors[rf.id] = rf

        self._check_mode_shift()
        if self.mode == "Mesh Mode":
            self.update_mesh()

    def _extract_risk_factors_from_application(self, application: LoanApplication) -> List[CreditRiskFactor]:
        risk_factors = []
        borrower = next((b for b in self.borrowers if b.id == application.borrower_id), None)
        if not borrower: return risk_factors
            
        factors = []
        if borrower.credit_score < 600: factors.append(("low_credit_score", "high"))
        elif borrower.credit_score < 700: factors.append(("medium_credit_score", "medium"))
        else: factors.append(("good_credit_score", "low"))
            
        if borrower.debt_to_income_ratio > 0.43: factors.append(("high_debt_to_income", "high"))
        elif borrower.debt_to_income_ratio > 0.36: factors.append(("medium_debt_to_income", "medium"))
        else: factors.append(("low_debt_to_income", "low"))
            
        for i, (f_desc, r_lvl) in enumerate(factors):
            rf = CreditRiskFactor(
                id=f"rf_{application.id}_{i}",
                risk_factor=f_desc,
                risk_level=r_lvl,
                source_application_ids=[application.id],
                embedding=application.embedding if application.embedding is not None else np.random.randn(self.d),
                metadata={"timestamp": application.timestamp, "borrower_id": application.borrower_id}
            )
            risk_factors.append(rf)
        return risk_factors

    def _extract_risk_factors_from_decision(self, decision: CreditDecision) -> List[CreditRiskFactor]:
        risk_level = "low" if decision.decision == "approved" else "high" if decision.decision == "rejected" else "medium"
        return [CreditRiskFactor(
            id=f"rf_decision_{decision.id}",
            risk_factor=f"decision_outcome_{decision.decision}",
            risk_level=risk_level,
            source_application_ids=[decision.application_id],
            embedding=decision.embedding if decision.embedding is not None else np.random.randn(self.d),
            metadata={"timestamp": decision.timestamp, "risk_score": decision.risk_score}
        )]

    def _check_mode_shift(self):
        factors_list = list(self.risk_factors.values())
        self.mode = "Mesh Mode" if len(factors_list) >= self.mesh_threshold else "Cold Start"

    def update_mesh(self):
        factors_list = list(self.risk_factors.values())
        self.graph_engine.build_risk_knn_graph(factors_list)
        self.graph_engine.extract_risk_signatures(factors_list)
        self.retriever.build_index(factors_list)

    async def evaluate_credit_application(self, borrower: BorrowerProfile, application: LoanApplication) -> CreditDecision:
        if self.mode == "Cold Start":
            return CreditDecision(
                id=f"dec_{application.id}",
                application_id=application.id,
                borrower_id=borrower.id,
                decision="requires_human_decision",
                risk_score=0.5,
                confidence=0.0,
                reason="System is in Cold Start mode. Human evaluation required.",
                similar_cases=[],
                embedding=application.embedding if application.embedding is not None else np.random.randn(self.d),
                metadata={}
            )

        if self.deepseek_agent:
            try:
                # 1. Retrieve similar ACUs (Atomic Credit Units) from the mesh
                # This grounds the AI in historical specific risk factors
                similar_results = self.retriever.retrieve_risk_factors(
                    application.embedding if application.embedding is not None else np.random.randn(self.d),
                    self.graph_engine
                )
                similar_acus = [res[0] for res in similar_results[:5]]
                
                # 2. Let DeepSeek reason using the grounded ACU context
                decision = await self.deepseek_agent.make_credit_decision(borrower, application, similar_cases=similar_acus)
                return decision
            except Exception as e:
                print(f"DeepSeek grounded evaluation failed: {e}. Falling back to pending review.")
                return CreditDecision(
                    id=f"err_{application.id}", 
                    application_id=application.id, 
                    borrower_id=borrower.id, 
                    decision="requires_manual_review", 
                    risk_score=0.5, 
                    confidence=0.0, 
                    reason=f"AI System Error: {str(e)}", 
                    similar_cases=[]
                )

        # 3. If AI is not available, we require manual review in Mesh Mode (No heuristic fallback)
        return CreditDecision(
            id=f"dec_{application.id}",
            application_id=application.id,
            borrower_id=borrower.id,
            decision="requires_manual_review",
            risk_score=0.5,
            confidence=0.0,
            reason="AI Evaluation system offline. Manual review required for reliability.",
            similar_cases=[],
            embedding=application.embedding if application.embedding is not None else np.random.randn(self.d),
            metadata={}
        )

    def clear_memory(self, clear_storage: bool = False):
        if clear_storage and hasattr(self.retriever, 'clear_index'):
             self.retriever.clear_index()
             
        self.borrowers = []
        self.applications = []
        self.decisions = []
        self.risk_factors = {}
        self.mode = "Cold Start"
        self.graph_engine = CreditGraphEngine(k=self.k)
        self.retriever = CreditDecisionRetriever(d=self.d)

    def get_risk_factor_analytics(self) -> Dict[str, Any]:
        factors_list = list(self.risk_factors.values())
        total = len(factors_list)
        dist = {"low": 0, "medium": 0, "high": 0, "critical": 0}
        desc_counts = {}

        for rf in factors_list:
            lvl = rf.risk_level.lower()
            if lvl in dist:
                dist[lvl] += 1
            desc_counts[rf.risk_factor] = desc_counts.get(rf.risk_factor, 0) + 1

        top_desc = [
            {"description": k, "count": v} 
            for k, v in sorted(desc_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        ]

        return {
            "total_count": total,
            "level_distribution": dist,
            "top_descriptions": top_desc
        }

    def _find_similar_cases(self, query_embedding: np.ndarray) -> List[str]:
        factors_list = list(self.risk_factors.values())
        if self.mode == "Cold Start" or not factors_list:
            if not factors_list: return []
            embeddings = np.stack([a.embedding for a in factors_list])
            scores = np.stack([a.embedding for a in factors_list]) @ (query_embedding / np.linalg.norm(query_embedding))
            top_idx = np.argsort(scores)[-5:][::-1]
            return [factors_list[i].source_application_ids[0] for i in top_idx if factors_list[i].source_application_ids]
        else:
            results = self.retriever.find_similar_risk_cases(query_embedding)
            return [res[0].source_application_ids[0] for res in results[:5] if res[0].source_application_ids]
