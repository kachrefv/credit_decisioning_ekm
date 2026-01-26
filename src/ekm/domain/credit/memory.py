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

    def get_risk_graph(self) -> Dict[str, Any]:
        """Returns graph data for visualization."""
        return self.graph_engine.get_graph_data()

    def consolidate_risk_factors(self, merge_threshold: float = 0.92) -> Dict[str, Any]:
        """
        Robustly consolidates similar risk factors.
        Ensures semantic integrity (within risk level), topological consistency (graph-aware), 
        and stability (importance-based seeding).
        """
        import time
        start_time = time.time()
        
        # 1. Importance-based Seeding: Sort by number of applications
        factors_list = sorted(
            list(self.risk_factors.values()), 
            key=lambda x: len(x.source_application_ids), 
            reverse=True
        )
        original_count = len(factors_list)
        
        if original_count < 2:
            return {
                "original_count": original_count,
                "consolidated_count": original_count,
                "merged_count": 0,
                "merge_threshold": merge_threshold,
                "duration_seconds": 0.0,
                "status": "No sufficient nodes for consolidation"
            }
        
        # Compute similarity matrix for semantic check
        embeddings = np.stack([rf.embedding for rf in factors_list])
        norm_embeddings = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-9)
        sim_matrix = np.clip(norm_embeddings @ norm_embeddings.T, -1.0, 1.0)
        
        # Pre-fetch neighbors for topological check
        node_neighbors = {}
        for rf in factors_list:
            if rf.id in self.graph_engine.graph:
                node_neighbors[rf.id] = set(self.graph_engine.graph.neighbors(rf.id))
            else:
                node_neighbors[rf.id] = set()

        merged_mask = np.zeros(len(factors_list), dtype=bool)
        consolidated_factors = {}
        
        for i in range(len(factors_list)):
            if merged_mask[i]:
                continue
            
            seed_rf = factors_list[i]
            similar_indices = []
            
            for j in range(i + 1, len(factors_list)):
                if merged_mask[j]:
                    continue
                
                candidate_rf = factors_list[j]
                
                # Semantic Similarity Check
                is_semantically_similar = sim_matrix[i, j] > merge_threshold
                
                # 2. Risk-Level Partitioning: MUST be same risk level
                is_risk_compatible = candidate_rf.risk_level == seed_rf.risk_level
                
                # 3. Topological Consistency: Jaccard Similarity of neighbors
                # Only merge if they share at least one neighbor OR if both are isolates
                neighbors_i = node_neighbors[seed_rf.id]
                neighbors_j = node_neighbors[candidate_rf.id]
                
                has_shared_struct = False
                if not neighbors_i and not neighbors_j:
                    has_shared_struct = True # Both isolates
                elif neighbors_i.intersection(neighbors_j):
                    has_shared_struct = True # Share at least one neighbor
                
                if is_semantically_similar and is_risk_compatible and has_shared_struct:
                    similar_indices.append(j)
            
            if similar_indices:
                # 4. Metadata Aggregation & Representative Selection
                merged_embeddings = [seed_rf.embedding]
                merged_source_ids = list(seed_rf.source_application_ids)
                
                # Combine expert notes if any
                combined_notes = [seed_rf.metadata.get("expert_notes", "")]
                earliest_ts = seed_rf.metadata.get("timestamp", time.time())
                
                for idx in similar_indices:
                    merged_mask[idx] = True
                    merged_embeddings.append(factors_list[idx].embedding)
                    merged_source_ids.extend(factors_list[idx].source_application_ids)
                    
                    if "expert_notes" in factors_list[idx].metadata:
                        combined_notes.append(factors_list[idx].metadata["expert_notes"])
                    
                    ts = factors_list[idx].metadata.get("timestamp", time.time())
                    if ts < earliest_ts:
                        earliest_ts = ts
                
                # Create consolidated factor (Medoid approach: keeps Seed ID and description)
                consolidated_rf = CreditRiskFactor(
                    id=seed_rf.id,
                    risk_factor=seed_rf.risk_factor,
                    risk_level=seed_rf.risk_level,
                    source_application_ids=list(set(merged_source_ids)),
                    embedding=np.mean(merged_embeddings, axis=0),
                    metadata={
                        **seed_rf.metadata,
                        "consolidated_from": len(similar_indices) + 1,
                        "consolidation_timestamp": time.time(),
                        "original_timestamp": earliest_ts,
                        "aggregated_notes": "; ".join(filter(None, set(combined_notes)))
                    }
                )
                consolidated_factors[consolidated_rf.id] = consolidated_rf
            else:
                consolidated_factors[seed_rf.id] = seed_rf
            
            merged_mask[i] = True
        
        # 5. Atomic Update: swap internal state
        self.risk_factors = consolidated_factors
        
        # Rebuild topological signatures and graph
        # This is CRITICAL after consolidation to reflect new node structure
        if self.mode == "Mesh Mode" or original_count > 0:
            self.update_mesh()
        
        duration = time.time() - start_time
        consolidated_count = len(consolidated_factors)
        
        return {
            "original_count": original_count,
            "consolidated_count": consolidated_count,
            "merged_count": original_count - consolidated_count,
            "merge_threshold": merge_threshold,
            "duration_seconds": round(duration, 3),
            "status": "Robust consolidation complete"
        }
