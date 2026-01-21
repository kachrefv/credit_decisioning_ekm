from typing import List, Tuple
import numpy as np
from .models import CreditRiskFactor
from ...core.retrieval import EKMRetriever
from qdrant_client.http import models as qmodels

class CreditDecisionRetriever(EKMRetriever):
    def __init__(self, d: int = 768, candidate_size: int = 100, collection_name: str = "credit_risk_factors"):
        super().__init__(d=d, candidate_size=candidate_size, collection_name=collection_name)
        self.risk_factors_map = {}

    def build_index(self, risk_factors: List[CreditRiskFactor]):
        import time
        import hashlib
        self.risk_factors_map = {rf.id: rf for rf in risk_factors}
        batch_size = 100
        for i in range(0, len(risk_factors), batch_size):
            batch = risk_factors[i:i + batch_size]
            points = []
            for rf in batch:
                # Use deterministic hash for stable IDs across restarts
                rf_hash = int(hashlib.md5(rf.id.encode()).hexdigest(), 16) % (2**63)
                points.append(qmodels.PointStruct(
                    id=rf_hash,
                    vector=rf.embedding.tolist(),
                    payload={
                        "risk_factor_id": rf.id, 
                        "risk_factor": rf.risk_factor,
                        "risk_level": rf.risk_level,
                        "source_application_ids": rf.source_application_ids
                    }
                ))
            self.client.upsert(collection_name=self.collection_name, points=points)
            time.sleep(1)

    def clear_index(self):
        try:
            self.client.delete_collection(self.collection_name)
        except Exception:
            pass
        self._ensure_collection()

    def retrieve_risk_factors(self, query_embedding: np.ndarray, graph_engine) -> List[Tuple[CreditRiskFactor, float]]:
        # This mirrors the parent logic but with a different map
        search_result = self.client.query_points(
            collection_name=self.collection_name,
            query=query_embedding.tolist(),
            limit=self.candidate_size
        )
        candidates = []
        for res in search_result.points if hasattr(search_result, 'points') else search_result:
            rf_id = res.payload.get("risk_factor_id")
            if rf_id in self.risk_factors_map:
                candidates.append(self.risk_factors_map[rf_id])
        
        if not candidates: return []

        q = query_embedding @ self.W_Q
        ks = np.stack([cand.structural_signature @ self.W_K for cand in candidates])
        vs = np.stack([cand.embedding @ self.W_V for cand in candidates])
        scores = (q @ ks.T) / np.sqrt(self.d)
        
        B = np.zeros((len(candidates), len(candidates)))
        for i, c_i in enumerate(candidates):
            for j, c_j in enumerate(candidates):
                if graph_engine.graph.has_edge(c_i.id, c_j.id):
                    B[i, j] = graph_engine.graph[c_i.id][c_j.id]['weight']
                else: B[i, j] = -1e9
        
        final_scores = np.exp(scores - np.max(scores))
        final_scores /= np.sum(final_scores)
        return sorted(zip(candidates, final_scores), key=lambda x: x[1], reverse=True)

    def find_similar_risk_cases(self, query_embedding: np.ndarray, risk_level: str = None) -> List[Tuple[CreditRiskFactor, float]]:
        search_result = self.client.query_points(
            collection_name=self.collection_name,
            query=query_embedding.tolist(),
            limit=self.candidate_size
        )
        candidates = []
        for res in search_result.points if hasattr(search_result, 'points') else search_result:
            rf_id = res.payload.get("risk_factor_id")
            if rf_id in self.risk_factors_map:
                rf = self.risk_factors_map[rf_id]
                if risk_level is None or rf.risk_level == risk_level:
                    candidates.append(rf)

        if not candidates: return []

        query_norm = query_embedding / np.linalg.norm(query_embedding)
        results = []
        for rf in candidates:
            rf_norm = rf.embedding / np.linalg.norm(rf.embedding)
            similarity = np.dot(query_norm, rf_norm)
            results.append((rf, similarity))
        return sorted(results, key=lambda x: x[1], reverse=True)
