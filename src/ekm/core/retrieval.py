import numpy as np
from typing import List, Tuple
from .models import AKU
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
import os

class EKMRetriever:
    def __init__(self, d: int = 768, candidate_size: int = 100, collection_name: str = "ekm_akus"):
        self.d = d
        self.candidate_size = candidate_size
        self.collection_name = collection_name
        
        # Load credentials from environment
        qdrant_url = os.getenv("QDRANT_URL", "https://2394163a-1fee-4cd2-a40d-e4765397680f.europe-west3-0.gcp.cloud.qdrant.io:6333")
        qdrant_api_key = os.getenv("QDRANT_API_KEY", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.2ktTYUzfKgTBlBvU1d2nrWsWP96z3lvdN1OOEWjxY_Y")
        
        self.client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key, timeout=120)
        
        self._ensure_collection()
        self.akus_map = {} 
        
        self.W_Q = np.random.randn(d, d)
        self.W_K = np.random.randn(d + 13, d) 
        self.W_V = np.random.randn(d, d)

    def _ensure_collection(self):
        try:
            collections = self.client.get_collections().collections
            exists = any(c.name == self.collection_name for c in collections)
            if not exists:
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=qmodels.VectorParams(size=self.d, distance=qmodels.Distance.COSINE),
                )
        except Exception as e:
            print(f"Error checking Qdrant collection: {e}")

    def build_index(self, akus: List[AKU]):
        import time
        self.akus_map = {aku.id: aku for aku in akus}
        
        batch_size = 100
        for i in range(0, len(akus), batch_size):
            batch = akus[i:i + batch_size]
            points = []
            for aku in batch:
                points.append(qmodels.PointStruct(
                    id=hash(aku.id) % (10**10),
                    vector=aku.embedding.tolist(),
                    payload={"aku_id": aku.id, "proposition": aku.proposition}
                ))
            
            self.client.upsert(collection_name=self.collection_name, points=points)
            time.sleep(1)

    def retrieve(self, query_embedding: np.ndarray, graph_engine) -> List[Tuple[AKU, float]]:
        search_result = self.client.query_points(
            collection_name=self.collection_name,
            query=query_embedding.tolist(),
            limit=self.candidate_size
        )
        
        candidates = []
        for res in search_result.points if hasattr(search_result, 'points') else search_result:
            aku_id = res.payload.get("aku_id")
            if aku_id in self.akus_map:
                candidates.append(self.akus_map[aku_id])
        
        if not candidates:
            return []

        q = query_embedding @ self.W_Q
        ks = np.stack([cand.structural_signature @ self.W_K for cand in candidates])
        vs = np.stack([cand.embedding @ self.W_V for cand in candidates])
        
        scores = (q @ ks.T) / np.sqrt(self.d)
        
        B = np.zeros((len(candidates), len(candidates)))
        for i, c_i in enumerate(candidates):
            for j, c_j in enumerate(candidates):
                if graph_engine.graph.has_edge(c_i.id, c_j.id):
                    B[i, j] = graph_engine.graph[c_i.id][c_j.id]['weight']
                else:
                    B[i, j] = -1e9 
        
        final_scores = np.exp(scores - np.max(scores))
        final_scores /= np.sum(final_scores)
        
        results = sorted(zip(candidates, final_scores), key=lambda x: x[1], reverse=True)
        return results
