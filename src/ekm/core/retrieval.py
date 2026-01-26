import numpy as np
from typing import List, Tuple
from .models import AKU
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
import os
from .tensor_ops import TensorOperations

class EKMRetriever:
    def __init__(self, d: int = 768, candidate_size: int = 100, collection_name: str = "ekm_akus",
                 projection_dim: int = 64):
        self.d = d
        self.candidate_size = candidate_size
        self.collection_name = collection_name
        self.projection_dim = projection_dim

        # Load credentials from environment
        qdrant_url = os.getenv("QDRANT_URL", "https://2394163a-1fee-4cd2-a40d-e4765397680f.europe-west3-0.gcp.cloud.qdrant.io:6333")
        qdrant_api_key = os.getenv("QDRANT_API_KEY", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.2ktTYUzfKgTBlBvU1d2nrWsWP96z3lvdN1OOEWjxY_Y")

        self.client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key, timeout=120)

        self._ensure_collection()
        self.akus_map = {}

        # Initialize proper attention weight matrices with orthogonal initialization
        self.W_Q = self._orthogonal_init(d, d)
        self.W_K = self._orthogonal_init(d + 13, d)  # +13 for topological features
        self.W_V = self._orthogonal_init(d, d)

        # Initialize tensor operations for attention mechanism
        self.tensor_ops = TensorOperations(embedding_dim=d, projection_dim=projection_dim, k_sparse=10)

    def _orthogonal_init(self, rows: int, cols: int) -> np.ndarray:
        """Initialize weight matrices with orthogonal initialization."""
        flat = np.random.randn(rows * cols)
        u, _, v = np.linalg.svd(flat.reshape((rows, cols)), full_matrices=False)
        return u @ v  # Orthogonal matrix

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
                    payload={"aku_id": aku.id, "proposition": aku.proposition, "structural_signature": aku.structural_signature.tolist() if aku.structural_signature is not None else []}
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

        # Enhanced attention mechanism with proper tensor operations
        q = query_embedding @ self.W_Q
        ks = np.stack([cand.structural_signature @ self.W_K for cand in candidates])
        vs = np.stack([cand.embedding @ self.W_V for cand in candidates])

        # Compute raw attention scores
        raw_scores = (q @ ks.T) / np.sqrt(self.d)

        # Apply softmax to get attention probabilities
        shifted_scores = raw_scores - np.max(raw_scores)
        attention_weights = np.exp(shifted_scores)
        attention_weights /= np.sum(attention_weights)

        # Incorporate graph connectivity weights (mesh-aware re-ranking)
        final_scores = attention_weights.copy()
        for i, c_i in enumerate(candidates):
            for j, c_j in enumerate(candidates):
                if graph_engine.graph.has_edge(c_i.id, c_j.id):
                    # Apply mesh connectivity weight as described in the report
                    edge_weight = graph_engine.graph[c_i.id][c_j.id]['weight']
                    # Modulate attention by connectivity strength
                    final_scores[i] *= (1 + edge_weight * 0.1)  # Small modulation factor

        # Normalize final scores
        final_scores = final_scores / np.sum(final_scores) if np.sum(final_scores) > 0 else final_scores

        results = sorted(zip(candidates, final_scores), key=lambda x: x[1], reverse=True)
        return results
