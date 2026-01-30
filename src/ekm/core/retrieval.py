import numpy as np
from typing import List, Tuple
from .models import AKU
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
import os
from .tensor_ops import TensorOperations
from .config import get_config

class EKMRetriever:
    def __init__(self, d: int = 768, candidate_size: int = 100, collection_name: str = "ekm_akus",
                 projection_dim: int = 64, config_path: str = None, id_payload_key: str = "aku_id"):
        # Load configuration
        self.config = get_config(config_path)

        self.d = d or self.config.embedding_dim
        self.candidate_size = candidate_size or self.config.candidate_size
        self.collection_name = collection_name
        self.projection_dim = projection_dim or self.config.projection_dim
        self.id_payload_key = id_payload_key

        # Load credentials from configuration
        qdrant_url = self.config.qdrant_url
        qdrant_api_key = self.config.qdrant_api_key

        if not qdrant_url or not qdrant_api_key:
            raise ValueError("QDRANT_URL and QDRANT_API_KEY must be set in configuration")

        self.client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key, timeout=300)

        self._ensure_collection()
        self.akus_map = {}

        # Initialize proper attention weight matrices with orthogonal initialization
        self.W_Q = self._orthogonal_init(d, d)
        self.W_K = self._orthogonal_init(d + 13, d)  # +13 for topological features
        self.W_V = self._orthogonal_init(d, d)

        # Initialize tensor operations for attention mechanism with higher-order terms
        self.tensor_ops = TensorOperations(
            embedding_dim=self.d,
            projection_dim=self.projection_dim,
            k_sparse=10,
            higher_order_terms=self.config.enable_higher_order_terms
        )

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

        batch_size = 20
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
            aku_id = res.payload.get(self.id_payload_key)
            if aku_id in self.akus_map:
                candidates.append(self.akus_map[aku_id])

        if not candidates:
            return []

        # Enhanced attention mechanism using Tensor Operations (Fixed 'Mesh Grounding')
        # This replaces the simple Q-K dot product with a proper tensor contraction
        
        # Pre-project all candidates for the tensor operation
        psi_query = query_embedding @ self.tensor_ops.psi_matrix
        psi_candidates = np.stack([cand.embedding @ self.tensor_ops.psi_matrix for cand in candidates])
        
        # Compute attention scores using tensor interactions
        final_scores = np.zeros(len(candidates))
        for i, cand in enumerate(candidates):
            # Compute the pattern tensor that would exist between query and candidate
            sim_qc = float(query_embedding @ cand.embedding / (np.linalg.norm(query_embedding) * np.linalg.norm(cand.embedding) + 1e-9))
            T_qc = self.tensor_ops.compute_pattern_tensor(
                query_embedding, cand.embedding, 
                semantic_similarity=sim_qc,
                temporal_weight=1.0, 
                alpha=0.6, beta=0.1, gamma=0.3
            )
            
            # Grounding score: q^T * T_qc * c (Proper tensor contraction)
            final_scores[i] = psi_query @ T_qc @ psi_candidates[i]

        # Incorporate graph connectivity weights (mesh-aware re-ranking)
        for i, c_i in enumerate(candidates):
            neighbor_bonus = 0.0
            for j, c_j in enumerate(candidates):
                if i != j and graph_engine.graph.has_edge(c_i.id, c_j.id):
                    edge_weight = graph_engine.graph[c_i.id][c_j.id].get('weight', 1.0)
                    neighbor_bonus += edge_weight * 0.2
            final_scores[i] *= (1 + neighbor_bonus)

        # Apply softmax for normalization
        exp_scores = np.exp(final_scores - np.max(final_scores))
        attention_weights = exp_scores / np.sum(exp_scores)

        results = sorted(zip(candidates, attention_weights), key=lambda x: x[1], reverse=True)
        return results
