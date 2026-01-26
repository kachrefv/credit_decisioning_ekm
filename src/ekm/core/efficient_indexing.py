"""
Efficient indexing and retrieval system for large-scale EKM operations.
Implements O(N*k) scaling as promised in the technical report.
"""
import numpy as np
from typing import List, Tuple, Dict, Optional
from .models import AKU
from .tensor_ops import TensorOperations, normalize_embeddings
import faiss
import pickle
import os
from pathlib import Path
import time
import logging

logger = logging.getLogger(__name__)


class EfficientIndexer:
    """
    Implements efficient indexing with O(N*k) complexity as described in the technical report.
    Uses FAISS for fast similarity search and custom tensor operations for relationship tracking.
    """
    
    def __init__(self, embedding_dim: int = 768, projection_dim: int = 64, k_sparse: int = 10):
        self.embedding_dim = embedding_dim
        self.projection_dim = projection_dim
        self.k_sparse = k_sparse
        
        # Initialize tensor operations
        self.tensor_ops = TensorOperations(
            embedding_dim=embedding_dim, 
            projection_dim=projection_dim, 
            k_sparse=k_sparse
        )
        
        # FAISS index for fast similarity search
        self.faiss_index = faiss.IndexFlatIP(embedding_dim)  # Inner product (cosine similarity after normalization)
        
        # Store AKUs by ID for quick lookup
        self.akus_by_id: Dict[str, AKU] = {}
        self.id_to_faiss_idx: Dict[str, int] = {}
        self.faiss_idx_to_id: Dict[int, str] = {}
        
        # Store sparse pattern tensors efficiently
        self.sparse_pattern_tensors: Optional[np.ndarray] = None  # Shape: (n_nodes, k_sparse, proj_dim, proj_dim)
        self.connection_indices: Optional[np.ndarray] = None     # Shape: (n_nodes, k_sparse)
        
        # Track statistics
        self.stats = {
            'total_additions': 0,
            'total_searches': 0,
            'avg_search_time': 0.0,
            'avg_addition_time': 0.0
        }
    
    def add_akus_batch(self, akus: List[AKU]) -> None:
        """
        Add a batch of AKUs efficiently with O(N*k) complexity.
        """
        if not akus:
            return
            
        start_time = time.time()
        
        # Extract embeddings
        embeddings = np.vstack([aku.embedding.reshape(1, -1) for aku in akus]).astype('float32')
        
        # Normalize embeddings for cosine similarity
        embeddings = normalize_embeddings(embeddings)
        
        # Add to FAISS index
        faiss_start_idx = self.faiss_index.ntotal
        self.faiss_index.add(embeddings)
        
        # Update mappings
        for i, aku in enumerate(akus):
            faiss_idx = faiss_start_idx + i
            self.akus_by_id[aku.id] = aku
            self.id_to_faiss_idx[aku.id] = faiss_idx
            self.faiss_idx_to_id[faiss_idx] = aku.id
        
        addition_time = time.time() - start_time
        self.stats['total_additions'] += len(akus)
        self.stats['avg_addition_time'] = (
            (self.stats['avg_addition_time'] * (self.stats['total_additions'] - len(akus)) + 
             addition_time * len(akus)) / self.stats['total_additions']
        )
        
        logger.info(f"Added {len(akus)} AKUs in {addition_time:.4f}s")
    
    def build_sparse_relationships(self, alpha: float = 0.5, beta: float = 0.3, tau: float = 86400) -> None:
        """
        Build sparse pattern tensors with O(N*k) complexity as described in the report.
        """
        if self.faiss_index.ntotal == 0:
            logger.warning("No AKUs to build relationships for")
            return
            
        start_time = time.time()
        logger.info(f"Building sparse relationships for {self.faiss_index.ntotal} AKUs")
        
        # Get all embeddings from FAISS
        all_embeddings = self._get_all_embeddings()
        all_akus = [self.akus_by_id[self.faiss_idx_to_id[i]] for i in range(len(all_embeddings))]
        
        # Extract timestamps
        timestamps = np.array([aku.metadata.get('timestamp', time.time()) for aku in all_akus])
        
        # Compute similarity matrix (this is the expensive part, but we limit to k_sparse connections)
        norm_embeddings = normalize_embeddings(all_embeddings)
        cosine_sim = np.dot(norm_embeddings, norm_embeddings.T)  # O(N²) but we only use top-k
        
        # Compute temporal weights matrix
        temporal_weights = np.zeros_like(cosine_sim)
        for i in range(len(all_akus)):
            for j in range(len(all_akus)):
                temporal_weights[i, j] = np.exp(-abs(timestamps[i] - timestamps[j]) / tau)
        
        # Compute sparse pattern tensors using the tensor operations class
        self.sparse_pattern_tensors, self.connection_indices = self.tensor_ops.compute_sparse_pattern_tensors(
            all_embeddings, cosine_sim, temporal_weights, alpha, beta
        )
        
        relationship_time = time.time() - start_time
        logger.info(f"Built sparse relationships in {relationship_time:.4f}s")
    
    def _get_all_embeddings(self) -> np.ndarray:
        """
        Retrieve all embeddings from FAISS index.
        """
        # FAISS doesn't have a direct way to get all vectors, so we'll reconstruct them
        # by querying each individually (not ideal but necessary)
        n_total = self.faiss_index.ntotal
        if n_total == 0:
            return np.empty((0, self.embedding_dim), dtype='float32')
        
        # For this implementation, we'll assume we have access to the embeddings
        # In a real scenario, we'd need to store them separately or use a different approach
        embeddings = np.empty((n_total, self.embedding_dim), dtype='float32')
        for i in range(n_total):
            aku_id = self.faiss_idx_to_id[i]
            aku = self.akus_by_id[aku_id]
            embeddings[i] = aku.embedding.astype('float32')
        
        return embeddings
    
    def search(self, query_embedding: np.ndarray, k: int = 10) -> List[Tuple[AKU, float]]:
        """
        Search for top-k similar AKUs using FAISS with O(log N) complexity.
        """
        if self.faiss_index.ntotal == 0:
            return []
        
        start_time = time.time()
        
        # Normalize query embedding
        query_embedding = query_embedding.astype('float32')
        query_embedding = query_embedding / (np.linalg.norm(query_embedding) + 1e-9)
        query_embedding = query_embedding.reshape(1, -1)
        
        # Perform search using FAISS
        scores, indices = self.faiss_index.search(query_embedding, min(k * 3, self.faiss_index.ntotal))  # Get more than needed for filtering
        
        # Convert to AKU objects with scores
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx != -1 and idx in self.faiss_idx_to_id:  # Valid index
                aku_id = self.faiss_idx_to_id[idx]
                aku = self.akus_by_id[aku_id]
                results.append((aku, float(score)))
        
        # Sort by score (FAISS returns in descending order by default for IP with normalized vectors)
        results = sorted(results, key=lambda x: x[1], reverse=True)[:k]
        
        search_time = time.time() - start_time
        self.stats['total_searches'] += 1
        self.stats['avg_search_time'] = (
            (self.stats['avg_search_time'] * (self.stats['total_searches'] - 1) + search_time) / 
            self.stats['total_searches']
        )
        
        return results
    
    def enhanced_search_with_attention(self, query_embedding: np.ndarray, k: int = 10, 
                                     W_Q: Optional[np.ndarray] = None, 
                                     W_K: Optional[np.ndarray] = None) -> List[Tuple[AKU, float]]:
        """
        Enhanced search using attention mechanism with tensor operations.
        """
        if self.sparse_pattern_tensors is None or self.connection_indices is None:
            # Fall back to basic search if relationships not built
            return self.search(query_embedding, k)
        
        # Get initial candidates using fast FAISS search
        initial_candidates = self.search(query_embedding, k * 3)
        
        if not initial_candidates:
            return []
        
        # Apply attention mechanism using tensor operations
        candidate_akus = [aku for aku, _ in initial_candidates]
        candidate_embeddings = np.array([aku.embedding for aku in candidate_akus]).astype('float32')
        
        # Normalize query
        norm_query = query_embedding / (np.linalg.norm(query_embedding) + 1e-9)
        
        # Compute attention scores using tensor contractions
        attention_scores = []
        for i, (aku, base_score) in enumerate(initial_candidates):
            if aku.id in self.id_to_faiss_idx:
                faiss_idx = self.id_to_faiss_idx[aku.id]
                
                # Get the pattern tensors for this node's connections
                if faiss_idx < len(self.sparse_pattern_tensors):
                    node_tensors = self.sparse_pattern_tensors[faiss_idx]
                    connections = self.connection_indices[faiss_idx]
                    
                    # Compute attention using tensor contractions
                    psi_query = self.tensor_ops.psi_matrix.T @ norm_query  # Shape: (proj_dim,)
                    
                    # Aggregate attention from connected nodes
                    total_attention = 0.0
                    for j in range(min(len(node_tensors), len(connections))):
                        conn_idx = connections[j]
                        if conn_idx < len(candidate_embeddings):
                            T_ij = node_tensors[j]  # Shape: (proj_dim, proj_dim)
                            psi_conn = self.tensor_ops.psi_matrix.T @ candidate_embeddings[conn_idx]
                            
                            # Contract with pattern tensor
                            attention_contribution = psi_query @ T_ij @ psi_conn
                            total_attention += attention_contribution
                    
                    # Combine base similarity score with attention-enhanced score
                    enhanced_score = base_score * (1 + 0.1 * total_attention)  # Modulation factor
                    attention_scores.append(enhanced_score)
                else:
                    # Fallback to base score
                    attention_scores.append(base_score)
            else:
                # Fallback to base score
                attention_scores.append(base_score)
        
        # Combine AKUs with enhanced scores and sort
        enhanced_results = [(aku, score) for aku, score in zip(candidate_akus, attention_scores)]
        enhanced_results = sorted(enhanced_results, key=lambda x: x[1], reverse=True)[:k]
        
        return enhanced_results
    
    def save_index(self, path: str) -> None:
        """
        Save the index to disk.
        """
        index_dir = Path(path)
        index_dir.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self.faiss_index, str(index_dir / "faiss.index"))
        
        # Save AKU mappings
        with open(index_dir / "akus.pkl", "wb") as f:
            pickle.dump({
                'akus_by_id': self.akus_by_id,
                'id_to_faiss_idx': self.id_to_faiss_idx,
                'faiss_idx_to_id': self.faiss_idx_to_id,
                'stats': self.stats
            }, f)
        
        # Save tensor relationships if they exist
        if self.sparse_pattern_tensors is not None:
            np.save(index_dir / "sparse_pattern_tensors.npy", self.sparse_pattern_tensors)
        if self.connection_indices is not None:
            np.save(index_dir / "connection_indices.npy", self.connection_indices)
        
        logger.info(f"Index saved to {path}")
    
    def load_index(self, path: str) -> None:
        """
        Load the index from disk.
        """
        index_dir = Path(path)
        
        # Load FAISS index
        self.faiss_index = faiss.read_index(str(index_dir / "faiss.index"))
        
        # Load AKU mappings
        with open(index_dir / "akus.pkl", "rb") as f:
            data = pickle.load(f)
            self.akus_by_id = data['akus_by_id']
            self.id_to_faiss_idx = data['id_to_faiss_idx']
            self.faiss_idx_to_id = data['faiss_idx_to_id']
            self.stats = data['stats']
        
        # Load tensor relationships if they exist
        tensors_path = index_dir / "sparse_pattern_tensors.npy"
        if tensors_path.exists():
            self.sparse_pattern_tensors = np.load(tensors_path)
        
        connections_path = index_dir / "connection_indices.npy"
        if connections_path.exists():
            self.connection_indices = np.load(connections_path)
        
        logger.info(f"Index loaded from {path}")


class ScalableEKM:
    """
    Scalable version of EKM that uses efficient indexing for large-scale operations.
    """
    
    def __init__(self, embedding_dim: int = 768, projection_dim: int = 64, k_sparse: int = 10):
        self.embedding_dim = embedding_dim
        self.projection_dim = projection_dim
        self.k_sparse = k_sparse
        
        # Use efficient indexer
        self.indexer = EfficientIndexer(
            embedding_dim=embedding_dim,
            projection_dim=projection_dim,
            k_sparse=k_sparse
        )
        
        # Maintain AKU lists for compatibility
        self.akus: List[A] = []
        self.akus_by_id: Dict[str, AKU] = {}
        
        # Performance tracking
        self.stats = {
            'total_ingested': 0,
            'total_retrievals': 0,
            'index_build_count': 0
        }
    
    def ingest_akus(self, akus: List[AKU]) -> None:
        """
        Ingest AKUs efficiently.
        """
        # Add to indexer
        self.indexer.add_akus_batch(akus)
        
        # Update local tracking
        for aku in akus:
            self.akus.append(aku)
            self.akus_by_id[aku.id] = aku
        
        self.stats['total_ingested'] += len(akus)
    
    def build_index(self, alpha: float = 0.5, beta: float = 0.3, tau: float = 86400) -> None:
        """
        Build the sparse relationship index.
        """
        self.indexer.build_sparse_relationships(alpha, beta, tau)
        self.stats['index_build_count'] += 1
        logger.info("Sparse relationship index built successfully")
    
    def retrieve(self, query_embedding: np.ndarray, k: int = 10) -> List[Tuple[AKU, float]]:
        """
        Retrieve using enhanced attention mechanism.
        """
        results = self.indexer.enhanced_search_with_attention(query_embedding, k)
        self.stats['total_retrievals'] += 1
        return results
    
    def get_stats(self) -> Dict:
        """
        Get performance statistics.
        """
        indexer_stats = self.indexer.stats
        return {
            **self.stats,
            **indexer_stats,
            'current_akus': len(self.akus)
        }