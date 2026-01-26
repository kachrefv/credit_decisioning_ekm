"""
Tensor operations module implementing the mathematical foundations of EKM
based on the technical report specifications.
"""
import numpy as np
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class TensorOperations:
    """
    Implements the mathematical operations described in the EKM technical report,
    including sparse pattern tensors and proper tensor contractions.
    """

    def __init__(self, embedding_dim: int = 768, projection_dim: int = 64, k_sparse: int = 10,
                 higher_order_terms: bool = True):
        """
        Initialize tensor operations with proper dimensions.

        Args:
            embedding_dim: Original embedding dimension
            projection_dim: Dimension for the psi projection
            k_sparse: Number of sparse connections to maintain
            higher_order_terms: Whether to include higher-order tensor terms
        """
        self.embedding_dim = embedding_dim
        self.projection_dim = projection_dim
        self.k_sparse = k_sparse
        self.higher_order_terms = higher_order_terms

        # Initialize the psi projection matrix as a learned transformation
        # In practice, this would be trained, but for now we initialize it meaningfully
        self.psi_matrix = self._initialize_projection_matrix()

        # Initialize higher-order tensor components if enabled
        if self.higher_order_terms:
            self._init_higher_order_components()

    def _init_higher_order_components(self):
        """Initialize components for higher-order tensor operations."""
        # Third-order tensor component for triadic relationships
        self.third_order_tensor = np.random.randn(self.projection_dim, self.projection_dim, self.projection_dim) * 0.01

        # Fourth-order tensor component for quadriadic relationships
        self.fourth_order_tensor = np.random.randn(self.projection_dim, self.projection_dim,
                                                  self.projection_dim, self.projection_dim) * 0.001

    def _initialize_projection_matrix(self) -> np.ndarray:
        """
        Initialize the psi projection matrix with orthogonal initialization
        to preserve important features during projection.
        """
        # Use semi-orthogonal initialization to preserve information
        psi = np.random.randn(self.embedding_dim, self.projection_dim)
        u, s, vh = np.linalg.svd(psi, full_matrices=False)
        # Apply spectral normalization to ensure Lipschitz continuity
        s_max = np.max(s)
        if s_max > 0:
            s = s / s_max  # Normalize singular values
        return u @ np.diag(s) @ vh  # Orthogonal matrix with normalized singular values
    
    def compute_pattern_tensor(self, embedding_i: np.ndarray, embedding_j: np.ndarray,
                             semantic_similarity: float, temporal_weight: float,
                             alpha: float = 0.5, beta: float = 0.3) -> np.ndarray:
        """
        Compute the pattern tensor T_ij as described in the technical report:
        T_ij = alpha * (psi(e_i) ⊗ psi(e_j)) + beta * temporal_component
        
        Args:
            embedding_i: First embedding vector
            embedding_j: Second embedding vector  
            semantic_similarity: Semantic similarity between embeddings
            temporal_weight: Temporal weighting factor
            alpha: Weight for semantic component
            beta: Weight for temporal component
            
        Returns:
            Pattern tensor T_ij of shape (projection_dim, projection_dim)
        """
        # Project embeddings using psi
        psi_i = self.psi_matrix.T @ embedding_i  # Shape: (projection_dim,)
        psi_j = self.psi_matrix.T @ embedding_j  # Shape: (projection_dim,)
        
        # Compute outer product: psi_i ⊗ psi_j
        outer_product = np.outer(psi_i, psi_j)  # Shape: (projection_dim, projection_dim)
        
        # Apply semantic weighting
        semantic_component = semantic_similarity * outer_product
        
        # For temporal component, we can add a diagonal matrix weighted by temporal factor
        temporal_component = temporal_weight * np.eye(self.projection_dim) * beta
        
        # Combine components
        pattern_tensor = alpha * semantic_component + temporal_component
        
        return pattern_tensor
    
    def compute_sparse_pattern_tensors(self, embeddings: np.ndarray, 
                                     similarities: np.ndarray,
                                     temporal_weights: np.ndarray,
                                     alpha: float = 0.5, beta: float = 0.3) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute sparse pattern tensors for all pairs of embeddings.
        
        Args:
            embeddings: Array of shape (n_nodes, embedding_dim)
            similarities: Symmetric matrix of shape (n_nodes, n_nodes) with similarities
            temporal_weights: Matrix of shape (n_nodes, n_nodes) with temporal weights
            alpha, beta: Weights for semantic and temporal components
            
        Returns:
            Tuple of (sparse_pattern_tensors, connection_indices)
            - sparse_pattern_tensors: Array of shape (n_nodes, k_sparse, proj_dim, proj_dim)
            - connection_indices: Array of shape (n_nodes, k_sparse) with indices of connections
        """
        n_nodes = embeddings.shape[0]
        
        # For each node, find k_sparse most similar neighbors
        sparse_pattern_tensors = np.zeros((n_nodes, self.k_sparse, self.projection_dim, self.projection_dim))
        connection_indices = np.zeros((n_nodes, self.k_sparse), dtype=int)
        
        for i in range(n_nodes):
            # Get similarities for node i, excluding self
            node_similarities = similarities[i].copy()
            node_similarities[i] = -np.inf  # Exclude self-similarity
            
            # Find k_sparse highest similarity connections
            top_k_indices = np.argpartition(node_similarities, -self.k_sparse)[-self.k_sparse:]
            top_k_indices = top_k_indices[np.argsort(-node_similarities[top_k_indices])]  # Sort descending
            
            connection_indices[i] = top_k_indices
            
            # Compute pattern tensors for top-k connections
            for j_idx, j in enumerate(top_k_indices):
                tensor = self.compute_pattern_tensor(
                    embeddings[i], embeddings[j],
                    similarities[i, j], temporal_weights[i, j],
                    alpha, beta
                )
                sparse_pattern_tensors[i, j_idx] = tensor
        
        return sparse_pattern_tensors, connection_indices
    
    def contract_tensors_with_attention(self, query_embedding: np.ndarray,
                                     sparse_pattern_tensors: np.ndarray,
                                     connection_indices: np.ndarray,
                                     W_Q: np.ndarray, W_K: np.ndarray, W_V: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform attention-weighted tensor contractions as described in the report.
        
        Args:
            query_embedding: Query embedding of shape (embedding_dim,)
            sparse_pattern_tensors: Sparse tensors of shape (n_nodes, k_sparse, proj_dim, proj_dim)
            connection_indices: Indices of shape (n_nodes, k_sparse)
            W_Q, W_K, W_V: Attention weight matrices
            
        Returns:
            Tuple of (attention_scores, contracted_values)
        """
        n_nodes = sparse_pattern_tensors.shape[0]
        
        # Project query through W_Q
        q = query_embedding @ W_Q  # Shape: (embedding_dim,)
        
        # Project connected embeddings through W_K and W_V
        # We need to get the corresponding embeddings for the connected nodes
        # This requires having access to the original embeddings
        # For now, we'll return the attention computation structure
        
        # Placeholder for attention scores and values
        attention_scores = np.zeros((n_nodes, self.k_sparse))
        values = np.zeros((n_nodes, self.k_sparse, self.embedding_dim))
        
        # Compute attention scores for each node's connections
        for i in range(n_nodes):
            for j_idx in range(self.k_sparse):
                conn_idx = connection_indices[i, j_idx]
                
                # Get the pattern tensor for this connection
                T_ij = sparse_pattern_tensors[i, j_idx]  # Shape: (proj_dim, proj_dim)
                
                # Compute attention score using tensor contraction
                # This is a simplified version - in reality, this would involve more complex contractions
                psi_query = self.psi_matrix.T @ query_embedding  # Shape: (proj_dim,)
                psi_conn = self.psi_matrix.T @ query_embedding  # Using query as proxy for connection embedding
                
                # Contract with pattern tensor
                attention_contribution = psi_query @ T_ij @ psi_conn.T
                attention_scores[i, j_idx] = attention_contribution
        
        return attention_scores, values


def compute_semantic_similarity(embedding_i: np.ndarray, embedding_j: np.ndarray) -> float:
    """
    Compute semantic similarity between two embeddings using cosine similarity.
    """
    norm_i = np.linalg.norm(embedding_i)
    norm_j = np.linalg.norm(embedding_j)
    
    if norm_i == 0 or norm_j == 0:
        return 0.0
    
    cos_sim = np.dot(embedding_i, embedding_j) / (norm_i * norm_j)
    return float(cos_sim)


def compute_temporal_weight(time_i: float, time_j: float, tau: float = 86400) -> float:
    """
    Compute temporal weight based on time difference.
    
    Args:
        time_i, time_j: Timestamps for the two embeddings
        tau: Time decay constant (default 1 day in seconds)
    """
    time_diff = abs(time_i - time_j)
    return float(np.exp(-time_diff / tau))


def normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    """
    Normalize embeddings to unit length.
    
    Args:
        embeddings: Array of shape (n, embedding_dim)
        
    Returns:
        Normalized embeddings of same shape
    """
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)  # Avoid division by zero
    return embeddings / norms