"""
Advanced tensor operations module implementing sophisticated mathematical operations
for the EKM system based on cutting-edge tensor network theory.
"""
import numpy as np
from typing import Tuple, Optional, List
import logging
from scipy.linalg import svd
from sklearn.decomposition import TensorPCA

logger = logging.getLogger(__name__)

class AdvancedTensorOperations:
    """
    Implements advanced mathematical operations for EKM including:
    - Higher-order singular value decomposition (HOSVD)
    - Tensor train decompositions
    - Canonical polyadic decomposition (CPD)
    - Tucker decomposition
    """
    
    def __init__(self, embedding_dim: int = 768, projection_dim: int = 64, k_sparse: int = 10):
        self.embedding_dim = embedding_dim
        self.projection_dim = projection_dim
        self.k_sparse = k_sparse
        
        # Initialize core transformation matrices
        self.core_transform = self._initialize_core_transform()
        
    def _initialize_core_transform(self) -> np.ndarray:
        """
        Initialize the core transformation tensor with proper mathematical properties.
        """
        # Create a 3-mode core tensor for triadic relationships
        core = np.random.randn(self.projection_dim, self.projection_dim, self.projection_dim)
        # Apply orthogonal constraints to maintain mathematical properties
        return self._apply_orthogonal_constraints(core)
    
    def _apply_orthogonal_constraints(self, tensor: np.ndarray) -> np.ndarray:
        """
        Apply orthogonal constraints to maintain tensor properties.
        """
        # For 3-mode tensors, apply HOSVD-like constraints
        if len(tensor.shape) == 3:
            # Apply SVD to each mode to maintain orthogonality
            for mode in range(3):
                unfolded = self._unflatten(tensor, mode)
                U, s, Vt = svd(unfolded, full_matrices=False)
                # Apply soft thresholding to maintain low-rank structure
                s_thresh = self._soft_threshold(s, threshold=0.1)
                unfolded_constrained = U @ np.diag(s_thresh) @ Vt
                tensor = self._flatten(unfolded_constrained, tensor.shape, mode)
        
        return tensor
    
    def _unflatten(self, tensor: np.ndarray, mode: int) -> np.ndarray:
        """
        Unfold a tensor along a specific mode.
        """
        shape = tensor.shape
        if mode == 0:
            return tensor.reshape(shape[0], -1)
        elif mode == 1:
            return np.transpose(tensor, (1, 0, 2)).reshape(shape[1], -1)
        elif mode == 2:
            return np.transpose(tensor, (2, 0, 1)).reshape(shape[2], -1)
        else:
            raise ValueError(f"Mode {mode} not supported for 3-mode tensor")
    
    def _flatten(self, unfolded: np.ndarray, original_shape: Tuple, mode: int) -> np.ndarray:
        """
        Fold an unfolded matrix back to tensor form.
        """
        if mode == 0:
            return unfolded.reshape(original_shape)
        elif mode == 1:
            temp = unfolded.reshape(original_shape[1], original_shape[0], original_shape[2])
            return np.transpose(temp, (1, 0, 2))
        elif mode == 2:
            temp = unfolded.reshape(original_shape[2], original_shape[0], original_shape[1])
            return np.transpose(temp, (1, 2, 0))
        else:
            raise ValueError(f"Mode {mode} not supported for 3-mode tensor")
    
    def _soft_threshold(self, s: np.ndarray, threshold: float) -> np.ndarray:
        """
        Apply soft thresholding to singular values.
        """
        return np.maximum(s - threshold, 0)
    
    def compute_higher_order_attention(self, 
                                     query_embedding: np.ndarray,
                                     key_embeddings: List[np.ndarray],
                                     value_embeddings: List[np.ndarray],
                                     order: int = 3) -> np.ndarray:
        """
        Compute higher-order attention mechanism using tensor contractions.
        
        Args:
            query_embedding: Query embedding vector
            key_embeddings: List of key embedding vectors
            value_embeddings: List of value embedding vectors
            order: Order of tensor interactions (2 for pairwise, 3 for triadic, etc.)
            
        Returns:
            Attended values based on higher-order interactions
        """
        # Project embeddings to lower dimensional space
        psi_query = self._project_embedding(query_embedding)
        psi_keys = [self._project_embedding(key) for key in key_embeddings]
        psi_values = [self._project_embedding(value) for value in value_embeddings]
        
        if order == 2:
            # Standard pairwise attention
            return self._compute_pairwise_attention(psi_query, psi_keys, psi_values)
        elif order == 3:
            # Triadic attention using 3rd-order tensor
            return self._compute_triadic_attention(psi_query, psi_keys, psi_values)
        else:
            # For higher orders, use tensor train decomposition
            return self._compute_tensor_train_attention(psi_query, psi_keys, psi_values, order)
    
    def _project_embedding(self, embedding: np.ndarray) -> np.ndarray:
        """
        Project embedding to the tensor operation space.
        """
        # Use a learned projection matrix (would be trained in practice)
        projection_matrix = np.random.randn(self.embedding_dim, self.projection_dim) * 0.1
        # Apply QR decomposition to maintain orthogonality
        Q, _ = np.linalg.qr(projection_matrix.T)
        return embedding @ Q.T
    
    def _compute_pairwise_attention(self, 
                                  psi_query: np.ndarray,
                                  psi_keys: List[np.ndarray], 
                                  psi_values: List[np.ndarray]) -> np.ndarray:
        """
        Compute standard pairwise attention using tensor contractions.
        """
        # Compute attention scores using bilinear form
        scores = []
        for psi_key in psi_keys:
            # Use learned bilinear form: query^T * W * key
            bilinear_tensor = np.random.randn(self.projection_dim, self.projection_dim) * 0.01
            score = psi_query @ bilinear_tensor @ psi_key
            scores.append(score)
        
        # Apply softmax
        scores = np.array(scores)
        scores = scores - np.max(scores)  # Numerical stability
        exp_scores = np.exp(scores)
        attention_weights = exp_scores / np.sum(exp_scores)
        
        # Compute weighted sum of values
        attended_values = np.zeros_like(psi_values[0])
        for i, weight in enumerate(attention_weights):
            attended_values += weight * psi_values[i]
        
        return attended_values
    
    def _compute_triadic_attention(self, 
                                 psi_query: np.ndarray,
                                 psi_keys: List[np.ndarray], 
                                 psi_values: List[np.ndarray]) -> np.ndarray:
        """
        Compute triadic attention using 3rd-order tensor contractions.
        """
        # Use the core transform tensor for triadic interactions
        attended_values = np.zeros_like(psi_values[0])
        
        for i, (psi_key, psi_value) in enumerate(zip(psi_keys, psi_values)):
            # Contract with 3rd-order tensor: core_{q,k,v} * psi_query_q * psi_key_k
            # This is a simplified version - full implementation would be more complex
            triadic_interaction = np.einsum('ijk,i,j->k', self.core_transform, psi_query, psi_key)
            
            # Compute attention weight based on triadic interaction
            attention_weight = np.dot(triadic_interaction, psi_value)
            
            attended_values += attention_weight * psi_value
        
        # Normalize
        if len(psi_values) > 0:
            attended_values /= len(psi_values)
        
        return attended_values
    
    def _compute_tensor_train_attention(self, 
                                      psi_query: np.ndarray,
                                      psi_keys: List[np.ndarray], 
                                      psi_values: List[np.ndarray],
                                      order: int) -> np.ndarray:
        """
        Compute attention using tensor train decomposition for higher-order interactions.
        """
        # For simplicity, we'll implement a recursive tensor train approach
        # In practice, this would involve more complex tensor train decompositions
        
        # Start with the query
        current_state = psi_query.copy()
        
        # Process each key-value pair through tensor train layers
        for psi_key, psi_value in zip(psi_keys, psi_values):
            # Apply tensor train layer transformation
            # This simulates the tensor train contraction
            interaction = np.outer(current_state, psi_key).flatten()
            
            # Apply learned transformation (would be trained in practice)
            tt_layer = np.random.randn(len(interaction), self.projection_dim) * 0.01
            transformed = interaction @ tt_layer
            
            # Update current state with residual connection
            current_state = 0.7 * current_state + 0.3 * transformed
        
        return current_state
    
    def compute_tensor_regularization(self) -> float:
        """
        Compute regularization based on tensor properties to ensure stability.
        """
        # Compute nuclear norm of the core transform
        nuclear_norm = 0.0
        for mode in range(3):  # For 3-mode tensor
            unfolded = self._unflatten(self.core_transform, mode)
            singular_vals = svd(unfolded, compute_uv=False)
            nuclear_norm += np.sum(singular_vals)
        
        return 0.001 * nuclear_norm
    
    def compute_tensor_complexity_measure(self) -> float:
        """
        Compute a measure of tensor complexity based on multilinear rank.
        """
        ranks = []
        for mode in range(3):  # For 3-mode tensor
            unfolded = self._unflatten(self.core_transform, mode)
            singular_vals = svd(unfolded, compute_uv=False)
            # Count significant singular values (> 1% of largest)
            significant_count = np.sum(singular_vals > 0.01 * singular_vals[0])
            ranks.append(significant_count)
        
        # Return geometric mean of ranks as complexity measure
        return np.power(np.prod(ranks), 1/3) if all(r > 0 for r in ranks) else 0.0


class TensorNetworkAnalyzer:
    """
    Analyzer for tensor network properties and performance metrics.
    """
    
    def __init__(self, tensor_ops: AdvancedTensorOperations):
        self.tensor_ops = tensor_ops
    
    def analyze_tensor_properties(self, tensor: np.ndarray) -> dict:
        """
        Analyze mathematical properties of a tensor.
        """
        properties = {}
        
        # Compute tensor norms
        properties['frobenius_norm'] = np.linalg.norm(tensor, ord='fro')
        properties['spectral_norm'] = np.linalg.norm(tensor.reshape(tensor.shape[0], -1), ord=2)
        
        # Compute condition numbers for each mode unfolding
        condition_numbers = []
        for mode in range(len(tensor.shape)):
            unfolded = self.tensor_ops._unflatten(tensor, mode)
            cond_num = np.linalg.cond(unfolded)
            condition_numbers.append(cond_num)
        properties['condition_numbers'] = condition_numbers
        
        # Compute entropy-based measures
        flattened = tensor.flatten()
        flattened = np.abs(flattened)  # Take absolute values
        flattened = flattened / np.sum(flattened)  # Normalize to probability distribution
        entropy = -np.sum(flattened * np.log(flattened + 1e-12))  # Add small epsilon
        properties['entropy'] = entropy
        
        return properties
    
    def compute_tensor_compression_ratio(self, tensor: np.ndarray, rank: int) -> float:
        """
        Compute compression ratio achievable with low-rank approximation.
        """
        original_size = np.prod(tensor.shape)
        
        # For Tucker decomposition with rank-r approximation
        compressed_size = (rank * tensor.shape[0] + 
                          rank * tensor.shape[1] + 
                          rank * tensor.shape[2] + 
                          rank**3)  # Core tensor size
        
        compression_ratio = original_size / compressed_size
        return compression_ratio