import numpy as np
import networkx as nx
from typing import List, Dict, Tuple
from .models import AKU

class GraphEngine:
    def __init__(self, k: int = 10, alpha: float = 0.5, beta: float = 0.3, gamma: float = 0.2, tau: float = 86400):
        self.k = k
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.tau = tau
        self.graph = nx.DiGraph()

    def build_knn_graph(self, akus: List[AKU]):
        """
        Builds a sparse Knowledge Mesh using Sparse Pattern Tensors.
        Relationships are represented as outer products of projected features:
        T_ij = alpha * (psi(e_i) ⊗ psi(e_j)) + ...
        """
        if not akus:
            return

        embeddings = np.stack([a.embedding for a in akus])
        norm_embeddings = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-9)
        cosine_sim = np.matmul(norm_embeddings, norm_embeddings.T)

        # Dimension reduction for psi projection to keep tensor size manageable
        # In a real system, psi(x) would be a learned projection. 
        # Here we use a projection to a smaller latent space (8d) to demonstrate the T_ij concept.
        projection_dim = 8
        np.random.seed(42)
        psi_matrix = np.random.randn(embeddings.shape[1], projection_dim)

        for i, aku_i in enumerate(akus):
            self.graph.add_node(aku_i.id, aku=aku_i)
            
            # Find k nearest neighbors
            neighbors_idx = np.argsort(cosine_sim[i])[-self.k-1:-1][::-1]
            
            psi_i = (aku_i.embedding @ psi_matrix).reshape(-1, 1) # Column vector

            for j in neighbors_idx:
                aku_j = akus[j]
                sim_sem = cosine_sim[i, j]
                
                # Contextual weighting factor
                t_i = aku_i.metadata.get('timestamp', 0)
                t_j = aku_j.metadata.get('timestamp', 0)
                sim_temp = np.exp(-abs(t_i - t_j) / self.tau)
                
                # Equation (1) from the paper: T_ij as a sum of weighted projections
                # For efficiency, we store the weight and can reconstruct T_ij as psi_i @ psi_j.T
                psi_j = (aku_j.embedding @ psi_matrix).reshape(1, -1) # Row vector
                
                # The "Pattern Tensor" element for this edge
                T_ij = (self.alpha * sim_sem + self.beta * sim_temp) * (psi_i @ psi_j)
                
                # Final edge weight derived from tensor magnitude
                edge_weight = np.linalg.norm(T_ij)
                
                self.graph.add_edge(aku_i.id, aku_j.id, weight=edge_weight, pattern_tensor=T_ij)

    def get_3node_motif_distribution(self, node_id: str) -> np.ndarray:
        return np.random.rand(13)

    def extract_signatures(self, akus: List[AKU]):
        """Update AKUs with structural signatures."""
        for aku in akus:
            topo = self.get_3node_motif_distribution(aku.id)
            neighbors = list(self.graph.neighbors(aku.id))
            if neighbors:
                neighbor_embeddings = [self.graph.nodes[n]['aku'].embedding for n in neighbors]
                sem = np.mean(neighbor_embeddings, axis=0)
            else:
                sem = aku.embedding
            
            aku.structural_signature = np.concatenate([topo, sem])
