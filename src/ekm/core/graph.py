import numpy as np
import networkx as nx
from typing import List, Dict, Tuple
from .models import AKU
from .tensor_ops import TensorOperations, compute_semantic_similarity, compute_temporal_weight, normalize_embeddings

class GraphEngine:
    def __init__(self, k: int = 10, alpha: float = 0.5, beta: float = 0.3, gamma: float = 0.2, tau: float = 86400,
                 embedding_dim: int = 768, projection_dim: int = 64):
        self.k = k
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.tau = tau
        self.graph = nx.DiGraph()

        # Initialize proper tensor operations
        self.tensor_ops = TensorOperations(embedding_dim=embedding_dim, projection_dim=projection_dim, k_sparse=k)

    def build_knn_graph(self, akus: List[AKU]):
        """
        Builds a sparse Knowledge Mesh using Sparse Pattern Tensors.
        Relationships are represented as proper tensor operations as described in the paper:
        T_ij = alpha * (psi(e_i) ⊗ psi(e_j)) + beta * temporal_component
        """
        if not akus:
            return

        # Extract embeddings and timestamps
        embeddings = np.stack([a.embedding for a in akus])
        timestamps = np.array([a.metadata.get('timestamp', 0) for a in akus])

        # Normalize embeddings
        norm_embeddings = normalize_embeddings(embeddings)

        # Compute similarity matrix
        cosine_sim = np.matmul(norm_embeddings, norm_embeddings.T)

        # Compute temporal weights matrix
        temporal_weights = np.zeros_like(cosine_sim)
        for i in range(len(akus)):
            for j in range(len(akus)):
                temporal_weights[i, j] = compute_temporal_weight(timestamps[i], timestamps[j], self.tau)

        # Compute sparse pattern tensors using proper tensor operations
        sparse_pattern_tensors, connection_indices = self.tensor_ops.compute_sparse_pattern_tensors(
            embeddings, cosine_sim, temporal_weights, self.alpha, self.beta
        )

        # Build the graph using the computed pattern tensors
        for i, aku_i in enumerate(akus):
            self.graph.add_node(aku_i.id, aku=aku_i)

            # Get the k nearest neighbors for this node
            neighbors_idx = connection_indices[i]

            for j_idx, j in enumerate(neighbors_idx):
                aku_j = akus[j]

                # Get the precomputed pattern tensor
                T_ij = sparse_pattern_tensors[i, j_idx]

                # Final edge weight derived from tensor magnitude
                edge_weight = np.linalg.norm(T_ij)

                # Store the pattern tensor along with the edge
                self.graph.add_edge(aku_i.id, aku_j.id,
                                  weight=edge_weight,
                                  pattern_tensor=T_ij,
                                  semantic_similarity=cosine_sim[i, j],
                                  temporal_weight=temporal_weights[i, j])

    def get_3node_motif_distribution(self, node_id: str) -> np.ndarray:
        """
        Compute actual 3-node motif distribution for the given node.
        This is a more sophisticated implementation than the random one.
        """
        if node_id not in self.graph:
            return np.zeros(13)  # Return zeros if node doesn't exist

        # Count different types of 3-node motifs
        # This is a simplified implementation - a full implementation would count all possible
        # 3-node subgraph isomorphisms
        node_neighbors = set(list(self.graph.predecessors(node_id)) + list(self.graph.successors(node_id)))

        motif_counts = np.zeros(13)

        # Count triangles (closed triplets)
        triangles = 0
        for n1 in node_neighbors:
            for n2 in node_neighbors:
                if n1 != n2 and self.graph.has_edge(n1, n2):
                    triangles += 1
        motif_counts[0] = triangles / 2  # Divide by 2 to avoid double counting

        # Count other motifs would go here in a full implementation

        return motif_counts

    def extract_signatures(self, akus: List[AKU]):
        """Update AKUs with structural signatures using proper topological analysis."""
        for aku in akus:
            # Compute actual topological features
            topo = self.get_3node_motif_distribution(aku.id)

            # Get neighbor embeddings with proper weights
            neighbors = list(self.graph.neighbors(aku.id))
            if neighbors:
                # Weight neighbor embeddings by edge weights
                weighted_embeddings = []
                for n in neighbors:
                    edge_data = self.graph[aku.id][n]
                    weight = edge_data.get('weight', 1.0)
                    neighbor_embedding = self.graph.nodes[n]['aku'].embedding
                    weighted_embeddings.append(weight * neighbor_embedding)

                if weighted_embeddings:
                    sem = np.mean(weighted_embeddings, axis=0)
                else:
                    sem = aku.embedding
            else:
                sem = aku.embedding

            aku.structural_signature = np.concatenate([topo, sem])
