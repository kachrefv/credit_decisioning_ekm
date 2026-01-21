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
        Builds a sparse k-NN graph from AKUs using Equation (1).
        w_ij = Norm(alpha * cos(e_i, e_j) + beta * exp(-|t_i - t_j|/tau) + gamma * c_ij)
        """
        if not akus:
            return

        embeddings = np.stack([a.embedding for a in akus])
        norm_embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        cosine_sim = np.matmul(norm_embeddings, norm_embeddings.T)

        for i, aku_i in enumerate(akus):
            self.graph.add_node(aku_i.id, aku=aku_i)
            
            neighbors_idx = np.argsort(cosine_sim[i])[-self.k-1:-1][::-1]
            
            weights = []
            neighbor_ids = []
            
            for j in neighbors_idx:
                aku_j = akus[j]
                sim_sem = cosine_sim[i, j]
                
                t_i = aku_i.metadata.get('timestamp', 0)
                t_j = aku_j.metadata.get('timestamp', 0)
                sim_temp = np.exp(-abs(t_i - t_j) / self.tau)
                
                sim_log = aku_i.metadata.get('logical_scores', {}).get(aku_j.id, 0.5)
                
                weight = self.alpha * sim_sem + self.beta * sim_temp + self.gamma * sim_log
                weights.append(weight)
                neighbor_ids.append(aku_j.id)
            
            if weights:
                w_min, w_max = min(weights), max(weights)
                if w_max > w_min:
                    normalized_weights = [(w - w_min) / (w_max - w_min) for w in weights]
                else:
                    normalized_weights = [1.0 for w in weights]
                
                for idx, neighbor_id in enumerate(neighbor_ids):
                    self.graph.add_edge(aku_i.id, neighbor_id, weight=normalized_weights[idx])

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
