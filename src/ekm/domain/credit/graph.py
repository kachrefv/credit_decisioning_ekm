import numpy as np
import networkx as nx
from typing import List, Dict, Tuple
from .models import CreditRiskFactor

class CreditGraphEngine:
    def __init__(self, k: int = 10, alpha: float = 0.5, beta: float = 0.3, gamma: float = 0.2, tau: float = 86400):
        self.k = k
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.tau = tau
        self.graph = nx.DiGraph()

    def build_risk_knn_graph(self, risk_factors: List[CreditRiskFactor]):
        if not risk_factors:
            return

        embeddings = np.stack([a.embedding for a in risk_factors])
        norm_embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        cosine_sim = np.matmul(norm_embeddings, norm_embeddings.T)

        for i, risk_i in enumerate(risk_factors):
            self.graph.add_node(risk_i.id, risk_factor=risk_i)
            neighbors_idx = np.argsort(cosine_sim[i])[-self.k-1:-1][::-1]

            weights = []
            neighbor_ids = []

            for j in neighbors_idx:
                risk_j = risk_factors[j]
                sim_sem = cosine_sim[i, j]
                t_i = risk_i.metadata.get('timestamp', 0)
                t_j = risk_j.metadata.get('timestamp', 0)
                sim_temp = np.exp(-abs(t_i - t_j) / self.tau)
                risk_dependency = risk_i.metadata.get('risk_dependencies', {}).get(risk_j.id, 0.5)
                risk_weight_i = self._risk_level_to_weight(risk_i.risk_level)
                risk_weight_j = self._risk_level_to_weight(risk_j.risk_level)
                
                weight = (self.alpha * sim_sem + self.beta * sim_temp + self.gamma * risk_dependency) * (risk_weight_i + risk_weight_j) / 2
                weights.append(weight)
                neighbor_ids.append(risk_j.id)

            if weights:
                w_min, w_max = min(weights), max(weights)
                if w_max > w_min:
                    normalized_weights = [(w - w_min) / (w_max - w_min) for w in weights]
                else:
                    normalized_weights = [1.0 for w in weights]

                for idx, neighbor_id in enumerate(neighbor_ids):
                    self.graph.add_edge(risk_i.id, neighbor_id, weight=normalized_weights[idx])

    def _risk_level_to_weight(self, risk_level: str) -> float:
        weights = {"low": 0.3, "medium": 0.6, "high": 0.8, "critical": 1.0}
        return weights.get(risk_level, 0.5)

    def get_3node_risk_motif_distribution(self, node_id: str) -> np.ndarray:
        return np.random.rand(13)

    def extract_risk_signatures(self, risk_factors: List[CreditRiskFactor]):
        for risk in risk_factors:
            topo = self.get_3node_risk_motif_distribution(risk.id)
            neighbors = list(self.graph.neighbors(risk.id))
            if neighbors:
                neighbor_embeddings = [self.graph.nodes[n]['risk_factor'].embedding for n in neighbors]
                sem = np.mean(neighbor_embeddings, axis=0)
            else:
                sem = risk.embedding
            risk.structural_signature = np.concatenate([topo, sem])

    def get_graph_data(self) -> Dict:
        """Type-safe graph export for visualization"""
        nodes = []
        links = []
        
        # Safe node access
        for node_id in self.graph.nodes():
            node_data = self.graph.nodes[node_id]
            risk_factor = node_data.get('risk_factor')
            if risk_factor:
                 nodes.append({
                    "id": node_id,
                    "group": risk_factor.risk_level,
                    "risk_factor": risk_factor.risk_factor,
                    "val": 1 # For visualization size
                })

        # Safe edge access 
        for u, v, data in self.graph.edges(data=True):
             links.append({
                "source": u,
                "target": v,
                "value": data.get('weight', 0.5)
            })
            
        return {"nodes": nodes, "links": links}
