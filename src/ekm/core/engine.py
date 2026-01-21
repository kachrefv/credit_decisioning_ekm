import numpy as np
from typing import List, Dict, Tuple
from .models import Episode, AKU, GKU
from .graph import GraphEngine
from .retrieval import EKMRetriever
from .consolidation import ConsolidationEngine

class EKM:
    def __init__(self, d: int = 768, k: int = 10, mesh_threshold: int = 1000):
        self.d = d
        self.k = k
        self.mesh_threshold = mesh_threshold
        
        self.episodes = []
        self.akus = []
        self.gkus = []
        
        self.graph_engine = GraphEngine(k=k)
        self.retriever = EKMRetriever(d=d)
        self.consolidation = ConsolidationEngine()
        
        self.mode = "Cold Start"

    def ingest_episodes(self, episodes: List[Episode]):
        self.episodes.extend(episodes)
        new_akus = []
        for ep in episodes:
            aku = AKU(
                id=f"aku_{ep.id}",
                proposition=ep.content,
                source_episode_ids=[ep.id],
                embedding=ep.embedding,
                metadata=ep.metadata
            )
            new_akus.append(aku)
        
        self.akus.extend(new_akus)
        self._check_mode_shift()
        
        if self.mode == "Mesh Mode":
            self.update_mesh()

    def _check_mode_shift(self):
        if len(self.akus) >= self.mesh_threshold:
            self.mode = "Mesh Mode"
        else:
            self.mode = "Cold Start"

    def update_mesh(self):
        self.graph_engine.build_knn_graph(self.akus)
        self.graph_engine.extract_signatures(self.akus)
        self.retriever.build_index(self.akus)

    def retrieve(self, query_text: str, query_embedding: np.ndarray) -> List[Tuple[AKU, float]]:
        if self.mode == "Cold Start" or not self.akus:
            if not self.akus: return []
            embeddings = np.stack([a.embedding for a in self.akus])
            norm_q = query_embedding / np.linalg.norm(query_embedding)
            norm_e = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
            scores = np.matmul(norm_e, norm_q)
            results = sorted(zip(self.akus, scores), key=lambda x: x[1], reverse=True)[:10]
            return results
        else:
            return self.retriever.retrieve(query_embedding, self.graph_engine)

    def consolidate(self):
        self.akus = self.consolidation.sleep_consolidation(self.akus, self.graph_engine)
        
        if len(self.akus) > self.k:
            n_clusters = max(2, len(self.akus) // 10)
            embeddings = np.stack([a.embedding for a in self.akus])
            cluster_ids = self.consolidation.nystrom_spectral_clustering(embeddings, n_clusters=n_clusters)
            
            new_gkus = []
            for c_id in range(n_clusters):
                member_akus = [self.akus[i] for i, cid in enumerate(cluster_ids) if cid == c_id]
                if member_akus:
                    gku = GKU(
                        id=f"gku_{c_id}",
                        label=f"Cluster {c_id}",
                        aku_ids=[a.id for a in member_akus],
                        centroid=np.mean([a.embedding for a in member_akus], axis=0)
                    )
                    new_gkus.append(gku)
            self.gkus = new_gkus
            
        self.update_mesh()
