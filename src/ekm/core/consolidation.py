import numpy as np
from sklearn.cluster import SpectralClustering, KMeans
from typing import List
from .models import AKU, GKU

class ConsolidationEngine:
    def __init__(self, merge_threshold: float = 0.92):
        self.merge_threshold = merge_threshold

    def nystrom_spectral_clustering(self, embeddings: np.ndarray, n_clusters: int, m: int = 100):
        N = embeddings.shape[0]
        if N <= m:
            sc = SpectralClustering(n_clusters=n_clusters, affinity='cosine')
            return sc.fit_predict(embeddings)

        indices = np.random.choice(N, m, replace=False)
        landmarks = embeddings[indices]
        
        def cosine_kernel(X, Y):
            X_norm = X / np.linalg.norm(X, axis=1, keepdims=True)
            Y_norm = Y / np.linalg.norm(Y, axis=1, keepdims=True)
            return (X_norm @ Y_norm.T + 1) / 2

        A = cosine_kernel(landmarks, landmarks)
        U, S, _ = np.linalg.svd(A)
        S_inv = 1.0 / (S + 1e-6)
        
        K_all_landmarks = cosine_kernel(embeddings, landmarks)
        V_approx = K_all_landmarks @ U @ np.diag(np.sqrt(S_inv))
        
        km = KMeans(n_clusters=n_clusters)
        clusters = km.fit_predict(V_approx)
        return clusters

    def sleep_consolidation(self, akus: List[AKU], graph_engine) -> List[AKU]:
        if not akus:
            return []

        merged_ids = set()
        new_akus = []
        
        embeddings = np.stack([a.embedding for a in akus])
        norm_embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        sim_matrix = np.matmul(norm_embeddings, norm_embeddings.T)

        for i in range(len(akus)):
            if akus[i].id in merged_ids:
                continue
                
            for j in range(i + 1, len(akus)):
                if akus[j].id in merged_ids:
                    continue
                
                if sim_matrix[i, j] > self.merge_threshold:
                    neighbors_i = set(graph_engine.graph.neighbors(akus[i].id))
                    neighbors_j = set(graph_engine.graph.neighbors(akus[j].id))
                    
                    if neighbors_i == neighbors_j:
                        akus[i].source_episode_ids.extend(akus[j].source_episode_ids)
                        akus[i].embedding = (akus[i].embedding + akus[j].embedding) / 2
                        akus[i].proposition += f" (Consolidated with {akus[j].id})"
                        merged_ids.add(akus[j].id)
            
            new_akus.append(akus[i])
        return new_akus
