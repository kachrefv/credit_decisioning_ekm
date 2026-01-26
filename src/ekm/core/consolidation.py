import numpy as np
from sklearn.cluster import SpectralClustering, KMeans
from sklearn.metrics import silhouette_score
from typing import List
from .models import AKU, GKU
import logging

logger = logging.getLogger(__name__)

class ConsolidationEngine:
    def __init__(self, merge_threshold: float = 0.92, consolidation_method: str = 'spectral'):
        self.merge_threshold = merge_threshold
        self.consolidation_method = consolidation_method

    def nystrom_spectral_clustering(self, embeddings: np.ndarray, n_clusters: int, m: int = 100):
        """
        Improved Nyström spectral clustering with better numerical stability.
        """
        N = embeddings.shape[0]
        if N <= m:
            # For small datasets, use standard spectral clustering
            sc = SpectralClustering(n_clusters=n_clusters, affinity='cosine', random_state=42)
            return sc.fit_predict(embeddings)

        # Select landmark points using k-means++ for better coverage
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=m, init='k-means++', random_state=42)
        kmeans.fit(embeddings)
        landmark_indices = kmeans.predict(embeddings)  # Get closest landmark for each point
        unique_landmarks = np.unique(kmeans.labels_)

        # Use the centroids as landmarks
        landmarks = kmeans.cluster_centers_

        def cosine_kernel(X, Y):
            X_norm = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
            Y_norm = Y / (np.linalg.norm(Y, axis=1, keepdims=True) + 1e-9)
            return np.clip(X_norm @ Y_norm.T, -1.0, 1.0)  # Clamp to [-1, 1] for numerical stability

        # Compute kernel matrix for landmarks
        A = cosine_kernel(landmarks, landmarks)

        # Add regularization for numerical stability
        A_reg = A + 1e-6 * np.eye(A.shape[0])

        # Compute SVD
        U, S, _ = np.linalg.svd(A_reg)
        S_inv_sqrt = np.diag(1.0 / np.sqrt(S + 1e-9))  # Add small epsilon to avoid division by zero

        # Compute kernel between all points and landmarks
        K_all_landmarks = cosine_kernel(embeddings, landmarks)

        # Compute Nyström approximation
        V_approx = K_all_landmarks @ U @ S_inv_sqrt

        # Perform clustering on the approximated embeddings
        km = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = km.fit_predict(V_approx)
        return clusters

    def optimize_cluster_number(self, embeddings: np.ndarray, max_clusters: int = 10) -> int:
        """
        Optimize the number of clusters using silhouette analysis.
        """
        if embeddings.shape[0] < 2:
            return 1
        if embeddings.shape[0] < max_clusters:
            max_clusters = embeddings.shape[0] - 1

        if max_clusters < 2:
            return 1

        best_n_clusters = 2
        best_score = -1

        for n_clusters in range(2, max_clusters + 1):
            try:
                labels = self.nystrom_spectral_clustering(embeddings, n_clusters)
                if len(np.unique(labels)) > 1:  # Need at least 2 clusters for silhouette
                    score = silhouette_score(embeddings, labels)
                    if score > best_score:
                        best_score = score
                        best_n_clusters = n_clusters
            except:
                continue  # Skip if clustering fails

        return best_n_clusters

    def sleep_consolidation(self, akus: List[AKU], graph_engine) -> List[AKU]:
        """
        Enhanced sleep consolidation with better merging heuristics.
        """
        if not akus:
            return []

        logger.info(f"Starting sleep consolidation for {len(akus)} AKUs")

        # Create a copy to work with
        working_akus = akus.copy()

        # Compute similarity matrix once
        embeddings = np.stack([a.embedding for a in working_akus])
        norm_embeddings = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-9)
        sim_matrix = np.clip(norm_embeddings @ norm_embeddings.T, -1.0, 1.0)  # Cosine similarity

        # Track which AKUs have been merged
        merged_mask = np.zeros(len(working_akus), dtype=bool)
        consolidated_akus = []

        # More sophisticated merging logic
        for i in range(len(working_akus)):
            if merged_mask[i]:
                continue

            current_aku = working_akus[i]
            similar_indices = []

            # Find all highly similar AKUs that could be merged
            for j in range(i + 1, len(working_akus)):
                if merged_mask[j]:
                    continue

                if sim_matrix[i, j] > self.merge_threshold:
                    # Additional checks for merging eligibility
                    neighbors_i = set(graph_engine.graph.neighbors(working_akus[i].id))
                    neighbors_j = set(graph_engine.graph.neighbors(working_akus[j].id))

                    # Only merge if they have similar neighborhood structures
                    if len(neighbors_i.intersection(neighbors_j)) > 0 or len(neighbors_i) == 0 and len(neighbors_j) == 0:
                        similar_indices.append(j)

            if similar_indices:
                # Merge all similar AKUs into the current one
                merged_source_ids = current_aku.source_episode_ids.copy()
                merged_embeddings = [current_aku.embedding]
                merged_propositions = [current_aku.proposition]

                for idx in similar_indices:
                    merged_mask[idx] = True
                    merged_source_ids.extend(working_akus[idx].source_episode_ids)
                    merged_embeddings.append(working_akus[idx].embedding)
                    merged_propositions.append(working_akus[idx].proposition)

                # Create consolidated AKU with averaged embedding
                consolidated_aku = AKU(
                    id=current_aku.id,
                    proposition="; ".join(merged_propositions[:5]) + ("..." if len(merged_propositions) > 5 else ""),  # Limit length
                    source_episode_ids=merged_source_ids,
                    embedding=np.mean(merged_embeddings, axis=0),
                    metadata={
                        **current_aku.metadata,
                        'consolidated_from': len(similar_indices) + 1,
                        'consolidation_timestamp': np.datetime64('now')
                    }
                )
                consolidated_akus.append(consolidated_aku)
            else:
                # No merging needed, keep original
                consolidated_akus.append(current_aku)

            # Mark current as processed
            merged_mask[i] = True

        logger.info(f"Sleep consolidation completed: {len(akus)} -> {len(consolidated_akus)} AKUs")
        return consolidated_akus
