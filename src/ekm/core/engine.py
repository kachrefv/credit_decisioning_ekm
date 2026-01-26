import numpy as np
from typing import List, Dict, Tuple
from .models import Episode, AKU, GKU
from .graph import GraphEngine
from .retrieval import EKMRetriever
from .consolidation import ConsolidationEngine
from .efficient_indexing import ScalableEKM
from .config import get_config
import time
import logging

logger = logging.getLogger(__name__)

class EKM:
    def __init__(self, d: int = 768, k: int = 10, mesh_threshold: int = 1000,
                 embedding_dim: int = 768, projection_dim: int = 64, use_scalable_index: bool = True,
                 config_path: str = None):
        # Load configuration
        self.config = get_config(config_path)

        self.d = d or self.config.embedding_dim
        self.k = k or self.config.k_sparse
        self.mesh_threshold = mesh_threshold or self.config.mesh_threshold
        self.use_scalable_index = use_scalable_index and self.config.use_scalable_index

        self.episodes = []
        self.akus = []
        self.gkus = []

        if self.use_scalable_index:
            # Use the scalable EKM with efficient indexing
            self.scalable_ekm = ScalableEKM(
                embedding_dim=self.d,
                projection_dim=projection_dim or self.config.projection_dim,
                k_sparse=self.k,
                higher_order_terms=self.config.enable_higher_order_terms
            )
        else:
            # Use the traditional approach
            self.graph_engine = GraphEngine(
                k=self.k,
                embedding_dim=self.d,
                projection_dim=projection_dim or self.config.projection_dim
            )
            self.retriever = EKMRetriever(
                d=self.d,
                projection_dim=projection_dim or self.config.projection_dim,
                candidate_size=self.config.candidate_size
            )

        self.consolidation = ConsolidationEngine()

        self.mode = "Cold Start"

        # Performance tracking
        self.stats = {
            'total_ingested': 0,
            'total_retrievals': 0,
            'total_consolidations': 0,
            'avg_retrieval_time': 0.0,
            'avg_ingestion_time': 0.0
        }

    def ingest_episodes(self, episodes: List[Episode]):
        start_time = time.time()

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
        self.stats['total_ingested'] += len(episodes)

        # Add to scalable index if enabled
        if self.use_scalable_index:
            self.scalable_ekm.ingest_akus(new_akus)

        self._check_mode_shift()

        if self.mode == "Mesh Mode":
            self.update_mesh()

        ingestion_time = time.time() - start_time
        self.stats['avg_ingestion_time'] = (
            (self.stats['avg_ingestion_time'] * (self.stats['total_ingested'] - len(episodes)) +
             ingestion_time * len(episodes)) / self.stats['total_ingested']
        )

    def _check_mode_shift(self):
        if len(self.akus) >= self.mesh_threshold:
            self.mode = "Mesh Mode"
        else:
            self.mode = "Cold Start"

    def update_mesh(self):
        """
        Update the knowledge mesh with proper tensor operations and graph analysis.
        """
        logger.info(f"Updating mesh with {len(self.akus)} AKUs")

        if self.use_scalable_index:
            # Use scalable indexing
            self.scalable_ekm.build_index()
        else:
            # Traditional approach
            self.graph_engine.build_knn_graph(self.akus)
            self.graph_engine.extract_signatures(self.akus)
            self.retriever.build_index(self.akus)

        logger.info("Mesh update completed")

    def retrieve(self, query_text: str, query_embedding: np.ndarray) -> List[Tuple[AKU, float]]:
        """
        Retrieve relevant AKUs using enhanced attention mechanism.
        """
        start_time = time.time()

        if self.mode == "Cold Start" or not self.akus:
            if not self.akus:
                return []
            embeddings = np.stack([a.embedding for a in self.akus])
            norm_q = query_embedding / np.linalg.norm(query_embedding)
            norm_e = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
            scores = np.matmul(norm_e, norm_q)
            results = sorted(zip(self.akus, scores), key=lambda x: x[1], reverse=True)[:10]
        else:
            if self.use_scalable_index:
                # Use scalable retrieval
                results = self.scalable_ekm.retrieve(query_embedding)
            else:
                # Use traditional retrieval
                results = self.retriever.retrieve(query_embedding, self.graph_engine)

        retrieval_time = time.time() - start_time
        self.stats['total_retrievals'] += 1

        # Update average retrieval time
        self.stats['avg_retrieval_time'] = (
            (self.stats['avg_retrieval_time'] * (self.stats['total_retrievals'] - 1) +
             retrieval_time) / self.stats['total_retrievals']
        )

        return results

    def consolidate(self):
        """
        Perform sleep consolidation with enhanced clustering algorithms.
        """
        logger.info(f"Starting consolidation of {len(self.akus)} AKUs")
        start_time = time.time()

        if self.use_scalable_index:
            # For scalable EKM, consolidation is handled differently
            # We'll rebuild the index which incorporates consolidation principles
            self.scalable_ekm.build_index()
        else:
            # Perform consolidation using improved algorithms
            self.akus = self.consolidation.sleep_consolidation(self.akus, self.graph_engine)

            if len(self.akus) > self.k:
                n_clusters = max(2, min(len(self.akus) // 10, 50))  # Cap clusters to prevent too many
                embeddings = np.stack([a.embedding for a in self.akus])

                # Use the improved clustering algorithm
                cluster_ids = self.consolidation.nystrom_spectral_clustering(embeddings, n_clusters=n_clusters)

                new_gkus = []
                for c_id in range(n_clusters):
                    member_akus = [self.akus[i] for i, cid in enumerate(cluster_ids) if cid == c_id]
                    if member_akus:
                        gku = GKU(
                            id=f"gku_{c_id}",
                            label=f"Cluster {c_id}",
                            aku_ids=[a.id for a in member_akus],
                            centroid=np.mean([a.embedding for a in member_akus], axis=0),
                            metadata={
                                'creation_time': time.time(),
                                'member_count': len(member_akus),
                                'cluster_quality': self._calculate_cluster_quality(member_akus)
                            }
                        )
                        new_gkus.append(gku)
                self.gkus = new_gkus

            self.update_mesh()

        consolidation_time = time.time() - start_time
        self.stats['total_consolidations'] += 1

        logger.info(f"Consolidation completed in {consolidation_time:.2f}s")

    def _calculate_cluster_quality(self, akus: List[AKU]) -> float:
        """
        Calculate the quality of a cluster based on internal cohesion and external separation.
        """
        if len(akus) < 2:
            return 1.0  # Perfect quality for single-item clusters

        embeddings = np.stack([aku.embedding for aku in akus])
        centroid = np.mean(embeddings, axis=0)

        # Calculate internal cohesion (average distance to centroid)
        distances_to_centroid = [np.linalg.norm(emb - centroid) for emb in embeddings]
        avg_internal_distance = np.mean(distances_to_centroid)

        # For external separation, we'd normally compare to other clusters
        # For simplicity, we'll use inverse of internal distance as quality measure
        # Higher quality means tighter clusters
        quality = 1.0 / (1.0 + avg_internal_distance)
        return min(quality, 1.0)  # Cap at 1.0

    def get_performance_stats(self) -> Dict:
        """
        Get performance statistics for the EKM system.
        """
        stats = {
            **self.stats,
            'current_akus': len(self.akus),
            'current_gkus': len(self.gkus),
            'current_episodes': len(self.episodes),
            'mode': self.mode
        }

        if self.use_scalable_index:
            stats.update(self.scalable_ekm.get_stats())

        return stats
