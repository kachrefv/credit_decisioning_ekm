"""
Comprehensive test suite for the optimized Credithos EKM system.
Includes unit tests, integration tests, and performance validation.
"""
import unittest
import numpy as np
import time
import sys
import os
import tempfile
import shutil
from typing import List

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.ekm.core.models import Episode, AKU, GKU
from src.ekm.core.engine import EKM
from src.ekm.core.tensor_ops import TensorOperations, compute_semantic_similarity, compute_temporal_weight, normalize_embeddings
from src.ekm.core.graph import GraphEngine
from src.ekm.core.retrieval import EKMRetriever
from src.ekm.core.consolidation import ConsolidationEngine
from src.ekm.core.efficient_indexing import EfficientIndexer, ScalableEKM
from src.ekm.benchmarking.performance_benchmarks import PerformanceBenchmarkSuite


class TestTensorOperations(unittest.TestCase):
    """Test the tensor operations module."""
    
    def setUp(self):
        self.tensor_ops = TensorOperations(embedding_dim=768, projection_dim=64, k_sparse=10)
        self.embedding_i = np.random.randn(768)
        self.embedding_j = np.random.randn(768)
    
    def test_psi_matrix_initialization(self):
        """Test that the psi matrix is properly initialized."""
        self.assertEqual(self.tensor_ops.psi_matrix.shape, (768, 64))
        # Check that it's close to orthogonal (preserves information)
        product = self.tensor_ops.psi_matrix.T @ self.tensor_ops.psi_matrix
        identity = np.eye(64)
        np.testing.assert_array_almost_equal(product, identity, decimal=1)
    
    def test_compute_pattern_tensor(self):
        """Test the pattern tensor computation."""
        semantic_sim = compute_semantic_similarity(self.embedding_i, self.embedding_j)
        temporal_weight = compute_temporal_weight(1000, 1005)  # 5 second difference
        
        tensor = self.tensor_ops.compute_pattern_tensor(
            self.embedding_i, self.embedding_j, 
            semantic_sim, temporal_weight
        )
        
        self.assertEqual(tensor.shape, (64, 64))
        self.assertIsInstance(tensor, np.ndarray)
    
    def test_sparse_pattern_tensors(self):
        """Test the sparse pattern tensor computation."""
        embeddings = np.stack([self.embedding_i, self.embedding_j, np.random.randn(768)])
        similarities = np.array([
            [1.0, 0.5, 0.3],
            [0.5, 1.0, 0.7],
            [0.3, 0.7, 1.0]
        ])
        temporal_weights = np.ones((3, 3))  # All ones for simplicity
        
        sparse_tensors, indices = self.tensor_ops.compute_sparse_pattern_tensors(
            embeddings, similarities, temporal_weights
        )
        
        self.assertEqual(sparse_tensors.shape, (3, 10, 64, 64))  # 3 nodes, k_sparse=10, 64x64 tensors
        self.assertEqual(indices.shape, (3, 10))
    
    def test_normalize_embeddings(self):
        """Test embedding normalization function."""
        embeddings = np.random.randn(5, 768)
        normalized = normalize_embeddings(embeddings)
        
        # Check that all embeddings have unit length
        norms = np.linalg.norm(normalized, axis=1)
        np.testing.assert_array_almost_equal(norms, np.ones(5), decimal=5)


class TestGraphEngine(unittest.TestCase):
    """Test the graph engine with tensor operations."""
    
    def setUp(self):
        self.graph_engine = GraphEngine(k=5, embedding_dim=768, projection_dim=64)
        
        # Create test AKUs
        self.akus = []
        for i in range(10):
            embedding = np.random.randn(768)
            aku = AKU(
                id=f"aku_{i}",
                proposition=f"Test proposition {i}",
                source_episode_ids=[f"ep_{i}"],
                embedding=embedding,
                metadata={"timestamp": time.time() + i}
            )
            self.akus.append(aku)
    
    def test_build_knn_graph(self):
        """Test building the KNN graph with tensor operations."""
        self.graph_engine.build_knn_graph(self.akus)
        
        # Check that the graph has the right number of nodes
        self.assertEqual(len(self.graph_engine.graph.nodes), len(self.akus))
        
        # Check that edges have the expected attributes
        for edge in self.graph_engine.graph.edges(data=True):
            self.assertIn('weight', edge[2])
            self.assertIn('pattern_tensor', edge[2])
            self.assertIn('semantic_similarity', edge[2])
            self.assertIn('temporal_weight', edge[2])
    
    def test_extract_signatures(self):
        """Test extracting structural signatures."""
        self.graph_engine.build_knn_graph(self.akus)
        self.graph_engine.extract_signatures(self.akus)
        
        # Check that all AKUs now have structural signatures
        for aku in self.akus:
            self.assertIsNotNone(aku.structural_signature)
            # Signature should include topological features (13) + semantic features (768)
            expected_length = 13 + 768
            self.assertEqual(len(aku.structural_signature), expected_length)


class TestEKMRetriever(unittest.TestCase):
    """Test the EKM retriever with attention mechanism."""
    
    def setUp(self):
        self.retriever = EKMRetriever(d=768, projection_dim=64)
        
        # Create test AKUs
        self.akus = []
        for i in range(20):
            embedding = np.random.randn(768)
            aku = AKU(
                id=f"aku_{i}",
                proposition=f"Test proposition {i}",
                source_episode_ids=[f"ep_{i}"],
                embedding=embedding,
                metadata={"timestamp": time.time() + i}
            )
            self.akus.append(aku)
    
    def test_build_index(self):
        """Test building the retrieval index."""
        self.retriever.build_index(self.akus)
        
        # Check that all AKUs are mapped
        self.assertEqual(len(self.retriever.akus_map), len(self.akus))
        
        # Check that each AKU is accessible by ID
        for aku in self.akus:
            self.assertIn(aku.id, self.retriever.akus_map)
    
    def test_retrieve(self):
        """Test the retrieval functionality."""
        # First build the index
        self.retriever.build_index(self.akus)
        
        # Create a graph engine for testing
        graph_engine = GraphEngine(k=5, embedding_dim=768, projection_dim=64)
        graph_engine.build_knn_graph(self.akus)
        graph_engine.extract_signatures(self.akus)
        
        # Test retrieval
        query_embedding = np.random.randn(768)
        results = self.retriever.retrieve(query_embedding, graph_engine)
        
        # Check results format
        self.assertIsInstance(results, list)
        self.assertTrue(len(results) <= len(self.akus))  # Should return at most all AKUs
        
        if results:
            aku, score = results[0]
            self.assertIsInstance(aku, AKU)
            self.assertIsInstance(score, float)


class TestConsolidationEngine(unittest.TestCase):
    """Test the consolidation engine."""
    
    def setUp(self):
        self.consolidation = ConsolidationEngine(merge_threshold=0.9)
        
        # Create similar AKUs to test consolidation
        self.akus = []
        base_embedding = np.random.randn(768)
        for i in range(20):
            # Add small noise to create similar embeddings
            embedding = base_embedding + np.random.randn(768) * 0.05  # Small variance
            aku = AKU(
                id=f"aku_{i}",
                proposition=f"Similar proposition {i}",
                source_episode_ids=[f"ep_{i}"],
                embedding=embedding,
                metadata={"timestamp": time.time() + i}
            )
            self.akus.append(aku)
    
    def test_nystrom_spectral_clustering(self):
        """Test the Nyström spectral clustering."""
        embeddings = np.stack([aku.embedding for aku in self.akus])
        
        # Test clustering with different numbers of clusters
        for n_clusters in [2, 3, 5]:
            labels = self.consolidation.nystrom_spectral_clustering(embeddings, n_clusters)
            
            self.assertEqual(len(labels), len(embeddings))
            self.assertEqual(len(np.unique(labels)), n_clusters)
    
    def test_optimize_cluster_number(self):
        """Test automatic cluster number optimization."""
        embeddings = np.stack([aku.embedding for aku in self.akus])
        
        n_clusters = self.consolidation.optimize_cluster_number(embeddings, max_clusters=5)
        
        self.assertIsInstance(n_clusters, int)
        self.assertGreaterEqual(n_clusters, 1)
        self.assertLessEqual(n_clusters, 5)
    
    def test_sleep_consolidation(self):
        """Test the sleep consolidation process."""
        # Create a simple graph engine for testing
        graph_engine = GraphEngine(k=5, embedding_dim=768, projection_dim=64)
        graph_engine.build_knn_graph(self.akus)
        
        original_count = len(self.akus)
        consolidated_akus = self.consolidation.sleep_consolidation(self.akus, graph_engine)
        
        # Consolidation should reduce the number of AKUs
        self.assertLessEqual(len(consolidated_akus), original_count)
        
        # Check that consolidated AKUs have proper metadata
        for aku in consolidated_akus:
            self.assertIn('consolidated_from', aku.metadata)


class TestEfficientIndexer(unittest.TestCase):
    """Test the efficient indexer for scalability."""
    
    def setUp(self):
        self.indexer = EfficientIndexer(embedding_dim=768, projection_dim=64, k_sparse=5)
        
        # Create test AKUs
        self.akus = []
        for i in range(50):  # Larger set for efficiency testing
            embedding = np.random.randn(768)
            aku = AKU(
                id=f"aku_{i}",
                proposition=f"Test proposition {i}",
                source_episode_ids=[f"ep_{i}"],
                embedding=embedding,
                metadata={"timestamp": time.time() + i}
            )
            self.akus.append(aku)
    
    def test_add_akus_batch(self):
        """Test batch addition of AKUs."""
        self.indexer.add_akus_batch(self.akus)
        
        # Check that all AKUs are indexed
        self.assertEqual(self.indexer.faiss_index.ntotal, len(self.akus))
        self.assertEqual(len(self.indexer.akus_by_id), len(self.akus))
        
        # Check mappings
        for aku in self.akus:
            self.assertIn(aku.id, self.indexer.akus_by_id)
            self.assertIn(aku.id, self.indexer.id_to_faiss_idx)
    
    def test_search_functionality(self):
        """Test search functionality."""
        self.indexer.add_akus_batch(self.akus)
        
        # Build relationships
        self.indexer.build_sparse_relationships()
        
        # Test search
        query_embedding = np.random.randn(768)
        results = self.indexer.search(query_embedding, k=5)
        
        self.assertIsInstance(results, list)
        self.assertLessEqual(len(results), 5)
        
        if results:
            aku, score = results[0]
            self.assertIsInstance(aku, AKU)
            self.assertIsInstance(score, float)
    
    def test_enhanced_search_with_attention(self):
        """Test enhanced search with attention mechanism."""
        self.indexer.add_akus_batch(self.akus)
        
        # Build relationships
        self.indexer.build_sparse_relationships()
        
        # Test enhanced search
        query_embedding = np.random.randn(768)
        results = self.indexer.enhanced_search_with_attention(query_embedding, k=5)
        
        self.assertIsInstance(results, list)
        self.assertLessEqual(len(results), 5)
        
        if results:
            aku, score = results[0]
            self.assertIsInstance(aku, AKU)
            self.assertIsInstance(score, float)
    
    def test_save_and_load_index(self):
        """Test saving and loading the index."""
        self.indexer.add_akus_batch(self.akus)
        self.indexer.build_sparse_relationships()
        
        # Create temporary directory for testing
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save index
            self.indexer.save_index(temp_dir)
            
            # Create new indexer and load
            new_indexer = EfficientIndexer(embedding_dim=768, projection_dim=64, k_sparse=5)
            new_indexer.load_index(temp_dir)
            
            # Verify loaded data
            self.assertEqual(new_indexer.faiss_index.ntotal, len(self.akus))
            self.assertEqual(len(new_indexer.akus_by_id), len(self.akus))
            
            # Test that search still works
            query_embedding = np.random.randn(768)
            results = new_indexer.search(query_embedding, k=3)
            self.assertIsInstance(results, list)


class TestScalableEKM(unittest.TestCase):
    """Test the scalable EKM implementation."""
    
    def setUp(self):
        self.scalable_ekm = ScalableEKM(embedding_dim=768, projection_dim=64, k_sparse=5)
        
        # Create test AKUs
        self.akus = []
        for i in range(30):
            embedding = np.random.randn(768)
            aku = AKU(
                id=f"aku_{i}",
                proposition=f"Test proposition {i}",
                source_episode_ids=[f"ep_{i}"],
                embedding=embedding,
                metadata={"timestamp": time.time() + i}
            )
            self.akus.append(aku)
    
    def test_ingest_and_build_index(self):
        """Test ingesting AKUs and building index."""
        self.scalable_ekm.ingest_akus(self.akus)
        
        # Check that AKUs are stored
        self.assertEqual(len(self.scalable_ekm.akus), len(self.akus))
        self.assertEqual(len(self.scalable_ekm.akus_by_id), len(self.akus))
        
        # Build index
        self.scalable_ekm.build_index()
        
        # Check stats are updated
        stats = self.scalable_ekm.get_stats()
        self.assertGreaterEqual(stats['index_build_count'], 1)
    
    def test_retrieve_functionality(self):
        """Test retrieval functionality."""
        self.scalable_ekm.ingest_akus(self.akus)
        self.scalable_ekm.build_index()
        
        # Test retrieval
        query_embedding = np.random.randn(768)
        results = self.scalable_ekm.retrieve(query_embedding, k=5)
        
        self.assertIsInstance(results, list)
        self.assertLessEqual(len(results), 5)
        
        if results:
            aku, score = results[0]
            self.assertIsInstance(aku, AKU)
            self.assertIsInstance(score, float)


class TestEKMIntegration(unittest.TestCase):
    """Integration tests for the main EKM engine."""
    
    def setUp(self):
        # Test with scalable indexing enabled
        self.ekm = EKM(d=768, k=5, mesh_threshold=10, use_scalable_index=True)
    
    def test_full_workflow(self):
        """Test the full workflow: ingest -> retrieve -> consolidate."""
        # Create test episodes
        episodes = []
        for i in range(15):  # Exceed threshold to trigger mesh mode
            embedding = np.random.randn(768)
            episode = Episode(
                id=f"ep_{i}",
                content=f"Test episode {i}",
                embedding=embedding,
                metadata={"timestamp": 1000 + i}
            )
            episodes.append(episode)
        
        # Ingest episodes
        self.ekm.ingest_episodes(episodes)
        
        # Verify mode transition
        self.assertEqual(self.ekm.mode, "Mesh Mode")
        
        # Test retrieval
        query_embedding = np.random.randn(768)
        results = self.ekm.retrieve("test query", query_embedding)
        
        self.assertIsInstance(results, list)
        if results:
            aku, score = results[0]
            self.assertIsInstance(aku, AKU)
            self.assertIsInstance(score, float)
        
        # Test consolidation
        original_count = len(self.ekm.akus)
        self.ekm.consolidate()
        after_consolidation_count = len(self.ekm.akus)
        
        # Consolidation should have run
        self.assertGreaterEqual(self.ekm.stats['total_consolidations'], 1)
        
        # Get performance stats
        stats = self.ekm.get_performance_stats()
        self.assertGreaterEqual(stats['total_ingested'], len(episodes))
        self.assertGreaterEqual(stats['total_retrievals'], 1)
        self.assertGreaterEqual(stats['total_consolidations'], 1)


class TestPerformanceBenchmarks(unittest.TestCase):
    """Test the performance benchmarking functionality."""
    
    def test_benchmark_suite_creation(self):
        """Test that benchmark suite can be created."""
        benchmark_suite = PerformanceBenchmarkSuite()
        self.assertIsInstance(benchmark_suite, PerformanceBenchmarkSuite)
        
        # Verify methods exist
        self.assertTrue(hasattr(benchmark_suite, 'benchmark_latency'))
        self.assertTrue(hasattr(benchmark_suite, 'benchmark_retrieval_precision'))
        self.assertTrue(hasattr(benchmark_suite, 'benchmark_scalability'))
        self.assertTrue(hasattr(benchmark_suite, 'benchmark_memory_efficiency'))
        self.assertTrue(hasattr(benchmark_suite, 'compare_with_baseline_rag'))
    
    def test_generate_report(self):
        """Test report generation."""
        benchmark_suite = PerformanceBenchmarkSuite()
        
        # Add some dummy results
        benchmark_suite.results = {
            'retrieval_precision': {
                'precision_at_k': 0.85,
                'std_precision': 0.05,
                'num_queries': 100,
                'time_elapsed': 2.5
            },
            'latency': {
                'avg_latency_ms': 150.0,
                'median_latency_ms': 140.0,
                'p95_latency_ms': 200.0,
                'p99_latency_ms': 250.0,
                'num_iterations': 1000
            }
        }
        
        report = benchmark_suite.generate_report()
        self.assertIsInstance(report, str)
        self.assertIn("Credithos EKM Performance Benchmarking Report", report)
        self.assertIn("Average Precision", report)
        self.assertIn("Latency Performance", report)


def run_comprehensive_test_suite():
    """Run all tests in the comprehensive suite."""
    print("Running comprehensive test suite for optimized Credithos EKM...")
    
    # Create a test suite
    loader = unittest.TestLoader()
    
    # Collect all test cases
    tensor_tests = loader.loadTestsFromTestCase(TestTensorOperations)
    graph_tests = loader.loadTestsFromTestCase(TestGraphEngine)
    retrieval_tests = loader.loadTestsFromTestCase(TestEKMRetriever)
    consolidation_tests = loader.loadTestsFromTestCase(TestConsolidationEngine)
    efficient_index_tests = loader.loadTestsFromTestCase(TestEfficientIndexer)
    scalable_ekm_tests = loader.loadTestsFromTestCase(TestScalableEKM)
    integration_tests = loader.loadTestsFromTestCase(TestEKMIntegration)
    benchmark_tests = loader.loadTestsFromTestCase(TestPerformanceBenchmarks)
    
    # Create suite with all tests
    suite = unittest.TestSuite([
        tensor_tests,
        graph_tests,
        retrieval_tests,
        consolidation_tests,
        efficient_index_tests,
        scalable_ekm_tests,
        integration_tests,
        benchmark_tests
    ])
    
    # Run the tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print(f"\n{'='*60}")
    print("COMPREHENSIVE TEST SUITE RESULTS")
    print(f"{'='*60}")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.failures:
        print("\nFAILURES:")
        for test, trace in result.failures:
            print(f"  {test}: {trace}")
    
    if result.errors:
        print("\nERRORS:")
        for test, trace in result.errors:
            print(f"  {test}: {trace}")
    
    if result.wasSuccessful():
        print("\n✅ All tests passed! Comprehensive test suite completed successfully.")
    else:
        print(f"\n❌ Test suite completed with issues.")
    
    return result


if __name__ == '__main__':
    run_comprehensive_test_suite()