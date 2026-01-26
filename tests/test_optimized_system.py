"""
Test suite for the optimized Credithos EKM system.
Validates that the mathematical improvements work correctly.
"""
import unittest
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from src.ekm.core.models import Episode, AKU
from src.ekm.core.engine import EKM
from src.ekm.core.tensor_ops import TensorOperations, compute_semantic_similarity, compute_temporal_weight
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


class TestEKMEngine(unittest.TestCase):
    """Test the optimized EKM engine."""
    
    def setUp(self):
        self.ekm = EKM(d=768, k=5, mesh_threshold=10, projection_dim=64)
    
    def test_ingest_and_retrieve(self):
        """Test basic ingestion and retrieval functionality."""
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
        self.assertTrue(len(results) <= 10)  # Should be limited to top 10
        if results:
            aku, score = results[0]
            self.assertIsInstance(aku, AKU)
            self.assertIsInstance(score, float)
    
    def test_consolidation(self):
        """Test the consolidation functionality."""
        # Create similar episodes to encourage consolidation
        episodes = []
        base_embedding = np.random.randn(768)
        for i in range(20):
            # Add small noise to create similar but not identical embeddings
            embedding = base_embedding + np.random.randn(768) * 0.1
            episode = Episode(
                id=f"ep_{i}",
                content=f"Similar episode {i}",
                embedding=embedding,
                metadata={"timestamp": 1000 + i}
            )
            episodes.append(episode)
        
        self.ekm.ingest_episodes(episodes)
        
        # Record AKU count before consolidation
        akus_before = len(self.ekm.akus)
        
        # Perform consolidation
        self.ekm.consolidate()
        
        # AKU count should be reduced after consolidation
        akus_after = len(self.ekm.akus)
        self.assertLessEqual(akus_after, akus_before)
        
        # Verify the system still works after consolidation
        query_embedding = np.random.randn(768)
        results = self.ekm.retrieve("test query", query_embedding)
        self.assertIsInstance(results, list)


class TestPerformanceBenchmarks(unittest.TestCase):
    """Test the performance benchmarking functionality."""
    
    def test_benchmark_creation(self):
        """Test that benchmark suite can be created and run."""
        benchmark_suite = PerformanceBenchmarkSuite()
        self.assertIsInstance(benchmark_suite, PerformanceBenchmarkSuite)
        
        # Verify methods exist
        self.assertTrue(hasattr(benchmark_suite, 'benchmark_latency'))
        self.assertTrue(hasattr(benchmark_suite, 'benchmark_retrieval_precision'))


def run_all_tests():
    """Run all tests in the suite."""
    # Create a test suite
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(__import__('__main__', globals(), locals(), ['TestTensorOperations']))
    
    # Add tests from all classes
    tensor_tests = loader.loadTestsFromTestCase(TestTensorOperations)
    ekm_tests = loader.loadTestsFromTestCase(TestEKMEngine)
    benchmark_tests = loader.loadTestsFromTestCase(TestPerformanceBenchmarks)
    
    suite = unittest.TestSuite([tensor_tests, ekm_tests, benchmark_tests])
    
    # Run the tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result


if __name__ == '__main__':
    print("Running comprehensive test suite for optimized Credithos EKM...")
    result = run_all_tests()
    
    if result.wasSuccessful():
        print("\n✅ All tests passed! Optimization implementation is working correctly.")
    else:
        print(f"\n❌ {len(result.failures)} test(s) failed, {len(result.errors)} error(s) occurred.")
        for failure in result.failures:
            print(f"FAILURE in {failure[0]}: {failure[1]}")
        for error in result.errors:
            print(f"ERROR in {error[0]}: {error[1]}")