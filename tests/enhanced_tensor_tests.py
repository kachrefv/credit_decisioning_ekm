"""
Comprehensive test suite for the enhanced EKM system with advanced tensor operations.
Tests mathematical sophistication, performance, and correctness of implementations.
"""
import unittest
import numpy as np
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.ekm.core.tensor_ops import TensorOperations
from src.ekm.core.advanced_tensor_ops import AdvancedTensorOperations, TensorNetworkAnalyzer
from src.ekm.core.models import Episode, AKU
from src.ekm.core.engine import EKM
from src.ekm.core.config import EKMConfig


class TestAdvancedTensorOperations(unittest.TestCase):
    """Test the advanced tensor operations module."""

    def setUp(self):
        self.advanced_ops = AdvancedTensorOperations(embedding_dim=768, projection_dim=64, k_sparse=10)
        self.embedding_i = np.random.randn(768)
        self.embedding_j = np.random.randn(768)

    def test_core_transform_initialization(self):
        """Test that the core transform tensor is properly initialized."""
        core_shape = self.advanced_ops.core_transform.shape
        expected_shape = (64, 64, 64)  # projection_dim ^ 3
        self.assertEqual(core_shape, expected_shape)

    def test_tensor_unfolding(self):
        """Test tensor unfolding and folding operations."""
        tensor = np.random.randn(10, 15, 20)
        
        # Test unfolding
        unfolded_0 = self.advanced_ops._unflatten(tensor, 0)
        self.assertEqual(unfolded_0.shape, (10, 15*20))
        
        unfolded_1 = self.advanced_ops._unflatten(tensor, 1)
        self.assertEqual(unfolded_1.shape, (15, 10*20))
        
        unfolded_2 = self.advanced_ops._unflatten(tensor, 2)
        self.assertEqual(unfolded_2.shape, (20, 10*15))
        
        # Test folding back
        folded_0 = self.advanced_ops._flatten(unfolded_0, tensor.shape, 0)
        np.testing.assert_array_almost_equal(folded_0, tensor)

    def test_soft_thresholding(self):
        """Test soft thresholding function."""
        s = np.array([1.0, 0.5, 0.1, 0.05])
        threshold = 0.2
        result = self.advanced_ops._soft_threshold(s, threshold)
        expected = np.array([0.8, 0.3, 0.0, 0.0])  # Each value minus threshold, clamped at 0
        np.testing.assert_array_almost_equal(result, expected)

    def test_higher_order_attention(self):
        """Test higher-order attention mechanisms."""
        query = np.random.randn(768)
        keys = [np.random.randn(768) for _ in range(5)]
        values = [np.random.randn(768) for _ in range(5)]
        
        # Test pairwise attention
        result_2nd_order = self.advanced_ops.compute_higher_order_attention(
            query, keys, values, order=2
        )
        self.assertEqual(result_2nd_order.shape, (64,))  # Projected dimension
        
        # Test triadic attention
        result_3rd_order = self.advanced_ops.compute_higher_order_attention(
            query, keys, values, order=3
        )
        self.assertEqual(result_3rd_order.shape, (64,))  # Projected dimension

    def test_tensor_regularization(self):
        """Test tensor regularization computation."""
        reg_value = self.advanced_ops.compute_tensor_regularization()
        self.assertIsInstance(reg_value, float)
        self.assertGreaterEqual(reg_value, 0.0)

    def test_tensor_complexity_measure(self):
        """Test tensor complexity measure."""
        complexity = self.advanced_ops.compute_tensor_complexity_measure()
        self.assertIsInstance(complexity, float)
        self.assertGreaterEqual(complexity, 0.0)


class TestTensorNetworkAnalyzer(unittest.TestCase):
    """Test the tensor network analyzer."""

    def setUp(self):
        self.advanced_ops = AdvancedTensorOperations(embedding_dim=10, projection_dim=5, k_sparse=3)
        self.analyzer = TensorNetworkAnalyzer(self.advanced_ops)

    def test_analyze_tensor_properties(self):
        """Test tensor property analysis."""
        tensor = np.random.randn(5, 5, 5)
        properties = self.analyzer.analyze_tensor_properties(tensor)
        
        self.assertIn('frobenius_norm', properties)
        self.assertIn('spectral_norm', properties)
        self.assertIn('condition_numbers', properties)
        self.assertIn('entropy', properties)
        
        self.assertIsInstance(properties['frobenius_norm'], float)
        self.assertIsInstance(properties['condition_numbers'], list)
        self.assertIsInstance(properties['entropy'], float)

    def test_tensor_compression_ratio(self):
        """Test tensor compression ratio calculation."""
        tensor = np.random.randn(10, 10, 10)
        ratio = self.analyzer.compute_tensor_compression_ratio(tensor, rank=5)
        
        self.assertIsInstance(ratio, float)
        self.assertGreater(ratio, 0.0)


class TestConfigIntegration(unittest.TestCase):
    """Test configuration integration with EKM system."""

    def test_config_validation(self):
        """Test configuration validation."""
        # Valid config should work
        config = EKMConfig(
            embedding_dim=768,
            projection_dim=64,
            k_sparse=10,
            alpha=0.5,
            beta=0.3,
            gamma=0.1
        )
        self.assertEqual(config.embedding_dim, 768)
        self.assertEqual(config.projection_dim, 64)

        # Invalid config should raise error
        with self.assertRaises(ValueError):
            EKMConfig(embedding_dim=-1)  # Negative embedding dim

        with self.assertRaises(ValueError):
            EKMConfig(alpha=1.5)  # Alpha > 1

    def test_config_from_dict(self):
        """Test creating config from dictionary."""
        config_dict = {
            'embedding_dim': 512,
            'projection_dim': 32,
            'k_sparse': 8,
            'alpha': 0.4,
            'beta': 0.3,
            'gamma': 0.2
        }
        config = EKMConfig.from_dict(config_dict)
        
        self.assertEqual(config.embedding_dim, 512)
        self.assertEqual(config.projection_dim, 32)
        self.assertEqual(config.alpha, 0.4)

    def test_config_serialization(self):
        """Test config serialization."""
        config = EKMConfig(
            embedding_dim=128,
            projection_dim=16,
            k_sparse=5
        )
        
        config_dict = config.to_dict()
        self.assertEqual(config_dict['embedding_dim'], 128)
        self.assertEqual(config_dict['projection_dim'], 16)


class TestEnhancedEKMSystem(unittest.TestCase):
    """Test the enhanced EKM system with all improvements."""

    def setUp(self):
        # Create a minimal config for testing (without external services)
        self.config = EKMConfig(
            embedding_dim=128,  # Smaller for faster tests
            projection_dim=16,
            k_sparse=5,
            mesh_threshold=5,  # Lower threshold for testing
            candidate_size=10,
            enable_higher_order_terms=True,
            use_scalable_index=True
        )

    def test_ekm_with_config(self):
        """Test EKM initialization with configuration."""
        # Note: This test would require mocking external services in a real scenario
        # For now, we'll test the parts that don't require external services
        ekm = EKM(
            d=self.config.embedding_dim,
            k=self.config.k_sparse,
            mesh_threshold=self.config.mesh_threshold,
            embedding_dim=self.config.embedding_dim,
            projection_dim=self.config.projection_dim,
            use_scalable_index=self.config.use_scalable_index
        )
        
        self.assertEqual(ekm.d, self.config.embedding_dim)
        self.assertEqual(ekm.k, self.config.k_sparse)

    def test_tensor_operations_enhancements(self):
        """Test enhanced tensor operations."""
        tensor_ops = TensorOperations(
            embedding_dim=128,
            projection_dim=16,
            k_sparse=5,
            higher_order_terms=True
        )
        
        # Test that higher-order components are initialized
        self.assertTrue(hasattr(tensor_ops, 'third_order_tensor'))
        self.assertTrue(hasattr(tensor_ops, 'fourth_order_tensor'))
        self.assertEqual(tensor_ops.third_order_tensor.shape, (16, 16, 16))
        self.assertEqual(tensor_ops.fourth_order_tensor.shape, (16, 16, 16, 16))

    def test_tensor_norm_regularization(self):
        """Test tensor norm regularization."""
        tensor_ops = TensorOperations(
            embedding_dim=64,
            projection_dim=8,
            k_sparse=3,
            higher_order_terms=True
        )
        
        reg_value = tensor_ops.compute_tensor_norm_regularization()
        self.assertIsInstance(reg_value, float)
        self.assertGreaterEqual(reg_value, 0.0)


class TestMathematicalCorrectness(unittest.TestCase):
    """Test mathematical correctness of tensor operations."""

    def test_orthogonal_projection(self):
        """Test that projection matrices maintain orthogonality."""
        tensor_ops = TensorOperations(
            embedding_dim=64,
            projection_dim=16,
            k_sparse=3
        )
        
        # Test that psi_matrix @ psi_matrix.T is approximately identity
        product = tensor_ops.psi_matrix @ tensor_ops.psi_matrix.T
        identity = np.eye(64)
        
        # Since psi_matrix is (64, 16), psi @ psi.T gives (64, 64) with rank 16
        # So we test that psi.T @ psi is identity (16, 16)
        product_small = tensor_ops.psi_matrix.T @ tensor_ops.psi_matrix
        expected_identity = np.eye(16)
        
        np.testing.assert_array_almost_equal(product_small, expected_identity, decimal=5)

    def test_tensor_contraction_shapes(self):
        """Test that tensor contractions produce correct shapes."""
        tensor_ops = TensorOperations(
            embedding_dim=32,
            projection_dim=8,
            k_sparse=3,
            higher_order_terms=True
        )
        
        # Create sample embeddings
        emb_i = np.random.randn(32)
        emb_j = np.random.randn(32)
        
        # Compute pattern tensor
        tensor = tensor_ops.compute_pattern_tensor(
            emb_i, emb_j, 
            semantic_similarity=0.8, 
            temporal_weight=0.5,
            alpha=0.5, beta=0.3, gamma=0.1
        )
        
        # Should be (projection_dim, projection_dim)
        self.assertEqual(tensor.shape, (8, 8))

    def test_sparse_pattern_tensors_shape(self):
        """Test that sparse pattern tensors have correct shapes."""
        tensor_ops = TensorOperations(
            embedding_dim=32,
            projection_dim=8,
            k_sparse=3
        )
        
        # Create sample embeddings
        embeddings = np.random.randn(5, 32)
        similarities = np.random.rand(5, 5)
        temporal_weights = np.random.rand(5, 5)
        
        sparse_tensors, indices = tensor_ops.compute_sparse_pattern_tensors(
            embeddings, similarities, temporal_weights
        )
        
        # Should have shape (n_nodes, k_sparse, proj_dim, proj_dim)
        self.assertEqual(sparse_tensors.shape, (5, 3, 8, 8))
        self.assertEqual(indices.shape, (5, 3))


def run_all_tests():
    """Run all tests in the suite."""
    # Create a test suite
    loader = unittest.TestLoader()
    
    # Add tests from all classes
    advanced_tensor_tests = loader.loadTestsFromTestCase(TestAdvancedTensorOperations)
    analyzer_tests = loader.loadTestsFromTestCase(TestTensorNetworkAnalyzer)
    config_tests = loader.loadTestsFromTestCase(TestConfigIntegration)
    enhanced_ekm_tests = loader.loadTestsFromTestCase(TestEnhancedEKMSystem)
    math_correctness_tests = loader.loadTestsFromTestCase(TestMathematicalCorrectness)

    suite = unittest.TestSuite([
        advanced_tensor_tests,
        analyzer_tests,
        config_tests,
        enhanced_ekm_tests,
        math_correctness_tests
    ])

    # Run the tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result


if __name__ == '__main__':
    print("Running comprehensive test suite for enhanced EKM system...")
    result = run_all_tests()

    if result.wasSuccessful():
        print("\n✅ All tests passed! Enhanced EKM system is working correctly.")
    else:
        print(f"\n❌ {len(result.failures)} test(s) failed, {len(result.errors)} error(s) occurred.")
        for failure in result.failures:
            print(f"FAILURE in {failure[0]}: {failure[1]}")
        for error in result.errors:
            print(f"ERROR in {error[0]}: {error[1]}")