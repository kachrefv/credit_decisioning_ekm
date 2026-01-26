"""
Test suite for configuration management in the EKM system.
Tests configuration loading, validation, and integration.
"""
import unittest
import tempfile
import os
import json
import yaml
from src.ekm.core.config import EKMConfig, ConfigManager


class TestEKMConfig(unittest.TestCase):
    """Test the EKM configuration class."""

    def test_default_config(self):
        """Test that default configuration is valid."""
        config = EKMConfig()
        
        # Check default values
        self.assertEqual(config.embedding_dim, 768)
        self.assertEqual(config.projection_dim, 64)
        self.assertEqual(config.k_sparse, 10)
        self.assertEqual(config.alpha, 0.5)
        self.assertEqual(config.beta, 0.3)
        self.assertEqual(config.gamma, 0.1)
        self.assertEqual(config.tau, 86400)
        self.assertEqual(config.mesh_threshold, 1000)
        self.assertEqual(config.candidate_size, 100)
        self.assertEqual(config.attention_temperature, 1.0)
        self.assertEqual(config.enable_higher_order_terms, True)
        self.assertEqual(config.use_scalable_index, True)

    def test_config_validation(self):
        """Test configuration validation."""
        # Valid config should work
        config = EKMConfig(embedding_dim=512, projection_dim=32)
        self.assertEqual(config.embedding_dim, 512)
        self.assertEqual(config.projection_dim, 32)

        # Invalid configs should raise errors
        with self.assertRaises(ValueError):
            EKMConfig(embedding_dim=0)  # Zero embedding dim
        
        with self.assertRaises(ValueError):
            EKMConfig(embedding_dim=-1)  # Negative embedding dim
        
        with self.assertRaises(ValueError):
            EKMConfig(alpha=1.5)  # Alpha > 1
        
        with self.assertRaises(ValueError):
            EKMConfig(alpha=-0.5)  # Alpha < 0
        
        with self.assertRaises(ValueError):
            EKMConfig(beta=2.0)  # Beta > 1
        
        with self.assertRaises(ValueError):
            EKMConfig(tau=0)  # Tau <= 0

    def test_config_from_dict(self):
        """Test creating config from dictionary."""
        config_dict = {
            'embedding_dim': 256,
            'projection_dim': 16,
            'k_sparse': 5,
            'alpha': 0.6,
            'beta': 0.2,
            'gamma': 0.1,
            'tau': 43200,  # 12 hours in seconds
            'mesh_threshold': 500,
            'candidate_size': 50,
            'enable_higher_order_terms': False,
            'use_scalable_index': False
        }
        
        config = EKMConfig.from_dict(config_dict)
        
        self.assertEqual(config.embedding_dim, 256)
        self.assertEqual(config.projection_dim, 16)
        self.assertEqual(config.k_sparse, 5)
        self.assertEqual(config.alpha, 0.6)
        self.assertEqual(config.beta, 0.2)
        self.assertEqual(config.gamma, 0.1)
        self.assertEqual(config.tau, 43200)
        self.assertEqual(config.mesh_threshold, 500)
        self.assertEqual(config.candidate_size, 50)
        self.assertEqual(config.enable_higher_order_terms, False)
        self.assertEqual(config.use_scalable_index, False)

    def test_config_to_dict(self):
        """Test converting config to dictionary."""
        config = EKMConfig(
            embedding_dim=128,
            projection_dim=8,
            k_sparse=3,
            alpha=0.7
        )
        
        config_dict = config.to_dict()
        
        self.assertEqual(config_dict['embedding_dim'], 128)
        self.assertEqual(config_dict['projection_dim'], 8)
        self.assertEqual(config_dict['k_sparse'], 3)
        self.assertEqual(config_dict['alpha'], 0.7)

    def test_config_serialization_yaml(self):
        """Test config serialization to YAML."""
        config = EKMConfig(
            embedding_dim=128,
            projection_dim=8,
            k_sparse=3
        )
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            temp_path = f.name
        
        try:
            config.to_file(temp_path, format='yaml')
            
            # Load the config back
            loaded_config = EKMConfig.from_file(temp_path)
            
            self.assertEqual(loaded_config.embedding_dim, 128)
            self.assertEqual(loaded_config.projection_dim, 8)
            self.assertEqual(loaded_config.k_sparse, 3)
        finally:
            os.unlink(temp_path)

    def test_config_serialization_json(self):
        """Test config serialization to JSON."""
        config = EKMConfig(
            embedding_dim=256,
            projection_dim=16,
            k_sparse=7
        )
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name
        
        try:
            config.to_file(temp_path, format='json')
            
            # Load the config back
            loaded_config = EKMConfig.from_file(temp_path)
            
            self.assertEqual(loaded_config.embedding_dim, 256)
            self.assertEqual(loaded_config.projection_dim, 16)
            self.assertEqual(loaded_config.k_sparse, 7)
        finally:
            os.unlink(temp_path)


class TestConfigManager(unittest.TestCase):
    """Test the configuration manager."""

    def setUp(self):
        # Clear any environment variables that might affect tests
        self.original_env = {}
        config_vars = [
            'EMBEDDING_DIM', 'PROJECTION_DIM', 'K_SPARSE', 'ALPHA', 'BETA',
            'GAMMA', 'TAU', 'MESH_THRESHOLD', 'CANDIDATE_SIZE', 'ATTENTION_TEMPERATURE',
            'QDRANT_URL', 'QDRANT_API_KEY', 'DEEPSEEK_API_KEY', 'ENABLE_HIGHER_ORDER_TERMS',
            'TENSOR_REGULARIZATION', 'USE_SCALABLE_INDEX', 'BATCH_SIZE', 'CACHE_ENABLED',
            'CACHE_SIZE', 'SIMILARITY_THRESHOLD', 'MIN_CLUSTER_SIZE', 'MAX_CLUSTER_SIZE'
        ]
        
        for var in config_vars:
            if var in os.environ:
                self.original_env[var] = os.environ[var]
                del os.environ[var]

    def tearDown(self):
        # Restore original environment variables
        for var, value in self.original_env.items():
            os.environ[var] = value

    def test_load_from_defaults(self):
        """Test loading config from defaults."""
        manager = ConfigManager()
        config = manager.load_config()
        
        # Should have default values
        self.assertEqual(config.embedding_dim, 768)
        self.assertEqual(config.projection_dim, 64)

    def test_override_with_env_vars(self):
        """Test overriding config with environment variables."""
        # Set environment variables
        os.environ['EMBEDDING_DIM'] = '512'
        os.environ['PROJECTION_DIM'] = '32'
        os.environ['K_SPARSE'] = '8'
        os.environ['ALPHA'] = '0.6'
        os.environ['BETA'] = '0.25'
        os.environ['ENABLE_HIGHER_ORDER_TERMS'] = 'false'
        os.environ['USE_SCALABLE_INDEX'] = 'false'
        
        manager = ConfigManager()
        config = manager.load_config()
        
        self.assertEqual(config.embedding_dim, 512)
        self.assertEqual(config.projection_dim, 32)
        self.assertEqual(config.k_sparse, 8)
        self.assertEqual(config.alpha, 0.6)
        self.assertEqual(config.beta, 0.25)
        self.assertEqual(config.enable_higher_order_terms, False)
        self.assertEqual(config.use_scalable_index, False)

    def test_convert_types_from_env(self):
        """Test type conversion from environment variables."""
        manager = ConfigManager()
        
        # Test integer conversion
        result = manager._convert_type('42', int)
        self.assertEqual(result, 42)
        
        # Test float conversion
        result = manager._convert_type('3.14', float)
        self.assertEqual(result, 3.14)
        
        # Test boolean conversion (true variants)
        for true_val in ['true', 'True', 'TRUE', '1', 'yes', 'Yes', 'YES', 'on', 'On', 'ON']:
            result = manager._convert_type(true_val, bool)
            self.assertTrue(result)
        
        # Test boolean conversion (false variants)
        for false_val in ['false', 'False', 'FALSE', '0', 'no', 'No', 'NO', 'off', 'Off', 'OFF', '']:
            result = manager._convert_type(false_val, bool)
            self.assertFalse(result)
        
        # Test string conversion
        result = manager._convert_type('hello', str)
        self.assertEqual(result, 'hello')


class TestConfigIntegration(unittest.TestCase):
    """Test configuration integration with other components."""

    def test_config_with_tensor_operations(self):
        """Test using config with tensor operations."""
        config = EKMConfig(
            embedding_dim=64,
            projection_dim=16,
            k_sparse=5,
            enable_higher_order_terms=True
        )
        
        from src.ekm.core.tensor_ops import TensorOperations
        
        tensor_ops = TensorOperations(
            embedding_dim=config.embedding_dim,
            projection_dim=config.projection_dim,
            k_sparse=config.k_sparse,
            higher_order_terms=config.enable_higher_order_terms
        )
        
        # Verify the tensor operations were initialized with correct parameters
        self.assertEqual(tensor_ops.embedding_dim, 64)
        self.assertEqual(tensor_ops.projection_dim, 16)
        self.assertEqual(tensor_ops.k_sparse, 5)
        self.assertTrue(hasattr(tensor_ops, 'third_order_tensor'))  # Higher-order terms enabled

    def test_config_with_different_parameters(self):
        """Test config with different parameter sets."""
        # Config 1: High-dimensional, sparse
        config1 = EKMConfig(
            embedding_dim=1024,
            projection_dim=128,
            k_sparse=20,
            alpha=0.7,
            beta=0.2,
            gamma=0.1
        )
        
        self.assertEqual(config1.embedding_dim, 1024)
        self.assertEqual(config1.projection_dim, 128)
        self.assertEqual(config1.k_sparse, 20)
        self.assertEqual(config1.alpha, 0.7)
        self.assertEqual(config1.beta, 0.2)
        self.assertEqual(config1.gamma, 0.1)
        
        # Config 2: Low-dimensional, dense
        config2 = EKMConfig(
            embedding_dim=128,
            projection_dim=8,
            k_sparse=3,
            alpha=0.3,
            beta=0.5,
            gamma=0.2,
            enable_higher_order_terms=False
        )
        
        self.assertEqual(config2.embedding_dim, 128)
        self.assertEqual(config2.projection_dim, 8)
        self.assertEqual(config2.k_sparse, 3)
        self.assertEqual(config2.alpha, 0.3)
        self.assertEqual(config2.beta, 0.5)
        self.assertEqual(config2.gamma, 0.2)
        self.assertFalse(config2.enable_higher_order_terms)


def run_all_tests():
    """Run all configuration tests."""
    # Create a test suite
    loader = unittest.TestLoader()
    
    # Add tests from all classes
    config_tests = loader.loadTestsFromTestCase(TestEKMConfig)
    manager_tests = loader.loadTestsFromTestCase(TestConfigManager)
    integration_tests = loader.loadTestsFromTestCase(TestConfigIntegration)

    suite = unittest.TestSuite([
        config_tests,
        manager_tests,
        integration_tests
    ])

    # Run the tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result


if __name__ == '__main__':
    print("Running configuration management test suite...")
    result = run_all_tests()

    if result.wasSuccessful():
        print("\n✅ All configuration tests passed!")
    else:
        print(f"\n❌ {len(result.failures)} test(s) failed, {len(result.errors)} error(s) occurred.")
        for failure in result.failures:
            print(f"FAILURE in {failure[0]}: {failure[1]}")
        for error in result.errors:
            print(f"ERROR in {error[0]}: {error[1]}")