"""
Integration tests for all components of the Credithos EKM system.
Tests that all components work together correctly.
"""
import unittest
import numpy as np
import time
import sys
import os
from typing import List, Dict, Any
import tempfile
import shutil

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.ekm.core.models import Episode, AKU, GKU
from src.ekm.core.engine import EKM
from src.ekm.core.tensor_ops import TensorOperations, compute_semantic_similarity, compute_temporal_weight
from src.ekm.core.graph import GraphEngine
from src.ekm.core.retrieval import EKMRetriever
from src.ekm.core.consolidation import ConsolidationEngine
from src.ekm.core.efficient_indexing import EfficientIndexer, ScalableEKM
from src.ekm.domain.credit.memory import CreditDecisionMemory
from src.ekm.domain.credit.models import BorrowerProfile, LoanApplication, CreditDecision
from src.ekm.api.app import to_borrower_model, to_application_model


class TestComponentIntegration(unittest.TestCase):
    """Integration tests for core components."""
    
    def setUp(self):
        """Set up test data for integration tests."""
        # Create synthetic episodes
        self.episodes = []
        for i in range(20):
            embedding = np.random.randn(768).astype(np.float32)
            episode = Episode(
                id=f"ep_{i}",
                content=f"Content for episode {i} with some meaningful information",
                embedding=embedding,
                metadata={"timestamp": time.time() - i*3600}  # Different timestamps
            )
            self.episodes.append(episode)
        
        # Create corresponding AKUs
        self.akus = []
        for i, episode in enumerate(self.episodes):
            aku = AKU(
                id=f"aku_{i}",
                proposition=f"Proposition based on episode {i}",
                source_episode_ids=[episode.id],
                embedding=episode.embedding,
                metadata=episode.metadata
            )
            self.akus.append(aku)
    
    def test_tensor_ops_graph_integration(self):
        """Test integration between tensor operations and graph engine."""
        # Create tensor operations
        tensor_ops = TensorOperations(embedding_dim=768, projection_dim=64, k_sparse=5)
        
        # Create graph engine with tensor ops
        graph_engine = GraphEngine(k=5, embedding_dim=768, projection_dim=64)
        
        # Build graph using AKUs
        graph_engine.build_knn_graph(self.akus)
        
        # Verify that pattern tensors are properly computed and stored
        edges_with_tensors = 0
        for u, v, data in graph_engine.graph.edges(data=True):
            if 'pattern_tensor' in data:
                edges_with_tensors += 1
                # Verify tensor shape
                self.assertEqual(data['pattern_tensor'].shape, (64, 64))
        
        # At least some edges should have pattern tensors
        self.assertGreater(edges_with_tensors, 0)
        
        # Extract signatures using the graph
        graph_engine.extract_signatures(self.akus)
        
        # Verify that all AKUs now have structural signatures
        for aku in self.akus:
            self.assertIsNotNone(aku.structural_signature)
            # Signature should include topological features (13) + semantic features (768)
            expected_length = 13 + 768
            self.assertEqual(len(aku.structural_signature), expected_length)
    
    def test_graph_retrieval_integration(self):
        """Test integration between graph engine and retrieval system."""
        # Create graph engine and build graph
        graph_engine = GraphEngine(k=5, embedding_dim=768, projection_dim=64)
        graph_engine.build_knn_graph(self.akus)
        graph_engine.extract_signatures(self.akus)
        
        # Create retrieval system
        retriever = EKMRetriever(d=768, projection_dim=64)
        retriever.build_index(self.akus)
        
        # Test retrieval with graph engine
        query_embedding = np.random.randn(768).astype(np.float32)
        results = retriever.retrieve(query_embedding, graph_engine)
        
        # Verify results format
        self.assertIsInstance(results, list)
        self.assertLessEqual(len(results), len(self.akus))
        
        if results:
            aku, score = results[0]
            self.assertIsInstance(aku, AKU)
            self.assertIsInstance(score, float)
            
            # Verify that the retrieved AKU has a structural signature
            self.assertIsNotNone(aku.structural_signature)
    
    def test_full_ekm_workflow_integration(self):
        """Test the full EKM workflow integration."""
        # Create EKM instance
        ekm = EKM(d=768, k=5, mesh_threshold=10, use_scalable_index=True)
        
        # Ingest episodes
        ekm.ingest_episodes(self.episodes)
        
        # Verify mode transition
        self.assertEqual(ekm.mode, "Mesh Mode")
        
        # Verify AKUs were created
        self.assertEqual(len(ekm.akus), len(self.episodes))
        
        # Test retrieval
        query_embedding = np.random.randn(768).astype(np.float32)
        results = ekm.retrieve("integration test query", query_embedding)
        
        self.assertIsInstance(results, list)
        if results:
            aku, score = results[0]
            self.assertIsInstance(aku, AKU)
            self.assertIsInstance(score, float)
        
        # Test consolidation
        original_count = len(ekm.akus)
        ekm.consolidate()
        after_consolidation_count = len(ekm.akus)
        
        # Consolidation should have run
        self.assertGreaterEqual(ekm.stats['total_consolidations'], 1)
        
        # Get performance stats
        stats = ekm.get_performance_stats()
        self.assertGreaterEqual(stats['total_ingested'], len(self.episodes))
        self.assertGreaterEqual(stats['total_retrievals'], 1 if results else 0)
        self.assertGreaterEqual(stats['total_consolidations'], 1)
    
    def test_scalable_ekm_integration(self):
        """Test integration with scalable EKM."""
        # Create scalable EKM
        scalable_ekm = ScalableEKM(embedding_dim=768, projection_dim=64, k_sparse=5)
        
        # Ingest AKUs
        scalable_ekm.ingest_akus(self.akus)
        
        # Build index
        scalable_ekm.build_index()
        
        # Test retrieval
        query_embedding = np.random.randn(768).astype(np.float32)
        results = scalable_ekm.retrieve(query_embedding, k=5)
        
        self.assertIsInstance(results, list)
        self.assertLessEqual(len(results), 5)
        
        if results:
            aku, score = results[0]
            self.assertIsInstance(aku, AKU)
            self.assertIsInstance(score, float)
        
        # Verify stats are updated
        stats = scalable_ekm.get_stats()
        self.assertGreaterEqual(stats['total_ingested'], len(self.akus))
        self.assertGreaterEqual(stats['index_build_count'], 1)


class TestDomainIntegration(unittest.TestCase):
    """Integration tests for domain-specific components."""
    
    def setUp(self):
        """Set up test data for domain integration tests."""
        # Create synthetic credit data
        self.borrowers = []
        for i in range(10):
            borrower = BorrowerProfile(
                id=f"borrower_{i}",
                name=f"Borrower {i}",
                credit_score=np.random.randint(300, 850),
                income=float(np.random.uniform(30000, 200000)),
                employment_years=float(np.random.uniform(0.5, 30)),
                debt_to_income_ratio=float(np.random.uniform(0.1, 0.6)),
                address=f"Address {i}",
                phone=f"Phone {i}",
                email=f"borrower{i}@example.com"
            )
            self.borrowers.append(borrower)
        
        self.applications = []
        for i in range(10):
            application = LoanApplication(
                id=f"app_{i}",
                borrower_id=f"borrower_{i}",
                loan_amount=float(np.random.uniform(5000, 500000)),
                loan_purpose=np.random.choice(["personal", "business", "mortgage", "auto"]),
                term_months=np.random.choice([12, 24, 36, 60, 120, 360]),
                interest_rate=float(np.random.uniform(0.03, 0.25))
            )
            self.applications.append(application)
        
        self.decisions = []
        for i in range(10):
            decision = CreditDecision(
                id=f"dec_{i}",
                application_id=f"app_{i}",
                borrower_id=f"borrower_{i}",
                decision=np.random.choice(["approved", "rejected", "requires_manual_review"]),
                risk_score=float(np.random.uniform(0.1, 0.9)),
                confidence=float(np.random.uniform(0.5, 1.0)),
                reason=f"Automated decision for application {i}",
                timestamp=time.time() - i*86400  # Different timestamps
            )
            self.decisions.append(decision)
    
    def test_credit_memory_integration(self):
        """Test integration of credit decision memory with core EKM."""
        # Create credit decision memory
        credit_memory = CreditDecisionMemory(
            d=768, k=5, mesh_threshold=5, deepseek_api_key=None  # No API key for testing
        )
        
        # Ingest credit data
        credit_memory.ingest_credit_data(self.borrowers, self.applications, self.decisions)
        
        # Verify data was ingested
        self.assertEqual(len(credit_memory.borrowers), len(self.borrowers))
        self.assertEqual(len(credit_memory.applications), len(self.applications))
        self.assertEqual(len(credit_memory.decisions), len(self.decisions))
        
        # Verify risk factors were extracted
        self.assertGreater(len(credit_memory.risk_factors), 0)
        
        # Test evaluation (should return manual review since no DeepSeek API)
        if self.borrowers and self.applications:
            decision = credit_memory.evaluate_credit_application(self.borrowers[0], self.applications[0])
            self.assertIsInstance(decision, CreditDecision)
            # Should be manual review since DeepSeek is not configured
            self.assertIn(decision.decision, ["requires_human_decision", "requires_manual_review"])
    
    def test_api_model_conversion_integration(self):
        """Test integration between API model conversion and domain models."""
        # Create a borrower request (simulating API input)
        from src.ekm.api.schemas import BorrowerRequest
        
        borrower_req = BorrowerRequest(
            id="api_borrower_1",
            name="API Borrower",
            credit_score=720,
            income=75000.0,
            employment_years=5.0,
            debt_to_income_ratio=0.3,
            address="123 API St",
            phone="555-0123",
            email="api@example.com",
            metadata={"source": "api_request"}
        )
        
        # Convert to domain model
        borrower_model = to_borrower_model(borrower_req)
        
        # Verify conversion worked
        self.assertIsInstance(borrower_model, BorrowerProfile)
        self.assertEqual(borrower_model.id, "api_borrower_1")
        self.assertEqual(borrower_model.name, "API Borrower")
        self.assertEqual(borrower_model.credit_score, 720)
        self.assertEqual(borrower_model.income, 75000.0)
        
        # Create an application request
        from src.ekm.api.schemas import LoanApplicationRequest
        
        app_req = LoanApplicationRequest(
            id="api_app_1",
            borrower_id="api_borrower_1",
            loan_amount=250000.0,
            loan_purpose="mortgage",
            term_months=360,
            interest_rate=0.045
        )
        
        # Convert to domain model
        app_model = to_application_model(app_req)
        
        # Verify conversion worked
        self.assertIsInstance(app_model, LoanApplication)
        self.assertEqual(app_model.id, "api_app_1")
        self.assertEqual(app_model.borrower_id, "api_borrower_1")
        self.assertEqual(app_model.loan_amount, 250000.0)
        self.assertEqual(app_model.loan_purpose, "mortgage")


class TestEfficientIndexingIntegration(unittest.TestCase):
    """Integration tests for efficient indexing components."""
    
    def setUp(self):
        """Set up test data for efficient indexing tests."""
        self.akus = []
        for i in range(50):  # Larger set for indexing tests
            embedding = np.random.randn(768).astype(np.float32)
            aku = AKU(
                id=f"index_aku_{i}",
                proposition=f"Index test proposition {i}",
                source_episode_ids=[f"ep_{i}"],
                embedding=embedding,
                metadata={"timestamp": time.time() - i*1000}
            )
            self.akus.append(aku)
    
    def test_efficient_indexer_full_workflow(self):
        """Test the full workflow of efficient indexer."""
        # Create indexer
        indexer = EfficientIndexer(embedding_dim=768, projection_dim=64, k_sparse=5)
        
        # Add AKUs in batches
        indexer.add_akus_batch(self.akus)
        
        # Verify AKUs were added
        self.assertEqual(indexer.faiss_index.ntotal, len(self.akus))
        self.assertEqual(len(indexer.akus_by_id), len(self.akus))
        
        # Build sparse relationships
        indexer.build_sparse_relationships()
        
        # Verify relationships were built
        self.assertIsNotNone(indexer.sparse_pattern_tensors)
        self.assertIsNotNone(indexer.connection_indices)
        
        # Verify tensor shapes
        expected_shape = (len(self.akus), 5, 64, 64)  # n_nodes, k_sparse, proj_dim, proj_dim
        self.assertEqual(indexer.sparse_pattern_tensors.shape, expected_shape)
        
        # Test basic search
        query_embedding = np.random.randn(768).astype(np.float32)
        basic_results = indexer.search(query_embedding, k=5)
        
        self.assertIsInstance(basic_results, list)
        self.assertLessEqual(len(basic_results), 5)
        
        if basic_results:
            aku, score = basic_results[0]
            self.assertIsInstance(aku, AKU)
            self.assertIsInstance(score, float)
        
        # Test enhanced search with attention
        enhanced_results = indexer.enhanced_search_with_attention(query_embedding, k=5)
        
        self.assertIsInstance(enhanced_results, list)
        self.assertLessEqual(len(enhanced_results), 5)
        
        if enhanced_results:
            aku, score = enhanced_results[0]
            self.assertIsInstance(aku, AKU)
            self.assertIsInstance(score, float)
    
    def test_index_persistence_integration(self):
        """Test persistence and loading of indexes."""
        # Create indexer and populate
        indexer = EfficientIndexer(embedding_dim=768, projection_dim=64, k_sparse=5)
        indexer.add_akus_batch(self.akus[:25])  # Use half for this test
        indexer.build_sparse_relationships()
        
        # Create temporary directory for testing
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save index
            indexer.save_index(temp_dir)
            
            # Verify files were created
            saved_files = os.listdir(temp_dir)
            self.assertIn("faiss.index", saved_files)
            self.assertIn("akus.pkl", saved_files)
            
            # Create new indexer and load
            new_indexer = EfficientIndexer(embedding_dim=768, projection_dim=64, k_sparse=5)
            new_indexer.load_index(temp_dir)
            
            # Verify loaded data
            self.assertEqual(new_indexer.faiss_index.ntotal, 25)
            self.assertEqual(len(new_indexer.akus_by_id), 25)
            
            # Verify tensor relationships were loaded if they existed
            if indexer.sparse_pattern_tensors is not None:
                self.assertIsNotNone(new_indexer.sparse_pattern_tensors)
                self.assertEqual(
                    new_indexer.sparse_pattern_tensors.shape,
                    indexer.sparse_pattern_tensors.shape
                )
            
            # Test that search still works on loaded index
            query_embedding = np.random.randn(768).astype(np.float32)
            results = new_indexer.search(query_embedding, k=3)
            self.assertIsInstance(results, list)
            self.assertLessEqual(len(results), 3)


class TestEndToEndIntegration(unittest.TestCase):
    """End-to-end integration tests."""
    
    def test_complete_credit_decisioning_workflow(self):
        """Test the complete credit decisioning workflow."""
        # Create credit decision memory
        credit_memory = CreditDecisionMemory(
            d=768, k=5, mesh_threshold=3, deepseek_api_key=None  # No API for testing
        )
        
        # Create test data
        borrower = BorrowerProfile(
            id="test_borrower",
            name="Test Borrower",
            credit_score=750,
            income=80000.0,
            employment_years=8.0,
            debt_to_income_ratio=0.25,
            address="123 Test St",
            phone="555-TEST",
            email="test@example.com"
        )
        
        application = LoanApplication(
            id="test_app",
            borrower_id="test_borrower",
            loan_amount=300000.0,
            loan_purpose="mortgage",
            term_months=360,
            interest_rate=0.04
        )
        
        # Initially, system should be in cold start
        self.assertEqual(credit_memory.mode, "Cold Start")
        
        # Evaluate application (should return human decision in cold start)
        decision = credit_memory.evaluate_credit_application(borrower, application)
        self.assertIsInstance(decision, CreditDecision)
        self.assertIn(decision.decision, ["requires_human_decision", "requires_manual_review"])
        
        # Add some historical data to move to mesh mode
        historical_borrowers = [borrower]  # Same borrower for simplicity
        historical_apps = [application]
        historical_decisions = [
            CreditDecision(
                id="hist_dec_1",
                application_id="test_app",
                borrower_id="test_borrower",
                decision="approved",
                risk_score=0.3,
                confidence=0.9,
                reason="Good credit history"
            )
        ]
        
        # Ingest historical data
        credit_memory.ingest_credit_data(historical_borrowers, historical_apps, historical_decisions)
        
        # Now system should be in mesh mode
        self.assertEqual(credit_memory.mode, "Mesh Mode")
        
        # Evaluate again (still manual review since no DeepSeek)
        decision2 = credit_memory.evaluate_credit_application(borrower, application)
        self.assertIsInstance(decision2, CreditDecision)
        
        # Check risk factor analytics
        analytics = credit_memory.get_risk_factor_analytics()
        self.assertIsInstance(analytics, dict)
        self.assertIn('total_count', analytics)
        self.assertIn('level_distribution', analytics)
    
    def test_tensor_to_graph_to_retrieval_pipeline(self):
        """Test the complete pipeline from tensor operations to graph to retrieval."""
        # Create tensor operations
        tensor_ops = TensorOperations(embedding_dim=768, projection_dim=64, k_sparse=5)
        
        # Create test embeddings
        embeddings = np.random.randn(20, 768).astype(np.float32)
        similarities = np.random.rand(20, 20).astype(np.float32)
        temporal_weights = np.random.rand(20, 20).astype(np.float32)
        
        # Compute sparse pattern tensors
        sparse_tensors, connection_indices = tensor_ops.compute_sparse_pattern_tensors(
            embeddings, similarities, temporal_weights
        )
        
        self.assertEqual(sparse_tensors.shape, (20, 5, 64, 64))
        self.assertEqual(connection_indices.shape, (20, 5))
        
        # Create AKUs with these embeddings
        akus = []
        for i in range(20):
            aku = AKU(
                id=f"pipeline_aku_{i}",
                proposition=f"Pipeline test {i}",
                source_episode_ids=[f"ep_{i}"],
                embedding=embeddings[i],
                metadata={"timestamp": time.time() - i*3600}
            )
            akus.append(aku)
        
        # Create graph using these AKUs
        graph_engine = GraphEngine(k=5, embedding_dim=768, projection_dim=64)
        graph_engine.build_knn_graph(akus)
        graph_engine.extract_signatures(akus)
        
        # Create retriever and index
        retriever = EKMRetriever(d=768, projection_dim=64)
        retriever.build_index(akus)
        
        # Test retrieval with the graph
        query_embedding = np.random.randn(768).astype(np.float32)
        results = retriever.retrieve(query_embedding, graph_engine)
        
        self.assertIsInstance(results, list)
        if results:
            aku, score = results[0]
            self.assertIsInstance(aku, AKU)
            self.assertIsInstance(score, float)
            # Verify the retrieved AKU has a structural signature
            self.assertIsNotNone(aku.structural_signature)


def run_integration_tests():
    """Run all integration tests."""
    print("Running integration tests for Credithos EKM system...")
    
    # Create a test suite
    loader = unittest.TestLoader()
    
    # Collect all integration test cases
    component_tests = loader.loadTestsFromTestCase(TestComponentIntegration)
    domain_tests = loader.loadTestsFromTestCase(TestDomainIntegration)
    indexing_tests = loader.loadTestsFromTestCase(TestEfficientIndexingIntegration)
    end_to_end_tests = loader.loadTestsFromTestCase(TestEndToEndIntegration)
    
    # Create suite with all tests
    suite = unittest.TestSuite([
        component_tests,
        domain_tests,
        indexing_tests,
        end_to_end_tests
    ])
    
    # Run the tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print(f"\n{'='*60}")
    print("INTEGRATION TEST SUITE RESULTS")
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
        print("\n✅ All integration tests passed! Components work together correctly.")
    else:
        print(f"\n❌ Integration test suite completed with issues.")
    
    return result


if __name__ == '__main__':
    run_integration_tests()