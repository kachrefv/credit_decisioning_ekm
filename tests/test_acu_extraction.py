"""
Tests for the enhanced ACU extraction system.

Validates:
1. Unique ACU embeddings (fixes Embedding Identity Problem)
2. AI-First Extraction integration
3. Post-Decision Feedback Loop
"""
import pytest
import numpy as np
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.ekm.domain.credit.models import BorrowerProfile, LoanApplication, CreditDecision
from src.ekm.domain.credit.memory import CreditDecisionMemory
from src.ekm.domain.credit.embedding_generator import (
    TextTemplateEmbeddingGenerator, 
    HybridEmbeddingGenerator
)


class TestACUEmbeddingGenerator:
    """Tests for the embedding generator that fixes the Embedding Identity Problem."""
    
    def test_text_template_generator_produces_unique_embeddings(self):
        """Different risk factors should produce different embeddings."""
        generator = TextTemplateEmbeddingGenerator(embedding_dim=768)
        
        emb1 = generator.generate("low_credit_score", "high", {})
        emb2 = generator.generate("high_debt_to_income", "high", {})
        emb3 = generator.generate("low_credit_score", "high", {})  # Same as emb1
        
        # Different risk factors should have different embeddings
        assert not np.allclose(emb1, emb2, atol=0.01), "Different risk factors should have different embeddings"
        
        # Same risk factor should produce identical embedding (deterministic)
        assert np.allclose(emb1, emb3), "Same risk factor should produce identical embedding"
    
    def test_embedding_dimension_is_correct(self):
        """Embeddings should have the correct dimension."""
        generator = TextTemplateEmbeddingGenerator(embedding_dim=768)
        emb = generator.generate("test_factor", "medium", {})
        
        assert emb.shape == (768,), f"Expected shape (768,), got {emb.shape}"
        assert emb.dtype == np.float32, f"Expected dtype float32, got {emb.dtype}"
    
    def test_hybrid_generator_blends_embeddings(self):
        """Hybrid generator should blend base and risk-specific embeddings."""
        generator = HybridEmbeddingGenerator(blend_ratio=0.5, embedding_dim=768)
        
        base_embedding = np.ones(768, dtype=np.float32)
        metadata = {"base_embedding": base_embedding.tolist()}
        
        emb = generator.generate("test_factor", "medium", metadata)
        
        # Should not be identical to base (blended with risk-specific)
        assert not np.allclose(emb, base_embedding / np.linalg.norm(base_embedding), atol=0.01)
        assert emb.shape == (768,)


class TestCreditDecisionMemoryExtraction:
    """Tests for the refactored CreditDecisionMemory ACU extraction."""
    
    @pytest.fixture
    def memory(self):
        """Create a memory instance with AI extraction disabled for sync tests."""
        return CreditDecisionMemory(
            d=768, 
            mesh_threshold=1000, 
            deepseek_api_key=None,  # Disable AI
            enable_ai_extraction=False
        )
    
    @pytest.fixture
    def sample_borrower(self):
        return BorrowerProfile(
            id="B-TEST001",
            name="Test Borrower",
            credit_score=580,
            income=50000.0,
            employment_years=3.0,
            debt_to_income_ratio=0.45,
            address="123 Test St",
            phone="555-1234",
            email="test@example.com",
            embedding=np.random.randn(768).astype(np.float32)
        )
    
    @pytest.fixture
    def sample_application(self, sample_borrower):
        return LoanApplication(
            id="A-TEST001",
            borrower_id=sample_borrower.id,
            loan_amount=25000.0,
            loan_purpose="Business expansion",
            term_months=36,
            interest_rate=8.5,
            embedding=np.random.randn(768).astype(np.float32)
        )
    
    def test_unique_acu_embeddings_from_same_application(self, memory, sample_borrower, sample_application):
        """
        Risk factors from the same application should have DIFFERENT embeddings.
        This is the core fix for the Embedding Identity Problem.
        """
        # Prepare memory with borrower
        memory.borrowers = [sample_borrower]
        
        # Extract risk factors (will generate 2: credit score + DTI)
        risk_factors = memory._extract_risk_factors_from_application(sample_application)
        
        assert len(risk_factors) >= 2, "Should extract at least 2 risk factors"
        
        # Get embeddings
        emb1 = risk_factors[0].embedding
        emb2 = risk_factors[1].embedding
        
        # They should NOT be identical
        assert not np.allclose(emb1, emb2, atol=0.01), \
            "Risk factors from same application should have DIFFERENT embeddings"
        
        # Verify metadata source
        for rf in risk_factors:
            assert rf.metadata.get("source") == "heuristic_extraction"
    
    def test_sync_ingestion_works(self, memory, sample_borrower, sample_application):
        """Synchronous ingestion should work for backward compatibility."""
        memory.ingest_credit_data_sync([sample_borrower], [sample_application], [])
        
        assert len(memory.risk_factors) >= 2
        assert len(memory.borrowers) == 1
        assert len(memory.applications) == 1


class TestAIExtraction:
    """Tests for AI-First extraction (mocked)."""
    
    @pytest.fixture
    def memory_with_mock_ai(self):
        """Create memory with mocked DeepSeek agent."""
        memory = CreditDecisionMemory(
            d=768,
            mesh_threshold=1000,
            deepseek_api_key=None,
            enable_ai_extraction=True
        )
        
        # Create mock agent
        mock_agent = MagicMock()
        mock_agent.extract_risk_factors = AsyncMock(return_value=[
            {"risk_factor": "volatile_industry_sector", "risk_level": "medium", "reasoning": "Test"},
            {"risk_factor": "seasonal_income_pattern", "risk_level": "low", "reasoning": "Test"}
        ])
        mock_agent.parse_decision_reason = AsyncMock(return_value=[
            {"risk_factor": "insufficient_collateral", "risk_level": "high"}
        ])
        
        memory.deepseek_agent = mock_agent
        return memory
    
    @pytest.mark.asyncio
    async def test_ai_extraction_creates_new_acus(self, memory_with_mock_ai):
        """AI extraction should create properly formatted ACUs."""
        borrower = BorrowerProfile(
            id="B-AI001", name="AI Test", credit_score=700, income=75000.0,
            employment_years=5.0, debt_to_income_ratio=0.30,
            address="123 AI St", phone="555-0000", email="ai@test.com",
            embedding=np.random.randn(768).astype(np.float32)
        )
        
        application = LoanApplication(
            id="A-AI001", borrower_id=borrower.id, loan_amount=50000.0,
            loan_purpose="Restaurant expansion", term_months=48, interest_rate=7.5,
            embedding=np.random.randn(768).astype(np.float32)
        )
        
        ai_factors = await memory_with_mock_ai._extract_ai_risk_factors(borrower, application)
        
        assert len(ai_factors) == 2
        assert ai_factors[0].risk_factor == "volatile_industry_sector"
        assert ai_factors[0].metadata.get("source") == "ai_extraction"
        assert ai_factors[0].embedding is not None
    
    @pytest.mark.asyncio
    async def test_decision_insight_ingestion(self, memory_with_mock_ai):
        """Post-decision feedback loop should create ACUs from decision reasoning."""
        decision = CreditDecision(
            id="D-INSIGHT001",
            application_id="A-001",
            borrower_id="B-001",
            decision="rejected",
            risk_score=0.8,
            confidence=0.9,
            reason="The borrower has insufficient collateral coverage for the requested amount.",
            similar_cases=[]
        )
        
        new_factors = await memory_with_mock_ai.ingest_decision_insight(decision)
        
        assert len(new_factors) == 1
        assert new_factors[0].risk_factor == "insufficient_collateral"
        assert new_factors[0].metadata.get("source") == "decision_insight"
        assert new_factors[0].id in memory_with_mock_ai.risk_factors


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
