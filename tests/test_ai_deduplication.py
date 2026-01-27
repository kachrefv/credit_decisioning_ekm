import pytest
import numpy as np
import sys
import os
from unittest.mock import AsyncMock, MagicMock

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.ekm.domain.credit.models import BorrowerProfile, LoanApplication
from src.ekm.domain.credit.memory import CreditDecisionMemory
from src.ekm.infra.deepseek import DeepSeekCreditAgent

class TestAIDeduplication:
    """Tests for deduplication of AI-extracted ACUs."""
    
    @pytest.mark.asyncio
    async def test_ai_acu_deduplication(self):
        # Setup mocks
        mock_agent = MagicMock(spec=DeepSeekCreditAgent)
        
        # Mocking extract_risk_factors to return the SAME risk for two different calls
        # Note: extract_risk_factors is async
        mock_agent.extract_risk_factors = AsyncMock(return_value=[
            {"risk_factor": "volatile_industry", "risk_level": "medium", "reasoning": "Tech is unstable"}
        ])
        
        memory = CreditDecisionMemory(d=768, enable_ai_extraction=True)
        memory.deepseek_agent = mock_agent  # Inject mock
        
        # Create two different applications
        b1 = BorrowerProfile(id="B1", name="Bob", credit_score=700, income=50000, employment_years=2, debt_to_income_ratio=0.3, address="", phone="", email="")
        a1 = LoanApplication(id="A1", borrower_id="B1", loan_amount=10000, loan_purpose="Startup", term_months=12, interest_rate=5.0)
        
        b2 = BorrowerProfile(id="B2", name="Alice", credit_score=720, income=55000, employment_years=3, debt_to_income_ratio=0.25, address="", phone="", email="")
        a2 = LoanApplication(id="A2", borrower_id="B2", loan_amount=12000, loan_purpose="Crypto", term_months=12, interest_rate=5.0)
        
        # Ingest App 1
        memory.borrowers = [b1]
        memory.applications = [a1]
        # We manually trigger _extract_ai_risk_factors logic via ingest loop simulation 
        # or just call ingest_credit_data if we mock correctly.
        
        await memory.ingest_credit_data([b1, b2], [a1, a2], [])
        
        # Check Risk Factors
        # Should have heuristic ones + AI ones
        # The AI one "volatile_industry" "medium" should be THERE ONCE
        
        target_id = "rf_volatile_industry_medium"
        assert target_id in memory.risk_factors, "AI extracted ACU not found"
        
        acu = memory.risk_factors[target_id]
        
        # KEY CHECK: Source IDs should contain BOTH A1 and A2
        print(f"ACU Source IDs: {acu.source_application_ids}")
        assert "A1" in acu.source_application_ids
        assert "A2" in acu.source_application_ids
        assert len(acu.source_application_ids) == 2, "ACU should link to both applications"
        
        # Ensure we don't have duplicates like rf_ai_A1_0
        for rid in memory.risk_factors.keys():
            assert not rid.startswith("rf_ai_"), f"Found un-consolidated AI factor ID: {rid}"

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
