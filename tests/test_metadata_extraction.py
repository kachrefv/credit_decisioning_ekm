import pytest
import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.ekm.domain.credit.models import BorrowerProfile, LoanApplication
from src.ekm.domain.credit.memory import CreditDecisionMemory

class TestMetadataExtraction:
    """Tests for ACU extraction from borrower metadata."""
    
    @pytest.fixture
    def memory(self):
        return CreditDecisionMemory(
            d=768, 
            mesh_threshold=1000, 
            deepseek_api_key=None,
            enable_ai_extraction=False
        )
    
    def test_metadata_becomes_risk_factors(self, memory):
        """Borrower metadata should be converted into structured risk factors."""
        borrower = BorrowerProfile(
            id="B-META001",
            name="Metadata Test",
            credit_score=700,
            income=50000.0,
            employment_years=3.0,
            debt_to_income_ratio=0.30,
            address="123 Test St",
            phone="555-1234",
            email="test@example.com",
            embedding=np.random.randn(768).astype(np.float32),
            metadata={
                "industry": "volatile tech",
                "residence_status": "renting",
                "timestamp": 123456789, # Should be ignored
                "source": "manual" # Should be ignored
            }
        )
        
        application = LoanApplication(
            id="A-META001",
            borrower_id=borrower.id,
            loan_amount=25000.0,
            loan_purpose="Business",
            term_months=36,
            interest_rate=8.5,
            embedding=np.random.randn(768).astype(np.float32)
        )
        
        memory.borrowers = [borrower]
        factors = memory._extract_risk_factors_from_application(application)
        
        # Should contain standard factors (good score, low DTI) + metadata factors
        factor_names = [f.risk_factor for f in factors]
        
        # Check standard factors
        assert "good_credit_score" in factor_names
        
        # Check metadata factors
        # Keys and values are lowercased and joined with underscore
        assert "industry_volatile_tech" in factor_names
        assert "residence_status_renting" in factor_names
        
        # Check ignored keys
        assert not any("timestamp" in name for name in factor_names)
        assert not any("source" in name for name in factor_names if name == "source_manual")

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
