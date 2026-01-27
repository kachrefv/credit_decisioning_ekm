import pytest
import sys
import os
from unittest.mock import AsyncMock, MagicMock

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.ekm.infra.deepseek import DeepSeekCreditAgent
from src.ekm.domain.credit.models import BorrowerProfile, LoanApplication

class TestAIMetadataPrompt:
    """Tests for metadata injection in DeepSeek prompts."""
    
    @pytest.mark.asyncio
    async def test_metadata_in_prompt(self):
        # Setup agent with mock client
        agent = DeepSeekCreditAgent(api_key="fake-key")
        agent.client = MagicMock()
        agent.client.chat.completions.create = AsyncMock(return_value=MagicMock(
            choices=[MagicMock(message=MagicMock(content='{"risk_factors": []}'))]
        ))
        
        # Borrower with metadata
        borrower = BorrowerProfile(
            id="B1", name="Test User", credit_score=700, income=50000, 
            employment_years=2, debt_to_income_ratio=0.3, address="", phone="", email="",
            metadata={"industry": "volatile_tech", "notes": "high_risk_sector"}
        )
        application = LoanApplication(
            id="A1", borrower_id="B1", loan_amount=10000, 
            loan_purpose="Startup", term_months=12, interest_rate=5.0
        )
        
        # Call extraction
        await agent.extract_risk_factors(borrower, application)
        
        # Verify call arguments
        call_args = agent.client.chat.completions.create.call_args
        messages = call_args.kwargs['messages']
        user_prompt = messages[1]['content']
        
        print("Generated Prompt:\n", user_prompt)
        
        # Assert metadata is present in prompt
        assert "Borrower Context / Metadata:" in user_prompt
        assert "- industry: volatile_tech" in user_prompt
        assert "- notes: high_risk_sector" in user_prompt

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
