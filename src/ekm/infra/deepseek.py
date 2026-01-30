import os
import json
from typing import List, Dict, Any, Optional
from openai import AsyncOpenAI
from ..domain.credit.models import BorrowerProfile, LoanApplication, CreditDecision, CreditRiskFactor

class DeepSeekCreditAgent:
    def __init__(self, api_key: Optional[str] = None):
        if api_key is None:
            api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            raise ValueError("DeepSeek API key is required.")
        
        self.client = AsyncOpenAI(api_key=api_key, base_url="https://api.deepseek.com")
        self.model = "deepseek-chat"

    async def train_on_historical_data(self, borrowers: List[BorrowerProfile], applications: List[LoanApplication], decisions: List[CreditDecision]) -> Dict[str, Any]:
        training_data = []
        for i, decision in enumerate(decisions):
            if i < len(applications) and i < len(borrowers):
                app, borrower = applications[i], borrowers[i]
                training_data.append({
                    "borrower": {"credit_score": borrower.credit_score, "income": borrower.income},
                    "application": {"loan_amount": app.loan_amount, "loan_purpose": app.loan_purpose},
                    "decision": {"decision": decision.decision, "reason": decision.reason}
                })
        
        prompt = f"Learn from these historical credit decisions:\n{json.dumps(training_data, indent=2)}"
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "system", "content": "You are intermediate system and need to learn patterns."}, {"role": "user", "content": prompt}],
                temperature=0.1
            )
            return {"success": True, "message": "Learned.", "response": response.choices[0].message.content}
        except Exception as e:
            return {"success": False, "message": str(e)}

    async def make_credit_decision(self, borrower: BorrowerProfile, application: LoanApplication, similar_cases: List[CreditRiskFactor] = []) -> CreditDecision:
        # Format similar cases into a readable context for the AI
        case_context = ""
        if similar_cases:
            case_context = "\n### Grounding Context: Similar Historical Risk Factors (ACUs)\n"
            for i, rf in enumerate(similar_cases):
                case_context += f"{i+1}. Factor: {rf.risk_factor}, Level: {rf.risk_level}\n"

        # Format borrower metadata for inclusion in the prompt
        borrower_metadata_context = self._format_metadata(borrower.metadata)

        system_prompt = (
            "You are a Senior Credit Underwriter. "
            "Evaluate the loan application based on the provided borrower profile and application details. "
            "Crucially, use the 'Grounding Context' provided below, which contains similar Atomic Credit Units (ACUs) "
            "from historical decisions, to maintain consistency with historical standards. "
            "Also consider the borrower metadata context provided. "
            "Return your final evaluation in strict JSON format."
        )

        user_prompt = (
            f"Borrower: {borrower.name}\n"
            f"Credit Score: {borrower.credit_score}\n"
            f"Income: ${borrower.income}\n"
            f"Employment Years: {borrower.employment_years}\n"
            f"Debt-to-Income Ratio: {borrower.debt_to_income_ratio}\n"
            f"Address: {borrower.address}\n"
            f"Phone: {borrower.phone}\n"
            f"Email: {borrower.email}\n"
            f"Borrower Metadata Context:\n{borrower_metadata_context}\n"
            f"Loan Amount: ${application.loan_amount}\n"
            f"Loan Purpose: {application.loan_purpose}\n"
            f"Term: {application.term_months} months\n"
            f"Interest Rate: {application.interest_rate}%\n"
            f"{case_context}\n\n"
            "Return JSON with the following fields: 'decision' (approved/rejected/requires_manual_review), "
            "'risk_score' (0.0 to 1.0), 'confidence' (0.0 to 1.0), 'reason' (detailed explanation)."
        )

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.2
            )
            
            content = response.choices[0].message.content
            # Handle potential markdown blocks
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            
            data = json.loads(content)
            return CreditDecision(
                id=f"ai_{application.id}", 
                application_id=application.id, 
                borrower_id=borrower.id,
                decision=data.get("decision", "requires_manual_review"),
                risk_score=data.get("risk_score", 0.5),
                confidence=data.get("confidence", 0.5),
                reason=data.get("reason", "AI Decision grounded in EKM."),
                similar_cases=[rf.source_application_ids[0] for rf in similar_cases if rf.source_application_ids]
            )
        except Exception as e:
            print(f"DeepSeek Evaluation Error: {e}")
            return CreditDecision(
                id=f"err_{application.id}", 
                application_id=application.id, 
                borrower_id=borrower.id, 
                decision="requires_manual_review", 
                risk_score=0.7, 
                confidence=0.0, 
                reason=f"AI System Error: {str(e)}", 
                similar_cases=[]
            )

    async def extract_risk_factors(self, borrower: BorrowerProfile, application: LoanApplication) -> List[Dict[str, Any]]:
        """
        Use AI to extract nuanced risk factors from unstructured application data.
        
        This enables "AI-First Extraction" by identifying risks that heuristics cannot catch,
        such as industry-specific concerns or unusual loan purposes.
        
        Args:
            borrower: The borrower profile.
            application: The loan application with unstructured fields like loan_purpose.
            
        Returns:
            A list of dicts, each with "risk_factor" (str) and "risk_level" (str).
        """
        system_prompt = (
            "You are an expert Credit Risk Analyst. Your task is to identify potential risk factors "
            "from credit application data. Focus on nuances that simple rules might miss, such as: "
            "industry volatility, unusual loan purposes, income-to-loan mismatches, or employment instability signals."
        )
        
        user_prompt = f"""Analyze the following loan application and identify any notable risk factors.

Borrower Details:
- Name: {borrower.name}
- Credit Score: {borrower.credit_score}
- Annual Income: ${borrower.income:,.2f}
- Employment Years: {borrower.employment_years}
- Debt-to-Income Ratio: {borrower.debt_to_income_ratio:.2%}

Borrower Context / Metadata:
{self._format_metadata(borrower.metadata)}

Loan Application:
- Loan Amount: ${application.loan_amount:,.2f}
- Loan Purpose: {application.loan_purpose}
- Term: {application.term_months} months
- Interest Rate: {application.interest_rate}%

Return a JSON object with a single key "risk_factors" containing a list of objects.
Each object should have:
- "risk_factor": A snake_case identifier (e.g., "high_loan_to_income_ratio", "volatile_industry_sector")
- "risk_level": One of "low", "medium", "high", "critical"
- "reasoning": A brief explanation (one sentence)

Example: {{"risk_factors": [{{"risk_factor": "stated_purpose_ambiguity", "risk_level": "medium", "reasoning": "Loan purpose is vague and could indicate hidden intent."}}]}}
If no specific risks are found beyond standard metrics, return an empty list."""

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.3
            )
            
            content = response.choices[0].message.content
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            
            data = json.loads(content)
            return data.get("risk_factors", [])
        except Exception as e:
            print(f"DeepSeek Risk Extraction Error: {e}")
            return []

    async def parse_decision_reason(self, reason: str) -> List[Dict[str, Any]]:
        """
        Parse an AI decision's reasoning to extract structured risk factors.
        
        This closes the "Reasoning Loop" by feeding insights from decisions
        back into the knowledge mesh as new ACUs.
        
        Args:
            reason: The free-text reasoning from a CreditDecision.
            
        Returns:
            A list of dicts, each with "risk_factor" (str) and "risk_level" (str).
        """
        if not reason or len(reason) < 20:
            return []
            
        system_prompt = (
            "You are a data extraction assistant. Your task is to parse a credit decision "
            "explanation and extract the specific risk factors mentioned."
        )
        
        user_prompt = f"""Extract risk factors from this credit decision reasoning:

"{reason}"

Return a JSON object with a single key "risk_factors" containing a list of objects.
Each object should have:
- "risk_factor": A snake_case identifier representing the risk (e.g., "insufficient_collateral", "short_employment_history")
- "risk_level": One of "low", "medium", "high", "critical"

Only extract explicitly mentioned risks. If no clear risks are mentioned, return an empty list."""

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.1
            )
            
            content = response.choices[0].message.content
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            
            data = json.loads(content)
            return data.get("risk_factors", [])
        except Exception as e:
            print(f"DeepSeek Reason Parsing Error: {e}")
            return []

    def _format_metadata(self, metadata: Dict[str, Any]) -> str:
        """Helper to format metadata dict into a readable string for the prompt."""
        if not metadata:
            return "None provided"
        
        lines = []
        for k, v in metadata.items():
            # Skip likely internal or irrelevant keys
            if k in ["timestamp", "source", "embedding", "id", "reasoning"]:
                continue
            lines.append(f"- {k}: {v}")
            
        return "\n".join(lines) if lines else "None provided"
