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

        system_prompt = (
            "You are a Senior Credit Underwriter. "
            "Evaluate the loan application based on the provided borrower profile and application details. "
            "Crucially, use the 'Grounding Context' provided below, which contains similar Atomic Credit Units (ACUs) "
            "from historical decisions, to maintain consistency with historical standards. "
            "Return your final evaluation in strict JSON format."
        )

        user_prompt = (
            f"Borrower: {borrower.name}\n"
            f"Credit Score: {borrower.credit_score}\n"
            f"Income: ${borrower.income}\n"
            f"Loan Amount: ${application.loan_amount}\n"
            f"Loan Purpose: {application.loan_purpose}\n"
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
