import numpy as np
from typing import List, Dict, Tuple
from .models import BorrowerProfile, LoanApplication, CreditDecision

def calculate_credit_metrics(borrower: BorrowerProfile, application: LoanApplication) -> Dict[str, float]:
    metrics = {}
    monthly_income = borrower.income / 12
    monthly_loan_payment = calculate_monthly_payment(
        application.loan_amount, 
        application.interest_rate / 100, 
        application.term_months
    )
    
    dti_ratio = monthly_loan_payment / monthly_income if monthly_income > 0 else float('inf')
    metrics['dti_ratio'] = dti_ratio
    
    if application.collateral_value and application.collateral_value > 0:
        ltv_ratio = application.loan_amount / application.collateral_value
        metrics['ltv_ratio'] = ltv_ratio
    else:
        metrics['ltv_ratio'] = float('inf')
    
    pti_ratio = monthly_loan_payment / monthly_income if monthly_income > 0 else float('inf')
    metrics['pti_ratio'] = pti_ratio
    
    risk_score = 0.5 
    
    if borrower.credit_score < 600:
        risk_score += 0.3
    elif borrower.credit_score < 700:
        risk_score += 0.1
    elif borrower.credit_score > 750:
        risk_score -= 0.1
    
    if dti_ratio > 0.43:
        risk_score += 0.2
    elif dti_ratio > 0.36:
        risk_score += 0.1
    
    if application.collateral_value:
        ltv_ratio = metrics.get('ltv_ratio', float('inf'))
        if ltv_ratio > 0.8:
            risk_score += 0.2
        elif ltv_ratio > 0.6:
            risk_score += 0.1
    
    metrics['combined_risk_score'] = max(0.0, min(1.0, risk_score))
    return metrics

def calculate_monthly_payment(loan_amount: float, annual_interest_rate: float, term_months: int) -> float:
    if term_months == 0:
        return loan_amount
    
    monthly_rate = annual_interest_rate / 12
    if monthly_rate == 0:
        return loan_amount / term_months
    
    payment = loan_amount * (monthly_rate * (1 + monthly_rate)**term_months) / ((1 + monthly_rate)**term_months - 1)
    return payment

def assess_borrower_risk_profile(borrower: BorrowerProfile) -> str:
    risk_factors = []
    
    if borrower.credit_score < 580:
        risk_factors.append("very_poor_credit")
    elif borrower.credit_score < 670:
        risk_factors.append("fair_credit")
    elif borrower.credit_score < 740:
        risk_factors.append("good_credit")
    elif borrower.credit_score < 800:
        risk_factors.append("very_good_credit")
    else:
        risk_factors.append("exceptional_credit")
    
    if borrower.debt_to_income_ratio > 0.43:
        risk_factors.append("high_dti")
    elif borrower.debt_to_income_ratio > 0.36:
        risk_factors.append("medium_dti")
    else:
        risk_factors.append("low_dti")
    
    if borrower.employment_years < 1:
        risk_factors.append("unstable_employment")
    elif borrower.employment_years < 2:
        risk_factors.append("recent_employment")
    else:
        risk_factors.append("stable_employment")
    
    if borrower.income < 30000:
        risk_factors.append("low_income")
    elif borrower.income < 75000:
        risk_factors.append("medium_income")
    else:
        risk_factors.append("high_income")
    
    high_risk_count = sum(1 for factor in risk_factors if "very_poor" in factor or "high" in factor or "unstable" in factor)
    medium_risk_count = sum(1 for factor in risk_factors if "fair" in factor or "medium" in factor or "recent" in factor)
    
    if high_risk_count >= 2:
        return "high_risk"
    elif high_risk_count == 1 or medium_risk_count >= 2:
        return "medium_risk"
    else:
        return "low_risk"

def generate_risk_explanation(decision: CreditDecision, borrower: BorrowerProfile, application: LoanApplication) -> str:
    metrics = calculate_credit_metrics(borrower, application)
    explanation_parts = []
    explanation_parts.append(f"The credit decision for {borrower.name} was '{decision.decision}' with a risk score of {decision.risk_score:.2f}.")
    
    if borrower.credit_score < 600:
        explanation_parts.append(f"The borrower's credit score of {borrower.credit_score} indicates poor creditworthiness.")
    elif borrower.credit_score < 700:
        explanation_parts.append(f"The borrower's credit score of {borrower.credit_score} indicates fair creditworthiness.")
    elif borrower.credit_score < 750:
        explanation_parts.append(f"The borrower's credit score of {borrower.credit_score} indicates good creditworthiness.")
    else:
        explanation_parts.append(f"The borrower's credit score of {borrower.credit_score} indicates excellent creditworthiness.")
    
    if metrics['dti_ratio'] > 0.43:
        explanation_parts.append(f"The debt-to-income ratio of {metrics['dti_ratio']:.2f} exceeds recommended limits (>43%).")
    elif metrics['dti_ratio'] > 0.36:
        explanation_parts.append(f"The debt-to-income ratio of {metrics['dti_ratio']:.2f} is elevated (>36%).")
    else:
        explanation_parts.append(f"The debt-to-income ratio of {metrics['dti_ratio']:.2f} is within acceptable limits.")
    
    income_multiple = application.loan_amount / borrower.income
    if income_multiple > 5:
        explanation_parts.append(f"The requested loan amount ({application.loan_amount:,.2f}) is high relative to the borrower's annual income ({borrower.income:,.2f}).")
    else:
        explanation_parts.append(f"The requested loan amount ({application.loan_amount:,.2f}) is reasonable relative to the borrower's annual income ({borrower.income:,.2f}).")
    
    if application.collateral_value:
        ltv = metrics['ltv_ratio']
        if ltv > 0.8:
            explanation_parts.append(f"The loan-to-value ratio of {ltv:.2f} is high, indicating limited equity cushion.")
        else:
            explanation_parts.append(f"The loan-to-value ratio of {ltv:.2f} is acceptable, providing adequate collateral coverage.")
    
    if decision.similar_cases:
        explanation_parts.append(f"This decision is consistent with {len(decision.similar_cases)} similar historical cases.")
    
    return " ".join(explanation_parts)

def detect_fraud_indicators(borrower: BorrowerProfile, application: LoanApplication) -> List[str]:
    indicators = []
    if application.loan_amount > borrower.income * 10:
        indicators.append("loan_amount_significantly_higher_than_income")
    if borrower.employment_years < 0.1:
        indicators.append("very_recent_employment_start")
    if "po box" in borrower.address.lower() or "mail drop" in borrower.address.lower():
        indicators.append("suspicious_address_type")
    if borrower.credit_score > 800 and borrower.income < 30000:
        indicators.append("inconsistent_credit_score_and_income")
    if borrower.credit_score > 750 and borrower.employment_years < 1:
        indicators.append("high_credit_score_with_short_employment")
    return indicators
