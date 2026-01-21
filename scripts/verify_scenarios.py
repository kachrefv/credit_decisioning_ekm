import requests
import json
import time

BASE_URL = "http://localhost:8000"

def get_token():
    res = requests.post(f"{BASE_URL}/auth/login", json={
        "email": "admin@ekm.com",
        "password": "admin123"
    })
    return res.json().get("access_token")

def run_scenarios():
    token = get_token()
    headers = {"Authorization": f"Bearer {token}"}
    
    print("--- SCENARIO 1: Thin File (New Borrower) ---")
    thin_borrower = {
        "id": f"thin_{int(time.time())}",
        "name": "Thin File User",
        "credit_score": 600,
        "income": 30000,
        "employment_years": 0.5,
        "debt_to_income_ratio": 0.45,
        "address": "123 Unknown St",
        "phone": "555-0000",
        "email": f"thin_{int(time.time())}@example.com"
    }
    
    thin_app = {
        "id": f"app_thin_{int(time.time())}",
        "borrower_id": thin_borrower["id"],
        "loan_amount": 50000,
        "loan_purpose": "Speculative Investment",
        "term_months": 12,
        "interest_rate": 0.15
    }
    
    res = requests.post(f"{BASE_URL}/decide", headers=headers, json={
        "borrower": thin_borrower,
        "application": thin_app
    })
    print(f"Result: {res.json().get('decision')} | Reason: {res.json().get('reason')}")

    print("\n--- SCENARIO 2: Policy Shift (Expert Grounding) ---")
    # 1. Submit an application that might be risky
    risky_borrower = {
        "id": f"risky_{int(time.time())}",
        "name": "Policy Shift User",
        "credit_score": 680,
        "income": 70000,
        "employment_years": 5,
        "debt_to_income_ratio": 0.38,
        "address": "456 Policy Blvd",
        "phone": "555-1111",
        "email": f"policy_{int(time.time())}@example.com"
    }
    
    risky_app = {
        "id": f"app_risky_{int(time.time())}",
        "borrower_id": risky_borrower["id"],
        "loan_amount": 20000,
        "loan_purpose": "Business Expansion",
        "term_months": 36,
        "interest_rate": 0.08
    }
    
    res_risky = requests.post(f"{BASE_URL}/decide", headers=headers, json={
        "borrower": risky_borrower,
        "application": risky_app
    })
    print(f"Initial AI Decision: {res_risky.json().get('decision')}")
    
    # 2. Overrule with Expert Notes
    print("Submitting expert overrule...")
    requests.post(f"{BASE_URL}/decisions/human", headers=headers, json={
        "application_id": risky_app["id"],
        "borrower_id": risky_borrower["id"],
        "decision": "approved",
        "risk_score": 0.2,
        "reason": "Expert override for business growth potential.",
        "expert_notes": "Borrower has significant industry experience not captured by DTI alone."
    })
    
    # 3. Submit a very similar application and check if the grounding changed
    print("Testing similar application grounded in expert notes...")
    similar_borrower = risky_borrower.copy()
    similar_borrower["id"] = f"similar_{int(time.time())}"
    similar_borrower["email"] = f"similar_{int(time.time())}@example.com"
    
    similar_app = risky_app.copy()
    similar_app["id"] = f"app_similar_{int(time.time())}"
    similar_app["borrower_id"] = similar_borrower["id"]
    
    res_similar = requests.post(f"{BASE_URL}/decide", headers=headers, json={
        "borrower": similar_borrower,
        "application": similar_app
    })
    print(f"Grounded AI Decision: {res_similar.json().get('decision')} | Confidence: {res_similar.json().get('confidence')}")

    print("\n--- SCENARIO 3: Threshold Breach (Critical Risk) ---")
    breach_borrower = {
        "id": f"breach_{int(time.time())}",
        "name": "High Risk User",
        "credit_score": 450,
        "income": 20000,
        "employment_years": 0.1,
        "debt_to_income_ratio": 0.9,
        "address": "999 Danger Rd",
        "phone": "555-9999",
        "email": f"breach_{int(time.time())}@example.com"
    }
    
    breach_app = {
        "id": f"app_breach_{int(time.time())}",
        "borrower_id": breach_borrower["id"],
        "loan_amount": 100000,
        "loan_purpose": "Vacation",
        "term_months": 12,
        "interest_rate": 0.25
    }
    
    res_breach = requests.post(f"{BASE_URL}/decide", headers=headers, json={
        "borrower": breach_borrower,
        "application": breach_app
    })
    print(f"Result: {res_breach.json().get('decision')} | Reason: {res_breach.json().get('reason')}")

if __name__ == "__main__":
    run_scenarios()
