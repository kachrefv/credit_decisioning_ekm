import requests
import json
import time

def test_hitl_flow():
    base_url = "http://localhost:8000"
    
    # 1. Login
    print("Logging in...")
    login_data = {"email": "admin@ekm.com", "password": "admin123"}
    login_res = requests.post(f"{base_url}/auth/login", json=login_data)
    token = login_res.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    # 2. Check current status
    status_res = requests.get(f"{base_url}/status", headers=headers)
    print(f"Initial Status: {status_res.json()}")

    # 3. Submit a decision (should be blocked in Cold Start)
    print("\nSubmitting /decide request...")
    decide_data = {
        "borrower": {
            "id": "b_hitl_1",
            "name": "HITL User",
            "credit_score": 750,
            "income": 100000,
            "employment_years": 5,
            "debt_to_income_ratio": 0.2,
            "address": "123 HITL St",
            "phone": "555-HITL",
            "email": "hitl@example.com"
        },
        "application": {
            "id": "a_hitl_1",
            "borrower_id": "b_hitl_1",
            "loan_amount": 10000,
            "loan_purpose": "Verification",
            "term_months": 12,
            "interest_rate": 0.05
        }
    }
    
    decide_res = requests.post(f"{base_url}/decide", json=decide_data, headers=headers)
    print(f"Decide Status Code: {decide_res.status_code}")
    decision = decide_res.json()
    print(f"Decision: {decision['decision']}")
    print(f"Reason: {decision['reason']}")

    if decision['decision'] == "requires_human_decision":
        print("\nSUCCESS: System correctly blocked auto-eval in Cold Start.")
        
        # 4. Submit human decision
        print("\nSubmitting /decisions/human request...")
        human_data = {
            "application_id": "a_hitl_1",
            "borrower_id": "b_hitl_1",
            "decision": "approved",
            "risk_score": 0.1,
            "reason": "Verified manually for Cold Start test.",
            "metadata": {"test": "hitl"}
        }
        human_res = requests.post(f"{base_url}/decisions/human", json=human_data, headers=headers)
        print(f"Human Decision Response: {human_res.json()['decision']}")
        
        # 5. Check status again (risk factors should have increased)
        status_after = requests.get(f"{base_url}/status/risk-factors", headers=headers)
        print(f"Risk Factors Count: {status_after.json()['analytics']['total_count']}")
    else:
        print("\nFAILURE: System did not block auto-eval.")

if __name__ == "__main__":
    test_hitl_flow()
