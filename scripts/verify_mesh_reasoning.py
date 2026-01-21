import requests
import json
import time

def verify():
    base_url = "http://localhost:8000"
    
    # 1. Login
    print("Logging in...")
    login_res = requests.post(f"{base_url}/auth/login", json={
        "email": "admin@ekm.com",
        "password": "admin123"
    })
    if login_res.status_code != 200:
        print(f"Login failed: {login_res.text}")
        return
    
    token = login_res.json().get("access_token")
    headers = {"Authorization": f"Bearer {token}"}
    
    # 2. Check Health / Mode
    print("\nChecking system state...")
    health_res = requests.get(f"{base_url}/health")
    print(f"Health: {health_res.json()}")
    
    # 3. Trigger Evaluation
    print("\nTriggering evaluation in Mesh Mode...")
    eval_payload = {
        "borrower": {
            "id": f"b_{int(time.time())}",
            "name": "Verification User",
            "credit_score": 650,
            "income": 45000,
            "employment_years": 3,
            "debt_to_income_ratio": 0.35,
            "address": "456 Test Ave",
            "phone": "555-0199",
            "email": "verify@example.com"
        },
        "application": {
            "id": f"app_{int(time.time())}",
            "borrower_id": f"b_{int(time.time())}",
            "loan_amount": 15000,
            "loan_purpose": "Debt Consolidation",
            "term_months": 36,
            "interest_rate": 0.12
        }
    }
    
    eval_res = requests.post(f"{base_url}/decide", headers=headers, json=eval_payload)
    print(f"Decision Status: {eval_res.status_code}")
    if eval_res.status_code == 200:
        decision_data = eval_res.json()
        print(f"Decision: {decision_data.get('decision')}")
        print(f"Reason: {decision_data.get('reason')}")
        print(f"Similar Cases Count: {len(decision_data.get('similar_cases', []))}")
        
        if decision_data.get('decision') == "requires_human_decision":
            print("FAILED: System still in Cold Start mode.")
        else:
            print("SUCCESS: System evaluated using AI (likely Mesh Mode).")
    else:
        print(f"Evaluation failed: {eval_res.text}")

if __name__ == "__main__":
    verify()
