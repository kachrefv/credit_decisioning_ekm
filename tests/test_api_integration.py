import json
import io
from fastapi.testclient import TestClient

def test_health_check(client: TestClient):
    """Verify that the API is up and running."""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "active"

def test_login_flow(client: TestClient):
    """Test the complete authentication flow (Register -> Login)."""
    # 1. Register
    reg_data = {
        "email": "test_qa@example.com",
        "password": "qa_password_123",
        "full_name": "QA Tester"
    }
    response = client.post("/auth/register", json=reg_data)
    # 200 or 400 if already exists is acceptable for this idempotent test
    assert response.status_code in [200, 400]

    # 2. Login
    login_data = {
        "username": "test_qa@example.com",
        "password": "qa_password_123"
    }
    response = client.post("/auth/token", data=login_data)
    assert response.status_code == 200
    token = response.json()
    assert "access_token" in token
    assert token["token_type"] == "bearer"

def test_bulk_training_no_mapping(authenticated_client: TestClient):
    """Test bulk upload failure when file format is wrong."""
    response = authenticated_client.post(
        "/train/bulk",
        files={"file": ("test.txt", b"dummy content", "text/plain")}
    )
    assert response.status_code == 400
    assert "Only CSV files are supported" in response.json()["detail"]

def test_bulk_training_with_mapping(authenticated_client: TestClient):
    """Test the new dynamic CSV mapping feature."""
    csv_content = (
        "full_name,user_email,annual_salary,outcome,justification\n"
        "John Doe,john@test.com,50000,approved,Good credit\n"
    )
    
    mapping = {
        "name": "full_name",
        "email": "user_email",
        "income": "annual_salary",
        "decision": "outcome",
        "reason": "justification"
    }
    
    file_obj = io.BytesIO(csv_content.encode('utf-8'))
    
    response = authenticated_client.post(
        "/train/bulk",
        files={"file": ("test_mapping.csv", file_obj, "text/csv")},
        data={"mapping": json.dumps(mapping)}
    )
    
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert "Processed 1 rows" in data["message"]

def test_credit_decision_flow(authenticated_client: TestClient):
    """Test the decision endpoint with a high-risk profile."""
    payload = {
        "borrower": {
            "name": "Risky Test User",
            "email": "risky@test.com",
            "income": 20000,
            "credit_score": 400,
            "employment_years": 0.5,
            "debt_to_income_ratio": 0.6
        },
        "application": {
            "loan_amount": 50000,
            "loan_purpose": "vacation",
            "term_months": 24
        }
    }
    
    response = authenticated_client.post("/decide", json=payload)
    
    # We expect a success code, even if the decision is 'rejected' or 'error' (graceful degradation)
    assert response.status_code == 200
    decision = response.json()
    
    assert "decision" in decision
    assert "risk_score" in decision
    assert "reason" in decision
    # Verify the structure of the response ID
    assert decision["id"].startswith("ai_") or decision["id"].startswith("err_")
