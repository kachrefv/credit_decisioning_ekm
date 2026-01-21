import requests
import json

def test_custom_auth():
    base_url = "http://localhost:8000"
    
    # 1. Test registration of a new user
    print("\n--- 1. Testing Registration ---")
    reg_data = {
        "email": "testuser@example.com",
        "password": "testpassword123",
        "name": "Test User"
    }
    try:
        reg_response = requests.post(f"{base_url}/auth/register", json=reg_data)
        print(f"Status: {reg_response.status_code}")
        print(f"Response: {reg_response.json()}")
    except Exception as e:
        print(f"Registration failed: {e}")

    # 2. Test login with admin user
    print("\n--- 2. Testing Login (Admin) ---")
    login_data = {
        "email": "admin@ekm.com",
        "password": "admin123"
    }
    
    try:
        login_response = requests.post(f"{base_url}/auth/login", json=login_data)
        print(f"Status: {login_response.status_code}")
        if login_response.status_code == 200:
            token_data = login_response.json()
            print("Login successful!")
            access_token = token_data.get("access_token")
            
            # 3. Test protected route
            print("\n--- 3. Testing Protected Route (/status) ---")
            headers = {"Authorization": f"Bearer {access_token}"}
            status_response = requests.get(f"{base_url}/status", headers=headers)
            print(f"Status: {status_response.status_code}")
            if status_response.status_code == 200:
                print(f"Route access successful: {status_response.json()}")
            else:
                print(f"Route access failed: {status_response.text}")
        else:
            print(f"Login failed: {login_response.text}")
    except Exception as e:
        print(f"Login test failed: {e}")

if __name__ == "__main__":
    # Ensure server is running
    test_custom_auth()
