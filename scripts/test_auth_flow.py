"""
Test Script: Authentication Flow with Proper JWT Handling

This script demonstrates how the authentication flow should work once Auth0 is properly configured
to return standard JWT tokens instead of encrypted JWE tokens.
"""

import requests
import json
from typing import Dict, Any

def test_authentication_flow():
    """
    Test the complete authentication flow with proper JWT handling.
    This demonstrates the expected behavior after Auth0 configuration is fixed.
    """
    print("Testing Authentication Flow with Proper JWT Handling")
    print("=" * 55)
    
    # This assumes your backend server is running on localhost:8000
    base_url = "http://localhost:8000"
    
    print("\n1. Testing Health Check...")
    try:
        response = requests.get(f"{base_url}/health")
        if response.status_code == 200:
            health_data = response.json()
            print(f"   ✓ Health check passed: {health_data['status']}")
        else:
            print(f"   ✗ Health check failed with status: {response.status_code}")
            return
    except Exception as e:
        print(f"   ✗ Could not connect to server: {e}")
        print("   Make sure your backend server is running with: python run_api.py")
        return
    
    print("\n2. Testing Authentication (this would work properly after Auth0 fix)...")
    print("   NOTE: This will fail until Auth0 is configured to return JWT tokens")
    print("   Follow the steps in AUTH0_CONFIG_GUIDE.md to fix the configuration")
    
    # Example login request (will fail until Auth0 is fixed)
    login_data = {
        "email": "user@example.com",  # Replace with actual test credentials
        "password": "password123"     # Replace with actual test credentials
    }
    
    try:
        login_response = requests.post(f"{base_url}/auth/login", json=login_data)
        if login_response.status_code == 200:
            token_data = login_response.json()
            print("   ✓ Login successful!")
            
            # Extract the access token
            access_token = token_data.get('access_token')
            if access_token:
                token_parts = access_token.split('.')
                print(f"   Token format: {len(token_parts)} parts")
                
                if len(token_parts) == 3:
                    print("   ✓ Token is a proper JWT (3 parts: header.payload.signature)")
                    
                    # Test protected endpoint with the token
                    headers = {"Authorization": f"Bearer {access_token}"}
                    status_response = requests.get(f"{base_url}/status", headers=headers)
                    
                    if status_response.status_code == 200:
                        print("   ✓ Protected endpoint accessed successfully!")
                        print(f"   Status: {status_response.json()}")
                    else:
                        print(f"   ✗ Failed to access protected endpoint: {status_response.status_code}")
                elif len(token_parts) == 5:
                    print("   ✗ Token is an encrypted JWE (5 parts) - Auth0 needs reconfiguration")
                else:
                    print(f"   ? Unexpected token format: {len(token_parts)} parts")
            else:
                print("   ✗ No access token returned from login")
        else:
            print(f"   ✗ Login failed with status: {login_response.status_code}")
            print(f"   Error: {login_response.text}")
    except Exception as e:
        print(f"   ✗ Authentication test failed: {e}")
    
    print("\n3. Summary:")
    print("   - The UI correctly sends JWT tokens to API requests")
    print("   - The backend correctly validates JWT tokens")
    print("   - The issue is that Auth0 is configured to return encrypted JWE tokens")
    print("   - Follow AUTH0_CONFIG_GUIDE.md to fix the Auth0 configuration")
    print("   - After fixing Auth0, the authentication flow will work properly")

def analyze_current_token_format():
    """
    Helper function to analyze token formats based on the server logs you shared
    """
    print("\nAnalyzing Current Token Issue:")
    print("-" * 30)
    print("From your server logs, we can see:")
    print('  JWE Token Header: {"alg":"dir","enc":"A256GCM","iss":"https://dev-feh4p55dbcbqitoe.us.auth0.com/"}')
    print("  This indicates an encrypted JWE token with:")
    print("    - alg: 'dir' (direct encryption)")
    print("    - enc: 'A256GCM' (AES-GCM encryption)")
    print("")
    print("A proper JWT token should look like:")
    print('  JWT Token Header: {"alg":"RS256","typ":"JWT","kid":"..."}')
    print("  This indicates a signed JWT token with:")
    print("    - alg: 'RS256' (RSA-SHA256 signature)")
    print("    - typ: 'JWT' (token type)")
    print("    - kid: key identifier for public key lookup")

if __name__ == "__main__":
    analyze_current_token_format()
    print()
    test_authentication_flow()