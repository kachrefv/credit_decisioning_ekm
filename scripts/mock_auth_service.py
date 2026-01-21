"""
Mock Auth0 Service for Testing JWT Configuration

This module provides a mock implementation that simulates proper JWT token handling
to demonstrate how the system would work once Auth0 is configured to return standard JWT tokens.
"""

import os
import json
from datetime import datetime, timedelta
import jwt
from jose import jwk, jwt as jose_jwt
from jose.utils import base64url_decode
from typing import Optional, Dict, Any
from dotenv import load_dotenv

load_dotenv()

class MockAuth0Service:
    """
    A mock service that simulates proper Auth0 JWT token behavior for testing purposes.
    This demonstrates how the system should work when Auth0 is properly configured.
    """
    
    def __init__(self):
        self.auth0_domain = os.getenv("AUTH0_DOMAIN", "dev-feh4p55dbcbqitoe.us.auth0.com")
        self.client_id = os.getenv("AUTH0_CLIENT_ID", "OPKKzYRX7TiB7lnXRB7YBKRUPgwqoSvI")
        self.client_secret = os.getenv("AUTH0_CLIENT_SECRET", "z2z5osUR-Mqb1kQzqJwHm2oMITC1nyscBuZ5A5nKlZPZhSMDJfVQD70C7qS44abN")
        self.audience = os.getenv("AUTH0_API_AUDIENCE", f"https://{self.auth0_domain}")
        
        # In a real scenario, you'd fetch the actual public key from Auth0's JWKS endpoint
        # For this mock, we'll create a simulated key structure
        self.mock_rsa_key = {
            "kty": "RSA",
            "use": "sig",
            "kid": "mock-key-id",
            "n": "some_mock_n_value_here_for_demonstration",
            "e": "AQAB"
        }
    
    def generate_mock_access_token(self, email: str) -> str:
        """
        Generate a mock JWT access token that follows the expected format.
        This simulates what a proper Auth0 JWT token would look like.
        """
        # Define the payload for the access token
        payload = {
            "iss": f"https://{self.auth0_domain}/",
            "sub": f"auth0|mock_user_{hash(email)}",
            "aud": [self.audience],
            "iat": int(datetime.now().timestamp()),
            "exp": int((datetime.now() + timedelta(hours=1)).timestamp()),  # 1 hour expiry
            "scope": "openid profile email",
            "permissions": [],  # Add any required permissions
            "gty": "password",  # Grant type
            "azp": self.client_id,
            "email": email,
            "email_verified": True
        }
        
        # In a real scenario, this would be signed with Auth0's private key
        # For this mock, we'll create a JWT with a dummy signature
        # NOTE: This is for demonstration only - never do this in production
        
        # Encode header and payload
        header = {"alg": "RS256", "typ": "JWT", "kid": "mock-key-id"}
        header_encoded = self._base64_encode(json.dumps(header).encode('utf-8'))
        payload_encoded = self._base64_encode(json.dumps(payload).encode('utf-8'))
        
        # Create a mock signature (in reality, this would be a real RSA signature)
        mock_signature = "mock_signature_for_demonstration_only"
        token = f"{header_encoded}.{payload_encoded}.{self._base64_encode(mock_signature.encode('utf-8'))}"
        
        return token
    
    def _base64_encode(self, data: bytes) -> str:
        """Helper to encode data in base64url format"""
        import base64
        encoded = base64.urlsafe_b64encode(data).decode('utf-8')
        # Remove padding
        return encoded.rstrip('=')
    
    def mock_login(self, email: str, password: str) -> Dict[str, Any]:
        """
        Mock login that returns a proper JWT access token instead of an encrypted JWE token.
        This simulates the correct behavior after Auth0 configuration is fixed.
        """
        # In a real implementation, you'd validate credentials against a user store
        # For this mock, we'll just generate tokens assuming valid credentials
        
        access_token = self.generate_mock_access_token(email)
        
        # Also generate a mock ID token
        id_payload = {
            "iss": f"https://{self.auth0_domain}/",
            "sub": f"auth0|mock_user_{hash(email)}",
            "aud": self.client_id,
            "iat": int(datetime.now().timestamp()),
            "exp": int((datetime.now() + timedelta(hours=1)).timestamp()),
            "email": email,
            "email_verified": True,
            "name": email.split('@')[0],  # Simple name derivation
        }
        
        id_header = {"alg": "RS256", "typ": "JWT", "kid": "mock-key-id"}
        id_header_encoded = self._base64_encode(json.dumps(id_header).encode('utf-8'))
        id_payload_encoded = self._base64_encode(json.dumps(id_payload).encode('utf-8'))
        id_mock_signature = "mock_id_token_signature"
        id_token = f"{id_header_encoded}.{id_payload_encoded}.{self._base64_encode(id_mock_signature.encode('utf-8'))}"
        
        return {
            "access_token": access_token,
            "id_token": id_token,
            "token_type": "Bearer",
            "expires_in": 3600,  # 1 hour
            "scope": "openid profile email"
        }

# Example usage for testing
if __name__ == "__main__":
    mock_auth = MockAuth0Service()
    
    # Simulate a login request
    result = mock_auth.mock_login("test@example.com", "password123")
    
    print("Mock Login Result:")
    print(f"Access Token Parts Count: {len(result['access_token'].split('.'))}")
    print(f"Access Token (first 100 chars): {result['access_token'][:100]}...")
    print(f"ID Token Parts Count: {len(result['id_token'].split('.'))}")
    
    # Verify this is a 3-part JWT (not 5-part JWE)
    access_parts = result['access_token'].split('.')
    if len(access_parts) == 3:
        print("✓ Access token is a proper JWT (3 parts)")
    else:
        print(f"✗ Access token has {len(access_parts)} parts - expected 3 for JWT")