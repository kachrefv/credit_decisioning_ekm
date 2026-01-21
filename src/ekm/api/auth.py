import os
from typing import Optional, Dict, Any
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session

from ekm.infra.database.config import get_db
from ekm.infra.database.repository import CreditRepository
from ekm.services.auth_service import AuthService

class TokenVerifier:
    """Verifies custom JWT tokens against local database."""
    
    def verify(self, token: str, db: Session) -> Dict[str, Any]:
        payload = AuthService.decode_token(token)
        if not payload:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Could not validate credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        email: str = payload.get("sub")
        if email is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Could not validate credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        repo = CreditRepository(db)
        user = repo.get_user_by_email(email)
        if user is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User not found",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        if not user.is_active:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Inactive user",
                headers={"WWW-Authenticate": "Bearer"},
            )
            
        return {
            "id": user.id,
            "email": user.email,
            "name": user.name,
            "picture": user.picture,
            "is_superuser": user.is_superuser
        }

auth_scheme = HTTPBearer()

def get_current_user(
    token: HTTPAuthorizationCredentials = Depends(auth_scheme),
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """FastAPI dependency to secure routes using custom JWT."""
    verifier = TokenVerifier()
    return verifier.verify(token.credentials, db)

class AuthManager:
    """Manager for login and registration logic."""
    
    @staticmethod
    def login(email, password, db: Session) -> dict:
        repo = CreditRepository(db)
        user = repo.get_user_by_email(email)
        
        if not user or not AuthService.verify_password(password, user.hashed_password):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect email or password",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        access_token = AuthService.create_access_token(data={"sub": user.email})
        return {
            "access_token": access_token,
            "token_type": "bearer",
            "expires_in": int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "1440")) * 60
        }

    @staticmethod
    def register(email, password, name=None, db: Session = None) -> dict:
        repo = CreditRepository(db)
        
        if repo.get_user_by_email(email):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email already registered"
            )
            
        hashed_password = AuthService.get_password_hash(password)
        user_data = {
            "email": email,
            "hashed_password": hashed_password,
            "name": name,
            "is_active": True
        }
        
        user = repo.create_user(user_data)
        return {
            "id": user.id,
            "email": user.email,
            "name": user.name,
            "message": "User created successfully"
        }
