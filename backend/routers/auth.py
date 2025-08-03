from fastapi import APIRouter, HTTPException, status, Depends
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from models import UserCreate, UserLogin, Token, User
from auth import create_user, login_user, get_current_user
from database import get_redis
import uuid
from datetime import datetime

router = APIRouter()
security = HTTPBearer()

@router.post("/register", response_model=User)
async def register(user: UserCreate):
    """Register a new user"""
    try:
        new_user = await create_user(user)
        return new_user
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Registration failed: {str(e)}"
        )

@router.post("/login", response_model=Token)
async def login(user_credentials: UserLogin):
    """Login user and return access token"""
    try:
        token = await login_user(user_credentials)
        return token
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Login failed: {str(e)}"
        )

@router.get("/me", response_model=User)
async def get_current_user_info(current_user: User = Depends(get_current_user)):
    """Get current user information"""
    return current_user

@router.post("/logout")
async def logout(current_user: User = Depends(get_current_user)):
    """Logout user (invalidate token)"""
    # In a real implementation, you might want to blacklist the token
    # For now, we'll just return success
    return {"message": "Successfully logged out"} 