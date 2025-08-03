import jwt
from datetime import datetime, timedelta
from passlib.context import CryptContext
from fastapi import HTTPException, status, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import uuid
from typing import Optional
from bson import ObjectId

from models import User, UserCreate, UserLogin, Token
from database import get_mongo_db

# Security configuration
SECRET_KEY = "your-secret-key-change-in-production"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
security = HTTPBearer()

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash"""
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    """Hash a password"""
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    """Create JWT access token"""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def verify_token(token: str) -> dict:
    """Verify JWT token and return payload"""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except jwt.PyJWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> User:
    """Get current user from token"""
    token = credentials.credentials
    payload = verify_token(token)
    user_id: str = payload.get("sub")
    if user_id is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Get user from MongoDB
    db = get_mongo_db()
    user_data = await db.users.find_one({"_id": ObjectId(user_id)})
    
    if not user_data:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    return User(
        id=str(user_data["_id"]),
        email=user_data["email"],
        full_name=user_data["full_name"],
        role=user_data["role"],
        created_at=user_data["created_at"],
        is_active=user_data.get("is_active", True)
    )

async def create_user(user: UserCreate) -> User:
    """Create a new user"""
    db = get_mongo_db()
    
    # Check if user already exists
    existing_user = await db.users.find_one({"email": user.email})
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )
    
    # Create user
    hashed_password = get_password_hash(user.password)
    
    user_data = {
        "email": user.email,
        "full_name": user.full_name,
        "role": user.role.value,
        "password_hash": hashed_password,
        "created_at": datetime.utcnow(),
        "is_active": True
    }
    
    # Store user data in MongoDB
    result = await db.users.insert_one(user_data)
    
    return User(
        id=str(result.inserted_id),
        email=user.email,
        full_name=user.full_name,
        role=user.role,
        created_at=user_data["created_at"],
        is_active=True
    )

async def authenticate_user(email: str, password: str) -> Optional[User]:
    """Authenticate user with email and password"""
    db = get_mongo_db()
    
    # Get user data from MongoDB
    user_data = await db.users.find_one({"email": email})
    if not user_data:
        return None
    
    # Verify password
    if not verify_password(password, user_data["password_hash"]):
        return None
    
    return User(
        id=str(user_data["_id"]),
        email=user_data["email"],
        full_name=user_data["full_name"],
        role=user_data["role"],
        created_at=user_data["created_at"],
        is_active=user_data.get("is_active", True)
    )

async def login_user(user_login: UserLogin) -> Token:
    """Login user and return access token"""
    user = await authenticate_user(user_login.email, user_login.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.id}, expires_delta=access_token_expires
    )
    
    return Token(access_token=access_token, token_type="bearer", user=user) 