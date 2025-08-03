from pydantic import BaseModel, EmailStr
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum

class UserRole(str, Enum):
    STUDENT = "student"
    TEACHER = "teacher"
    ADMIN = "admin"

class UserBase(BaseModel):
    email: EmailStr
    full_name: str
    role: UserRole = UserRole.STUDENT

class UserCreate(UserBase):
    password: str

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class User(UserBase):
    id: str
    created_at: datetime
    is_active: bool = True

    class Config:
        from_attributes = True

class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: User

class ChatMessage(BaseModel):
    content: str
    role: str  # "user" or "assistant"
    timestamp: datetime
    message_id: str

class ChatSession(BaseModel):
    id: str
    user_id: str
    title: str
    created_at: datetime
    updated_at: datetime
    messages: List[ChatMessage] = []

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    session_id: str
    message_id: str
    sources: List[Dict[str, Any]] = []
    route: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

class DocumentUpload(BaseModel):
    filename: str
    content: str
    category: Optional[str] = None

class DocumentInfo(BaseModel):
    id: str
    filename: str
    category: Optional[str]
    uploaded_at: datetime
    chunk_count: int

class SearchRequest(BaseModel):
    query: str
    limit: int = 5

class SearchResponse(BaseModel):
    results: List[Dict[str, Any]]
    total: int 