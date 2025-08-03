import redis
import faiss
import numpy as np
import pickle
import os
import motor.motor_asyncio
from typing import Optional, List, Dict, Any
import json

# Redis connection (for caching)
redis_client: Optional[redis.Redis] = None

# FAISS index (for vector storage)
faiss_index: Optional[faiss.Index] = None
faiss_documents: List[Dict[str, Any]] = []
faiss_metadata: List[Dict[str, Any]] = []

# MongoDB client (for user data and chat history)
mongo_client: Optional[motor.motor_asyncio.AsyncIOMotorClient] = None
mongo_db: Optional[motor.motor_asyncio.AsyncIOMotorDatabase] = None

# FAISS data directory
FAISS_DATA_DIR = "faiss_data"
FAISS_INDEX_FILE = os.path.join(FAISS_DATA_DIR, "faiss_index.bin")
FAISS_DOCUMENTS_FILE = os.path.join(FAISS_DATA_DIR, "documents.json")
FAISS_METADATA_FILE = os.path.join(FAISS_DATA_DIR, "metadata.json")

async def init_db():
    """Initialize database connections"""
    global redis_client, faiss_index, faiss_documents, faiss_metadata, mongo_client, mongo_db
    
    # Create FAISS data directory if it doesn't exist
    os.makedirs(FAISS_DATA_DIR, exist_ok=True)
    
    # Initialize Redis (for caching)
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
    redis_client = redis.from_url(redis_url, decode_responses=True)
    
    # Test Redis connection
    try:
        redis_client.ping()
        print("✅ Redis connected successfully")
    except Exception as e:
        print(f"❌ Redis connection failed: {e}")
        raise
    
    # Initialize MongoDB (for user data and chat history)
    mongo_url = os.getenv("MONGODB_URL", "mongodb://localhost:27017")
    mongo_db_name = os.getenv("MONGODB_DB", "ptithcm_rag")
    try:
        mongo_client = motor.motor_asyncio.AsyncIOMotorClient(mongo_url)
        mongo_db = mongo_client[mongo_db_name]
        # Test connection
        await mongo_client.admin.command('ping')
        print("✅ MongoDB connected successfully")
    except Exception as e:
        print(f"❌ MongoDB connection failed: {e}")
        raise
    
    # Initialize FAISS (for vector storage)
    try:
        load_faiss_data()
        print("✅ FAISS initialized successfully")
    except Exception as e:
        print(f"❌ FAISS initialization failed: {e}")
        raise

def load_faiss_data():
    """Load FAISS index and documents from disk"""
    global faiss_index, faiss_documents, faiss_metadata
    
    # Load documents and metadata
    if os.path.exists(FAISS_DOCUMENTS_FILE):
        with open(FAISS_DOCUMENTS_FILE, 'r', encoding='utf-8') as f:
            faiss_documents = json.load(f)
    
    if os.path.exists(FAISS_METADATA_FILE):
        with open(FAISS_METADATA_FILE, 'r', encoding='utf-8') as f:
            faiss_metadata = json.load(f)
    
    # Load or create FAISS index
    if os.path.exists(FAISS_INDEX_FILE):
        faiss_index = faiss.read_index(FAISS_INDEX_FILE)
        print(f"✅ Loaded existing FAISS index with {faiss_index.ntotal} vectors")
    else:
        # Create new index (assuming 384-dimensional vectors from all-MiniLM-L6-v2)
        dimension = 384
        faiss_index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
        print("✅ Created new FAISS index")

def save_faiss_data():
    """Save FAISS index and documents to disk"""
    global faiss_index, faiss_documents, faiss_metadata
    
    # Save documents and metadata
    with open(FAISS_DOCUMENTS_FILE, 'w', encoding='utf-8') as f:
        json.dump(faiss_documents, f, ensure_ascii=False, indent=2)
    
    with open(FAISS_METADATA_FILE, 'w', encoding='utf-8') as f:
        json.dump(faiss_metadata, f, ensure_ascii=False, indent=2)
    
    # Save FAISS index
    faiss.write_index(faiss_index, FAISS_INDEX_FILE)
    print(f"✅ Saved FAISS data: {faiss_index.ntotal} vectors, {len(faiss_documents)} documents")

def get_redis() -> redis.Redis:
    """Get Redis client"""
    if redis_client is None:
        raise Exception("Redis client not initialized")
    return redis_client

def get_faiss_index() -> faiss.Index:
    """Get FAISS index"""
    if faiss_index is None:
        raise Exception("FAISS index not initialized")
    return faiss_index

def get_faiss_documents() -> List[Dict[str, Any]]:
    """Get FAISS documents"""
    if faiss_documents is None:
        raise Exception("FAISS documents not initialized")
    return faiss_documents

def get_faiss_metadata() -> List[Dict[str, Any]]:
    """Get FAISS metadata"""
    if faiss_metadata is None:
        raise Exception("FAISS metadata not initialized")
    return faiss_metadata

def add_to_faiss(embeddings: np.ndarray, documents: List[str], metadata: List[Dict[str, Any]]):
    """Add embeddings, documents, and metadata to FAISS"""
    global faiss_index, faiss_documents, faiss_metadata
    
    # Add embeddings to FAISS index
    faiss_index.add(embeddings.astype('float32'))
    
    # Add documents and metadata
    faiss_documents.extend(documents)
    faiss_metadata.extend(metadata)
    
    # Save to disk
    save_faiss_data()

def get_mongo_db() -> motor.motor_asyncio.AsyncIOMotorDatabase:
    """Get MongoDB database"""
    if mongo_db is None:
        raise Exception("MongoDB client not initialized")
    return mongo_db 