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
    
    # Initialize empty lists
    faiss_documents = []
    faiss_metadata = []
    
    # Load documents and metadata with error handling
    try:
        if os.path.exists(FAISS_DOCUMENTS_FILE):
            with open(FAISS_DOCUMENTS_FILE, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                if content:  # Only load if file is not empty
                    faiss_documents = json.loads(content)
                    # Validate documents are strings
                    if not all(isinstance(doc, str) for doc in faiss_documents):
                        print("⚠️ Documents không đúng định dạng, bắt đầu mới")
                        faiss_documents = []
                    else:
                        print(f"✅ Loaded {len(faiss_documents)} documents")
                else:
                    print("⚠️ Documents file is empty, starting fresh")
    except Exception as e:
        print(f"⚠️ Error loading documents: {e}, starting fresh")
    
    try:
        if os.path.exists(FAISS_METADATA_FILE):
            with open(FAISS_METADATA_FILE, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                if content:  # Only load if file is not empty
                    faiss_metadata = json.loads(content)
                    print(f"✅ Loaded {len(faiss_metadata)} metadata entries")
                else:
                    print("⚠️ Metadata file is empty, starting fresh")
    except Exception as e:
        print(f"⚠️ Error loading metadata: {e}, starting fresh")
    
    # Validate consistency
    if len(faiss_documents) != len(faiss_metadata):
        print(f"⚠️ Data inconsistency: {len(faiss_documents)} documents vs {len(faiss_metadata)} metadata")
        print(f"⚠️ Using documents count ({len(faiss_documents)}) as reference")
        # Trim metadata to match documents count
        if len(faiss_metadata) > len(faiss_documents):
            faiss_metadata = faiss_metadata[:len(faiss_documents)]
            print(f"⚠️ Trimmed metadata to {len(faiss_metadata)} entries")
        else:
            print(f"⚠️ Metadata count is less than documents, this is unusual")
            # Don't clear data, just use what we have
    
    # Load or create FAISS index
    try:
        if os.path.exists(FAISS_INDEX_FILE) and len(faiss_documents) > 0:
            print(f"📁 Loading existing FAISS index from {FAISS_INDEX_FILE}")
            faiss_index = faiss.read_index(FAISS_INDEX_FILE)
            print(f"✅ Loaded existing FAISS index with {faiss_index.ntotal} vectors")
            
            # Validate index consistency
            if faiss_index.ntotal != len(faiss_documents):
                print(f"⚠️ Index inconsistency: {faiss_index.ntotal} vectors vs {len(faiss_documents)} documents")
                print("⚠️ Creating new index")
                dimension = 384
                faiss_index = faiss.IndexFlatIP(dimension)
        else:
            # Create new index (assuming 384-dimensional vectors from all-MiniLM-L6-v2)
            dimension = 384
            faiss_index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
            print("✅ Created new FAISS index")
    except Exception as e:
        print(f"⚠️ Error loading FAISS index: {e}, creating new one")
        dimension = 384
        faiss_index = faiss.IndexFlatIP(dimension)
        print("✅ Created new FAISS index")
    
    # Final validation
    print(f"📊 Final status:")
    print(f"  - Documents: {len(faiss_documents)}")
    print(f"  - Metadata: {len(faiss_metadata)}")
    print(f"  - FAISS vectors: {faiss_index.ntotal}")
    print(f"  - FAISS index type: {type(faiss_index)}")

def serialize_datetime(obj):
    """Helper function to serialize datetime objects to ISO format strings"""
    from datetime import datetime
    if isinstance(obj, datetime):
        return obj.isoformat()
    return obj

def save_faiss_data():
    """Save FAISS index and documents to disk"""
    global faiss_index, faiss_documents, faiss_metadata
    
    # Save documents and metadata
    with open(FAISS_DOCUMENTS_FILE, 'w', encoding='utf-8') as f:
        json.dump(faiss_documents, f, ensure_ascii=False, indent=2)
    
    # Serialize metadata to handle datetime objects
    serialized_metadata = []
    for meta in faiss_metadata:
        serialized_meta = {}
        for key, value in meta.items():
            serialized_meta[key] = serialize_datetime(value)
        serialized_metadata.append(serialized_meta)
    
    with open(FAISS_METADATA_FILE, 'w', encoding='utf-8') as f:
        json.dump(serialized_metadata, f, ensure_ascii=False, indent=2)
    
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

def add_to_faiss(embeddings: np.ndarray, documents: List[str], metadata: List[Dict[str, Any]], reset: bool = False):
    """Add embeddings, documents, and metadata to FAISS. Nếu reset=True thì ghi đè toàn bộ (dùng cho init), nếu False thì chỉ append tránh trùng lặp (dùng cho rag_engine)."""
    global faiss_index, faiss_documents, faiss_metadata
    import hashlib
    if reset:
        # Ghi đè toàn bộ dữ liệu
        print(f"🧹 Reset FAISS: clear all data and re-create index")
        dimension = 384
        faiss_index = faiss.IndexFlatIP(dimension)
        faiss_documents.clear()
        faiss_metadata.clear()
        faiss_index.add(embeddings.astype('float32'))
        faiss_documents.extend(documents)
        # Serialize metadata to handle datetime objects
        serialized_metadata = []
        for meta in metadata:
            serialized_meta = {}
            for key, value in meta.items():
                serialized_meta[key] = serialize_datetime(value)
            serialized_metadata.append(serialized_meta)
        faiss_metadata.extend(serialized_metadata)
        print(f"✅ Reset and added {len(documents)} documents")
        print(f"✅ FAISS index now has {faiss_index.ntotal} vectors")
        save_faiss_data()
        return
    # Append mode (giữ nguyên logic cũ)
    if faiss_index is None:
        dimension = 384
        faiss_index = faiss.IndexFlatIP(dimension)
        print(f"✅ Created new FAISS index (append mode)")
    existing_hashes = set(hashlib.md5(doc.encode('utf-8')).hexdigest() for doc in faiss_documents)
    new_embeddings = []
    new_documents = []
    new_metadata = []
    for i, doc in enumerate(documents):
        doc_hash = hashlib.md5(doc.encode('utf-8')).hexdigest()
        if doc_hash in existing_hashes:
            print(f"⚠️ Duplicate document detected, skipping: {doc[:60]}...")
            continue
        new_embeddings.append(embeddings[i])
        new_documents.append(doc)
        new_metadata.append(metadata[i])
        existing_hashes.add(doc_hash)
    if new_embeddings:
        faiss_index.add(np.array(new_embeddings).astype('float32'))
        faiss_documents.extend(new_documents)
        serialized_metadata = []
        for meta in new_metadata:
            serialized_meta = {}
            for key, value in meta.items():
                serialized_meta[key] = serialize_datetime(value)
            serialized_metadata.append(serialized_meta)
        faiss_metadata.extend(serialized_metadata)
        print(f"✅ Added {len(new_documents)} new documents (no duplicates)")
        print(f"✅ FAISS index now has {faiss_index.ntotal} vectors")
        save_faiss_data()
    else:
        print("ℹ️ No new documents to add (all were duplicates)")

def get_mongo_db() -> motor.motor_asyncio.AsyncIOMotorDatabase:
    """Get MongoDB database"""
    if mongo_db is None:
        raise Exception("MongoDB client not initialized")
    return mongo_db 