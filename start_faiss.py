#!/usr/bin/env python3
"""
Start script for RAG application with FAISS
This script initializes the application with FAISS vector database.
"""

import os
import sys
import asyncio
import uvicorn
from pathlib import Path

# Add backend to path
backend_path = Path(__file__).parent / "backend"
sys.path.append(str(backend_path))

async def init_application():
    """Initialize the application with FAISS"""
    try:
        print("🚀 Initializing RAG Application with FAISS")
        print("=" * 50)
        
        # Import and initialize database
        from database import init_db
        await init_db()
        print("✅ Database initialized")
        
        # Initialize FAISS with sample data if empty
        from database import get_faiss_index
        faiss_index = get_faiss_index()
        
        if faiss_index.ntotal == 0:
            print("📚 Initializing FAISS with sample data...")
            # Add sample data directly
            from sentence_transformers import SentenceTransformer
            from database import add_to_faiss
            
            embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            
            # Sample documents
            sample_docs = [
                "Học viện Công nghệ Bưu chính Viễn thông - Đại học Quốc gia TP.HCM (PTITHCM) là một học viện công lập chuyên về công nghệ thông tin tại Việt Nam.",
                "PTITHCM được thành lập năm 2006 và là một trong những học viện hàng đầu về đào tạo công nghệ thông tin tại Việt Nam.",
                "Học viện có các chương trình đào tạo đại học, thạc sĩ và tiến sĩ về các lĩnh vực công nghệ thông tin."
            ]
            
            sample_metadata = [
                {"source": "intro", "category": "general"},
                {"source": "history", "category": "general"},
                {"source": "programs", "category": "academic"}
            ]
            
            # Generate embeddings and add to FAISS
            embeddings = embedding_model.encode(sample_docs)
            add_to_faiss(embeddings, sample_docs, sample_metadata)
            print("✅ Sample data loaded")
        else:
            print(f"✅ FAISS index ready with {faiss_index.ntotal} vectors")
        
        print("🎉 Application ready to start!")
        return True
        
    except Exception as e:
        print(f"❌ Initialization failed: {e}")
        return False

def start_server():
    """Start the FastAPI server"""
    try:
        print("🌐 Starting FastAPI server...")
        
        # Change to backend directory
        os.chdir("backend")
        
        # Start uvicorn server
        uvicorn.run(
            "main:app",
            host="0.0.0.0",
            port=8000,
            reload=True,
            log_level="info"
        )
        
    except Exception as e:
        print(f"❌ Server start failed: {e}")
        sys.exit(1)

async def main():
    """Main function"""
    print("🔧 RAG Application with FAISS")
    print("=" * 40)
    
    # Initialize application
    success = await init_application()
    
    if success:
        # Start server
        start_server()
    else:
        print("❌ Failed to initialize application")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main()) 