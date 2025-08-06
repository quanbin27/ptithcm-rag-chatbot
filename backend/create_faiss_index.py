#!/usr/bin/env python3
"""
Script to create FAISS index from documents in faiss_data directory
"""

import json
import os
import numpy as np
import asyncio
from sentence_transformers import SentenceTransformer
from database import add_to_faiss, get_faiss_index, save_faiss_data, init_db

def load_documents():
    """
    Load documents and metadata from faiss_data directory
    """
    documents_path = "faiss_data/documents.json"
    metadata_path = "faiss_data/metadata.json"
    
    if not os.path.exists(documents_path) or not os.path.exists(metadata_path):
        print("❌ Không tìm thấy file documents.json hoặc metadata.json")
        print("💡 Vui lòng chạy init_faiss_data.py trước để tạo dữ liệu")
        return None, None
    
    try:
        with open(documents_path, 'r', encoding='utf-8') as f:
            documents = json.load(f)
        
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        # Validate data consistency
        if len(documents) != len(metadata):
            print(f"❌ Số lượng documents ({len(documents)}) không khớp với metadata ({len(metadata)})")
            return None, None
        
        print(f"✅ Đã tải {len(documents)} documents và {len(metadata)} metadata")
        return documents, metadata
        
    except Exception as e:
        print(f"❌ Lỗi khi tải dữ liệu: {e}")
        return None, None

async def create_faiss_index():
    """
    Create FAISS index from documents
    """
    print("🚀 Bắt đầu tạo FAISS index...")
    
    # Initialize database first
    print("📊 Khởi tạo database...")
    try:
        await init_db()
        print("✅ Database initialized")
    except Exception as e:
        print(f"❌ Lỗi khởi tạo database: {e}")
        return False
    
    # Load documents
    documents, metadata = load_documents()
    if not documents or not metadata:
        return False
    
    # Validate document format
    if not isinstance(documents, list) or not all(isinstance(doc, str) for doc in documents):
        print("❌ Documents không đúng định dạng (phải là list of strings)")
        return False
    
    # Validate consistency - use documents count as reference
    if len(documents) != len(metadata):
        print(f"⚠️ Data inconsistency: {len(documents)} documents vs {len(metadata)} metadata")
        print(f"⚠️ Using documents count ({len(documents)}) as reference")
        if len(metadata) > len(documents):
            metadata = metadata[:len(documents)]
            print(f"⚠️ Trimmed metadata to {len(metadata)} entries")
        else:
            print("❌ Metadata count is less than documents, cannot proceed")
            return False
    
    # Initialize embedding model
    print("📚 Khởi tạo embedding model...")
    try:
        embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        print("✅ Embedding model ready")
    except Exception as e:
        print(f"❌ Lỗi khởi tạo embedding model: {e}")
        return False
    
    # Generate embeddings
    print("🔄 Đang tạo embeddings...")
    try:
        embeddings = embedding_model.encode(documents, show_progress_bar=True)
        print(f"✅ Đã tạo {len(embeddings)} embeddings với shape {embeddings.shape}")
    except Exception as e:
        print(f"❌ Lỗi tạo embeddings: {e}")
        return False
    
    # Clear existing FAISS index
    print("🧹 Xóa FAISS index cũ...")
    try:
        # Delete existing index file if it exists
        import faiss
        index_file = "faiss_data/faiss_index.bin"
        if os.path.exists(index_file):
            os.remove(index_file)
            print("✅ Đã xóa index cũ")
    except Exception as e:
        print(f"⚠️  Lỗi khi xóa index cũ: {e}")
    
    # Add to FAISS
    print("📝 Thêm dữ liệu vào FAISS...")
    try:
        add_to_faiss(embeddings, documents, metadata)
        print("✅ Đã thêm dữ liệu vào FAISS")
    except Exception as e:
        print(f"❌ Lỗi thêm dữ liệu vào FAISS: {e}")
        return False
    
    # Verify
    try:
        final_index = get_faiss_index()
        print(f"✅ FAISS index hoàn thành với {final_index.ntotal} vectors")
        
        # Test search
        print("🧪 Kiểm tra tìm kiếm...")
        test_query = "tuyển sinh"
        test_embedding = embedding_model.encode([test_query])
        D, I = final_index.search(test_embedding, 2)
        
        print(f"Kết quả tìm kiếm cho '{test_query}':")
        for i, (distance, idx) in enumerate(zip(D[0], I[0])):
            if idx < len(documents):
                print(f"  {i+1}. Document {idx}: {documents[idx][:100]}... (distance: {distance:.4f})")
        
        return True
        
    except Exception as e:
        print(f"❌ Lỗi kiểm tra index: {e}")
        return False

async def main():
    """
    Main function
    """
    print("🔧 Tạo FAISS Index từ dữ liệu PTITHCM")
    print("=" * 50)
    
    success = await create_faiss_index()
    
    if success:
        print("\n🎉 Hoàn thành! FAISS index đã sẵn sàng cho RAG")
        print("Bạn có thể chạy ứng dụng với: python main.py")
    else:
        print("\n❌ Thất bại! Vui lòng kiểm tra lỗi và thử lại")

if __name__ == "__main__":
    asyncio.run(main()) 