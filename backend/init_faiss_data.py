#!/usr/bin/env python3
"""
Script to initialize FAISS data from text files in the data directory
"""

import json
import os
import numpy as np
import asyncio
from sentence_transformers import SentenceTransformer
from database import add_to_faiss, get_faiss_index, save_faiss_data, init_db
from langchain_text_splitters import RecursiveCharacterTextSplitter

def load_text_files():
    """
    Load text files from data directory and split into chunks
    """
    data_dir = "../data"
    documents = []
    metadata = []
    
    # Text splitter configuration
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        length_function=len,
        separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""]
    )
    
    # Process each text file
    for filename in os.listdir(data_dir):
        if filename.endswith('.txt'):
            file_path = os.path.join(data_dir, filename)
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Split content into chunks
                chunks = text_splitter.split_text(content)
                
                # Create metadata for each chunk
                for i, chunk in enumerate(chunks):
                    if len(chunk.strip()) < 50:  # Skip very short chunks
                        continue
                    
                    documents.append(chunk.strip())
                    metadata.append({
                        "source": filename,
                        "category": "general",
                        "chunk_index": i,
                        "total_chunks": len(chunks),
                        "uploaded_at": "2024-01-01T00:00:00.000000"  # Default timestamp
                    })
                
                print(f"✅ Processed {filename}: {len(chunks)} chunks")
                
            except Exception as e:
                print(f"❌ Error processing {filename}: {e}")
    
    print(f"📊 Total: {len(documents)} documents, {len(metadata)} metadata entries")
    return documents, metadata

async def create_faiss_index():
    """
    Create FAISS index from text files
    """
    print("🚀 Bắt đầu tạo FAISS index từ dữ liệu text...")
    
    # Initialize database first
    print("📊 Khởi tạo database...")
    try:
        await init_db()
        print("✅ Database initialized")
    except Exception as e:
        print(f"❌ Lỗi khởi tạo database: {e}")
        return False
    
    # Load documents from text files
    print("📚 Đang tải dữ liệu từ text files...")
    documents, metadata = load_text_files()
    
    if not documents or not metadata:
        print("❌ Không có dữ liệu để xử lý")
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
        print(f"✅ Đã tạo {len(embeddings)} embeddings")
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
            print(f"  {i+1}. Document {idx}: {documents[idx][:100]}... (distance: {distance:.4f})")
        
        return True
        
    except Exception as e:
        print(f"❌ Lỗi kiểm tra index: {e}")
        return False

async def main():
    """
    Main function
    """
    print("🔧 Tạo FAISS Index từ dữ liệu text PTITHCM")
    print("=" * 50)
    
    success = await create_faiss_index()
    
    if success:
        print("\n🎉 Hoàn thành! FAISS index đã sẵn sàng cho RAG")
        print("Bạn có thể chạy ứng dụng với: python main.py")
    else:
        print("\n❌ Thất bại! Vui lòng kiểm tra lỗi và thử lại")

if __name__ == "__main__":
    asyncio.run(main()) 