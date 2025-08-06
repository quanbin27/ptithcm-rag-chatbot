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
    
    # Category mapping based on filename and content
    def determine_category(filename: str, content: str) -> str:
        filename_lower = filename.lower()
        content_lower = content.lower()
        # Check for admission-related content
        admission_keywords = [
            "tuyển sinh", "đăng ký", "hồ sơ", "điểm chuẩn", "học phí", 
            "thủ tục nhập học", "chỉ tiêu", "phương thức tuyển sinh",
            "xét tuyển", "thi đánh giá", "thi tốt nghiệp", "học bổng"
        ]
        # Check for academic-related content
        academic_keywords = [
            "chương trình học", "môn học", "giảng viên", "lịch học", 
            "thời khóa biểu", "đào tạo", "ngành học", "kỹ thuật phần mềm",
            "hệ thống thông tin", "khoa học máy tính", "mạng máy tính",
            "công nghệ thông tin", "an toàn thông tin", "trí tuệ nhân tạo",
            "quy định sinh viên", "quy chế học vụ", "nghĩa vụ sinh viên", "nội quy giảng đường",
            "nội quy lớp học", "nội quy ra vào cổng", "thi cử", "khen thưởng", "kỷ luật sinh viên",
            "quy trình học tập", "quy định thi tốt nghiệp", "quy định bảo vệ đồ án",
            "nghiên cứu khoa học", "hoạt động học tập", "điểm rèn luyện", "điểm chuyên cần",
            "quy định về trang phục", "quy định về sử dụng tài sản trường",
            "quy định về bảo vệ môi trường", "quy định về an ninh trật tự",
            "quy định về đóng học phí", "quy định về nghỉ học", "quy định về bảo lưu kết quả",
            "quy định về chuyển ngành, chuyển trường", "quy định về xét tốt nghiệp",
            "quy định về học lại, thi lại", "quy định về học bổng, hỗ trợ sinh viên"
        ]
        # Check filename first
        if "tuyen_sinh" in filename_lower:
            return "admission"
        elif "so_tay" in filename_lower:
            return "academic"
        elif "gioi_thieu" in filename_lower:
            return "general"
        # Check content keywords
        admission_count = sum(1 for keyword in admission_keywords if keyword in content_lower)
        academic_count = sum(1 for keyword in academic_keywords if keyword in content_lower)
        # Determine category based on keyword frequency
        if admission_count > academic_count:
            return "admission"
        elif academic_count > 0:
            return "academic"
        else:
            return "general"
    
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
                    
                    # Gán category cho từng chunk dựa trên nội dung chunk
                    chunk_category = determine_category(filename, chunk)
                    documents.append(chunk.strip())
                    metadata.append({
                        "source": filename,
                        "category": chunk_category,
                        "chunk_index": i,
                        "total_chunks": len(chunks),
                        "uploaded_at": "2024-01-01T00:00:00.000000"  # Default timestamp
                    })
                
                print(f"✅ Processed {filename}: {len(chunks)} chunks (category by chunk)")
                
            except Exception as e:
                print(f"❌ Error processing {filename}: {e}")
    
    # Print category statistics
    category_counts = {}
    for meta in metadata:
        cat = meta.get("category", "unknown")
        category_counts[cat] = category_counts.get(cat, 0) + 1
    
    print(f"📊 Category distribution:")
    for category, count in category_counts.items():
        print(f"  - {category}: {count} chunks")
    
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
    
    # Clear existing FAISS index and data
    print("🧹 Xóa FAISS index và dữ liệu cũ...")
    try:
        index_file = "faiss_data/faiss_index.bin"
        documents_file = "faiss_data/documents.json"
        metadata_file = "faiss_data/metadata.json"
        for f in [index_file, documents_file, metadata_file]:
            if os.path.exists(f):
                os.remove(f)
                print(f"✅ Đã xóa {f}")
    except Exception as e:
        print(f"⚠️  Lỗi khi xóa dữ liệu cũ: {e}")
    # Add to FAISS (reset)
    print("📝 Thêm dữ liệu vào FAISS...")
    try:
        add_to_faiss(embeddings, documents, metadata, reset=True)
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
            if idx < len(documents):  # Kiểm tra index bounds
                print(f"  {i+1}. Document {idx}: {documents[idx][:100]}... (distance: {distance:.4f})")
            else:
                print(f"  {i+1}. Document {idx}: [INDEX OUT OF RANGE] (distance: {distance:.4f})")
        
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