#!/usr/bin/env python3
"""
Script khởi tạo dữ liệu mẫu cho PTITHCM RAG System
"""

import json
import os
import re
from typing import List, Dict, Any

def split_text_into_chunks(text: str, chunk_size: int = 500, overlap: int = 100) -> List[str]:
    """
    Chia văn bản thành các đoạn nhỏ với độ chồng lấp
    """
    words = text.split()
    chunks = []
    
    for i in range(0, len(words), chunk_size - overlap):
        chunk = ' '.join(words[i:i + chunk_size])
        if chunk.strip():
            chunks.append(chunk.strip())
    
    return chunks

def process_file(file_path: str, source_name: str) -> List[Dict[str, Any]]:
    """
    Xử lý một file và trả về danh sách các đoạn văn bản với metadata
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Remove empty lines and normalize text
    content = re.sub(r'\n\s*\n', '\n', content)
    content = re.sub(r'\s+', ' ', content)
    
    # Split into small chunks
    chunks = split_text_into_chunks(content)
    
    # Create metadata for each chunk
    documents = []
    for i, chunk in enumerate(chunks):
        doc = {
            "content": chunk,
            "source": source_name,
            "chunk_id": i,
            "file_path": file_path
        }
        documents.append(doc)
    
    return documents

def main():
    """
    Chuyển đổi dữ liệu từ thư mục data vào faiss_data
    """
    # Path
    data_dir = "../data"
    faiss_data_dir = "faiss_data"
    
    # Ensure faiss_data directory exists
    os.makedirs(faiss_data_dir, exist_ok=True)
    
    all_documents = []
    all_metadata = []
    
    # Process files in data directory
    data_files = [
        ("ptithcm_gioi_thieu.txt", "gioi_thieu"),
        ("ptithcm_tuyen_sinh.txt", "tuyen_sinh")
    ]
    
    for filename, source_name in data_files:
        file_path = os.path.join(data_dir, filename)
        if os.path.exists(file_path):
            print(f"Đang xử lý file: {filename}")
            documents = process_file(file_path, source_name)
            all_documents.extend(documents)
            print(f"Đã tạo {len(documents)} đoạn văn bản từ {filename}")
    
    # Create documents.json
    documents_content = [doc["content"] for doc in all_documents]
    with open(os.path.join(faiss_data_dir, "documents.json"), 'w', encoding='utf-8') as f:
        json.dump(documents_content, f, ensure_ascii=False, indent=2)
    
    # Create metadata.json
    metadata_content = []
    for doc in all_documents:
        metadata = {
            "source": doc["source"],
            "chunk_id": doc["chunk_id"],
            "file_path": doc["file_path"]
        }
        metadata_content.append(metadata)
    
    with open(os.path.join(faiss_data_dir, "metadata.json"), 'w', encoding='utf-8') as f:
        json.dump(metadata_content, f, ensure_ascii=False, indent=2)
    
    print(f"\nHoàn thành! Đã tạo {len(all_documents)} đoạn văn bản:")
    print(f"- documents.json: {len(documents_content)} đoạn")
    print(f"- metadata.json: {len(metadata_content)} metadata")
    
    # Print statistics by source
    source_stats = {}
    for doc in all_documents:
        source = doc["source"]
        source_stats[source] = source_stats.get(source, 0) + 1
    
    print("\nThống kê theo nguồn:")
    for source, count in source_stats.items():
        print(f"- {source}: {count} đoạn")

if __name__ == "__main__":
    main() 