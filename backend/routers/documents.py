from fastapi import APIRouter, HTTPException, status, Depends, UploadFile, File
from models import DocumentUpload, DocumentInfo, SearchRequest, SearchResponse
from auth import get_current_user
from rag_engine import RAGEngine
from database import get_mongo_db
from bson import ObjectId
import uuid
from datetime import datetime
from typing import List

router = APIRouter()

# Initialize RAG engine
rag_engine = RAGEngine()

@router.post("/upload", response_model=DocumentInfo)
async def upload_document(
    file: UploadFile = File(...),
    category: str = None,
    current_user = Depends(get_current_user)
):
    """Upload a document to the RAG system"""
    try:
        # Check user role - only teacher and admin can upload
        if current_user.role not in ["teacher", "admin"]:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Only teachers and administrators can upload documents"
            )
        
        # Check file type
        if not file.filename.endswith('.txt'):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Only .txt files are supported"
            )
        
        # Read file content
        content = await file.read()
        content_str = content.decode('utf-8')
        
        # Create document metadata
        metadata = {
            "source": file.filename,
            "category": category or "general",
            "uploaded_by": current_user.id,
            "uploaded_at": datetime.utcnow().isoformat()
        }
        
        # Add document to RAG system
        chunk_count = rag_engine.add_document(content_str, metadata)
        
        # Store document info in MongoDB
        db = get_mongo_db()
        doc_info = {
            "filename": file.filename,
            "category": category or "general",
            "uploaded_by": current_user.id,
            "uploaded_at": datetime.utcnow(),
            "chunk_count": chunk_count,
            "file_size": len(content_str)
        }
        
        result = await db.documents.insert_one(doc_info)
        
        return DocumentInfo(
            id=str(result.inserted_id),
            filename=file.filename,
            category=category or "general",
            uploaded_at=doc_info["uploaded_at"],
            chunk_count=chunk_count
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Document upload failed: {str(e)}"
        )

@router.get("/list", response_model=List[DocumentInfo])
async def list_documents(current_user = Depends(get_current_user)):
    """List all documents in the system - only teachers and admins"""
    try:
        # Check user role - students cannot access
        if current_user.role == "student":
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Students cannot access document management"
            )
        
        db = get_mongo_db()
        
        # Get all documents
        documents_cursor = db.documents.find().sort("uploaded_at", -1)
        
        documents = []
        async for doc_data in documents_cursor:
            document = DocumentInfo(
                id=str(doc_data["_id"]),
                filename=doc_data["filename"],
                category=doc_data.get("category"),
                uploaded_at=doc_data["uploaded_at"],
                chunk_count=doc_data["chunk_count"]
            )
            documents.append(document)
        
        return documents
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list documents: {str(e)}"
        )

@router.delete("/{doc_id}")
async def delete_document(
    doc_id: str,
    current_user = Depends(get_current_user)
):
    """Delete a document from the system - only teachers and admins"""
    try:
        # Check user role - students cannot delete
        if current_user.role == "student":
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Students cannot delete documents"
            )
        
        db = get_mongo_db()
        
        # Check if document exists
        doc_data = await db.documents.find_one({"_id": ObjectId(doc_id)})
        if not doc_data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Document not found"
            )
        
        # Check if user has permission (admin or document owner)
        if current_user.role != "admin" and doc_data.get("uploaded_by") != current_user.id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Permission denied. Only administrators or document owners can delete documents"
            )
        
        # Remove from MongoDB
        await db.documents.delete_one({"_id": ObjectId(doc_id)})
        
        # Note: In a production system, you would also remove from ChromaDB
        # This is simplified for demo purposes
        
        return {"message": "Document deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete document: {str(e)}"
        )

@router.post("/search", response_model=SearchResponse)
async def search_documents(
    search_request: SearchRequest,
    current_user = Depends(get_current_user)
):
    """Search documents using semantic search - only teachers and admins"""
    try:
        # Check user role - students cannot access
        if current_user.role == "student":
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Students cannot access document management"
            )
        
        # Use RAG engine for semantic search
        results = rag_engine.semantic_search(search_request.query, search_request.limit)
        
        return SearchResponse(
            results=results,
            total=len(results)
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Search failed: {str(e)}"
        )

@router.get("/stats")
async def get_document_stats(current_user = Depends(get_current_user)):
    """Get document statistics - only teachers and admins"""
    try:
        # Check user role - students cannot access
        if current_user.role == "student":
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Students cannot access document management"
            )
        
        db = get_mongo_db()
        
        # Get all documents
        documents_cursor = db.documents.find()
        
        total_documents = 0
        total_chunks = 0
        categories = {}
        
        async for doc_data in documents_cursor:
            total_documents += 1
            total_chunks += doc_data["chunk_count"]
            category = doc_data.get("category", "general")
            categories[category] = categories.get(category, 0) + 1
        
        return {
            "total_documents": total_documents,
            "total_chunks": total_chunks,
            "categories": categories
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get stats: {str(e)}"
        ) 