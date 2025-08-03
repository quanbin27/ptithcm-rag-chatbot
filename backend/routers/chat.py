from fastapi import APIRouter, HTTPException, status, Depends
from models import ChatRequest, ChatResponse, ChatSession, ChatMessage
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

@router.post("/send", response_model=ChatResponse)
async def send_message(
    chat_request: ChatRequest,
    current_user = Depends(get_current_user)
):
    """Send a message and get AI response"""
    try:
        db = get_mongo_db()
        
        # Get or create session
        if chat_request.session_id:
            session_id = chat_request.session_id
            # Verify session belongs to user
            session_data = await db.chat_sessions.find_one({
                "_id": ObjectId(session_id),
                "user_id": current_user.id
            })
            if not session_data:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Session not found or access denied"
                )
        else:
            # Create new session
            session_data = {
                "user_id": current_user.id,
                "title": chat_request.message[:50] + "..." if len(chat_request.message) > 50 else chat_request.message,
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow()
            }
            result = await db.chat_sessions.insert_one(session_data)
            session_id = str(result.inserted_id)
        
        # Get chat history
        chat_history = []
        messages_cursor = db.chat_messages.find({
            "session_id": session_id
        }).sort("timestamp", 1)
        
        async for msg in messages_cursor:
            chat_history.append({
                "role": msg["role"],
                "content": msg["content"]
            })
        
        # Process query with RAG
        rag_result = rag_engine.process_query(chat_request.message, chat_history)
        
        # Create message IDs
        user_message_id = str(uuid.uuid4())
        assistant_message_id = str(uuid.uuid4())
        
        # Create messages
        user_message = {
            "message_id": user_message_id,
            "session_id": session_id,
            "content": chat_request.message,
            "role": "user",
            "timestamp": datetime.utcnow()
        }
        
        assistant_message = {
            "message_id": assistant_message_id,
            "session_id": session_id,
            "content": rag_result["response"],
            "role": "assistant",
            "timestamp": datetime.utcnow(),
            "sources": rag_result["sources"]
        }
        
        # Store messages in MongoDB
        await db.chat_messages.insert_many([user_message, assistant_message])
        
        # Update session timestamp
        await db.chat_sessions.update_one(
            {"_id": ObjectId(session_id)},
            {"$set": {"updated_at": datetime.utcnow()}}
        )
        
        return ChatResponse(
            response=rag_result["response"],
            session_id=session_id,
            message_id=assistant_message_id,
            sources=rag_result["sources"],
            route=rag_result.get("route"),
            metadata=rag_result.get("metadata")
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Chat processing failed: {str(e)}"
        )

@router.get("/sessions", response_model=List[ChatSession])
async def get_chat_sessions(current_user = Depends(get_current_user)):
    """Get all chat sessions for current user"""
    try:
        db = get_mongo_db()
        
        # Get all sessions for user
        sessions_cursor = db.chat_sessions.find({"user_id": current_user.id}).sort("updated_at", -1)
        
        sessions = []
        async for session_data in sessions_cursor:
            # Get last few messages for preview
            recent_messages_cursor = db.chat_messages.find({
                "session_id": str(session_data["_id"])
            }).sort("timestamp", -1).limit(3)
            
            messages = []
            async for msg_data in recent_messages_cursor:
                messages.append(ChatMessage(
                    message_id=msg_data["message_id"],
                    content=msg_data["content"],
                    role=msg_data["role"],
                    timestamp=msg_data["timestamp"],
                    sources=msg_data.get("sources", [])
                ))
            
            # Reverse to get chronological order
            messages.reverse()
            
            session = ChatSession(
                id=str(session_data["_id"]),
                user_id=session_data["user_id"],
                title=session_data["title"],
                created_at=session_data["created_at"],
                updated_at=session_data["updated_at"],
                messages=messages
            )
            sessions.append(session)
        
        return sessions
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get chat sessions: {str(e)}"
        )

@router.get("/sessions/{session_id}/messages", response_model=List[ChatMessage])
async def get_session_messages(
    session_id: str,
    current_user = Depends(get_current_user)
):
    """Get all messages in a specific chat session"""
    try:
        db = get_mongo_db()
        
        # Verify session belongs to user
        session_data = await db.chat_sessions.find_one({
            "_id": ObjectId(session_id),
            "user_id": current_user.id
        })
        
        if not session_data:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Session not found or access denied"
            )
        
        # Get all messages
        messages_cursor = db.chat_messages.find({
            "session_id": session_id
        }).sort("timestamp", 1)
        
        messages = []
        async for msg_data in messages_cursor:
            messages.append(ChatMessage(
                message_id=msg_data["message_id"],
                content=msg_data["content"],
                role=msg_data["role"],
                timestamp=msg_data["timestamp"],
                sources=msg_data.get("sources", [])
            ))
        
        return messages
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get session messages: {str(e)}"
        )

@router.delete("/sessions/{session_id}")
async def delete_chat_session(
    session_id: str,
    current_user = Depends(get_current_user)
):
    """Delete a chat session"""
    try:
        db = get_mongo_db()
        
        # Verify session belongs to user
        session_data = await db.chat_sessions.find_one({
            "_id": ObjectId(session_id),
            "user_id": current_user.id
        })
        
        if not session_data:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Session not found or access denied"
            )
        
        # Delete session and all its messages
        await db.chat_sessions.delete_one({"_id": ObjectId(session_id)})
        await db.chat_messages.delete_many({"session_id": session_id})
        
        return {"message": "Session deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete session: {str(e)}"
        )

@router.get("/stats")
async def get_rag_stats(current_user = Depends(get_current_user)):
    """Get RAG engine statistics"""
    try:
        stats = rag_engine.get_stats()
        return stats
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get RAG stats: {str(e)}"
        ) 