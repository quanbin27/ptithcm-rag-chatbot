import google.generativeai as genai
import numpy as np
from sentence_transformers import SentenceTransformer
import redis
import json
import hashlib
from typing import List, Dict, Any, Optional, Tuple
import os
from datetime import datetime
import tiktoken
from langchain_text_splitters import RecursiveCharacterTextSplitter
import logging
from dotenv import load_dotenv
from dataclasses import dataclass, asdict
from enum import Enum
import asyncio

load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QueryRoute(Enum):
    ADMISSION = "admission"
    ACADEMIC = "academic"
    FACILITIES = "facilities"
    GENERAL = "general"

@dataclass
class SearchResult:
    content: str
    metadata: Dict[str, Any]
    score: float
    
@dataclass
class RAGConfig:
    chunk_size: int = 800
    chunk_overlap: int = 150
    search_limit: int = 5
    cache_ttl: int = 3600
    max_context_tokens: int = 4000
    similarity_threshold: float = 0.5

class RAGEngine:
    def __init__(self, config: RAGConfig = None):
        self.config = config or RAGConfig()
        
        # Initialize Google Gemini
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable is required")
        
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-2.0-flash')
        
        # Initialize tokenizer for context management
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        
        # Initialize sentence transformer for embeddings
        logger.info("Loading sentence transformer model...")
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        logger.info("Sentence transformer model loaded successfully")
        
        # Initialize FAISS safely
        self._init_faiss()
        
        # Initialize Redis for caching
        self._init_redis()
        
        # Initialize text splitter with configuration
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""]
        )
        
        # Semantic routing embeddings (precomputed for efficiency)
        self._init_semantic_routes()
    
    def _init_faiss(self):
        """Initialize FAISS with better error handling"""
        try:
            from database import get_faiss_index, get_faiss_documents, get_faiss_metadata
            self.faiss_index = get_faiss_index()
            self.faiss_documents = get_faiss_documents()
            self.faiss_metadata = get_faiss_metadata()
            logger.info(f"RAG Engine initialized with {len(self.faiss_documents)} documents")
        except Exception as e:
            logger.warning(f"FAISS not initialized: {e}")
            self.faiss_index = None
            self.faiss_documents = []
            self.faiss_metadata = []
    
    def _init_redis(self):
        """Initialize Redis with better error handling"""
        try:
            from database import get_redis
            self.redis_client = get_redis()
            # Test connection
            self.redis_client.ping()
            logger.info("Redis connection established")
        except Exception as e:
            logger.warning(f"Redis not available: {e}")
            self.redis_client = None
    
    def _init_semantic_routes(self):
        """Initialize semantic routing with embedding-based approach"""
        self.route_definitions = {
            QueryRoute.ADMISSION: [
                "tuyển sinh", "đăng ký nhập học", "hồ sơ xét tuyển", 
                "điểm chuẩn", "học phí", "thủ tục nhập học", "chỉ tiêu tuyển sinh",
                "phương thức tuyển sinh", "xét tuyển", "thi đánh giá năng lực",
                "thi tốt nghiệp THPT", "học bổng", "ký túc xá"
            ],
            QueryRoute.ACADEMIC: [
                "chương trình học", "môn học chuyên ngành", "giảng viên", 
                "lịch học", "thời khóa biểu", "đào tạo", "ngành học",
                "kỹ thuật phần mềm", "hệ thống thông tin", "khoa học máy tính",
                "mạng máy tính", "công nghệ thông tin", "an toàn thông tin",
                "trí tuệ nhân tạo", "khoa học dữ liệu"
            ],
            QueryRoute.FACILITIES: [
                "cơ sở vật chất", "thư viện trường", "phòng thí nghiệm", 
                "khuôn viên", "ký túc xá", "canteen", "tòa nhà giảng đường",
                "phòng học", "trung tâm dữ liệu", "nhà thi đấu"
            ],
            QueryRoute.GENERAL: [
                "giới thiệu trường", "lịch sử phát triển", "sứ mệnh tầm nhìn", 
                "liên hệ", "địa chỉ", "thành lập", "phát triển", "hợp tác quốc tế"
            ]
        }
        
        # Precompute route embeddings for faster routing
        self.route_embeddings = {}
        for route, examples in self.route_definitions.items():
            embeddings = self.embedding_model.encode(examples)
            self.route_embeddings[route] = np.mean(embeddings, axis=0)
    
    def _count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        return len(self.tokenizer.encode(text))
    
    def _ensure_faiss_initialized(self) -> bool:
        """Ensure FAISS is initialized, return success status"""
        if self.faiss_index is not None:
            return True
            
        try:
            # Try to initialize without async complications
            from database import get_faiss_index, get_faiss_documents, get_faiss_metadata
            self.faiss_index = get_faiss_index()
            self.faiss_documents = get_faiss_documents()
            self.faiss_metadata = get_faiss_metadata()
            logger.info(f"FAISS initialized with {len(self.faiss_documents)} documents")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize FAISS: {e}")
            return False
    
    def _get_cache_key(self, query: str, prefix: str = "rag_cache") -> str:
        """Generate cache key for query"""
        return f"{prefix}:{hashlib.md5(query.encode()).hexdigest()}"
    
    def chunk_document(self, content: str, metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Split document into chunks with enhanced metadata"""
        logger.info(f"Chunking document with {len(content)} characters")
        
        # Clean content
        content = content.strip()
        if not content:
            return []
        
        chunks = self.text_splitter.split_text(content)
        logger.info(f"Created {len(chunks)} chunks")
        
        chunk_data = []
        for i, chunk in enumerate(chunks):
            if len(chunk.strip()) < 50:  # Skip very short chunks
                continue
                
            chunk_metadata = {
                "chunk_index": i,
                "total_chunks": len(chunks),
                "source": metadata.get("source", "unknown") if metadata else "unknown",
                "category": metadata.get("category", "general") if metadata else "general",
                "uploaded_at": metadata.get("uploaded_at", datetime.utcnow().isoformat()) if metadata else datetime.utcnow().isoformat(),
                "chunk_size": len(chunk),
                "token_count": self._count_tokens(chunk)
            }
            
            chunk_data.append({
                "text": chunk.strip(),
                "metadata": chunk_metadata
            })
        
        logger.info(f"Final chunk count: {len(chunk_data)}")
        return chunk_data
    
    def add_document(self, content: str, metadata: Dict[str, Any] = None) -> int:
        """Add document to FAISS vector database with validation"""
        if not self._ensure_faiss_initialized():
            raise RuntimeError("Cannot add document: FAISS not available")
        
        if not content or len(content.strip()) < 100:
            raise ValueError("Document content too short or empty")
        
        if metadata is None:
            metadata = {}
        
        # Add timestamp if not provided
        if "uploaded_at" not in metadata:
            metadata["uploaded_at"] = datetime.utcnow().isoformat()
        
        logger.info(f"Adding document: {metadata.get('source', 'unknown')}")
        
        # Chunk the document
        chunks = self.chunk_document(content, metadata)
        
        if not chunks:
            logger.warning("No valid chunks created from document")
            return 0
        
        # Generate embeddings and add to FAISS
        texts = [chunk["text"] for chunk in chunks]
        metadatas = [chunk["metadata"] for chunk in chunks]
        
        try:
            # Generate embeddings
            logger.info("Generating embeddings...")
            embeddings = self.embedding_model.encode(texts, show_progress_bar=True)
            
            # Add to FAISS
            logger.info("Adding to FAISS...")
            from database import add_to_faiss
            add_to_faiss(embeddings, texts, metadatas)
            
            # Update local references
            self.faiss_documents.extend(texts)
            self.faiss_metadata.extend(metadatas)
            
            logger.info(f"Successfully added {len(chunks)} chunks to FAISS")
            return len(chunks)
            
        except Exception as e:
            logger.error(f"Error adding document to FAISS: {e}")
            raise
    
    def semantic_search(self, query: str, limit: int = None) -> List[SearchResult]:
        """Perform semantic search with improved caching and filtering"""
        if not self._ensure_faiss_initialized():
            logger.warning("FAISS not available, returning empty results")
            return []
        
        if limit is None:
            limit = self.config.search_limit
        
        # Check cache first
        if self.redis_client:
            cache_key = self._get_cache_key(query, "search")
            try:
                cached_result = self.redis_client.get(cache_key)
                if cached_result:
                    cached_data = json.loads(cached_result)
                    return [SearchResult(**item) for item in cached_data[:limit]]
            except Exception as e:
                logger.warning(f"Cache read error: {e}")
        
        # Generate query embedding
        query_embedding = self.embedding_model.encode([query])
        
        # Search in FAISS
        if self.faiss_index.ntotal == 0:
            logger.warning("FAISS index is empty")
            return []
        
        # Perform search with more results for filtering
        search_limit = min(limit * 2, self.faiss_index.ntotal)
        distances, indices = self.faiss_index.search(
            query_embedding.astype('float32'), 
            search_limit
        )
        
        # Format and filter results
        results = []
        for distance, idx in zip(distances[0], indices[0]):
            if idx < len(self.faiss_documents):
                # Convert distance to similarity score (lower distance = higher similarity)
                similarity_score = 1 / (1 + distance)
                
                # Filter by similarity threshold
                if similarity_score >= self.config.similarity_threshold:
                    results.append(SearchResult(
                        content=self.faiss_documents[idx],
                        metadata=self.faiss_metadata[idx],
                        score=similarity_score
                    ))
        
        # Sort by score and limit
        results.sort(key=lambda x: x.score, reverse=True)
        results = results[:limit]
        
        # Cache results
        if self.redis_client:
            try:
                cache_data = [asdict(result) for result in results]
                self.redis_client.setex(cache_key, self.config.cache_ttl, json.dumps(cache_data))
            except Exception as e:
                logger.warning(f"Cache write error: {e}")
        
        return results
    
    def semantic_route(self, query: str) -> QueryRoute:
        """Enhanced semantic routing using embeddings"""
        # Check cache first
        if self.redis_client:
            route_key = self._get_cache_key(query, "route")
            try:
                cached_route = self.redis_client.get(route_key)
                if cached_route:
                    return QueryRoute(cached_route.decode())
            except Exception as e:
                logger.warning(f"Route cache read error: {e}")
        
        # Generate query embedding
        query_embedding = self.embedding_model.encode([query])[0]
        
        # Calculate similarity with each route
        best_route = QueryRoute.GENERAL
        best_similarity = 0
        
        for route, route_embedding in self.route_embeddings.items():
            similarity = np.dot(query_embedding, route_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(route_embedding)
            )
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_route = route
        
        # Cache result
        if self.redis_client:
            try:
                self.redis_client.setex(route_key, self.config.cache_ttl, best_route.value)
            except Exception as e:
                logger.warning(f"Route cache write error: {e}")
        
        return best_route
    
    def _build_context(self, search_results: List[SearchResult]) -> Tuple[str, int]:
        """Build context string with token management"""
        context_parts = []
        total_tokens = 0
        seen_sources = set()
        
        for result in search_results:
            content_tokens = self._count_tokens(result.content)
            
            if total_tokens + content_tokens > self.config.max_context_tokens:
                break
            
            source = result.metadata.get('source', 'unknown')
            
            # Chỉ hiển thị nguồn nếu chưa thấy trước đó
            if source not in seen_sources:
                context_parts.append(f"[Nguồn: {source}]\n{result.content}")
                seen_sources.add(source)
            else:
                context_parts.append(result.content)
                
            total_tokens += content_tokens
        
        return "\n\n".join(context_parts), total_tokens
    
    def generate_response(self, query: str, search_results: List[SearchResult], chat_history: List[Dict[str, Any]] = None) -> str:
        """Generate response with improved context management"""
        # Build context with token management
        context_text, context_tokens = self._build_context(search_results)
        
        logger.info(f"Context built with {context_tokens} tokens from {len(search_results)} results")
        
        # Build system prompt
        system_prompt = """Bạn là trợ lý AI của Học viện Công nghệ Bưu chính Viễn thông - Đại học Quốc gia TP.HCM (PTITHCM). 
        
        Hướng dẫn trả lời:
        - Trả lời dựa trên thông tin được cung cấp
        - Nếu không có thông tin liên quan, nói rõ bạn không có thông tin về vấn đề đó
        - Luôn trả lời bằng tiếng Việt thân thiện và hữu ích
        - Có thể tham khảo cuộc trò chuyện trước đó nếu có
        - Nếu thông tin không đầy đủ, gợi ý người dùng liên hệ trực tiếp với học viện"""
        
        # Build conversation history (limit to manage tokens)
        conversation = []
        if chat_history:
            history_tokens = 0
            for msg in reversed(chat_history[-10:]):  # Last 10 messages, reversed
                msg_tokens = self._count_tokens(msg["content"])
                if history_tokens + msg_tokens > 1000:  # Limit history tokens
                    break
                conversation.insert(0, {
                    "role": msg["role"],
                    "parts": [msg["content"]]
                })
                history_tokens += msg_tokens
        
        # Add current query with context
        user_prompt = f"""Thông tin tham khảo:
{context_text}

Câu hỏi: {query}

Hãy trả lời dựa trên thông tin trên."""
        
        conversation.append({
            "role": "user", 
            "parts": [user_prompt]
        })
        
        # Generate response
        try:
            response = self.model.generate_content(
                conversation,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.7,
                    max_output_tokens=1000,
                )
            )
            return response.text
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return f"Xin lỗi, có lỗi xảy ra khi tạo câu trả lời. Vui lòng thử lại sau."
    
    def process_query(self, query: str, chat_history: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Process user query with enhanced pipeline"""
        if not query or len(query.strip()) < 3:
            return {
                "response": "Vui lòng nhập câu hỏi cụ thể hơn.",
                "sources": [],
                "route": QueryRoute.GENERAL.value,
                "metadata": {"error": "Query too short"}
            }
        
        try:
            # Semantic routing
            route = self.semantic_route(query)
            logger.info(f"Query routed to: {route.value}")
            
            # Search for relevant documents
            search_results = self.semantic_search(query)
            logger.info(f"Found {len(search_results)} relevant documents")
            
            # Generate response
            response = self.generate_response(query, search_results, chat_history)
            
            return {
                "response": response,
                "sources": [asdict(result) for result in search_results],
                "route": route.value,
                "metadata": {
                    "total_results": len(search_results),
                    "avg_score": np.mean([r.score for r in search_results]) if search_results else 0,
                    "timestamp": datetime.utcnow().isoformat()
                }
            }
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return {
                "response": "Xin lỗi, có lỗi xảy ra khi xử lý câu hỏi. Vui lòng thử lại sau.",
                "sources": [],
                "route": QueryRoute.GENERAL.value,
                "metadata": {"error": str(e)}
            }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get RAG engine statistics"""
        return {
            "total_documents": len(self.faiss_documents) if self.faiss_documents else 0,
            "faiss_initialized": self.faiss_index is not None,
            "redis_available": self.redis_client is not None,
            "config": asdict(self.config),
            "routes_available": [route.value for route in QueryRoute]
        }