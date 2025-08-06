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
import traceback

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
    search_limit: int = 10
    cache_ttl: int = 3600
    max_context_tokens: int = 5000
    similarity_threshold: float = 0.55

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
    
    @classmethod
    async def create(cls, config: RAGConfig = None):
        """Async factory method to create RAGEngine"""
        instance = cls(config)
        await instance._init_faiss_async()
        return instance
    
    def _init_faiss(self):
        """Initialize FAISS with better error handling"""
        try:
            from database import get_faiss_index, get_faiss_documents, get_faiss_metadata
            logger.info("🔄 Initializing FAISS...")
            
            self.faiss_index = get_faiss_index()
            logger.info(f"✅ FAISS index loaded: {self.faiss_index.ntotal} vectors")
            
            self.faiss_documents = get_faiss_documents()
            logger.info(f"✅ Documents loaded: {len(self.faiss_documents)} documents")
            
            self.faiss_metadata = get_faiss_metadata()
            logger.info(f"✅ Metadata loaded: {len(self.faiss_metadata)} metadata entries")
            
            # Validate data consistency
            if len(self.faiss_documents) != len(self.faiss_metadata):
                logger.error(f"Data inconsistency: {len(self.faiss_documents)} documents vs {len(self.faiss_metadata)} metadata")
                raise ValueError("Documents and metadata count mismatch")
            
            if self.faiss_index.ntotal == 0:
                logger.error("FAISS index is empty")
                raise ValueError("FAISS index is empty")
            
            if self.faiss_index.ntotal != len(self.faiss_documents):
                logger.error(f"Index-document mismatch: {self.faiss_index.ntotal} vectors vs {len(self.faiss_documents)} documents")
                raise ValueError("FAISS index and documents count mismatch")
                
        except Exception as e:
            logger.error(f"Failed to initialize FAISS: {e}")
            self.faiss_index = None
            self.faiss_documents = None
            self.faiss_metadata = None
    
    async def _init_faiss_async(self):
        """Async initialization of FAISS"""
        try:
            from database import init_db, get_faiss_index, get_faiss_documents, get_faiss_metadata
            logger.info("🔄 Initializing FAISS (async)...")
            
            # Initialize database first
            await init_db()
            logger.info("✅ Database initialized")
            
            self.faiss_index = get_faiss_index()
            logger.info(f"✅ FAISS index loaded: {self.faiss_index.ntotal} vectors")
            
            self.faiss_documents = get_faiss_documents()
            logger.info(f"✅ Documents loaded: {len(self.faiss_documents)} documents")
            
            self.faiss_metadata = get_faiss_metadata()
            logger.info(f"✅ Metadata loaded: {len(self.faiss_metadata)} metadata entries")
            
            # Validate data consistency
            if len(self.faiss_documents) != len(self.faiss_metadata):
                logger.error(f"Data inconsistency: {len(self.faiss_documents)} documents vs {len(self.faiss_metadata)} metadata")
                raise ValueError("Documents and metadata count mismatch")
            
            if self.faiss_index.ntotal == 0:
                logger.error("FAISS index is empty")
                raise ValueError("FAISS index is empty")
            
            if self.faiss_index.ntotal != len(self.faiss_documents):
                logger.error(f"Index-document mismatch: {self.faiss_index.ntotal} vectors vs {len(self.faiss_documents)} documents")
                raise ValueError("FAISS index and documents count mismatch")
                
        except Exception as e:
            logger.error(f"Failed to initialize FAISS: {e}")
            self.faiss_index = None
            self.faiss_documents = None
            self.faiss_metadata = None
    
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
        """Initialize semantic routing with embedding-based and keyword-based approach"""
        # Định nghĩa từ khóa đặc trưng cho từng category dựa trên data thực tế
        self.route_keywords = {
            QueryRoute.ADMISSION: [
                "tuyển sinh", "xét tuyển", "điểm chuẩn", "hồ sơ", "nhập học", "phương thức", "chỉ tiêu", "lịch tuyển sinh", "kênh thanh toán", "hotline tuyển sinh", "quy chế tuyển sinh", "mức điểm", "đăng ký xét tuyển", "học phí tuyển sinh", "thông báo tuyển sinh", "mã ngành", "tổ hợp xét tuyển", "ưu tiên tuyển sinh", "hướng dẫn tuyển sinh", "thông tin tuyển sinh", "thời gian tuyển sinh", "điều kiện xét tuyển", "kết quả tuyển sinh", "thủ tục nhập học", "xác nhận nhập học", "chuyển trường", "chuyển ngành"
            ],
            QueryRoute.ACADEMIC: [
                "quy chế học vụ", "chương trình học", "môn học", "tín chỉ", "giảng viên", "lịch học", "thời khóa biểu", "đào tạo", "ngành học", "học phí", "học bổng", "bảo lưu", "chuyển lớp", "chuyển ngành", "đăng ký học phần", "điểm thi", "thi lại", "học lại", "tốt nghiệp", "thực tập", "hướng dẫn sinh viên", "hoạt động sinh viên", "nghĩa vụ sinh viên", "quyền lợi sinh viên", "nội quy sinh viên", "quy định sinh viên", "học vụ", "học tập", "rèn luyện", "khen thưởng", "kỷ luật", "hỗ trợ sinh viên", "thủ tục sinh viên"
            ],
            QueryRoute.GENERAL: [
                "giới thiệu trường", "lịch sử phát triển", "sứ mệnh", "tầm nhìn", "giá trị cốt lõi", "liên hệ", "địa chỉ", "thành lập", "phát triển", "hợp tác quốc tế", "thành tích", "ngành đào tạo", "mô hình đào tạo", "chiến lược phát triển", "logo", "triết lý giáo dục", "thông tin liên hệ", "cơ sở đào tạo", "quy mô đào tạo", "thành tựu", "giải thưởng", "truyền thống", "tầm nhìn 2030", "sứ mạng", "mục tiêu", "mô hình quản trị", "hội nhập quốc tế"
            ]
        }
        # Semantic route definitions (dùng cho embedding)
        self.route_definitions = self.route_keywords
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
    
    def semantic_search(self, query: str, limit: int = None, category_filter: str = None) -> List[SearchResult]:
        """
        Perform semantic search with hybrid approach (semantic + keyword matching)
        """
        if not self._ensure_faiss_initialized():
            logger.warning("FAISS not available, returning empty results")
            return []
        
        limit = limit or self.config.search_limit
        search_limit = limit * 5  # Get more candidates for filtering
        
        logger.info(f"Searching for: '{query}' (limit: {limit}, category_filter: {category_filter})")
        
        # Check cache first
        cache_key = f"{query}_{category_filter}" if category_filter else query
        if self.redis_client:
            cache_key = self._get_cache_key(cache_key, "search")
            try:
                cached_result = self.redis_client.get(cache_key)
                if cached_result:
                    cached_data = json.loads(cached_result)
                    logger.info(f"Returning {len(cached_data)} cached results")
                    return [SearchResult(**item) for item in cached_data[:limit]]
            except Exception as e:
                logger.warning(f"Cache read error: {e}")
        
        # Generate query embedding
        query_embedding = self.embedding_model.encode([query])
        logger.info(f"Query embedding shape: {query_embedding.shape}")
        
        # Perform FAISS search
        logger.info(f"Searching top {search_limit} results from {self.faiss_index.ntotal} total vectors")
        D, I = self.faiss_index.search(query_embedding, search_limit)
        
        # Convert to list of (distance, index) pairs
        search_results = list(zip(D[0], I[0]))
        logger.info(f"Raw search results: {len(search_results)} items")
        
        # Hybrid scoring: combine semantic similarity with keyword matching
        hybrid_results = []
        query_lower = query.lower()
        query_words = set(query_lower.split())
        
        for distance, idx in search_results:
            if idx >= len(self.faiss_documents):
                continue
                
            document = self.faiss_documents[idx]
            metadata = self.faiss_metadata[idx] if idx < len(self.faiss_metadata) else {}
            
            # Semantic score (normalized)
            semantic_score = float(distance)
            
            # Keyword matching score
            doc_lower = document.lower()
            keyword_matches = sum(1 for word in query_words if word in doc_lower)
            exact_phrase_match = query_lower in doc_lower
            
            # Boost score for keyword matches
            keyword_boost = 0.0
            if exact_phrase_match:
                keyword_boost = 0.3  # Strong boost for exact phrase
            elif keyword_matches > 0:
                keyword_boost = 0.1 * keyword_matches  # Moderate boost per keyword
            
            # Hybrid score
            hybrid_score = semantic_score + keyword_boost
            
            # Apply category filter if specified
            if category_filter and metadata.get("category") != category_filter:
                continue
            
            hybrid_results.append(SearchResult(
                content=document,
                metadata=metadata,
                score=hybrid_score
            ))
        
        # Sort by hybrid score (descending)
        hybrid_results.sort(key=lambda x: x.score, reverse=True)
        
        # Apply similarity threshold
        filtered_results = [
            result for result in hybrid_results 
            if result.score >= self.config.similarity_threshold
        ]
        
        # Limit results
        final_results = filtered_results[:limit]
        
        logger.info(f"Category matches: {len([r for r in hybrid_results if category_filter and r.metadata.get('category') == category_filter])}, "
                   f"Similarity matches: {len(filtered_results)}, Final results: {len(final_results)}")
        
        # Log top results for debugging
        for i, result in enumerate(hybrid_results[:3]):
            logger.info(f"Top result {i+1}: score={result.score:.3f}, category={result.metadata.get('category', 'unknown')}")
            logger.info(f"Content preview: {result.content[:100]}...")
        
        # Cache results
        if self.redis_client:
            try:
                cache_data = [asdict(result) for result in final_results]
                self.redis_client.setex(cache_key, self.config.cache_ttl, json.dumps(cache_data))
            except Exception as e:
                logger.warning(f"Cache write error: {e}")
        
        return final_results
    
    def semantic_route(self, query: str) -> QueryRoute:
        """Keyword-prioritized semantic routing"""
        # Ưu tiên match keyword trước
        for route, keywords in self.route_keywords.items():
            for kw in keywords:
                if kw.lower() in query.lower():
                    return route
        # Nếu không match keyword thì dùng semantic
        query_embedding = self.embedding_model.encode([query])[0]
        best_route = QueryRoute.GENERAL
        best_similarity = 0
        for route, route_embedding in self.route_embeddings.items():
            similarity = np.dot(query_embedding, route_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(route_embedding)
            )
            if similarity > best_similarity:
                best_similarity = similarity
                best_route = route
        return best_route
    
    def semantic_route_multi(self, query: str) -> list:
        """Trả về danh sách các category phù hợp nhất với query (ưu tiên keyword, có thể nhiều category)"""
        matched_routes = []
        for route, keywords in self.route_keywords.items():
            for kw in keywords:
                if kw.lower() in query.lower():
                    matched_routes.append(route)
                    break  # Không cần check tiếp keyword cùng category
        if matched_routes:
            return matched_routes
        # Nếu không match keyword, dùng semantic (chỉ lấy best route)
        query_embedding = self.embedding_model.encode([query])[0]
        best_route = QueryRoute.GENERAL
        best_similarity = 0
        for route, route_embedding in self.route_embeddings.items():
            similarity = np.dot(query_embedding, route_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(route_embedding)
            )
            if similarity > best_similarity:
                best_similarity = similarity
                best_route = route
        return [best_route]

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
        """Process user query with enhanced pipeline (multi-category support)"""
        if not query or len(query.strip()) < 3:
            return {
                "response": "Vui lòng nhập câu hỏi cụ thể hơn.",
                "sources": [],
                "route": QueryRoute.GENERAL.value,
                "metadata": {"error": "Query too short"}
            }
        try:
            # Multi-category routing
            routes = self.semantic_route_multi(query)
            logger.info(f"Query routed to: {[r.value for r in routes]}")
            # Map route(s) to category filter(s)
            category_filters = []
            for route in routes:
                if route == QueryRoute.ADMISSION:
                    category_filters.append("admission")
                elif route == QueryRoute.ACADEMIC:
                    category_filters.append("academic")
            # GENERAL không filter để lấy kết quả rộng
            # Nếu có nhiều category, tìm kiếm lần lượt và gộp kết quả (ưu tiên không trùng lặp)
            search_results = []
            seen_doc_ids = set()
            for cat in category_filters:
                results = self.semantic_search(query, category_filter=cat)
                for r in results:
                    doc_id = r.metadata.get("chunk_index", None)  # hoặc id khác nếu có
                    if doc_id is None or doc_id not in seen_doc_ids:
                        search_results.append(r)
                        if doc_id is not None:
                            seen_doc_ids.add(doc_id)
            logger.info(f"Found {len(search_results)} relevant documents (category filters: {category_filters})")
            # Nếu không có kết quả ở các category này thì fallback sang GENERAL
            if not search_results and category_filters:
                logger.info("No results in specific categories, fallback to GENERAL")
                search_results = self.semantic_search(query, category_filter=None)
                logger.info(f"Found {len(search_results)} documents in GENERAL fallback")
            # Nếu vẫn không có kết quả, thử với threshold thấp hơn
            if not search_results:
                logger.info("No results found, trying with lower similarity threshold")
                original_threshold = self.config.similarity_threshold
                self.config.similarity_threshold = 0.1
                search_results = self.semantic_search(query)
                self.config.similarity_threshold = original_threshold
                logger.info(f"Found {len(search_results)} documents with lower threshold")
            # Generate response
            if search_results:
                response = self.generate_response(query, search_results, chat_history)
            else:
                response = f"Xin lỗi, tôi không tìm thấy thông tin liên quan đến '{query}' trong cơ sở dữ liệu. Vui lòng thử lại với từ khóa khác hoặc liên hệ trực tiếp với học viện để được hỗ trợ."
            return {
                "response": response,
                "sources": [asdict(result) for result in search_results],
                "route": [r.value for r in routes],
                "category_filter": category_filters,
                "metadata": {
                    "total_results": len(search_results),
                    "avg_score": np.mean([r.score for r in search_results]) if search_results else 0,
                    "timestamp": datetime.utcnow().isoformat()
                }
            }
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            logger.error(traceback.format_exc())
            return {
                "response": "Xin lỗi, có lỗi xảy ra khi xử lý câu hỏi. Vui lòng thử lại sau.",
                "sources": [],
                "route": QueryRoute.GENERAL.value,
                "metadata": {"error": str(e)}
            }
    
    def test_faiss_search(self, query: str) -> Dict[str, Any]:
        """Simple test function to check if FAISS search works"""
        logger.info(f"=== TESTING FAISS SEARCH FOR: '{query}' ===")
        
        if not self._ensure_faiss_initialized():
            return {"error": "FAISS not initialized"}
        
        try:
            # Simple search without any filtering
            query_embedding = self.embedding_model.encode([query])
            distances, indices = self.faiss_index.search(
                query_embedding.astype('float32'), 
                10  # Get top 10
            )
            
            results = []
            for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
                if idx < len(self.faiss_documents):
                    similarity = 1 / (1 + distance)
                    results.append({
                        "rank": i + 1,
                        "index": idx,
                        "distance": float(distance),
                        "similarity": float(similarity),
                        "content": self.faiss_documents[idx][:200] + "..."
                    })
            
            return {
                "query": query,
                "total_vectors": self.faiss_index.ntotal,
                "total_documents": len(self.faiss_documents),
                "results": results
            }
            
        except Exception as e:
            logger.error(f"Test search error: {e}")
            return {"error": str(e)}
    
    def debug_search(self, query: str) -> Dict[str, Any]:
        """Debug search functionality"""
        logger.info(f"=== DEBUG SEARCH FOR: '{query}' ===")
        
        # Check FAISS status
        if not self._ensure_faiss_initialized():
            return {"error": "FAISS not initialized"}
        
        logger.info(f"FAISS index has {self.faiss_index.ntotal} vectors")
        logger.info(f"Documents count: {len(self.faiss_documents)}")
        logger.info(f"Metadata count: {len(self.faiss_metadata)}")
        
        # Check category distribution
        category_counts = {}
        for meta in self.faiss_metadata:
            cat = meta.get("category", "unknown")
            category_counts[cat] = category_counts.get(cat, 0) + 1
        
        logger.info(f"Category distribution: {category_counts}")
        
        # Test raw search without filters
        try:
            query_embedding = self.embedding_model.encode([query])
            distances, indices = self.faiss_index.search(
                query_embedding.astype('float32'), 
                20  # Get top 20 results
            )
            
            logger.info(f"Raw search returned {len(indices[0])} results")
            
            # Show top 5 raw results
            raw_results = []
            for i, (distance, idx) in enumerate(zip(distances[0][:5], indices[0][:5])):
                if idx < len(self.faiss_documents):
                    similarity = 1 / (1 + distance)
                    raw_results.append({
                        "rank": i + 1,
                        "index": idx,
                        "distance": distance,
                        "similarity": similarity,
                        "category": self.faiss_metadata[idx].get("category", "unknown"),
                        "content_preview": self.faiss_documents[idx][:100] + "..."
                    })
            
            return {
                "faiss_status": "OK",
                "total_vectors": self.faiss_index.ntotal,
                "category_distribution": category_counts,
                "raw_search_results": raw_results
            }
            
        except Exception as e:
            logger.error(f"Debug search error: {e}")
            return {"error": str(e)}
    
    def get_stats(self) -> Dict[str, Any]:
        """Get RAG engine statistics"""
        return {
            "total_documents": len(self.faiss_documents) if self.faiss_documents else 0,
            "faiss_initialized": self.faiss_index is not None,
            "redis_available": self.redis_client is not None,
            "config": asdict(self.config),
            "routes_available": [route.value for route in QueryRoute]
        }