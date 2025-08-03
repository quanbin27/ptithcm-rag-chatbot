# PTITHCM RAG System

A Retrieval-Augmented Generation (RAG) system for the Posts and Telecommunications Institute of Technology (PTITHCM), providing intelligent Q&A capabilities based on university documents and information.

## 🌟 Features

- **Intelligent Q&A**: AI-powered chatbot that answers questions about PTITHCM using institute documents
- **Document Management**: Upload and manage institute documents (admissions, academic info, facilities)
- **Semantic Search**: Advanced vector search using FAISS for accurate document retrieval
- **Multi-language Support**: Vietnamese interface with Vietnamese document processing
- **User Authentication**: Secure login/registration system with JWT tokens
- **Chat History**: Persistent chat sessions with conversation history
- **Real-time Responses**: Fast response generation using Google Gemini AI
- **Modern UI**: Beautiful, responsive Vue.js frontend with Element Plus components

## 🏗️ Architecture

### Backend (FastAPI)
- **FastAPI**: Modern Python web framework
- **Google Gemini AI**: For response generation
- **FAISS**: Vector similarity search for document retrieval
- **MongoDB**: User data and chat history storage
- **Redis**: Caching layer for improved performance
- **Sentence Transformers**: Document embedding generation

### Frontend (Vue.js)
- **Vue 3**: Progressive JavaScript framework
- **Element Plus**: UI component library
- **Vue Router**: Client-side routing
- **Pinia**: State management
- **Vite**: Build tool and dev server

## 📋 Prerequisites

- Python 3.8+
- Node.js 16+
- Docker and Docker Compose
- Google Gemini API key

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone <repository-url>
cd rag_ptit
```

### 2. Set Up Environment Variables
```bash
cp env.example .env
```

Edit `.env` file with your configuration:
```env
# Google Gemini API Key
GOOGLE_API_KEY=your_google_gemini_api_key_here

# Redis Configuration
REDIS_URL=redis://redis:6379

# MongoDB Configuration
MONGODB_URL=mongodb://admin:password123@mongodb:27017
MONGODB_DB=ptithcm_rag

# JWT Secret
JWT_SECRET_KEY=your-secret-key-change-in-production

# Environment
ENVIRONMENT=development
```

### 3. Start Infrastructure Services
```bash
docker-compose up -d
```

This starts:
- Redis (port 6379)
- MongoDB (port 27017)

### 4. Set Up Backend

#### Install Python Dependencies
```bash
cd backend
pip install -r requirements.txt
```

#### Initialize FAISS Index
```bash
python create_faiss_index.py
```

#### Start Backend Server
```bash
python main.py
```

The backend will be available at `http://localhost:8000`

### 5. Set Up Frontend

#### Install Node.js Dependencies
```bash
cd frontend
npm install
```

#### Start Development Server
```bash
npm run dev
```

The frontend will be available at `http://localhost:3000`

## 📚 Available Data

The system comes pre-loaded with PTITHCM institute information:

- **Admissions Information** (`ptithcm_tuyen_sinh.txt`): Admission methods, quotas, requirements, tuition fees
- **Institute Introduction** (`ptithcm_gioi_thieu.txt`): History, mission, facilities, academic programs

## 🔧 API Endpoints

### Authentication
- `POST /api/auth/register` - User registration
- `POST /api/auth/login` - User login
- `GET /api/auth/profile` - Get user profile

### Chat
- `POST /api/chat/send` - Send message and get AI response
- `GET /api/chat/sessions` - Get user's chat sessions
- `GET /api/chat/sessions/{session_id}/messages` - Get session messages
- `DELETE /api/chat/sessions/{session_id}` - Delete chat session

### Documents
- `POST /api/documents/upload` - Upload new document
- `GET /api/documents` - List all documents
- `DELETE /api/documents/{document_id}` - Delete document

## 🛠️ Development

### Project Structure
```
rag_ptit/
├── backend/                 # FastAPI backend
│   ├── routers/            # API route handlers
│   ├── models.py           # Pydantic models
│   ├── database.py         # Database connections
│   ├── rag_engine.py       # RAG processing logic
│   ├── auth.py             # Authentication logic
│   └── main.py             # FastAPI app entry point
├── frontend/               # Vue.js frontend
│   ├── src/
│   │   ├── views/          # Page components
│   │   ├── stores/         # Pinia stores
│   │   ├── services/       # API services
│   │   └── router/         # Vue Router
│   └── index.html
├── data/                   # Institute documents
├── faiss_data/            # Vector database files
└── docker-compose.yml     # Infrastructure services
```

### Adding New Documents

1. Place your document in the `data/` directory
2. Run the FAISS index creation script:
   ```bash
   python backend/create_faiss_index.py
   ```
3. The document will be automatically chunked and indexed

### Customizing the RAG Engine

The RAG engine supports:
- **Semantic Routing**: Automatically categorizes queries
- **Configurable Chunking**: Adjustable document chunk sizes
- **Caching**: Redis-based response caching
- **Similarity Thresholds**: Configurable relevance filtering

## 🔒 Security

- JWT-based authentication
- Password hashing with bcrypt
- CORS configuration
- Input validation with Pydantic
- Environment variable configuration

## 📊 Performance

- **Vector Search**: FAISS for fast similarity search
- **Caching**: Redis for response and search result caching
- **Async Processing**: Non-blocking I/O operations
- **Token Management**: Efficient context management

## 🐳 Docker Deployment

### Production Docker Setup
```bash
# Build and run with Docker Compose
docker-compose -f docker-compose.prod.yml up -d
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License.

## 🆘 Support

For support and questions:
- Create an issue in the repository
- Contact the development team
- Check the documentation

## 🔄 Updates

The system is designed to be easily extensible:
- Add new document types
- Implement additional AI models
- Extend the chat interface
- Add new data sources

---

**Built with ❤️ for PTITHCM** 