# 🚀 OmniDoc AI - Smart Document Assistant

A powerful, dark-themed local web application for uploading PDF/TXT documents and interacting with them via AI. Features advanced semantic processing, hybrid search, and multi-provider LLM support.

## 🌟 Features

- **📄 Document Processing**: Support for PDF and TXT files with advanced text extraction
- **🧠 AI-Powered Q&A**: Deep document understanding with detailed reasoning chains
- **🔍 Hybrid Search**: Combines semantic and keyword-based search for better results
- **🎯 Challenge Mode**: Generate and evaluate comprehension questions
- **📚 Multiple LLM Providers**: OpenAI, Google Gemini, Anthropic Claude, and local models
- **🎨 Modern UI**: Dark-themed, responsive interface with modular components
- **🔗 Citations & References**: Track sources and provide document references
- **⚡ Fast Startup**: Pre-downloaded models for quick initialization

## 🏗️ Architecture

```
OmniDoc AI/
├── frontend/          # React + TypeScript + Vite
│   ├── src/
│   │   ├── components/    # Modular UI components
│   │   ├── context/       # React context for state management
│   │   └── services/      # API service layer
│   └── package.json
├── backend/           # FastAPI + Python
│   ├── api/           # API routes and models
│   ├── services/      # Core business logic
│   ├── requirements.txt
│   └── download_models.py  # Model pre-download script
└── README.md
```

## 🚀 Quick Start

### Prerequisites

- **Python 3.8+** with pip
- **Node.js 16+** with npm
- **Git**

### Option 1: Automated Setup (Recommended)

```bash
git clone <repository-url>
cd project
python setup.py
```

This will automatically:
- Install all dependencies
- Download AI models
- Create configuration files
- Test the setup

### Option 2: Manual Setup

### 1. Clone the Repository

```bash
git clone <repository-url>
cd project
```

### 2. Backend Setup

#### Install Dependencies
```bash
cd backend
pip install -r requirements.txt
```

#### Pre-download AI Models (Recommended)
This step downloads all required AI models and embeddings, ensuring fast backend startup:

```bash
python download_models.py
```

**What this does:**
- Downloads SentenceTransformer models (1.34GB total)
- Installs spaCy language model
- Downloads NLTK data
- Tests ChromaDB setup
- Verifies LLM provider packages

**Why pre-download?**
- Models are large (1-2GB total) and take time to download
- Prevents slow startup on first run
- Ensures all dependencies are working
- Models are cached locally for future use

#### Start the Backend

**Option 1: Fast Startup with Logs (Recommended)**
```bash
python start_backend.py
```
- Starts server immediately (~5 seconds)
- Loads models in background with progress logs
- Shows timing information for each model
- Best for development

**Option 2: Verbose Startup (Debug Mode)**
```bash
python start_backend_verbose.py
```
- Maximum logging detail
- Shows all startup logs and model loading progress
- Detailed timing and debugging information
- Best for troubleshooting

**Option 3: Standard Startup**
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```
- Loads all models before starting server
- Takes 2-3 minutes on first run
- Models cached for subsequent runs

**Option 4: Production Mode**
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 1
```
- No auto-reload
- Optimized for production

The backend will be available at `http://localhost:8000`

### 3. Frontend Setup

#### Install Dependencies
```bash
cd frontend
npm install
```

#### Start the Frontend
```bash
npm run dev
```

The frontend will be available at `http://localhost:5173`

## 🔧 Configuration

### API Keys

The application supports multiple LLM providers. You can configure API keys in two ways:

#### 1. Frontend Configuration (Recommended)
- Open the sidebar in the web interface
- Enter your API keys for the providers you want to use
- Keys are stored locally in your browser

#### 2. Environment Variables
Create a `.env` file in the backend directory:

```env
# OpenAI
OPENAI_API_KEY=your_openai_key_here

# Google Gemini
GOOGLE_API_KEY=your_google_key_here

# Anthropic Claude
ANTHROPIC_API_KEY=your_anthropic_key_here
```

### Supported LLM Providers

| Provider | Models | Requires API Key |
|----------|--------|------------------|
| OpenAI | GPT-3.5, GPT-4, GPT-4 Turbo | Yes |
| Google Gemini | Gemini Pro, Gemini Pro Vision | Yes |
| Anthropic Claude | Claude 3 Haiku, Sonnet, Opus | Yes |
| Local LLM | Llama 2, Mistral (via llama-cpp) | No |

## 📖 Usage

### 1. Upload Documents
- Drag and drop PDF or TXT files into the upload area
- Or click to browse and select files
- Documents are processed with semantic analysis

### 2. Ask Questions
- Switch to "Ask Anything" mode
- Type your question about the uploaded documents
- Get detailed answers with reasoning chains and citations

### 3. Challenge Mode
- Switch to "Challenge Me" mode
- The AI generates comprehension questions
- Test your understanding of the documents

### 4. View References
- Click on citations to see source documents
- View reasoning chains for AI responses
- Track confidence scores and metadata

## 🛠️ Development

### Project Structure

```
frontend/src/
├── components/           # React components
│   ├── ChatArea.tsx     # Main chat interface
│   ├── FileUpload.tsx   # Document upload
│   ├── Message.tsx      # Individual messages
│   ├── Sidebar.tsx      # Settings and API keys
│   └── SummaryCards.tsx # Document summaries
├── context/
│   └── AppContext.tsx   # Global state management
└── services/
    └── api.ts          # API communication

backend/
├── api/
│   ├── models.py       # Pydantic models
│   └── routes/         # API endpoints
├── services/
│   ├── document_processor.py  # Document processing
│   └── llm_service.py         # LLM integration
└── main.py             # FastAPI application
```

### Key Technologies

#### Frontend
- **React 18** with TypeScript
- **Vite** for fast development
- **Tailwind CSS** for styling
- **Lucide React** for icons

#### Backend
- **FastAPI** for API framework
- **ChromaDB** for vector database
- **SentenceTransformers** for embeddings
- **LangChain** for LLM orchestration
- **PyMuPDF** for PDF processing

### Adding New Features

1. **New LLM Provider**: Add to `llm_service.py` and update models
2. **New Document Type**: Extend `document_processor.py`
3. **New UI Component**: Create in `frontend/src/components/`
4. **New API Endpoint**: Add to `backend/api/routes/`

## 🐛 Troubleshooting

### Common Issues

#### Backend Won't Start
1. **ChromaDB Error**: Delete `chroma_db/` directory and restart
2. **Model Download Issues**: Run `python download_models.py`
3. **Port Already in Use**: Change port in uvicorn command

#### Frontend Issues
1. **API Connection**: Check backend is running on port 8000
2. **Build Errors**: Clear `node_modules` and reinstall
3. **CORS Issues**: Verify backend CORS configuration

#### Model Loading Issues
1. **Slow Startup**: Models are downloading - wait or pre-download
2. **Memory Issues**: Close other applications to free RAM
3. **Network Issues**: Check internet connection for model downloads

### Performance Tips

- **Pre-download models** using `download_models.py`
- **Use SSD storage** for faster model loading
- **Allocate sufficient RAM** (8GB+ recommended)
- **Close unnecessary applications** during first run

## 📝 API Documentation

Once the backend is running, visit:
- **Interactive API Docs**: `http://localhost:8000/docs`
- **Alternative Docs**: `http://localhost:8000/redoc`

### Key Endpoints

- `GET /health` - Health check
- `POST /api/documents/upload` - Upload documents
- `POST /api/chat/ask` - Ask questions
- `POST /api/chat/challenge` - Generate challenges
- `GET /api/llm/providers` - List LLM providers

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **HuggingFace** for transformer models
- **ChromaDB** for vector database
- **FastAPI** for the web framework
- **React** and **Vite** for the frontend

---

**Need help?** Check the troubleshooting section or open an issue on GitHub. 