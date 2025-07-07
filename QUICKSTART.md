# 🚀 OmniDoc AI - Quick Start Guide

Get OmniDoc AI running in 5 minutes!

## ⚡ Super Quick Setup

### Option 1: Automated Setup (Recommended)
```bash
# Clone and setup everything automatically
git clone <repository-url>
cd project
python setup.py
```

### Option 2: Manual Setup
```bash
# 1. Install backend dependencies
cd backend
pip install -r requirements.txt

# 2. Download AI models (this takes a few minutes)
python download_models.py

# 3. Install frontend dependencies
cd ../frontend
npm install

# 4. Start the backend (Fast startup with logs)
cd ../backend
python start_backend.py

# 5. Start the frontend (in a new terminal)
cd ../frontend
npm run dev
```

## 🎯 What You'll Get

- **Frontend**: http://localhost:5173
- **Backend API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs

## 📝 First Steps

1. **Upload a Document**: Drag and drop a PDF or TXT file
2. **Ask Questions**: Type questions about your document
3. **Try Challenge Mode**: Switch to "Challenge Me" for comprehension questions
4. **Configure API Keys**: Add your LLM API keys in the sidebar (optional)

## 🔑 API Keys (Optional)

You can use OmniDoc AI without API keys for basic functionality, or add them for enhanced AI responses:

- **OpenAI**: https://platform.openai.com/
- **Google Gemini**: https://makersuite.google.com/
- **Anthropic Claude**: https://console.anthropic.com/

## 🐛 Common Issues

| Issue | Solution |
|-------|----------|
| Backend won't start | Run `python download_models.py` in backend directory |
| Frontend can't connect | Make sure backend is running on port 8000 |
| Models downloading slowly | This is normal on first run - they're cached after download |
| Port already in use | Change port: `uvicorn main:app --port 8001` |

## 📚 Next Steps

- Read the full [README.md](README.md) for detailed documentation
- Check the [troubleshooting section](README.md#troubleshooting) if you have issues
- Explore the [API documentation](http://localhost:8000/docs) when the backend is running

---

**Need help?** Check the troubleshooting section in the main README or open an issue on GitHub. 