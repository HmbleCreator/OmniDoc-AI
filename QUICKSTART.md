# 🚀 OmniDoc AI - Quick Start Guide

## 1. Backend Setup
```bash
cd backend
pip install -r requirements.txt
python download_models.py
python main.py  # Starts backend at http://localhost:8000
```

## 2. Frontend Setup
```bash
cd frontend
npm install
npm run dev  # Starts frontend at http://localhost:5173
```

## 3. Usage
- Upload PDF/TXT documents
- View auto-summary
- Use "Ask Anything" for Q&A
- Use "Challenge Me" for logic-based questions
- All answers include references to the document

## 4. API Keys
- Add your LLM API keys in the sidebar (frontend) or in `backend/.env`

## 5. Running Tests
```bash
cd tests
pytest  # or python test_backend.py
```

## 6. Troubleshooting
| Issue                  | Solution                                      |
|------------------------|-----------------------------------------------|
| Backend won't start    | Run `python download_models.py` in backend    |
| Frontend can't connect | Make sure backend is running on port 8000     |
| Model download slow    | Wait for first run; models are cached         |
| Port already in use    | Change port in `main.py` or frontend config   |
| ChromaDB errors        | Delete `chroma_db/` and restart backend       |

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

## 📚 Next Steps

- Read the full [README.md](README.md) for detailed documentation
- Check the [troubleshooting section](README.md#troubleshooting) if you have issues
- Explore the [API documentation](http://localhost:8000/docs) when the backend is running

---

**Need help?** Check the troubleshooting section in the main README or open an issue on GitHub. 