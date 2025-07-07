#!/usr/bin/env python3
"""
Optimized Backend Startup Script for OmniDoc AI

This script starts the backend with optimized model loading:
1. Starts the FastAPI server immediately
2. Pre-loads models in the background
3. Shows startup progress and logs

Usage:
    python start_backend.py
"""

import asyncio
import threading
import time
import uvicorn
import logging
import sys
from pathlib import Path

# Add backend directory to path
backend_dir = Path(__file__).parent
sys.path.insert(0, str(backend_dir))

# Configure logging to show all messages
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

def preload_models():
    """Pre-load all models in background thread"""
    try:
        print("\n🔄 Pre-loading AI models in background...")
        print("   (Server is already running - you can use the app now!)")
        
        # Import and initialize document processor
        from services.document_processor import EnhancedDocumentProcessor
        
        # This will trigger lazy loading of all models
        processor = EnhancedDocumentProcessor()
        
        # Pre-load each model with progress indicators
        print("   📥 Loading embedding model...")
        start_time = time.time()
        _ = processor.embedding_model
        elapsed = time.time() - start_time
        print(f"   ✅ Embedding model loaded ({elapsed:.2f}s)")
        
        print("   📥 Loading cross-encoder...")
        start_time = time.time()
        _ = processor.cross_encoder
        elapsed = time.time() - start_time
        print(f"   ✅ Cross-encoder loaded ({elapsed:.2f}s)")
        
        print("   📥 Loading KeyBERT...")
        start_time = time.time()
        _ = processor.keybert
        elapsed = time.time() - start_time
        print(f"   ✅ KeyBERT loaded ({elapsed:.2f}s)")
        
        print("   📥 Loading YAKE...")
        start_time = time.time()
        _ = processor.yake_extractor
        elapsed = time.time() - start_time
        print(f"   ✅ YAKE loaded ({elapsed:.2f}s)")
        
        total_time = time.time() - start_time
        print(f"\n🎉 All models pre-loaded successfully!")
        print(f"   Total model loading time: {total_time:.2f}s")
        print(f"   All AI features are now fully available!")
        
    except Exception as e:
        print(f"\n⚠️  Model pre-loading failed: {e}")
        print("   Models will load on first use instead")

def start_backend():
    """Start the FastAPI backend with full logging"""
    print("\n🚀 Starting OmniDoc AI Backend...")
    print("   FastAPI server starting immediately...")
    print("   Models will load in background")
    print("   Access the app at: http://localhost:8000")
    print("   API docs at: http://localhost:8000/docs")
    print("\n" + "="*60)
    
    # Start uvicorn server with full logging
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,  # Disable reload for production-like startup
        log_level="info",
        access_log=True,
        use_colors=True
    )

def main():
    """Main function"""
    print("⚡ OmniDoc AI - Optimized Backend Startup")
    print("="*60)
    print("   Fast startup with full logging enabled")
    print("   Models load in background for immediate server access")
    print("="*60)
    
    # Start model pre-loading in background thread
    model_thread = threading.Thread(target=preload_models, daemon=True)
    model_thread.start()
    
    # Give models a moment to start loading
    time.sleep(1)
    
    # Start the backend
    start_backend()

if __name__ == "__main__":
    main() 