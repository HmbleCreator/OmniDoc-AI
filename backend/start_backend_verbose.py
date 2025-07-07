#!/usr/bin/env python3
"""
Verbose Backend Startup Script for OmniDoc AI

This script starts the backend with maximum logging detail:
1. Shows all startup logs and model loading progress
2. Displays detailed timing information
3. Provides comprehensive debugging information

Usage:
    python start_backend_verbose.py
"""

import asyncio
import threading
import time
import uvicorn
import logging
import sys
import os
from pathlib import Path

# Add backend directory to path
backend_dir = Path(__file__).parent
sys.path.insert(0, str(backend_dir))

# Configure verbose logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

# Enable all relevant loggers
logging.getLogger('uvicorn').setLevel(logging.INFO)
logging.getLogger('fastapi').setLevel(logging.INFO)
logging.getLogger('sentence_transformers').setLevel(logging.INFO)
logging.getLogger('chromadb').setLevel(logging.INFO)
logging.getLogger('services').setLevel(logging.DEBUG)

def preload_models_verbose():
    """Pre-load all models with detailed logging"""
    try:
        print("\n" + "="*80)
        print("🔄 PRE-LOADING AI MODELS (VERBOSE MODE)")
        print("="*80)
        print("   Server is already running - you can use the app now!")
        print("   Models are loading in background for optimal performance...")
        print("="*80)
        
        # Import and initialize document processor
        print("\n📦 Importing document processor...")
        from services.document_processor import EnhancedDocumentProcessor
        
        print("🔧 Initializing document processor...")
        processor = EnhancedDocumentProcessor()
        
        # Pre-load each model with detailed timing
        models_to_load = [
            ("embedding_model", "Main embedding model (intfloat/e5-large-v2)"),
            ("cross_encoder", "Cross-encoder for reranking"),
            ("keybert", "KeyBERT for key term extraction"),
            ("yake_extractor", "YAKE for keyword extraction")
        ]
        
        total_start_time = time.time()
        
        for model_name, description in models_to_load:
            print(f"\n📥 Loading {description}...")
            print(f"   Model: {model_name}")
            start_time = time.time()
            
            try:
                # Access the model property to trigger loading
                model = getattr(processor, model_name)
                elapsed = time.time() - start_time
                print(f"   ✅ {description} loaded successfully")
                print(f"   ⏱️  Loading time: {elapsed:.2f} seconds")
                
                # Show model info if available
                if hasattr(model, 'model_name'):
                    print(f"   📋 Model name: {model.model_name}")
                if hasattr(model, 'max_seq_length'):
                    print(f"   📏 Max sequence length: {model.max_seq_length}")
                    
            except Exception as e:
                elapsed = time.time() - start_time
                print(f"   ❌ Failed to load {description}")
                print(f"   ⏱️  Time before failure: {elapsed:.2f} seconds")
                print(f"   🔍 Error: {e}")
        
        total_time = time.time() - total_start_time
        print(f"\n" + "="*80)
        print(f"🎉 MODEL LOADING COMPLETE")
        print(f"="*80)
        print(f"   Total loading time: {total_time:.2f} seconds")
        print(f"   Average per model: {total_time/len(models_to_load):.2f} seconds")
        print(f"   All AI features are now fully available!")
        print(f"   Server has been running since the beginning!")
        print(f"="*80)
        
    except Exception as e:
        print(f"\n❌ CRITICAL ERROR in model pre-loading:")
        print(f"   Error: {e}")
        print(f"   Models will load on first use instead")
        print(f"   Check the logs above for more details")

def start_backend_verbose():
    """Start the FastAPI backend with maximum logging"""
    print("\n" + "="*80)
    print("🚀 STARTING OMNIDOC AI BACKEND (VERBOSE MODE)")
    print("="*80)
    print("   FastAPI server starting immediately...")
    print("   Models will load in background with detailed progress")
    print("   Access the app at: http://localhost:8000")
    print("   API docs at: http://localhost:8000/docs")
    print("   Health check at: http://localhost:8000/health")
    print("="*80)
    
    # Start uvicorn server with maximum logging
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="debug",
        access_log=True,
        use_colors=True,
        log_config=None  # Use our custom logging config
    )

def main():
    """Main function"""
    print("🔍 OmniDoc AI - Verbose Backend Startup")
    print("="*80)
    print("   Maximum logging enabled for debugging and monitoring")
    print("   Fast startup with detailed model loading progress")
    print("   All logs will be displayed in real-time")
    print("="*80)
    
    # Show system info
    print(f"\n💻 System Information:")
    print(f"   Python version: {sys.version}")
    print(f"   Working directory: {os.getcwd()}")
    print(f"   Backend directory: {backend_dir}")
    
    # Start model pre-loading in background thread
    model_thread = threading.Thread(target=preload_models_verbose, daemon=True)
    model_thread.start()
    
    # Give models a moment to start loading
    time.sleep(1)
    
    # Start the backend
    start_backend_verbose()

if __name__ == "__main__":
    main() 