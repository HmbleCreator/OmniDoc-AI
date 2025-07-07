#!/usr/bin/env python3
"""
Model Pre-download Script for OmniDoc AI

This script downloads all required AI models and embeddings before running the backend.
This ensures the backend starts quickly without needing to download models at runtime.

Usage:
    python download_models.py
"""

import os
import sys
import time
from pathlib import Path

def print_step(step_num: int, description: str):
    """Print a formatted step message"""
    print(f"\n{'='*60}")
    print(f"STEP {step_num}: {description}")
    print(f"{'='*60}")

def download_sentence_transformers():
    """Download SentenceTransformer models"""
    print_step(1, "Downloading SentenceTransformer Models")
    
    try:
        from sentence_transformers import SentenceTransformer
        
        models = [
            "intfloat/e5-large-v2",      # Main embedding model
            "all-MiniLM-L6-v2",          # KeyBERT model
            "cross-encoder/ms-marco-MiniLM-L-6-v2"  # Cross-encoder for reranking
        ]
        
        for model_name in models:
            print(f"\n📥 Downloading: {model_name}")
            start_time = time.time()
            
            model = SentenceTransformer(model_name)
            
            # Test the model with a simple query
            test_text = "This is a test sentence for model verification."
            embeddings = model.encode(test_text)
            
            elapsed = time.time() - start_time
            print(f"✅ {model_name} - Downloaded and tested successfully ({elapsed:.2f}s)")
            print(f"   Embedding dimension: {len(embeddings)}")
            
    except Exception as e:
        print(f"❌ Error downloading SentenceTransformer models: {e}")
        return False
    
    return True

def download_spacy_model():
    """Download spaCy model"""
    print_step(2, "Downloading spaCy Model")
    
    try:
        import spacy
        
        # Check if model is already installed
        try:
            nlp = spacy.load("en_core_web_sm")
            print("✅ spaCy model 'en_core_web_sm' already installed")
            return True
        except OSError:
            pass
        
        print("📥 Installing spaCy model 'en_core_web_sm'...")
        os.system("python -m spacy download en_core_web_sm")
        
        # Verify installation
        nlp = spacy.load("en_core_web_sm")
        test_text = "This is a test sentence."
        doc = nlp(test_text)
        print(f"✅ spaCy model installed and tested successfully")
        print(f"   Processed {len(doc)} tokens")
        
    except Exception as e:
        print(f"❌ Error downloading spaCy model: {e}")
        return False
    
    return True

def download_nltk_data():
    """Download NLTK data"""
    print_step(3, "Downloading NLTK Data")
    
    try:
        import nltk
        
        # Download required NLTK data
        nltk_data = [
            "punkt",
            "stopwords",
            "averaged_perceptron_tagger",
            "wordnet"
        ]
        
        for data_name in nltk_data:
            print(f"📥 Downloading NLTK data: {data_name}")
            nltk.download(data_name, quiet=True)
            print(f"✅ {data_name} downloaded")
        
        # Test NLTK functionality
        from nltk.tokenize import word_tokenize
        from nltk.corpus import stopwords
        
        test_text = "This is a test sentence for NLTK verification."
        tokens = word_tokenize(test_text)
        stops = set(stopwords.words('english'))
        
        print(f"✅ NLTK tested successfully")
        print(f"   Tokens: {tokens}")
        print(f"   Stop words loaded: {len(stops)} words")
        
    except Exception as e:
        print(f"❌ Error downloading NLTK data: {e}")
        return False
    
    return True

def test_chromadb():
    """Test ChromaDB functionality"""
    print_step(4, "Testing ChromaDB Setup")
    
    try:
        import chromadb
        
        # Test ChromaDB client creation
        test_path = "./test_chroma_db"
        client = chromadb.PersistentClient(path=test_path)
        
        # Create a test collection
        collection = client.create_collection(name="test_collection")
        
        # Add some test data
        collection.add(
            documents=["This is a test document"],
            metadatas=[{"source": "test"}],
            ids=["test_id"]
        )
        
        # Query the collection
        results = collection.query(
            query_texts=["test document"],
            n_results=1
        )
        
        print(f"✅ ChromaDB tested successfully")
        print(f"   Found {len(results['documents'][0])} documents")
        
        # Clean up test data
        import shutil
        if os.path.exists(test_path):
            shutil.rmtree(test_path)
        
    except Exception as e:
        print(f"❌ Error testing ChromaDB: {e}")
        return False
    
    return True

def verify_llm_providers():
    """Verify LLM provider packages"""
    print_step(5, "Verifying LLM Provider Packages")
    
    providers = {
        "openai": "openai",
        "anthropic": "anthropic", 
        "google": "google.generativeai",
        "llama": "llama_cpp"
    }
    
    all_ok = True
    
    for provider, module in providers.items():
        try:
            __import__(module)
            print(f"✅ {provider.capitalize()} package available")
        except ImportError:
            print(f"⚠️  {provider.capitalize()} package not installed")
            all_ok = False
    
    return all_ok

def main():
    """Main function to run all downloads"""
    print("🚀 OmniDoc AI - Model Pre-download Script")
    print("This script will download all required AI models and verify the setup.")
    print("\nNote: This may take several minutes depending on your internet connection.")
    
    # Create cache directory if it doesn't exist
    cache_dir = Path.home() / ".cache" / "omnidoc_ai"
    cache_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n📁 Cache directory: {cache_dir}")
    
    success_count = 0
    total_steps = 5
    
    # Run all download steps
    if download_sentence_transformers():
        success_count += 1
    
    if download_spacy_model():
        success_count += 1
    
    if download_nltk_data():
        success_count += 1
    
    if test_chromadb():
        success_count += 1
    
    if verify_llm_providers():
        success_count += 1
    
    # Summary
    print(f"\n{'='*60}")
    print(f"📊 DOWNLOAD SUMMARY")
    print(f"{'='*60}")
    print(f"✅ Successful: {success_count}/{total_steps}")
    print(f"❌ Failed: {total_steps - success_count}/{total_steps}")
    
    if success_count == total_steps:
        print(f"\n🎉 All models downloaded successfully!")
        print(f"   Your OmniDoc AI backend is ready to start.")
        print(f"   Run: cd backend && uvicorn main:app --reload --host 0.0.0.0 --port 8000")
    else:
        print(f"\n⚠️  Some downloads failed. Please check the errors above.")
        print(f"   You may need to install missing packages or check your internet connection.")
    
    print(f"\n💡 Tip: Models are cached locally and won't be downloaded again on subsequent runs.")

if __name__ == "__main__":
    main() 