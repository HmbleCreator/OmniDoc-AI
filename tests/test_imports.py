#!/usr/bin/env python3

import sys
import traceback

def test_imports():
    """Test all critical imports"""
    imports_to_test = [
        "chromadb",
        "fastapi",
        "uvicorn",
        "sentence_transformers",
        "langchain",
        "pdfplumber",
        "fitz",  # PyMuPDF
        "spacy",
        "nltk",
        "keybert",
        "yake",
        "sklearn",
        "rank_bm25",
        "google.generativeai",
        "anthropic",
        "llama_cpp",
        "textstat"
    ]
    
    failed_imports = []
    
    for module in imports_to_test:
        try:
            __import__(module)
            print(f"✅ {module}")
        except ImportError as e:
            print(f"❌ {module}: {e}")
            failed_imports.append(module)
        except Exception as e:
            print(f"⚠️ {module}: {e}")
            failed_imports.append(module)
    
    if failed_imports:
        print(f"\nFailed imports: {failed_imports}")
        return False
    else:
        print("\nAll imports successful!")
        return True

def test_chromadb():
    """Test ChromaDB specifically"""
    try:
        import chromadb
        print(f"ChromaDB version: {chromadb.__version__}")
        
        # Test the new client format
        client = chromadb.PersistentClient(path="./test_chroma_db")
        print("✅ ChromaDB PersistentClient created successfully")
        
        # Clean up
        import shutil
        import os
        if os.path.exists("./test_chroma_db"):
            shutil.rmtree("./test_chroma_db")
        
        return True
    except Exception as e:
        print(f"❌ ChromaDB test failed: {e}")
        traceback.print_exc()
        return False

def test_document_processor():
    """Test document processor import"""
    try:
        from services.document_processor import EnhancedDocumentProcessor
        print("✅ Document processor imported successfully")
        return True
    except Exception as e:
        print(f"❌ Document processor import failed: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Testing imports...")
    print("=" * 50)
    
    imports_ok = test_imports()
    print("\n" + "=" * 50)
    
    print("Testing ChromaDB...")
    chromadb_ok = test_chromadb()
    print("\n" + "=" * 50)
    
    print("Testing document processor...")
    processor_ok = test_document_processor()
    print("\n" + "=" * 50)
    
    if imports_ok and chromadb_ok and processor_ok:
        print("🎉 All tests passed! Backend should work.")
    else:
        print("❌ Some tests failed. Check the errors above.") 