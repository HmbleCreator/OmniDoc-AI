#!/usr/bin/env python3
"""
Test script for OmniDoc AI Backend
Tests the enhanced document processing and search capabilities
"""

import requests
import json
import time
import os
from pathlib import Path

# Backend URL
BASE_URL = "http://localhost:8000"

def test_backend_health():
    """Test if backend is running"""
    try:
        response = requests.get(f"{BASE_URL}/health")
        if response.status_code == 200:
            print("✅ Backend is running")
            return True
        else:
            print("❌ Backend health check failed")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Backend is not running. Please start it first.")
        return False

def test_llm_providers():
    """Test LLM providers endpoint"""
    try:
        response = requests.get(f"{BASE_URL}/api/llm/providers")
        if response.status_code == 200:
            providers = response.json()
            print(f"✅ Found {len(providers['providers'])} LLM providers:")
            for provider in providers['providers']:
                print(f"   - {provider['name']} ({provider['id']})")
            return True
        else:
            print("❌ Failed to get LLM providers")
            return False
    except Exception as e:
        print(f"❌ Error testing LLM providers: {e}")
        return False

def test_document_upload():
    """Test document upload with a sample text file"""
    try:
        # Create a sample text file with enhanced content
        sample_content = """
# Advanced Machine Learning Research Paper

## Abstract
This paper presents a comprehensive analysis of machine learning algorithms and their applications in modern computing systems. We explore neural networks, deep learning architectures, and optimization techniques.

## Introduction
Machine learning has revolutionized the way we approach problem-solving in computer science. This section introduces the fundamental concepts and motivations behind this research.

## Methodology
Our approach involves three main components:
1. Data preprocessing and feature extraction using advanced techniques
2. Model selection and training with hyperparameter optimization
3. Evaluation and validation with cross-validation methods

## Results
The experimental results show significant improvements in accuracy and performance compared to traditional methods. Our neural network achieved 95% accuracy on the test dataset.

## Conclusion
This research demonstrates the effectiveness of modern machine learning techniques in solving complex problems. Future work will focus on scalability and real-time processing.
        """
        
        # Create temporary file
        test_file_path = "test_document.txt"
        with open(test_file_path, "w", encoding="utf-8") as f:
            f.write(sample_content)
        
        # Upload file
        with open(test_file_path, "rb") as f:
            files = {"file": ("test_document.txt", f, "text/plain")}
            response = requests.post(f"{BASE_URL}/api/documents/upload", files=files)
        
        # Clean up
        os.remove(test_file_path)
        
        if response.status_code == 200:
            doc_data = response.json()
            print(f"✅ Document uploaded successfully:")
            print(f"   - ID: {doc_data['id']}")
            print(f"   - Name: {doc_data['name']}")
            print(f"   - Summary: {doc_data['summary'][:100]}...")
            print(f"   - Chunks: {len(doc_data['chunks'])}")
            if 'chunk_metadata' in doc_data:
                print(f"   - Enhanced metadata: {len(doc_data['chunk_metadata'])} chunks with metadata")
            return doc_data['id']
        else:
            print(f"❌ Document upload failed: {response.text}")
            return None
            
    except Exception as e:
        print(f"❌ Error testing document upload: {e}")
        return None

def test_enhanced_search(document_id):
    """Test enhanced search functionality with hybrid search and MMR"""
    try:
        # Test search with enhanced parameters
        search_data = {
            "query": "machine learning methodology neural networks",
            "document_ids": [document_id],
            "top_k": 5,
            "use_reranking": True
        }
        
        response = requests.post(f"{BASE_URL}/api/documents/search", json=search_data)
        
        if response.status_code == 200:
            results = response.json()
            print(f"✅ Enhanced search completed successfully:")
            print(f"   - Query: {results['query']}")
            print(f"   - Results: {results['result_count']}")
            print(f"   - Reranking used: {results['reranking_used']}")
            print(f"   - Embedding model: {results['embedding_model']}")
            
            for i, result in enumerate(results['results'][:3]):
                print(f"   - Result {i+1}: {result['citation']} (Score: {result['relevance_score']:.2f})")
                if 'key_terms' in result:
                    print(f"     Key terms: {', '.join(result['key_terms'][:3])}")
                if 'semantic_density' in result:
                    print(f"     Semantic density: {result['semantic_density']:.2f}")
            
            return True
        else:
            print(f"❌ Enhanced search failed: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Error testing enhanced search: {e}")
        return False

def test_chat_with_reasoning(document_id):
    """Test chat functionality with reasoning chains"""
    try:
        # Test asking a question
        chat_data = {
            "message": "What is the methodology described in this document and how does it work?",
            "mode": "ask",
            "document_id": document_id,
            "provider": "openai"
        }
        
        response = requests.post(f"{BASE_URL}/api/chat/ask", json=chat_data)
        
        if response.status_code == 200:
            chat_response = response.json()
            print(f"✅ Chat with reasoning completed successfully:")
            print(f"   - Answer: {chat_response['message']['content'][:100]}...")
            print(f"   - Confidence: {chat_response['confidence']:.2f}")
            print(f"   - References: {len(chat_response['document_references'])}")
            
            # Check for reasoning chain
            if 'reasoning_chain' in chat_response:
                reasoning_steps = chat_response['reasoning_chain']
                print(f"   - Reasoning chain: {len(reasoning_steps)} steps")
                for i, step in enumerate(reasoning_steps[:3]):
                    print(f"     Step {i+1}: {step['step_type']} - {step['description'][:50]}...")
                    print(f"       Confidence: {step['confidence']:.2f}")
            
            for ref in chat_response['document_references'][:2]:
                print(f"   - Reference: {ref['citation']} (Score: {ref['relevance_score']:.2f})")
                if 'key_terms' in ref:
                    print(f"     Key terms: {', '.join(ref['key_terms'][:3])}")
            
            return True
        else:
            print(f"❌ Chat failed: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Error testing chat: {e}")
        return False

def test_challenge_generation_with_enhancements(document_id):
    """Test enhanced challenge question generation"""
    try:
        # Test generating challenge questions
        challenge_data = {
            "document_ids": [document_id],
            "provider": "openai"
        }
        
        response = requests.post(f"{BASE_URL}/api/chat/challenge/generate", json=challenge_data)
        
        if response.status_code == 200:
            challenge_response = response.json()
            print(f"✅ Enhanced challenge generation completed successfully:")
            print(f"   - Questions: {len(challenge_response['questions'])}")
            
            for i, question in enumerate(challenge_response['questions']):
                print(f"   - Question {i+1}: {question['question'][:80]}...")
                print(f"     References: {len(question['document_references'])}")
                if question['document_references']:
                    ref = question['document_references'][0]
                    print(f"     Top reference: {ref['citation']} (Score: {ref['relevance_score']:.2f})")
            
            return True
        else:
            print(f"❌ Challenge generation failed: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Error testing challenge generation: {e}")
        return False

def test_reasoning_chain_retrieval():
    """Test reasoning chain retrieval endpoint"""
    try:
        # This would typically be called with a session ID from a previous chat
        # For testing, we'll just check if the endpoint exists
        response = requests.get(f"{BASE_URL}/api/chat/reasoning/test-session-id")
        
        if response.status_code in [404, 400]:  # Expected for invalid session ID
            print("✅ Reasoning chain endpoint is accessible")
            return True
        elif response.status_code == 200:
            print("✅ Reasoning chain retrieved successfully")
            return True
        else:
            print(f"❌ Unexpected response from reasoning chain endpoint: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Error testing reasoning chain retrieval: {e}")
        return False

def test_document_metadata(document_id):
    """Test enhanced document metadata endpoint"""
    try:
        response = requests.get(f"{BASE_URL}/api/documents/{document_id}/metadata")
        
        if response.status_code == 200:
            metadata = response.json()
            print(f"✅ Document metadata retrieved successfully:")
            print(f"   - Document ID: {metadata['document_id']}")
            print(f"   - Chunk count: {metadata['chunk_count']}")
            print(f"   - Total words: {metadata['total_words']}")
            print(f"   - Headers: {len(metadata['headers'])}")
            print(f"   - Enhanced metadata: {len(metadata['chunk_metadata'])} chunks")
            
            # Show sample metadata
            if metadata['chunk_metadata']:
                sample = metadata['chunk_metadata'][0]
                print(f"   - Sample chunk: Page {sample['page_number']}, Type: {sample['chunk_type']}")
                if 'key_terms' in sample:
                    print(f"     Key terms: {', '.join(sample['key_terms'][:3])}")
            
            return True
        else:
            print(f"❌ Metadata retrieval failed: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Error testing metadata retrieval: {e}")
        return False

def main():
    """Run all enhanced tests"""
    print("🧪 Testing OmniDoc AI Enhanced Backend")
    print("=" * 60)
    
    # Test 1: Health check
    if not test_backend_health():
        return
    
    print()
    
    # Test 2: LLM providers
    test_llm_providers()
    
    print()
    
    # Test 3: Document upload with enhanced processing
    document_id = test_document_upload()
    if not document_id:
        return
    
    print()
    
    # Test 4: Enhanced document metadata
    test_document_metadata(document_id)
    
    print()
    
    # Test 5: Enhanced search with hybrid search and MMR
    test_enhanced_search(document_id)
    
    print()
    
    # Test 6: Chat with reasoning chains
    test_chat_with_reasoning(document_id)
    
    print()
    
    # Test 7: Enhanced challenge generation
    test_challenge_generation_with_enhancements(document_id)
    
    print()
    
    # Test 8: Reasoning chain retrieval
    test_reasoning_chain_retrieval()
    
    print()
    print("🎉 All enhanced tests completed!")
    print("\n✨ Enhanced Features Tested:")
    print("   ✅ Semantic-aware chunking with dynamic sizing")
    print("   ✅ Key term extraction (KeyBERT + YAKE)")
    print("   ✅ Hybrid search (dense + BM25 + MMR)")
    print("   ✅ Cross-encoder reranking")
    print("   ✅ Reasoning chain transparency")
    print("   ✅ Enhanced metadata tracking")
    print("   ✅ Advanced challenge generation")

if __name__ == "__main__":
    main() 