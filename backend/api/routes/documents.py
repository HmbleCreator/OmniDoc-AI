from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, Query, Request
from fastapi.responses import JSONResponse
import os
import tempfile
import uuid
from typing import List, Optional
from datetime import datetime

from api.models import DocumentResponse, DocumentUpload
from services.document_processor import EnhancedDocumentProcessor

router = APIRouter()

# Initialize enhanced document processor
doc_processor = EnhancedDocumentProcessor()

# In-memory storage for documents (in production, use a database)
documents_db = {}

def extract_api_keys(request: Request) -> dict:
    """Extract API keys from request headers"""
    return {
        'openai': request.headers.get('X-OpenAI-Key', ''),
        'gemini': request.headers.get('X-Gemini-Key', ''),
        'claude': request.headers.get('X-Claude-Key', ''),
        'local': request.headers.get('X-Local-Path', '')
    }

@router.post("/upload", response_model=DocumentResponse)
async def upload_document(file: UploadFile = File(...), request: Request = None, provider: str = Query('openai')):
    """Upload and process a document (PDF or TXT) with enhanced metadata"""
    
    # Extract API keys from headers if available
    api_keys = extract_api_keys(request) if request else {}
    
    # Validate file type
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")
    
    file_extension = os.path.splitext(file.filename)[1].lower()
    if file_extension not in ['.pdf', '.txt']:
        raise HTTPException(
            status_code=400, 
            detail="Unsupported file type. Only PDF and TXT files are supported."
        )
    
    # Validate file size (max 10MB)
    if file.size and file.size > 10 * 1024 * 1024:
        raise HTTPException(
            status_code=400,
            detail="File too large. Maximum size is 10MB."
        )
    
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        # Process the document with enhanced metadata and API keys for summary
        doc_data = doc_processor.process_document(temp_file_path, file.filename, api_keys, provider=provider)
        
        # Store in memory (in production, save to database)
        documents_db[doc_data["id"]] = {
            "id": doc_data["id"],
            "name": doc_data["name"],
            "type": doc_data["type"],
            "content": doc_data["content"],
            "summary": doc_data["summary"],
            "upload_time": doc_data["upload_time"],
            "chunks": doc_data["chunks"],
            "chunk_metadata": doc_data["chunk_metadata"],
            "headers": doc_data["headers"]
        }
        
        # Clean up temporary file
        os.unlink(temp_file_path)
        
        response_data = DocumentResponse(
            id=doc_data["id"],
            name=doc_data["name"],
            type=doc_data["type"],
            content=doc_data["content"],
            summary=doc_data["summary"],
            upload_time=doc_data["upload_time"],
            chunks=doc_data["chunks"]
        )
        
        print(f"✅ Document uploaded successfully: {doc_data['name']}")
        print(f"   ID: {doc_data['id']}")
        print(f"   Summary: {doc_data['summary'][:100]}...")
        
        return response_data
        
    except Exception as e:
        # Clean up temporary file if it exists
        if 'temp_file_path' in locals():
            try:
                os.unlink(temp_file_path)
            except:
                pass
        
        raise HTTPException(
            status_code=500,
            detail=f"Error processing document: {str(e)}"
        )

@router.get("/", response_model=List[DocumentResponse])
async def list_documents():
    """Get list of all uploaded documents"""
    documents = []
    for doc_id, doc_data in documents_db.items():
        documents.append(DocumentResponse(
            id=doc_data["id"],
            name=doc_data["name"],
            type=doc_data["type"],
            content=doc_data["content"],
            summary=doc_data["summary"],
            upload_time=doc_data["upload_time"],
            chunks=doc_data["chunks"]
        ))
    
    return documents

@router.get("/{document_id}", response_model=DocumentResponse)
async def get_document(document_id: str):
    """Get a specific document by ID with enhanced metadata"""
    if document_id not in documents_db:
        raise HTTPException(status_code=404, detail="Document not found")
    
    doc_data = documents_db[document_id]
    return DocumentResponse(
        id=doc_data["id"],
        name=doc_data["name"],
        type=doc_data["type"],
        content=doc_data["content"],
        summary=doc_data["summary"],
        upload_time=doc_data["upload_time"],
        chunks=doc_data["chunks"]
    )

@router.get("/{document_id}/metadata")
async def get_document_metadata(document_id: str):
    """Get enhanced metadata for a document"""
    if document_id not in documents_db:
        raise HTTPException(status_code=404, detail="Document not found")
    
    doc_data = documents_db[document_id]
    return {
        "document_id": document_id,
        "chunk_metadata": doc_data.get("chunk_metadata", []),
        "headers": doc_data.get("headers", []),
        "chunk_count": len(doc_data["chunks"]),
        "total_words": sum(len(chunk.split()) for chunk in doc_data["chunks"])
    }

@router.delete("/{document_id}")
async def delete_document(document_id: str):
    """Delete a document and its associated data"""
    if document_id not in documents_db:
        raise HTTPException(status_code=404, detail="Document not found")
    
    try:
        # Remove from vector database
        doc_processor.delete_document(document_id)
        
        # Remove from memory
        del documents_db[document_id]
        
        return {"message": "Document deleted successfully"}
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error deleting document: {str(e)}"
        )

@router.get("/{document_id}/chunks")
async def get_document_chunks(document_id: str):
    """Get all chunks for a specific document with enhanced metadata"""
    if document_id not in documents_db:
        raise HTTPException(status_code=404, detail="Document not found")
    
    try:
        chunks = doc_processor.get_document_chunks(document_id)
        return {
            "document_id": document_id,
            "chunks": chunks,
            "chunk_count": len(chunks)
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving document chunks: {str(e)}"
        )

@router.post("/search")
async def search_documents(
    query: str, 
    document_ids: Optional[List[str]] = Query(None),
    top_k: int = Query(5, ge=1, le=20),
    use_reranking: bool = Query(True)
):
    """Search for relevant chunks across documents with enhanced reranking"""
    try:
        if use_reranking:
            results = doc_processor.search_with_reranking(query, document_ids, top_k)
        else:
            # Fallback to basic search
            results = doc_processor.search_similar_chunks(query, document_ids, top_k)
        
        return {
            "query": query,
            "results": results,
            "result_count": len(results),
            "reranking_used": use_reranking,
            "embedding_model": doc_processor.embedding_model_name
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error searching documents: {str(e)}"
        )

@router.get("/search/suggestions")
async def get_search_suggestions(document_id: Optional[str] = None):
    """Get search suggestions based on document content"""
    try:
        suggestions = []
        
        if document_id and document_id in documents_db:
            # Get headers and key phrases from specific document
            doc_data = documents_db[document_id]
            headers = doc_data.get("headers", [])
            
            for header in headers[:5]:  # Top 5 headers
                suggestions.append({
                    "type": "header",
                    "text": f"What is {header['text']}?",
                    "source": header['text']
                })
        else:
            # Get suggestions from all documents
            for doc_data in documents_db.values():
                headers = doc_data.get("headers", [])
                for header in headers[:2]:  # Top 2 headers per doc
                    suggestions.append({
                        "type": "header",
                        "text": f"What is {header['text']}?",
                        "source": f"{doc_data['name']}: {header['text']}"
                    })
        
        return {
            "suggestions": suggestions[:10],  # Limit to 10 suggestions
            "total_documents": len(documents_db)
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error generating search suggestions: {str(e)}"
        ) 