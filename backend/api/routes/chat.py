import json
from fastapi import APIRouter, HTTPException, Body, Query, Request
from typing import List, Dict, Any, Optional
import uuid
from datetime import datetime
import dataclasses
import traceback

from api.models import (
    ChatRequest, ChatResponse, ChatMessage, 
    ChallengeRequest, ChallengeResponse, ChallengeQuestion,
    ChallengeAnswer, ChallengeEvaluation, ChallengeEvaluationRequest,
    DocumentReference
)
from services.llm_service import LLMService
from services.document_processor import EnhancedDocumentProcessor, ReasoningStep

router = APIRouter()

# Initialize document processor (LLM service will be initialized per request)
doc_processor = EnhancedDocumentProcessor()

# In-memory storage for sessions (in production, use a database)
sessions_db = {}

def extract_api_keys(request: Request) -> Dict[str, str]:
    """Extract API keys from request headers"""
    return {
        'openai': request.headers.get('X-OpenAI-Key', ''),
        'gemini': request.headers.get('X-Gemini-Key', ''),
        'claude': request.headers.get('X-Claude-Key', ''),
        'local': request.headers.get('X-Local-Path', '')
    }

@router.post("/ask", response_model=ChatResponse)
async def ask_question(request: ChatRequest, http_request: Request):
    try:
        # Extract API keys from headers
        api_keys = extract_api_keys(http_request)
        llm_service = LLMService(api_keys)
        session_id = str(uuid.uuid4())
        reasoning_steps = []

        # Step 1: Query Analysis
        analysis_step = ReasoningStep(
            step_id=str(uuid.uuid4()),
            step_type="analysis",
            description="Analyzing user query for key concepts and intent",
            input_data={"query": request.message},
            output_data={"key_terms": extract_key_terms(request.message)},
            confidence=0.9,
            timestamp=datetime.now()
        )
        reasoning_steps.append(analysis_step)

        # Step 2: Document Retrieval with Hybrid Search
        context_chunks = []
        citations = []
        if request.document_id:
            search_results = doc_processor.hybrid_search_with_mmr(
                request.message, [request.document_id], top_k=5, diversity_weight=0.3
            )
        else:
            search_results = doc_processor.hybrid_search_with_mmr(
                request.message, top_k=5, diversity_weight=0.3
            )
        context_chunks = [result["chunk"] for result in search_results]
        citations = [result["citation"] for result in search_results]

        retrieval_step = ReasoningStep(
            step_id=str(uuid.uuid4()),
            step_type="retrieval",
            description="Retrieved relevant document chunks using hybrid search (dense + BM25 + MMR)",
            input_data={"query": request.message, "document_id": request.document_id},
            output_data={
                "chunks_retrieved": len(context_chunks),
                "search_method": "hybrid_mmr",
                "top_citations": citations[:3]
            },
            confidence=0.85,
            timestamp=datetime.now()
        )
        reasoning_steps.append(retrieval_step)

        if not context_chunks:
            no_context_step = ReasoningStep(
                step_id=str(uuid.uuid4()),
                step_type="evaluation",
                description="No relevant document context found",
                input_data={"query": request.message},
                output_data={"reason": "No matching content in uploaded documents"},
                confidence=1.0,
                timestamp=datetime.now()
            )
            reasoning_steps.append(no_context_step)
            doc_processor.log_reasoning_chain(session_id, reasoning_steps)
            reasoning_chain = [
                {k: str(v) for k, v in step.__dict__.items()}
                for step in reasoning_steps
            ]
            return ChatResponse(
                message=ChatMessage(
                    id=str(uuid.uuid4()),
                    type="ai",
                    content="No relevant document context found. Please upload a document or try a different question.",
                    timestamp=datetime.now(),
                    reasoning="No document context available for the question.",
                    document_ref=None,
                    show_reasoning=False
                ),
                reasoning="No document context available for the question.",
                document_references=[],
                confidence=0.0,
                reasoning_chain=reasoning_chain
            )

        # Step 3: Context Synthesis
        synthesis_step = ReasoningStep(
            step_id=str(uuid.uuid4()),
            step_type="synthesis",
            description="Synthesizing retrieved context for answer generation",
            input_data={"context_chunks": len(context_chunks)},
            output_data={"synthesis_method": "context_aggregation"},
            confidence=0.8,
            timestamp=datetime.now()
        )
        reasoning_steps.append(synthesis_step)

        # Step 4: Get AI response with enhanced context
        response = llm_service.ask_question(
            question=request.message,
            context=context_chunks,
            provider=request.provider.value if hasattr(request.provider, 'value') else str(request.provider)
        )
        print("[API Debug] llm_service.ask_question response:", response)

        # Step 5: Response Evaluation
        evaluation_step = ReasoningStep(
            step_id=str(uuid.uuid4()),
            step_type="evaluation",
            description="Evaluating response quality and relevance",
            input_data={"response_length": len(response["answer"] or "")},
            output_data={
                "confidence": response.get("confidence", 0.8),
                "reasoning_quality": "high" if response.get("reasoning") else "medium"
            },
            confidence=response.get("confidence", 0.8),
            timestamp=datetime.now()
        )
        reasoning_steps.append(evaluation_step)

        # Debug: Print top 3 search_results before building enhanced_references
        print("[DEBUG] Top 3 search_results:", [
            {
                'citation': r.get('citation'),
                'relevance_score': r.get('relevance_score'),
                'chunk_preview': r.get('chunk', '')[:100]
            } for r in search_results[:3]
        ])
        # Build enhanced references directly from the top search results (no deduplication)
        enhanced_references = []
        for result in search_results[:3]:
            raw_key_terms = result.get("key_terms", [])
            if isinstance(raw_key_terms, str):
                key_terms = [k.strip() for k in raw_key_terms.split(",") if k.strip()]
            else:
                key_terms = raw_key_terms
            enhanced_references.append(DocumentReference(
                citation=result["citation"],
                relevance_score=result["relevance_score"],
                chunk_preview=result["chunk"][:200] + "..." if len(result["chunk"]) > 200 else result["chunk"],
                key_terms=key_terms
            ))
        enhanced_references_dicts = [ref.dict() for ref in enhanced_references]
        # Debug: Print constructed enhanced_references_dicts
        print("[DEBUG] enhanced_references_dicts:", enhanced_references_dicts)

        # Prepare document_ref for ChatMessage
        try:
            if enhanced_references_dicts and "citation" in enhanced_references_dicts[0]:
                document_ref_val = enhanced_references_dicts[0]["citation"]
            else:
                document_ref_val = None
        except Exception as e:
            print("[API ERROR] Exception preparing documentRef:", e)
            traceback.print_exc()
            document_ref_val = None

        # Construct chat_message
        try:
            chat_message = ChatMessage(
                id=str(uuid.uuid4()),
                type=response.get("type", "ai"),
                content=response.get("answer", ""),
                timestamp=datetime.now(),
                reasoning=response.get("reasoning", ""),
                document_ref=document_ref_val,
                show_reasoning=False
            )
        except Exception as e:
            print("[API ERROR] Exception in chat_message construction:", e)
            print("[API ERROR] Args passed:", {
                'type': response.get("type", "ai"),
                'content': response.get("answer", ""),
                'documentRef': document_ref_val,
                'reasoning': response.get("reasoning", "")
            })
            traceback.print_exc()
            chat_message = ChatMessage(
                id=str(uuid.uuid4()),
                type="ai",
                content="An error occurred while constructing the chat message.",
                timestamp=datetime.now(),
                reasoning="",
                document_ref=None,
                show_reasoning=False
            )

        # Ensure confidence is a float
        try:
            confidence_val = float(response.get("confidence", 0.7))
        except Exception:
            confidence_val = 0.7

        doc_processor.log_reasoning_chain(session_id, reasoning_steps)
        reasoning_chain = [
            {k: str(v) for k, v in step.__dict__.items()}
            for step in reasoning_steps
        ]

        # Final ChatResponse
        try:
            return ChatResponse(
                message=chat_message,
                reasoning=response.get("reasoning", ""),
                document_references=enhanced_references_dicts,
                confidence=confidence_val,
                reasoning_chain=reasoning_chain
            )
        except Exception as e:
            print("[API ERROR] Exception in ChatResponse construction:", e)
            print("[API ERROR] Data:", json.dumps({
                "message": str(chat_message),
                "reasoning": str(response.get("reasoning", "")),
                "document_references": str(enhanced_references_dicts),
                "confidence": str(confidence_val),
                "reasoning_chain": str(reasoning_chain)
            }, default=str))
            traceback.print_exc()
            # Fallback to minimal response
            return ChatResponse(
                message=ChatMessage(
                    id=str(uuid.uuid4()),
                    type="ai",
                    content="Test response (fallback)",
                    timestamp=datetime.now(),
                    reasoning="Test reasoning",
                    document_ref=None,
                    show_reasoning=False
                ),
                reasoning="Test reasoning",
                document_references=[],
                confidence=1.0,
                reasoning_chain=[]
            )
    except Exception as e:
        print("[ASK ENDPOINT ERROR] Exception in ask endpoint:", e)
        traceback.print_exc()
        # Fallback to minimal response
        return ChatResponse(
            message=ChatMessage(
                id=str(uuid.uuid4()),
                type="ai",
                content="Test response (outer fallback)",
                timestamp=datetime.now(),
                reasoning="Test reasoning",
                document_ref=None,
                show_reasoning=False
            ),
            reasoning="Test reasoning",
            document_references=[],
            confidence=1.0,
            reasoning_chain=[]
        )

def extract_key_terms(text: str) -> List[str]:
    """Extract key terms from query for analysis"""
    # Simple key term extraction
    import re
    words = re.findall(r'\b\w+\b', text.lower())
    # Filter out common words and short terms
    stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'what', 'when', 'where', 'why', 'how', 'this', 'that', 'these', 'those'}
    key_terms = [word for word in words if len(word) > 3 and word not in stop_words]
    return key_terms[:5]  # Top 5 key terms

@router.post("/challenge/generate", response_model=ChallengeResponse)
async def generate_challenge_questions(request: ChallengeRequest, http_request: Request):
    """Generate challenge questions based on uploaded documents with enhanced context and key term extraction"""
    
    # Extract API keys from headers
    api_keys = extract_api_keys(http_request)
    
    # Initialize LLM service with API keys
    llm_service = LLMService(api_keys)
    
    session_id = str(uuid.uuid4())
    reasoning_steps = []
    
    # Step 1: Document Analysis
    all_chunks = []
    key_terms = []
    
    # If multiple documents, prioritize the most relevant one for challenge generation
    if len(request.document_ids) > 1:
        # For now, use the first document to avoid mixing content
        # In the future, this could be enhanced with document relevance scoring
        primary_doc_id = request.document_ids[0]
        chunks = doc_processor.get_document_chunks(primary_doc_id)
        doc_chunks = [chunk["text"] for chunk in chunks]
        all_chunks.extend(doc_chunks)
        
        # Extract key terms from primary document
        for chunk in chunks:
            chunk_key_terms = chunk["metadata"].get("key_terms", [])
            key_terms.extend(chunk_key_terms)
    else:
        # Single document - process normally
        for doc_id in request.document_ids:
            # Get document chunks with metadata
            chunks = doc_processor.get_document_chunks(doc_id)
            doc_chunks = [chunk["text"] for chunk in chunks]
            all_chunks.extend(doc_chunks)
            
            # Extract key terms from document
            for chunk in chunks:
                chunk_key_terms = chunk["metadata"].get("key_terms", [])
                key_terms.extend(chunk_key_terms)
    
    analysis_step = ReasoningStep(
        step_id=str(uuid.uuid4()),
        step_type="analysis",
        description="Analyzing documents for key concepts and generating questions",
        input_data={"document_count": len(request.document_ids), "total_chunks": len(all_chunks)},
        output_data={"key_terms_extracted": len(set(key_terms))},
        confidence=0.9,
        timestamp=datetime.now()
    )
    reasoning_steps.append(analysis_step)
    
    if not all_chunks:
        raise HTTPException(
            status_code=400,
            detail="No document content found for generating challenge questions."
        )
    
    try:
        # Step 2: Question Generation with Key Terms
        questions_data = llm_service.generate_challenge_questions(
            context=all_chunks,
            provider=request.provider.value,
            key_terms=key_terms[:10]  # Top 10 key terms
        )
        
        generation_step = ReasoningStep(
            step_id=str(uuid.uuid4()),
            step_type="generation",
            description="Generated challenge questions using key term analysis",
            input_data={"key_terms_used": key_terms[:10]},
            output_data={"questions_generated": len(questions_data)},
            confidence=0.85,
            timestamp=datetime.now()
        )
        reasoning_steps.append(generation_step)
        
        # Step 3: Convert to ChallengeQuestion objects with enhanced metadata
        questions = []
        for i, q_data in enumerate(questions_data[:3]):  # Limit to 3 questions
            # Find most relevant document context for this question
            relevant_chunks = doc_processor.hybrid_search_with_mmr(
                q_data["question"], 
                request.document_ids, 
                top_k=2
            )
            
            def ensure_list_key_terms(val):
                if isinstance(val, str):
                    return [k.strip() for k in val.split(",") if k.strip()]
                return val

            question = ChallengeQuestion(
                id=str(uuid.uuid4()),
                question=q_data["question"],
                correct_answer=q_data["correct_answer"],
                explanation=q_data["explanation"],
                document_references=[
                    DocumentReference(
                        citation=chunk["citation"],
                        relevance_score=chunk["relevance_score"],
                        chunk_preview=chunk.get("text", "")[:200] + "..." if len(chunk.get("text", "")) > 200 else chunk.get("text", ""),
                        key_terms=ensure_list_key_terms(chunk.get("key_terms", []))
                    )
                    for chunk in relevant_chunks
                ]
            )
            questions.append(question)
        
        # Log reasoning chain
        doc_processor.log_reasoning_chain(session_id, reasoning_steps)
        
        return ChallengeResponse(questions=questions)
        
    except Exception as e:
        error_step = ReasoningStep(
            step_id=str(uuid.uuid4()),
            step_type="error",
            description=f"Error generating challenge questions: {str(e)}",
            input_data={"document_ids": request.document_ids},
            output_data={"error": str(e)},
            confidence=0.0,
            timestamp=datetime.now()
        )
        reasoning_steps.append(error_step)
        
        # Log reasoning chain
        doc_processor.log_reasoning_chain(session_id, reasoning_steps)
        
        raise HTTPException(
            status_code=500,
            detail=f"Error generating challenge questions: {str(e)}"
        )

@router.post("/challenge/evaluate", response_model=ChallengeEvaluation)
async def evaluate_challenge_answer(
    question_id: str,
    user_answer: str,
    correct_answer: str,
    question_text: str,
    document_context: str,
    http_request: Request,
    provider: str = "openai"
):
    """Evaluate a user's answer to a challenge question with enhanced feedback and similarity scoring"""
    
    # Extract API keys from headers
    api_keys = extract_api_keys(http_request)
    
    # Initialize LLM service with API keys
    llm_service = LLMService(api_keys)
    
    session_id = str(uuid.uuid4())
    reasoning_steps = []
    
    # Step 1: Answer Analysis
    analysis_step = ReasoningStep(
        step_id=str(uuid.uuid4()),
        step_type="analysis",
        description="Analyzing user answer for content and structure",
        input_data={"user_answer_length": len(user_answer)},
        output_data={"analysis_method": "content_structure"},
        confidence=0.8,
        timestamp=datetime.now()
    )
    reasoning_steps.append(analysis_step)
    
    try:
        # Step 2: Enhanced Evaluation with Multiple Metrics
        evaluation = llm_service.evaluate_challenge_answer(
            question=question_text,
            correct_answer=correct_answer,
            user_answer=user_answer,
            context=document_context,
            provider=provider
        )
        
        # Step 3: Calculate Additional Metrics
        # Cosine similarity between user answer and correct answer
        user_embedding = doc_processor.embedding_model.encode([user_answer])[0]
        correct_embedding = doc_processor.embedding_model.encode([correct_answer])[0]
        
        from sklearn.metrics.pairwise import cosine_similarity
        similarity_score = cosine_similarity([user_embedding], [correct_embedding])[0][0]
        
        # Keyword overlap
        user_words = set(user_answer.lower().split())
        correct_words = set(correct_answer.lower().split())
        keyword_overlap = len(user_words.intersection(correct_words)) / len(correct_words) if correct_words else 0
        
        # Enhanced evaluation metrics
        enhanced_evaluation = {
            "is_correct": evaluation["is_correct"],
            "score": evaluation["score"],
            "feedback": evaluation["feedback"],
            "correct_answer": correct_answer,
            "reasoning": evaluation["reasoning"],
            "similarity_score": float(similarity_score),
            "keyword_overlap": keyword_overlap,
            "comprehensive_score": (evaluation["score"] + similarity_score + keyword_overlap) / 3
        }
        
        evaluation_step = ReasoningStep(
            step_id=str(uuid.uuid4()),
            step_type="evaluation",
            description="Evaluated answer using multiple metrics (LLM + similarity + keyword overlap)",
            input_data={"evaluation_methods": ["llm", "cosine_similarity", "keyword_overlap"]},
            output_data={"comprehensive_score": enhanced_evaluation["comprehensive_score"]},
            confidence=0.9,
            timestamp=datetime.now()
        )
        reasoning_steps.append(evaluation_step)
        
        # Log reasoning chain
        doc_processor.log_reasoning_chain(session_id, reasoning_steps)
        
        return ChallengeEvaluation(**enhanced_evaluation)
        
    except Exception as e:
        error_step = ReasoningStep(
            step_id=str(uuid.uuid4()),
            step_type="error",
            description=f"Error evaluating answer: {str(e)}",
            input_data={"question_id": question_id},
            output_data={"error": str(e)},
            confidence=0.0,
            timestamp=datetime.now()
        )
        reasoning_steps.append(error_step)
        
        # Log reasoning chain
        doc_processor.log_reasoning_chain(session_id, reasoning_steps)
        
        raise HTTPException(
            status_code=500,
            detail=f"Error evaluating answer: {str(e)}"
        )

@router.get("/reasoning/{session_id}")
async def get_reasoning_chain(session_id: str):
    """Get reasoning chain for a specific session"""
    reasoning_chain = doc_processor.get_reasoning_chain(session_id)
    if not reasoning_chain:
        raise HTTPException(status_code=404, detail="Reasoning chain not found")
    
    return reasoning_chain

@router.get("/suggestions")
async def get_chat_suggestions(document_id: Optional[str] = None):
    """Get chat suggestions based on document content"""
    try:
        suggestions = []
        
        if document_id:
            # Get suggestions from specific document
            doc_chunks = doc_processor.get_document_chunks(document_id)
            if doc_chunks:
                # Generate questions based on document content
                sample_chunks = [chunk["text"] for chunk in doc_chunks[:3]]
                
                # Simple question generation based on content
                for chunk in sample_chunks:
                    # Extract key phrases and generate questions
                    sentences = chunk.split('.')
                    for sentence in sentences[:2]:  # First 2 sentences
                        if len(sentence.split()) > 5:
                            suggestions.append({
                                "type": "question",
                                "text": f"Can you explain: {sentence.strip()}?",
                                "category": "comprehension"
                            })
                            break
        else:
            # General suggestions
            suggestions = [
                {"type": "question", "text": "What are the main topics covered in this document?", "category": "overview"},
                {"type": "question", "text": "Can you summarize the key findings?", "category": "summary"},
                {"type": "question", "text": "What are the limitations mentioned?", "category": "analysis"},
                {"type": "question", "text": "How does this compare to other approaches?", "category": "comparison"}
            ]
        
        return {
            "suggestions": suggestions[:5],  # Limit to 5 suggestions
            "document_id": document_id
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error generating chat suggestions: {str(e)}"
        )

@router.post("/session/start")
async def start_session():
    """Start a new chat session"""
    session_id = str(uuid.uuid4())
    sessions_db[session_id] = {
        "id": session_id,
        "title": "New Chat",
        "created_at": datetime.now(),
        "last_activity": datetime.now(),
        "messages": [],
        "documents": []
    }
    
    return {
        "session_id": session_id,
        "message": "Session started successfully"
    }

@router.get("/session/{session_id}")
async def get_session(session_id: str):
    """Get session information and messages"""
    if session_id not in sessions_db:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = sessions_db[session_id]
    return {
        "session_id": session_id,
        "title": session["title"],
        "created_at": session["created_at"],
        "last_activity": session["last_activity"],
        "messages": session["messages"],
        "document_count": len(session["documents"])
    }

@router.post("/session/{session_id}/message")
async def add_message_to_session(session_id: str, message: ChatMessage):
    """Add a message to a session"""
    if session_id not in sessions_db:
        raise HTTPException(status_code=404, detail="Session not found")
    
    sessions_db[session_id]["messages"].append(message.dict())
    sessions_db[session_id]["last_activity"] = datetime.now()
    
    return {"message": "Message added to session"}

@router.delete("/session/{session_id}")
async def delete_session(session_id: str):
    """Delete a chat session"""
    if session_id not in sessions_db:
        raise HTTPException(status_code=404, detail="Session not found")
    
    del sessions_db[session_id]
    return {"message": "Session deleted successfully"}

@router.get("/sessions")
async def list_sessions():
    """List all chat sessions"""
    sessions = []
    for session_id, session_data in sessions_db.items():
        sessions.append({
            "id": session_id,
            "title": session_data["title"],
            "created_at": session_data["created_at"],
            "last_activity": session_data["last_activity"],
            "message_count": len(session_data["messages"]),
            "document_count": len(session_data["documents"])
        })
    
    return sessions

@router.post("/challenge/next-question", response_model=ChallengeQuestion)
async def get_next_challenge_question(request: ChallengeRequest, http_request: Request):
    """Get the next challenge question in interactive mode"""
    
    # Extract API keys from headers
    api_keys = extract_api_keys(http_request)
    
    # Initialize LLM service with API keys
    llm_service = LLMService(api_keys)
    
    session_id = str(uuid.uuid4())
    reasoning_steps = []
    
    # Step 1: Document Analysis
    all_chunks = []
    key_terms = []
    
    # If multiple documents, prioritize the most relevant one for challenge generation
    if len(request.document_ids) > 1:
        # For now, use the first document to avoid mixing content
        # In the future, this could be enhanced with document relevance scoring
        primary_doc_id = request.document_ids[0]
        chunks = doc_processor.get_document_chunks(primary_doc_id)
        doc_chunks = [chunk["text"] for chunk in chunks]
        all_chunks.extend(doc_chunks)
        
        # Extract key terms from primary document
        for chunk in chunks:
            chunk_key_terms = chunk["metadata"].get("key_terms", [])
            key_terms.extend(chunk_key_terms)
    else:
        # Single document - process normally
        for doc_id in request.document_ids:
            # Get document chunks with metadata
            chunks = doc_processor.get_document_chunks(doc_id)
            doc_chunks = [chunk["text"] for chunk in chunks]
            all_chunks.extend(doc_chunks)
            
            # Extract key terms from document
            for chunk in chunks:
                chunk_key_terms = chunk["metadata"].get("key_terms", [])
                key_terms.extend(chunk_key_terms)
    
    analysis_step = ReasoningStep(
        step_id=str(uuid.uuid4()),
        step_type="analysis",
        description="Analyzing documents for next challenge question",
        input_data={"document_count": len(request.document_ids), "total_chunks": len(all_chunks)},
        output_data={"key_terms_extracted": len(set(key_terms))},
        confidence=0.9,
        timestamp=datetime.now()
    )
    reasoning_steps.append(analysis_step)
    
    if not all_chunks:
        raise HTTPException(
            status_code=400,
            detail="No document content found for generating challenge questions."
        )
    
    try:
        # Step 2: Generate a single question
        questions_data = llm_service.generate_challenge_questions(
            context=all_chunks,
            provider=request.provider.value,
            key_terms=key_terms[:10]  # Top 10 key terms
        )
        
        if not questions_data:
            raise HTTPException(
                status_code=500,
                detail="Failed to generate challenge question."
            )
        
        # Take the first question
        q_data = questions_data[0]
        
        generation_step = ReasoningStep(
            step_id=str(uuid.uuid4()),
            step_type="generation",
            description="Generated single challenge question",
            input_data={"key_terms_used": key_terms[:10]},
            output_data={"question_generated": True},
            confidence=0.85,
            timestamp=datetime.now()
        )
        reasoning_steps.append(generation_step)
        
        # Step 3: Find relevant document context for this question
        relevant_chunks = doc_processor.hybrid_search_with_mmr(
            q_data["question"], 
            request.document_ids, 
            top_k=2
        )
        
        def ensure_list_key_terms(val):
            if isinstance(val, str):
                return [k.strip() for k in val.split(",") if k.strip()]
            return val

        question = ChallengeQuestion(
            id=str(uuid.uuid4()),
            question=q_data["question"],
            correct_answer=q_data["correct_answer"],
            explanation=q_data["explanation"],
            document_references=[
                DocumentReference(
                    citation=chunk["citation"],
                    relevance_score=chunk["relevance_score"],
                    chunk_preview=chunk.get("text", "")[:200] + "..." if len(chunk.get("text", "")) > 200 else chunk.get("text", ""),
                    key_terms=ensure_list_key_terms(chunk.get("key_terms", []))
                )
                for chunk in relevant_chunks
            ]
        )
        
        # Log reasoning chain
        doc_processor.log_reasoning_chain(session_id, reasoning_steps)
        
        return question
        
    except Exception as e:
        error_step = ReasoningStep(
            step_id=str(uuid.uuid4()),
            step_type="error",
            description=f"Error generating challenge question: {str(e)}",
            input_data={"document_ids": request.document_ids},
            output_data={"error": str(e)},
            confidence=0.0,
            timestamp=datetime.now()
        )
        reasoning_steps.append(error_step)
        
        # Log reasoning chain
        doc_processor.log_reasoning_chain(session_id, reasoning_steps)
        
        raise HTTPException(
            status_code=500,
            detail=f"Error generating challenge question: {str(e)}"
        )

@router.post("/challenge/evaluate-answer", response_model=ChallengeEvaluation)
async def evaluate_challenge_answer_interactive(
    request: ChallengeEvaluationRequest,
    http_request: Request
):
    """Evaluate a user's answer to a challenge question with detailed feedback"""
    
    # Extract API keys from headers
    api_keys = extract_api_keys(http_request)
    
    # Initialize LLM service with API keys
    llm_service = LLMService(api_keys)
    
    session_id = str(uuid.uuid4())
    reasoning_steps = []
    
    # Step 1: Answer Analysis
    analysis_step = ReasoningStep(
        step_id=str(uuid.uuid4()),
        step_type="analysis",
        description="Analyzing user answer for content and structure",
        input_data={"user_answer_length": len(request.user_answer)},
        output_data={"analysis_method": "content_structure"},
        confidence=0.8,
        timestamp=datetime.now()
    )
    reasoning_steps.append(analysis_step)
    
    try:
        # Step 2: Enhanced Evaluation with Multiple Metrics
        evaluation = llm_service.evaluate_challenge_answer(
            question=request.question_text,
            correct_answer=request.correct_answer,
            user_answer=request.user_answer,
            context=request.document_context,
            provider=request.provider
        )
        
        # Step 3: Calculate Additional Metrics
        # Cosine similarity between user answer and correct answer
        user_embedding = doc_processor.embedding_model.encode([request.user_answer])[0]
        correct_embedding = doc_processor.embedding_model.encode([request.correct_answer])[0]
        
        from sklearn.metrics.pairwise import cosine_similarity
        similarity_score = cosine_similarity([user_embedding], [correct_embedding])[0][0]
        
        # Keyword overlap
        user_words = set(request.user_answer.lower().split())
        correct_words = set(request.correct_answer.lower().split())
        keyword_overlap = len(user_words.intersection(correct_words)) / len(correct_words) if correct_words else 0
        
        # Enhanced evaluation metrics
        enhanced_evaluation = {
            "question_id": request.question_id,
            "is_correct": evaluation["is_correct"],
            "score": evaluation["score"],
            "feedback": evaluation["feedback"],
            "correct_answer": request.correct_answer,
            "reasoning": evaluation["reasoning"],
            "similarity_score": float(similarity_score),
            "keyword_overlap": keyword_overlap,
            "comprehensive_score": (evaluation["score"] + similarity_score + keyword_overlap) / 3
        }
        
        evaluation_step = ReasoningStep(
            step_id=str(uuid.uuid4()),
            step_type="evaluation",
            description="Evaluated answer using multiple metrics (LLM + similarity + keyword overlap)",
            input_data={"evaluation_methods": ["llm", "cosine_similarity", "keyword_overlap"]},
            output_data={"comprehensive_score": enhanced_evaluation["comprehensive_score"]},
            confidence=0.9,
            timestamp=datetime.now()
        )
        reasoning_steps.append(evaluation_step)
        
        # Log reasoning chain
        doc_processor.log_reasoning_chain(session_id, reasoning_steps)
        
        return ChallengeEvaluation(**enhanced_evaluation)
        
    except Exception as e:
        error_step = ReasoningStep(
            step_id=str(uuid.uuid4()),
            step_type="error",
            description=f"Error evaluating answer: {str(e)}",
            input_data={"question_id": request.question_id},
            output_data={"error": str(e)},
            confidence=0.0,
            timestamp=datetime.now()
        )
        reasoning_steps.append(error_step)
        
        # Log reasoning chain
        doc_processor.log_reasoning_chain(session_id, reasoning_steps)
        
        raise HTTPException(
            status_code=500,
            detail=f"Error evaluating answer: {str(e)}"
        )

@router.get("/test")
def chat_test():
    from backend.api.models import ChatMessage, ChatResponse
    from datetime import datetime
    import uuid
    return ChatResponse(
        message=ChatMessage(
            id=str(uuid.uuid4()),
            type="ai",
            content="Test response",
            timestamp=datetime.now(),
            reasoning="Test reasoning",
            document_ref=None,
            show_reasoning=False
        ),
        reasoning="Test reasoning",
        document_references=[],
        confidence=1.0,
        reasoning_chain=[]
    ) 