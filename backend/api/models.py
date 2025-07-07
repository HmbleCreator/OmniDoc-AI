from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum

class LLMProvider(str, Enum):
    OPENAI = "openai"
    GEMINI = "gemini"
    CLAUDE = "claude"
    LOCAL = "local"

class ChatMode(str, Enum):
    ASK = "ask"
    CHALLENGE = "challenge"

class DocumentUpload(BaseModel):
    filename: str
    content: str
    file_type: str = Field(..., description="pdf or txt")

class DocumentResponse(BaseModel):
    id: str
    name: str
    type: str
    content: str
    summary: str
    upload_time: datetime
    chunks: List[str] = []

class ChatMessage(BaseModel):
    id: Optional[str] = None
    type: Optional[str] = None
    content: Optional[str] = None
    timestamp: Optional[datetime] = None
    reasoning: Optional[str] = None
    document_ref: Optional[Any] = None
    show_reasoning: Optional[bool] = None

    class Config:
        arbitrary_types_allowed = True

class ChatRequest(BaseModel):
    message: str
    mode: ChatMode = ChatMode.ASK
    document_id: Optional[str] = None
    session_id: Optional[str] = None
    provider: LLMProvider = LLMProvider.OPENAI

class ChatResponse(BaseModel):
    message: ChatMessage
    reasoning: Optional[str] = None
    document_references: Optional[List[Dict[str, Any]]] = []
    confidence: Optional[float] = 0.0
    reasoning_chain: Optional[List[Dict[str, str]]] = []

    class Config:
        arbitrary_types_allowed = True

class DocumentReference(BaseModel):
    citation: str
    relevance_score: float
    chunk_preview: Optional[str] = None
    key_terms: List[str] = []

class ChallengeQuestion(BaseModel):
    id: str
    question: str
    correct_answer: str
    explanation: str
    document_references: List[DocumentReference] = []

class ChallengeRequest(BaseModel):
    document_ids: List[str]
    provider: LLMProvider = LLMProvider.OPENAI

class ChallengeResponse(BaseModel):
    questions: List[ChallengeQuestion]

class ChallengeAnswer(BaseModel):
    question_id: str
    user_answer: str

class ChallengeEvaluation(BaseModel):
    question_id: str
    is_correct: bool
    score: float
    feedback: str
    correct_answer: str
    reasoning: str

class ChallengeEvaluationRequest(BaseModel):
    question_id: str
    user_answer: str
    correct_answer: str
    question_text: str
    document_context: str
    provider: str = "openai"

class LLMStatus(str, Enum):
    READY = "ready"
    NOT_CONFIGURED = "not_configured"
    ERROR = "error"
    LOADING = "loading"

class LLMConfigRequest(BaseModel):
    provider: LLMProvider
    api_key: str
    model: Optional[str] = None

class LLMConfigResponse(BaseModel):
    message: str
    provider: str
    status: LLMStatus

class LLMConfig(BaseModel):
    provider: LLMProvider
    api_key: str
    model: Optional[str] = None

class SessionInfo(BaseModel):
    id: str
    title: str
    created_at: datetime
    last_activity: datetime
    document_count: int
    message_count: int

class ErrorResponse(BaseModel):
    error: str
    details: Optional[str] = None 