from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum

class EmbeddingModel(str, Enum):
    ADA_002 = "text-embedding-ada-002"
    SMALL_3 = "text-embedding-3-small"
    LARGE_3 = "text-embedding-3-large"

class ChatModel(str, Enum):
    GPT_4 = "gpt-4"
    GPT_4_TURBO = "gpt-4-turbo"
    GPT_35_TURBO = "gpt-35-turbo"
    FINANCIAL_LLM = "financial-llm"
    GROK_BETA = "grok-beta"
    DEEPSEEK_CHAT = "deepseek-chat"

class DocumentType(str, Enum):
    FORM_10K = "10-K"
    FORM_10Q = "10-Q"
    ANNUAL_REPORT = "annual-report"
    EARNINGS_REPORT = "earnings-report"
    OTHER = "other"

class DocumentStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

class Citation(BaseModel):
    document_id: str
    document_name: str
    page_number: Optional[int] = None
    section: Optional[str] = None
    confidence_score: float = Field(ge=0.0, le=1.0)
    text_snippet: str
    url: Optional[str] = None

class ChatMessage(BaseModel):
    role: str = Field(..., description="Role of the message sender (user, assistant, system)")
    content: str = Field(..., description="Content of the message")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    citations: List[Citation] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)

class ChatRequest(BaseModel):
    message: str = Field(..., description="User message")
    session_id: Optional[str] = None
    chat_model: ChatModel = ChatModel.GPT_4
    embedding_model: EmbeddingModel = EmbeddingModel.SMALL_3
    temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    max_tokens: int = Field(default=4000, ge=1, le=8000)
    use_knowledge_base: bool = True
    exercise_type: Optional[str] = Field(None, description="Exercise 1, 2, or 3")

class ChatResponse(BaseModel):
    response: str
    session_id: str
    citations: List[Citation]
    metadata: Dict[str, Any] = Field(default_factory=dict)
    token_usage: Dict[str, int] = Field(default_factory=dict)

class DocumentUploadRequest(BaseModel):
    file_name: str
    document_type: DocumentType
    company_name: Optional[str] = None
    filing_date: Optional[datetime] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

class DocumentUploadResponse(BaseModel):
    document_id: str
    status: DocumentStatus
    message: str
    processing_started_at: datetime

class DocumentInfo(BaseModel):
    document_id: str
    file_name: str
    document_type: DocumentType
    company_name: Optional[str] = None
    filing_date: Optional[datetime] = None
    status: DocumentStatus
    uploaded_at: datetime
    processed_at: Optional[datetime] = None
    chunk_count: Optional[int] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

class KnowledgeBaseStats(BaseModel):
    total_documents: int
    total_chunks: int
    last_updated: datetime
    documents_by_type: Dict[DocumentType, int]
    processing_queue_size: int

class KnowledgeBaseUpdateRequest(BaseModel):
    source_urls: List[str] = Field(default_factory=list)
    auto_update_enabled: bool = True
    update_frequency_hours: int = Field(default=24, ge=1, le=168)
    credibility_threshold: float = Field(default=0.7, ge=0.0, le=1.0)

class AdminMetrics(BaseModel):
    total_requests: int
    total_tokens_used: int
    average_response_time: float
    error_rate: float
    active_sessions: int
    knowledge_base_stats: KnowledgeBaseStats
    model_usage: Dict[str, int]
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class EvaluationResult(BaseModel):
    metric_name: str
    score: float
    details: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class SessionInfo(BaseModel):
    session_id: str
    user_id: Optional[str] = None
    created_at: datetime
    last_activity: datetime
    message_count: int
    total_tokens: int
    metadata: Dict[str, Any] = Field(default_factory=dict)
