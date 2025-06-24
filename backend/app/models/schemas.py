from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, TYPE_CHECKING
from datetime import datetime
from enum import Enum

if TYPE_CHECKING:
    from .schemas import PerformanceBenchmark, ReasoningChain

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
    id: str = Field(default_factory=lambda: str(__import__('uuid').uuid4()))
    content: str = Field(..., description="Content of the citation")
    source: str = Field(..., description="Source name or title")
    document_id: str
    document_title: str = Field(..., description="Title of the source document")
    page_number: Optional[int] = None
    section_title: Optional[str] = None
    confidence: str = Field(default="medium", description="Confidence level: low, medium, high")
    url: Optional[str] = None
    credibility_score: Optional[float] = Field(default=0.5, ge=0.0, le=1.0, description="Source credibility score")
    
    document_name: Optional[str] = None
    section: Optional[str] = None
    confidence_score: Optional[float] = Field(default=0.5, ge=0.0, le=1.0)
    text_snippet: Optional[str] = None

class ChatMessage(BaseModel):
    role: str = Field(..., description="Role of the message sender (user, assistant, system)")
    content: str = Field(..., description="Content of the message")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    citations: List[Citation] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)

class ChatRequest(BaseModel):
    message: str = Field(..., description="User message")
    session_id: Optional[str] = None
    chat_model: str = Field(default="chat4omini", description="Chat model deployment name")
    embedding_model: str = Field(default="text-embedding-ada-002", description="Embedding model deployment name")
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


class VerificationLevel(str, Enum):
    BASIC = "basic"
    THOROUGH = "thorough"
    COMPREHENSIVE = "comprehensive"

class QARequest(BaseModel):
    question: str = Field(..., description="The financial question to answer")
    session_id: Optional[str] = None
    chat_model: str = Field(default="chat4omini", description="Chat model deployment name")
    embedding_model: str = Field(default="text-embedding-3-small", description="Embedding model deployment name")
    temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    verification_level: VerificationLevel = VerificationLevel.THOROUGH
    context: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional context for the question")
    max_tokens: int = Field(default=4000, ge=1, le=8000)
    credibility_check_enabled: bool = Field(default=False, description="Whether to perform credibility checks on sources")

class QAResponse(BaseModel):
    answer: str = Field(..., description="The comprehensive answer to the question")
    session_id: str
    confidence_score: float = Field(ge=0.0, le=1.0, description="Overall confidence in the answer")
    citations: List[Citation] = Field(default_factory=list)
    sub_questions: List[str] = Field(default_factory=list, description="Sub-questions that were researched")
    verification_details: Dict[str, Any] = Field(default_factory=dict, description="Source verification details")
    performance_benchmark: Optional["PerformanceBenchmark"] = None
    reasoning_chain: Optional["ReasoningChain"] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    token_usage: Dict[str, int] = Field(default_factory=dict)

class QuestionDecompositionRequest(BaseModel):
    question: str = Field(..., description="The complex question to decompose")
    chat_model: str = Field(default="chat4omini", description="Chat model deployment name")
    context: Optional[Dict[str, Any]] = Field(default_factory=dict)

class QuestionDecompositionResponse(BaseModel):
    original_question: str
    sub_questions: List[str] = Field(..., description="List of researchable sub-questions")
    reasoning: str = Field(..., description="Explanation of how the question was decomposed")
    session_id: str
    metadata: Dict[str, Any] = Field(default_factory=dict)

class SourceInfo(BaseModel):
    id: str = Field(default_factory=lambda: str(__import__('uuid').uuid4()))
    url: str = Field(..., description="URL of the source")
    title: str = Field(..., description="Title of the source")
    content: str = Field(..., description="Content excerpt from the source")
    metadata: Dict[str, Any] = Field(default_factory=dict)

class VerifiedSource(BaseModel):
    source_id: str
    url: str
    title: str
    content: str
    credibility_score: float = Field(ge=0.0, le=1.0)
    credibility_explanation: str = Field(..., description="Explanation of credibility assessment")
    trust_indicators: List[str] = Field(default_factory=list)
    red_flags: List[str] = Field(default_factory=list)
    verification_status: str = Field(..., description="verified, questionable, or unverified")

class SourceVerificationRequest(BaseModel):
    sources: List[SourceInfo] = Field(..., description="List of sources to verify")
    context: Optional[Dict[str, Any]] = Field(default_factory=dict)

class SourceVerificationResponse(BaseModel):
    verified_sources: List[VerifiedSource] = Field(..., description="List of verified sources with credibility scores")
    overall_credibility_score: float = Field(ge=0.0, le=1.0, description="Average credibility across all sources")
    verification_summary: str = Field(..., description="Summary of verification results")
    session_id: str
    metadata: Dict[str, Any] = Field(default_factory=dict)

class PerformanceBenchmark(BaseModel):
    question_id: str = Field(..., description="Unique identifier for the question")
    question: str = Field(..., description="The original question")
    complexity_score: int = Field(ge=1, le=5, description="Question complexity score (1=simple, 5=very complex)")
    estimated_manual_time: float = Field(..., description="Estimated manual research time in minutes")
    ai_processing_time: float = Field(..., description="Actual AI processing time in minutes")
    efficiency_gain: float = Field(ge=0, description="Percentage improvement over manual research")
    source_count: int = Field(ge=0, description="Number of sources analyzed")
    accuracy_score: float = Field(ge=0.0, le=1.0, description="Answer accuracy score")
    confidence_score: float = Field(ge=0.0, le=1.0, description="Confidence in the answer")
    verification_level: VerificationLevel
    session_id: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = Field(default_factory=dict)

class ReasoningStep(BaseModel):
    step_number: int = Field(ge=1, description="Sequential step number")
    description: str = Field(..., description="Description of this reasoning step")
    action_type: str = Field(..., description="Type of action: search, analyze, synthesize, verify")
    sources_consulted: List[str] = Field(default_factory=list, description="Sources referenced in this step")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence in this step")
    duration_ms: int = Field(ge=0, description="Duration of this step in milliseconds")
    output: str = Field(..., description="Output or result of this step")
    metadata: Dict[str, Any] = Field(default_factory=dict)

class ReasoningChain(BaseModel):
    question_id: str = Field(..., description="Unique identifier for the question")
    question: str = Field(..., description="The original question")
    reasoning_steps: List[ReasoningStep] = Field(..., description="Sequential reasoning steps")
    total_duration_ms: int = Field(ge=0, description="Total reasoning duration in milliseconds")
    final_confidence: float = Field(ge=0.0, le=1.0, description="Final confidence in the answer")
    session_id: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class PerformanceMetrics(BaseModel):
    total_questions: int = Field(ge=0, description="Total questions processed")
    average_efficiency_gain: float = Field(ge=0, description="Average efficiency improvement percentage")
    average_accuracy_score: float = Field(ge=0.0, le=1.0, description="Average answer accuracy")
    average_processing_time: float = Field(ge=0, description="Average AI processing time in minutes")
    complexity_breakdown: Dict[int, int] = Field(default_factory=dict, description="Question count by complexity level")
    time_saved_minutes: float = Field(ge=0, description="Total time saved vs manual research")
    session_id: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
