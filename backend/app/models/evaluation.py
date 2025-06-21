from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Any
from enum import Enum

class EvaluationMetric(str, Enum):
    """Evaluation metrics for RAG responses"""
    RELEVANCE = "relevance"
    GROUNDEDNESS = "groundedness"
    COHERENCE = "coherence"
    FLUENCY = "fluency"
    FINANCIAL_ACCURACY = "financial_accuracy"
    CITATION_QUALITY = "citation_quality"
    RESPONSE_TIME = "response_time"

@dataclass
class EvaluationResult:
    """Result of an evaluation metric"""
    metric: str
    score: float
    reasoning: str
    timestamp: datetime
    session_id: str
    query: str
    response: str
    sources: List[str]
    model_used: str
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class FinancialEvaluationContext:
    """Context for financial document evaluation"""
    query: str
    response: str
    sources: List[Dict[str, Any]]
    document_types: List[str]
    financial_metrics: Optional[Dict[str, Any]] = None
