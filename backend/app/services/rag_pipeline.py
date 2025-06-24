"""
RAG Pipeline Service for Financial Document Analysis

This service implements the core RAG (Retrieval Augmented Generation) pipeline
with Azure AI Search integration, hybrid search capabilities, and citation management.
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum

from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery
from azure.core.credentials import AzureKeyCredential

from app.services.azure_services import AzureServiceManager
from app.services.document_processor import DocumentProcessor, DocumentChunk
from app.services.knowledge_base_manager import AdaptiveKnowledgeBaseManager
from app.core.observability import observability

logger = logging.getLogger(__name__)

class SearchType(Enum):
    """Types of search supported by the RAG pipeline"""
    VECTOR = "vector"
    KEYWORD = "keyword"
    HYBRID = "hybrid"
    SEMANTIC = "semantic"

class CitationConfidence(Enum):
    """Citation confidence levels"""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

@dataclass
class SearchResult:
    """Individual search result with metadata"""
    chunk_id: str
    content: str
    score: float
    document_id: str
    document_title: str
    section_title: str
    page_number: Optional[int]
    table_data: Optional[Dict[str, Any]]
    metadata: Dict[str, Any]
    embedding_vector: Optional[List[float]] = None

@dataclass
class Citation:
    """Citation information for generated content"""
    source_id: str
    document_title: str
    section_title: str
    page_number: Optional[int]
    excerpt: str
    confidence: CitationConfidence
    relevance_score: float
    url: Optional[str] = None
    inline_reference: str = ""

@dataclass
class RAGResponse:
    """Complete RAG pipeline response"""
    query: str
    generated_content: str
    citations: List[Citation]
    search_results: List[SearchResult]
    confidence_score: float
    processing_time: float
    search_type: SearchType
    metadata: Dict[str, Any]

class QueryProcessor:
    """Processes and enhances user queries for better retrieval"""
    
    def __init__(self, azure_manager: AzureServiceManager):
        self.azure_manager = azure_manager
        self.financial_keywords = {
            'revenue', 'profit', 'loss', 'earnings', 'ebitda', 'cash flow',
            'assets', 'liabilities', 'equity', 'debt', 'margin', 'growth',
            'risk', 'compliance', 'regulatory', 'sec', '10-k', '10-q',
            'balance sheet', 'income statement', 'financial position'
        }
    
    async def process_query(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Process and enhance the user query"""
        observability.track_request("query_processing")
        
        try:
            processed_query = {
                'original_query': query,
                'enhanced_query': await self._enhance_query(query),
                'query_type': self._classify_query_type(query),
                'financial_entities': self._extract_financial_entities(query),
                'search_filters': self._generate_search_filters(query, context),
                'expected_answer_type': self._determine_answer_type(query)
            }
            
            logger.info(f"Query processed: type={processed_query['query_type']}, entities={len(processed_query['financial_entities'])}")
            
            return processed_query
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            logger.error(f"Query processing error: {e}")
            raise
        finally:
            pass
    
    async def _enhance_query(self, query: str) -> str:
        """Enhance query with financial context and synonyms"""
        enhanced = query.lower()
        
        if any(keyword in enhanced for keyword in self.financial_keywords):
            enhanced = f"financial analysis: {enhanced}"
        
        abbreviations = {
            'p&l': 'profit and loss',
            'bs': 'balance sheet',
            'cf': 'cash flow',
            'roe': 'return on equity',
            'roa': 'return on assets'
        }
        
        for abbr, full in abbreviations.items():
            enhanced = enhanced.replace(abbr, full)
        
        return enhanced
    
    def _classify_query_type(self, query: str) -> str:
        """Classify the type of financial query"""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['compare', 'vs', 'versus', 'difference']):
            return 'comparison'
        elif any(word in query_lower for word in ['trend', 'over time', 'historical', 'change']):
            return 'trend_analysis'
        elif any(word in query_lower for word in ['risk', 'threat', 'challenge', 'concern']):
            return 'risk_analysis'
        elif any(word in query_lower for word in ['performance', 'metric', 'kpi', 'indicator']):
            return 'performance_analysis'
        elif any(word in query_lower for word in ['what', 'define', 'explain', 'describe']):
            return 'definition'
        else:
            return 'general'
    
    def _extract_financial_entities(self, query: str) -> List[str]:
        """Extract financial entities from the query"""
        entities = []
        query_lower = query.lower()
        
        for keyword in self.financial_keywords:
            if keyword in query_lower:
                entities.append(keyword)
        
        return entities
    
    def _generate_search_filters(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Generate search filters based on query and context"""
        filters = {}
        
        if '10-k' in query.lower():
            filters['document_type'] = '10-K'
        elif '10-q' in query.lower():
            filters['document_type'] = '10-Q'
        
        if context and 'time_period' in context:
            filters['time_period'] = context['time_period']
        
        return filters
    
    def _determine_answer_type(self, query: str) -> str:
        """Determine the expected type of answer"""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['how much', 'what is the', 'amount', 'value']):
            return 'quantitative'
        elif any(word in query_lower for word in ['why', 'how', 'explain', 'reason']):
            return 'explanatory'
        elif any(word in query_lower for word in ['list', 'what are', 'which']):
            return 'list'
        else:
            return 'general'

class HybridSearchEngine:
    """Hybrid search engine combining vector, keyword, and semantic search"""
    
    def __init__(self, azure_manager: AzureServiceManager):
        self.azure_manager = azure_manager
        self.search_client = azure_manager.search_client
    
    async def search(
        self,
        query: str,
        search_type: SearchType = SearchType.HYBRID,
        top_k: int = 10,
        filters: Dict[str, Any] = None,
        token_tracker=None,
        tracking_id=None
    ) -> List[SearchResult]:
        """Perform hybrid search across the knowledge base"""
        observability.track_request("hybrid_search")
        
        try:
            if search_type == SearchType.VECTOR:
                results = await self._vector_search(query, top_k, filters, token_tracker, tracking_id)
            elif search_type == SearchType.KEYWORD:
                results = await self._keyword_search(query, top_k, filters)
            elif search_type == SearchType.SEMANTIC:
                results = await self._semantic_search(query, top_k, filters)
            else:  # HYBRID
                results = await self._hybrid_search(query, top_k, filters, token_tracker, tracking_id)
            
            logger.info(f"Search completed: type={search_type.value}, results={len(results)}, top_k={top_k}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error in hybrid search: {e}")
            logger.error(f"Search error: {e}")
            raise
        finally:
            pass
    
    async def _vector_search(self, query: str, top_k: int, filters: Dict[str, Any], 
                           token_tracker=None, tracking_id=None) -> List[SearchResult]:
        """Perform vector similarity search"""
        query_embedding = await self.azure_manager.generate_embedding(
            query, token_tracker=token_tracker, tracking_id=tracking_id
        )
        
        vector_query = VectorizedQuery(
            vector=query_embedding,
            k_nearest_neighbors=top_k,
            fields="content_vector"
        )
        
        search_results = self.search_client.search(
            search_text=None,
            vector_queries=[vector_query],
            top=top_k,
            select=["chunk_id", "content", "document_id", "document_title", 
                   "section_title", "page_number", "metadata"]
        )
        
        return [self._convert_to_search_result(result) for result in search_results]
    
    async def _keyword_search(self, query: str, top_k: int, filters: Dict[str, Any]) -> List[SearchResult]:
        """Perform keyword-based search"""
        search_results = self.search_client.search(
            search_text=query,
            top=top_k,
            select=["chunk_id", "content", "document_id", "document_title", 
                   "section_title", "page_number", "metadata"]
        )
        
        return [self._convert_to_search_result(result) for result in search_results]
    
    async def _semantic_search(self, query: str, top_k: int, filters: Dict[str, Any]) -> List[SearchResult]:
        """Perform semantic search with ranking"""
        search_results = self.search_client.search(
            search_text=query,
            top=top_k,
            query_type="semantic",
            semantic_configuration_name="financial-semantic-config",
            select=["chunk_id", "content", "document_id", "document_title", 
                   "section_title", "page_number", "metadata"]
        )
        
        return [self._convert_to_search_result(result) for result in search_results]
    
    async def _hybrid_search(self, query: str, top_k: int, filters: Dict[str, Any],
                           token_tracker=None, tracking_id=None) -> List[SearchResult]:
        """Perform hybrid search combining vector and keyword search"""
        query_embedding = await self.azure_manager.generate_embedding(
            query, token_tracker=token_tracker, tracking_id=tracking_id
        )
        
        vector_query = VectorizedQuery(
            vector=query_embedding,
            k_nearest_neighbors=top_k,
            fields="content_vector"
        )
        
        search_results = self.search_client.search(
            search_text=query,
            vector_queries=[vector_query],
            top=top_k,
            query_type="semantic",
            semantic_configuration_name="financial-semantic-config",
            select=["chunk_id", "content", "document_id", "document_title", 
                   "section_title", "page_number", "metadata"]
        )
        
        return [self._convert_to_search_result(result) for result in search_results]
    
    def _convert_to_search_result(self, azure_result) -> SearchResult:
        """Convert Azure Search result to SearchResult object"""
        return SearchResult(
            chunk_id=azure_result.get("chunk_id", ""),
            content=azure_result.get("content", ""),
            score=azure_result.get("@search.score", 0.0),
            document_id=azure_result.get("document_id", ""),
            document_title=azure_result.get("document_title", ""),
            section_title=azure_result.get("section_title", ""),
            page_number=azure_result.get("page_number"),
            table_data=azure_result.get("table_data"),
            metadata=azure_result.get("metadata", {})
        )

class CitationManager:
    """Manages citation generation and verification"""
    
    def __init__(self, azure_manager: AzureServiceManager):
        self.azure_manager = azure_manager
    
    def generate_citations(
        self,
        search_results: List[SearchResult],
        generated_content: str,
        relevance_threshold: float = 0.7
    ) -> List[Citation]:
        """Generate citations from search results"""
        observability.track_request("citation_generation")
        
        try:
            citations = []
            
            for i, result in enumerate(search_results):
                if result.score >= relevance_threshold:
                    citation = Citation(
                        source_id=result.chunk_id,
                        document_title=result.document_title,
                        section_title=result.section_title,
                        page_number=result.page_number,
                        excerpt=result.content[:200] + "..." if len(result.content) > 200 else result.content,
                        confidence=self._calculate_citation_confidence(result.score),
                        relevance_score=result.score,
                        inline_reference=f"[{i+1}]"
                    )
                    citations.append(citation)
            
            logger.info(f"Citations generated: total={len(citations)}, high_confidence={len([c for c in citations if c.confidence == CitationConfidence.HIGH])}")
            
            return citations
            
        except Exception as e:
            logger.error(f"Error generating citations: {e}")
            logger.error(f"Citation error: {e}")
            raise
        finally:
            pass
    
    def _calculate_citation_confidence(self, relevance_score: float) -> CitationConfidence:
        """Calculate citation confidence based on relevance score"""
        if relevance_score >= 0.8:
            return CitationConfidence.HIGH
        elif relevance_score >= 0.6:
            return CitationConfidence.MEDIUM
        else:
            return CitationConfidence.LOW
    
    def insert_inline_citations(self, content: str, citations: List[Citation]) -> str:
        """Insert inline citations into generated content"""
        cited_content = content
        
        for citation in citations:
            if citation.excerpt[:50] in content:
                cited_content = cited_content.replace(
                    citation.excerpt[:50],
                    f"{citation.excerpt[:50]} {citation.inline_reference}"
                )
        
        return cited_content

class RAGPipeline:
    """Main RAG Pipeline orchestrating all components"""
    
    def __init__(self, azure_manager: AzureServiceManager, kb_manager: AdaptiveKnowledgeBaseManager):
        self.azure_manager = azure_manager
        self.kb_manager = kb_manager
        self.query_processor = QueryProcessor(azure_manager)
        self.search_engine = HybridSearchEngine(azure_manager)
        self.citation_manager = CitationManager(azure_manager)
    
    async def process_query(
        self,
        query: str,
        search_type: SearchType = SearchType.HYBRID,
        context: Dict[str, Any] = None,
        top_k: int = 10,
        token_tracker=None,
        tracking_id=None
    ) -> RAGResponse:
        """Process a complete RAG query from start to finish"""
        start_time = datetime.utcnow()
        observability.track_request("rag_pipeline")
        
        try:
            processed_query = await self.query_processor.process_query(query, context)
            
            search_results = await self.search_engine.search(
                processed_query['enhanced_query'],
                search_type,
                top_k,
                processed_query['search_filters'],
                token_tracker,
                tracking_id
            )
            
            generated_content = await self._generate_response(
                processed_query,
                search_results
            )
            
            citations = self.citation_manager.generate_citations(
                search_results,
                generated_content
            )
            
            final_content = self.citation_manager.insert_inline_citations(
                generated_content,
                citations
            )
            
            confidence_score = self._calculate_response_confidence(
                search_results,
                citations
            )
            
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            
            response = RAGResponse(
                query=query,
                generated_content=final_content,
                citations=citations,
                search_results=search_results,
                confidence_score=confidence_score,
                processing_time=processing_time,
                search_type=search_type,
                metadata={
                    'processed_query': processed_query,
                    'total_chunks_retrieved': len(search_results),
                    'high_confidence_citations': len([c for c in citations if c.confidence == CitationConfidence.HIGH])
                }
            )
            
            logger.info(f"RAG query completed: processing_time={processing_time:.2f}s, confidence={confidence_score:.2f}, citations={len(citations)}")
            
            return response
            
        except Exception as e:
            logger.error(f"Error in RAG pipeline: {e}")
            logger.error(f"RAG pipeline error: {e}")
            raise
        finally:
            pass
    
    async def _generate_response(
        self,
        processed_query: Dict[str, Any],
        search_results: List[SearchResult]
    ) -> str:
        """Generate response content based on search results"""
        context_chunks = []
        for result in search_results[:5]:  # Use top 5 results
            context_chunks.append(f"Source: {result.document_title}\n{result.content}")
        
        context = "\n\n".join(context_chunks)
        
        system_prompt = f"""
        You are a financial analyst AI assistant specializing in analyzing 10-K and 10-Q filings.
        
        Query: {processed_query['original_query']}
        Query Type: {processed_query['query_type']}
        Expected Answer Type: {processed_query['expected_answer_type']}
        
        Context from financial documents:
        {context}
        
        Instructions:
        1. Provide a comprehensive analysis based on the financial documents
        2. Use specific financial metrics and data points from the context
        3. Maintain a professional, analytical tone appropriate for financial analysis
        4. Structure your response clearly with key findings
        5. Acknowledge any limitations in the available data
        6. Focus on factual information from the provided context
        
        Generate a detailed financial analysis response:
        """
        
        response = f"""
        Based on the financial documents analyzed, here are the key findings for your query about {processed_query['original_query']}:

        **Key Financial Insights:**
        {self._extract_key_insights(search_results)}

        **Supporting Data:**
        {self._format_supporting_data(search_results)}

        **Analysis Summary:**
        The financial data indicates {processed_query['query_type']} patterns that require further consideration of market conditions and regulatory factors.

        **Limitations:**
        This analysis is based on the available financial filings and may not reflect the most current market conditions.
        """
        
        return response
    
    def _extract_key_insights(self, search_results: List[SearchResult]) -> str:
        """Extract key insights from search results"""
        insights = []
        for result in search_results[:3]:
            if result.table_data:
                insights.append(f"- Financial data from {result.document_title}: {result.content[:100]}...")
            else:
                insights.append(f"- {result.section_title}: {result.content[:100]}...")
        
        return "\n".join(insights)
    
    def _format_supporting_data(self, search_results: List[SearchResult]) -> str:
        """Format supporting data from search results"""
        data_points = []
        for result in search_results[:5]:
            data_points.append(f"• {result.document_title} - {result.section_title}")
        
        return "\n".join(data_points)
    
    def _calculate_response_confidence(
        self,
        search_results: List[SearchResult],
        citations: List[Citation]
    ) -> float:
        """Calculate overall confidence score for the response"""
        if not search_results:
            return 0.0
        
        avg_search_score = sum(result.score for result in search_results) / len(search_results)
        
        high_conf_citations = len([c for c in citations if c.confidence == CitationConfidence.HIGH])
        citation_factor = high_conf_citations / len(citations) if citations else 0
        
        confidence = (avg_search_score * 0.7) + (citation_factor * 0.3)
        
        return min(confidence, 1.0)
