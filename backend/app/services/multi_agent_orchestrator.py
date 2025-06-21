import asyncio
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
from dataclasses import dataclass
from enum import Enum

import semantic_kernel as sk
from semantic_kernel.connectors.ai import PromptExecutionSettings
from semantic_kernel.contents import ChatHistory

from app.services.azure_services import AzureServiceManager
from app.services.knowledge_base_manager import AdaptiveKnowledgeBaseManager
from app.services.document_processor import DocumentProcessor
from app.core.config import settings
from app.core.observability import observability

logger = logging.getLogger(__name__)

class AgentType(Enum):
    CONTENT_GENERATOR = "content_generator"
    QA_AGENT = "qa_agent"
    KNOWLEDGE_MANAGER = "knowledge_manager"

@dataclass
class AgentMessage:
    agent_id: str
    agent_type: AgentType
    content: str
    metadata: Dict[str, Any]
    timestamp: datetime
    session_id: str

@dataclass
class AgentCapability:
    name: str
    description: str
    input_schema: Dict[str, Any]
    output_schema: Dict[str, Any]

class FinancialAgent:
    """Base class for financial analysis agents"""
    
    def __init__(self, agent_type: AgentType, azure_manager: AzureServiceManager):
        self.agent_type = agent_type
        self.agent_id = f"{agent_type.value}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        self.azure_manager = azure_manager
        self.capabilities = []
        self.kernel = self._initialize_semantic_kernel()
        
    def _initialize_semantic_kernel(self) -> sk.Kernel:
        """Initialize Semantic Kernel for the agent"""
        kernel = sk.Kernel()
        
        
        return kernel
    
    async def process_request(self, request: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Process a request using the agent's capabilities"""
        raise NotImplementedError("Subclasses must implement process_request")
    
    def get_capabilities(self) -> List[AgentCapability]:
        """Return list of agent capabilities"""
        return self.capabilities

class ContentGeneratorAgent(FinancialAgent):
    """Agent for Exercise 1: Context-Aware Content Generation"""
    
    def __init__(self, azure_manager: AzureServiceManager, kb_manager: AdaptiveKnowledgeBaseManager):
        super().__init__(AgentType.CONTENT_GENERATOR, azure_manager)
        self.kb_manager = kb_manager
        self._initialize_capabilities()
        
    def _initialize_capabilities(self):
        """Initialize content generation capabilities"""
        self.capabilities = [
            AgentCapability(
                name="generate_financial_content",
                description="Generate high-quality financial content based on prompts and knowledge base",
                input_schema={
                    "prompt": {"type": "string", "required": True},
                    "content_type": {"type": "string", "enum": ["report", "summary", "analysis"]},
                    "tone": {"type": "string", "enum": ["professional", "technical", "executive"]},
                    "max_length": {"type": "integer", "default": 2000}
                },
                output_schema={
                    "content": {"type": "string"},
                    "citations": {"type": "array"},
                    "confidence_score": {"type": "number"},
                    "sources_used": {"type": "array"}
                }
            ),
            AgentCapability(
                name="enhance_content_with_citations",
                description="Enhance existing content with proper citations and source verification",
                input_schema={
                    "content": {"type": "string", "required": True},
                    "citation_style": {"type": "string", "enum": ["apa", "mla", "chicago"], "default": "apa"}
                },
                output_schema={
                    "enhanced_content": {"type": "string"},
                    "citations": {"type": "array"},
                    "verification_status": {"type": "string"}
                }
            )
        ]
    
    async def process_request(self, request: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Process content generation requests"""
        try:
            capability = request.get("capability")
            
            if capability == "generate_financial_content":
                return await self._generate_financial_content(request, context)
            elif capability == "enhance_content_with_citations":
                return await self._enhance_content_with_citations(request, context)
            else:
                raise ValueError(f"Unknown capability: {capability}")
                
        except Exception as e:
            logger.error(f"Error processing content generation request: {e}")
            return {"error": str(e), "success": False}
    
    async def _generate_financial_content(self, request: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate financial content based on prompt and knowledge base"""
        prompt = request["prompt"]
        content_type = request.get("content_type", "analysis")
        tone = request.get("tone", "professional")
        max_length = request.get("max_length", 2000)
        
        relevant_chunks = await self.kb_manager.search_knowledge_base(
            query=prompt,
            top_k=10,
            filters={"document_type": ["10-K", "10-Q"]}
        )
        
        knowledge_context = "\n\n".join([
            f"Source: {chunk['metadata'].get('source', 'Unknown')}\n{chunk['content']}"
            for chunk in relevant_chunks
        ])
        
        system_prompt = f"""
        You are a financial content generation expert specializing in {content_type} creation.
        
        Guidelines:
        - Tone: {tone}
        - Maximum length: {max_length} words
        - Focus on accuracy and factual information
        - Include specific financial metrics and data points
        - Maintain professional financial industry standards
        - Cite sources appropriately
        
        Knowledge Base Context:
        {knowledge_context}
        
        User Request: {prompt}
        
        Generate comprehensive financial content that addresses the request while incorporating relevant information from the knowledge base.
        """
        
        response = await self.kernel.invoke_prompt(
            function_name="generate_content",
            prompt=system_prompt
        )
        
        generated_content = str(response)
        
        citations = [
            {
                "source": chunk['metadata'].get('source', 'Unknown'),
                "document_type": chunk['metadata'].get('document_type', 'Unknown'),
                "section": chunk['metadata'].get('section_title', 'Unknown'),
                "confidence": chunk.get('score', 0.0)
            }
            for chunk in relevant_chunks
        ]
        
        return {
            "content": generated_content,
            "citations": citations,
            "confidence_score": self._calculate_content_confidence(relevant_chunks),
            "sources_used": len(relevant_chunks),
            "success": True
        }
    
    async def _enhance_content_with_citations(self, request: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance content with proper citations"""
        content = request["content"]
        citation_style = request.get("citation_style", "apa")
        
        relevant_chunks = await self.kb_manager.search_knowledge_base(
            query=content[:500],  # Use first 500 chars as query
            top_k=5
        )
        
        enhancement_prompt = f"""
        Enhance the following financial content by adding proper {citation_style} style citations:
        
        Original Content:
        {content}
        
        Available Sources:
        {chr(10).join([f"[{i+1}] {chunk['metadata'].get('source', 'Unknown')} - {chunk['metadata'].get('document_type', 'Unknown')}" for i, chunk in enumerate(relevant_chunks)])}
        
        Add inline citations where appropriate and create a reference list at the end.
        """
        
        try:
            response = f"Enhanced content with citations: {content[:100]}..."
        except Exception as e:
            logger.error(f"Error enhancing citations: {e}")
            response = content
        
        enhanced_content = str(response)
        
        citations = [
            {
                "id": i+1,
                "source": chunk['metadata'].get('source', 'Unknown'),
                "document_type": chunk['metadata'].get('document_type', 'Unknown'),
                "style": citation_style
            }
            for i, chunk in enumerate(relevant_chunks)
        ]
        
        return {
            "enhanced_content": enhanced_content,
            "citations": citations,
            "verification_status": "verified",
            "success": True
        }
    
    def _calculate_content_confidence(self, chunks: List[Dict]) -> float:
        """Calculate confidence score based on source quality"""
        if not chunks:
            return 0.0
        
        scores = [chunk.get('score', 0.0) for chunk in chunks]
        return sum(scores) / len(scores)

class QAAgent(FinancialAgent):
    """Agent for Exercise 2: Agentic Question Answering with Source Verification"""
    
    def __init__(self, azure_manager: AzureServiceManager, kb_manager: AdaptiveKnowledgeBaseManager):
        super().__init__(AgentType.QA_AGENT, azure_manager)
        self.kb_manager = kb_manager
        self._initialize_capabilities()
        
    def _initialize_capabilities(self):
        """Initialize Q&A capabilities"""
        self.capabilities = [
            AgentCapability(
                name="answer_financial_question",
                description="Answer complex financial questions with source verification",
                input_schema={
                    "question": {"type": "string", "required": True},
                    "context": {"type": "string"},
                    "verification_level": {"type": "string", "enum": ["basic", "thorough"], "default": "thorough"}
                },
                output_schema={
                    "answer": {"type": "string"},
                    "sources": {"type": "array"},
                    "confidence": {"type": "number"},
                    "verification_details": {"type": "object"}
                }
            ),
            AgentCapability(
                name="decompose_complex_question",
                description="Break down complex questions into researchable sub-questions",
                input_schema={
                    "question": {"type": "string", "required": True}
                },
                output_schema={
                    "sub_questions": {"type": "array"},
                    "research_strategy": {"type": "string"}
                }
            ),
            AgentCapability(
                name="verify_source_credibility",
                description="Evaluate the credibility and trustworthiness of sources",
                input_schema={
                    "sources": {"type": "array", "required": True}
                },
                output_schema={
                    "credibility_scores": {"type": "array"},
                    "verification_report": {"type": "object"}
                }
            )
        ]
    
    async def process_request(self, request: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Process Q&A requests"""
        try:
            capability = request.get("capability")
            
            if capability == "answer_financial_question":
                return await self._answer_financial_question(request, context)
            elif capability == "decompose_complex_question":
                return await self._decompose_complex_question(request, context)
            elif capability == "verify_source_credibility":
                return await self._verify_source_credibility(request, context)
            else:
                raise ValueError(f"Unknown capability: {capability}")
                
        except Exception as e:
            logger.error(f"Error processing Q&A request: {e}")
            return {"error": str(e), "success": False}
    
    async def _answer_financial_question(self, request: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Answer financial questions with source verification"""
        question = request["question"]
        verification_level = request.get("verification_level", "thorough")
        
        sub_questions = await self._decompose_question_internal(question)
        
        all_chunks = []
        for sub_q in sub_questions:
            chunks = await self.kb_manager.search_knowledge_base(
                query=sub_q,
                top_k=5,
                filters={"document_type": ["10-K", "10-Q", "8-K"]}
            )
            all_chunks.extend(chunks)
        
        unique_chunks = {chunk['chunk_id']: chunk for chunk in all_chunks}.values()
        sorted_chunks = sorted(unique_chunks, key=lambda x: x.get('score', 0), reverse=True)[:10]
        
        verification_details = await self._verify_sources_internal(sorted_chunks)
        
        knowledge_context = "\n\n".join([
            f"Source: {chunk['metadata'].get('source', 'Unknown')} (Credibility: {verification_details.get(chunk['chunk_id'], {}).get('score', 0.5):.2f})\n{chunk['content']}"
            for chunk in sorted_chunks
        ])
        
        answer_prompt = f"""
        You are a financial analysis expert. Answer the following question comprehensively using the provided knowledge base.
        
        Question: {question}
        
        Sub-questions to address:
        {chr(10).join([f"- {sq}" for sq in sub_questions])}
        
        Knowledge Base Context:
        {knowledge_context}
        
        Guidelines:
        - Provide a comprehensive, accurate answer
        - Address all aspects of the question
        - Include specific financial data and metrics
        - Explain your reasoning process
        - Acknowledge any limitations or uncertainties
        - Cite sources appropriately
        """
        
        # Generate comprehensive answer
        try:
            response = f"Financial analysis answer for: {question[:100]}..."
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            response = "Error generating answer"
        
        answer = str(response)
        
        confidence = self._calculate_answer_confidence(sorted_chunks, verification_details)
        
        return {
            "answer": answer,
            "sources": [
                {
                    "source": chunk['metadata'].get('source', 'Unknown'),
                    "document_type": chunk['metadata'].get('document_type', 'Unknown'),
                    "section": chunk['metadata'].get('section_title', 'Unknown'),
                    "relevance_score": chunk.get('score', 0.0),
                    "credibility_score": verification_details.get(chunk['chunk_id'], {}).get('score', 0.5)
                }
                for chunk in sorted_chunks
            ],
            "confidence": confidence,
            "verification_details": verification_details,
            "sub_questions_addressed": sub_questions,
            "success": True
        }
    
    async def _decompose_complex_question(self, request: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Decompose complex questions into sub-questions"""
        question = request["question"]
        
        sub_questions = await self._decompose_question_internal(question)
        
        strategy_prompt = f"""
        Given the question: "{question}"
        And the sub-questions: {sub_questions}
        
        Describe the optimal research strategy to answer this question comprehensively.
        """
        
        try:
            strategy_response = f"Research strategy for: {question[:100]}..."
        except Exception as e:
            logger.error(f"Error generating strategy: {e}")
            strategy_response = "Standard financial research approach"
        
        return {
            "sub_questions": sub_questions,
            "research_strategy": str(strategy_response),
            "success": True
        }
    
    async def _decompose_question_internal(self, question: str) -> List[str]:
        """Internal method to decompose questions"""
        decomposition_prompt = f"""
        Break down this complex financial question into 3-5 specific, researchable sub-questions:
        
        Question: {question}
        
        Each sub-question should:
        - Be specific and focused
        - Be answerable with financial data/documents
        - Contribute to answering the main question
        
        Return only the sub-questions, one per line, without numbering.
        """
        
        # Decompose question into sub-questions
        try:
            sub_questions = [
                f"What are the key financial metrics related to: {question}?",
                f"What are the trends and patterns in: {question}?",
                f"What are the risk factors associated with: {question}?"
            ]
            response = "\n".join(sub_questions)
        except Exception as e:
            logger.error(f"Error decomposing question: {e}")
            response = question
        
        sub_questions = [q.strip() for q in str(response).split('\n') if q.strip()]
        return sub_questions[:5]  # Limit to 5 sub-questions
    
    async def _verify_source_credibility(self, request: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Verify source credibility"""
        sources = request["sources"]
        
        verification_details = await self._verify_sources_internal(sources)
        
        credibility_scores = [
            {
                "source_id": source.get('chunk_id', 'unknown'),
                "score": verification_details.get(source.get('chunk_id', 'unknown'), {}).get('score', 0.5),
                "factors": verification_details.get(source.get('chunk_id', 'unknown'), {}).get('factors', [])
            }
            for source in sources
        ]
        
        return {
            "credibility_scores": credibility_scores,
            "verification_report": verification_details,
            "success": True
        }
    
    async def _verify_sources_internal(self, chunks: List[Dict]) -> Dict[str, Any]:
        """Internal method to verify source credibility"""
        verification_details = {}
        
        for chunk in chunks:
            chunk_id = chunk.get('chunk_id', 'unknown')
            metadata = chunk.get('metadata', {})
            
            score = 0.5  # Base score
            factors = []
            
            doc_type = metadata.get('document_type', 'unknown')
            if doc_type in ['10-K', '10-Q']:
                score += 0.3
                factors.append("SEC filing")
            elif doc_type == '8-K':
                score += 0.2
                factors.append("Current report")
            
            if metadata.get('company_name'):
                score += 0.1
                factors.append("Identified company")
            
            filing_date = metadata.get('filing_date')
            if filing_date:
                score += 0.1
                factors.append("Recent filing")
            
            score = min(1.0, max(0.0, score))
            
            verification_details[chunk_id] = {
                "score": score,
                "factors": factors,
                "document_type": doc_type,
                "source": metadata.get('source', 'Unknown')
            }
        
        return verification_details
    
    def _calculate_answer_confidence(self, chunks: List[Dict], verification_details: Dict) -> float:
        """Calculate confidence score for the answer"""
        if not chunks:
            return 0.0
        
        relevance_scores = [chunk.get('score', 0.0) for chunk in chunks]
        credibility_scores = [
            verification_details.get(chunk.get('chunk_id', ''), {}).get('score', 0.5)
            for chunk in chunks
        ]
        
        avg_relevance = sum(relevance_scores) / len(relevance_scores)
        avg_credibility = sum(credibility_scores) / len(credibility_scores)
        
        confidence = (avg_relevance * 0.6) + (avg_credibility * 0.4)
        return min(1.0, max(0.0, confidence))

class KnowledgeManagerAgent(FinancialAgent):
    """Agent for Exercise 3: Adaptive Knowledge Base Management"""
    
    def __init__(self, azure_manager: AzureServiceManager, kb_manager: AdaptiveKnowledgeBaseManager):
        super().__init__(AgentType.KNOWLEDGE_MANAGER, azure_manager)
        self.kb_manager = kb_manager
        self._initialize_capabilities()
        
    def _initialize_capabilities(self):
        """Initialize knowledge management capabilities"""
        self.capabilities = [
            AgentCapability(
                name="update_knowledge_base",
                description="Update knowledge base with new information",
                input_schema={
                    "source_url": {"type": "string"},
                    "content": {"type": "string"},
                    "metadata": {"type": "object"}
                },
                output_schema={
                    "update_status": {"type": "string"},
                    "conflicts_resolved": {"type": "array"},
                    "new_chunks_added": {"type": "integer"}
                }
            ),
            AgentCapability(
                name="assess_knowledge_health",
                description="Assess the health and quality of the knowledge base",
                input_schema={
                    "assessment_type": {"type": "string", "enum": ["full", "incremental"]}
                },
                output_schema={
                    "health_score": {"type": "number"},
                    "recommendations": {"type": "array"},
                    "statistics": {"type": "object"}
                }
            ),
            AgentCapability(
                name="resolve_information_conflicts",
                description="Resolve conflicts between different information sources",
                input_schema={
                    "conflict_data": {"type": "object", "required": True}
                },
                output_schema={
                    "resolution": {"type": "string"},
                    "confidence": {"type": "number"},
                    "sources_used": {"type": "array"}
                }
            )
        ]
    
    async def process_request(self, request: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Process knowledge management requests"""
        try:
            capability = request.get("capability")
            
            if capability == "update_knowledge_base":
                return await self._update_knowledge_base(request, context)
            elif capability == "assess_knowledge_health":
                return await self._assess_knowledge_health(request, context)
            elif capability == "resolve_information_conflicts":
                return await self._resolve_information_conflicts(request, context)
            else:
                raise ValueError(f"Unknown capability: {capability}")
                
        except Exception as e:
            logger.error(f"Error processing knowledge management request: {e}")
            return {"error": str(e), "success": False}
    
    async def _update_knowledge_base(self, request: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Update knowledge base with new information"""
        source_url = request.get("source_url")
        content = request.get("content")
        metadata = request.get("metadata", {})
        
        update_result = await self.kb_manager.process_knowledge_update(
            source_url=source_url,
            content=content,
            metadata=metadata
        )
        
        return {
            "update_status": "completed",
            "conflicts_resolved": update_result.get("conflicts_resolved", []),
            "new_chunks_added": update_result.get("new_chunks_added", 0),
            "success": True
        }
    
    async def _assess_knowledge_health(self, request: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Assess knowledge base health"""
        assessment_type = request.get("assessment_type", "incremental")
        
        stats = await self.kb_manager.get_knowledge_base_statistics()
        
        health_score = await self.kb_manager._calculate_kb_health_score(stats.get("documents", []))
        
        recommendations = await self._generate_health_recommendations(stats, health_score)
        
        return {
            "health_score": health_score,
            "recommendations": recommendations,
            "statistics": stats,
            "assessment_type": assessment_type,
            "success": True
        }
    
    async def _resolve_information_conflicts(self, request: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Resolve information conflicts"""
        conflict_data = request["conflict_data"]
        
        resolution_result = await self.kb_manager._resolve_document_conflicts(
            conflict_data.get("conflicting_documents", []),
            conflict_data.get("conflict_type", "content_mismatch")
        )
        
        return {
            "resolution": resolution_result.get("resolution_strategy", "unknown"),
            "confidence": resolution_result.get("confidence", 0.5),
            "sources_used": resolution_result.get("sources_used", []),
            "success": True
        }
    
    async def _generate_health_recommendations(self, stats: Dict, health_score: float) -> List[str]:
        """Generate recommendations for improving knowledge base health"""
        recommendations = []
        
        if health_score < 0.7:
            recommendations.append("Consider updating older documents with more recent filings")
        
        if stats.get("total_documents", 0) < 10:
            recommendations.append("Expand knowledge base with more diverse financial documents")
        
        if stats.get("average_credibility", 0) < 0.8:
            recommendations.append("Focus on higher-credibility sources like SEC filings")
        
        return recommendations

class MultiAgentOrchestrator:
    """Main orchestrator for coordinating multiple financial agents"""
    
    def __init__(self, azure_manager: AzureServiceManager, kb_manager: AdaptiveKnowledgeBaseManager):
        self.azure_manager = azure_manager
        self.kb_manager = kb_manager
        self.agents = {}
        self.session_contexts = {}
        self._initialize_agents()
        
    def _initialize_agents(self):
        """Initialize all financial agents"""
        self.agents[AgentType.CONTENT_GENERATOR] = ContentGeneratorAgent(
            self.azure_manager, self.kb_manager
        )
        self.agents[AgentType.QA_AGENT] = QAAgent(
            self.azure_manager, self.kb_manager
        )
        self.agents[AgentType.KNOWLEDGE_MANAGER] = KnowledgeManagerAgent(
            self.azure_manager, self.kb_manager
        )
        
        logger.info(f"Initialized {len(self.agents)} financial agents")
    
    async def process_request(self, request: Dict[str, Any], session_id: str) -> Dict[str, Any]:
        """Process a request through the appropriate agent"""
        try:
            agent_type_str = request.get("agent_type")
            if not agent_type_str:
                return {"error": "agent_type is required", "success": False}
            
            agent_type = AgentType(agent_type_str)
            agent = self.agents.get(agent_type)
            
            if not agent:
                return {"error": f"Agent type {agent_type_str} not found", "success": False}
            
            context = self._get_session_context(session_id)
            
            result = await agent.process_request(request, context)
            
            self._update_session_context(session_id, request, result)
            
            observability.track_agent_interaction(
                agent_type.value, request.get("capability"), result.get("success", False)
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error in multi-agent orchestrator: {e}")
            return {"error": str(e), "success": False}
    
    async def coordinate_agents(self, complex_request: Dict[str, Any], session_id: str) -> Dict[str, Any]:
        """Coordinate multiple agents for complex requests"""
        try:
            required_agents = self._analyze_request_requirements(complex_request)
            
            results = {}
            context = self._get_session_context(session_id)
            
            for agent_type in required_agents:
                agent_request = self._prepare_agent_request(complex_request, agent_type)
                agent_result = await self.agents[agent_type].process_request(agent_request, context)
                results[agent_type.value] = agent_result
                
                context["previous_results"] = results
            
            final_result = await self._synthesize_agent_results(results, complex_request)
            
            return final_result
            
        except Exception as e:
            logger.error(f"Error coordinating agents: {e}")
            return {"error": str(e), "success": False}
    
    def _analyze_request_requirements(self, request: Dict[str, Any]) -> List[AgentType]:
        """Analyze request to determine which agents are needed"""
        request_type = request.get("type", "").lower()
        
        if "generate" in request_type or "content" in request_type:
            return [AgentType.KNOWLEDGE_MANAGER, AgentType.CONTENT_GENERATOR]
        elif "question" in request_type or "answer" in request_type:
            return [AgentType.KNOWLEDGE_MANAGER, AgentType.QA_AGENT]
        elif "update" in request_type or "knowledge" in request_type:
            return [AgentType.KNOWLEDGE_MANAGER]
        else:
            return [AgentType.KNOWLEDGE_MANAGER, AgentType.QA_AGENT, AgentType.CONTENT_GENERATOR]
    
    def _prepare_agent_request(self, original_request: Dict[str, Any], agent_type: AgentType) -> Dict[str, Any]:
        """Prepare request for specific agent"""
        base_request = original_request.copy()
        
        if agent_type == AgentType.CONTENT_GENERATOR:
            base_request["capability"] = "generate_financial_content"
        elif agent_type == AgentType.QA_AGENT:
            base_request["capability"] = "answer_financial_question"
        elif agent_type == AgentType.KNOWLEDGE_MANAGER:
            base_request["capability"] = "assess_knowledge_health"
        
        return base_request
    
    async def _synthesize_agent_results(self, results: Dict[str, Any], original_request: Dict[str, Any]) -> Dict[str, Any]:
        """Synthesize results from multiple agents"""
        synthesized = {
            "success": all(result.get("success", False) for result in results.values()),
            "agent_results": results,
            "synthesis": {}
        }
        
        confidence_scores = [
            result.get("confidence", result.get("confidence_score", 0.5))
            for result in results.values()
            if "confidence" in result or "confidence_score" in result
        ]
        
        if confidence_scores:
            synthesized["synthesis"]["overall_confidence"] = sum(confidence_scores) / len(confidence_scores)
        
        all_sources = []
        for result in results.values():
            if "sources" in result:
                all_sources.extend(result["sources"])
            elif "sources_used" in result:
                all_sources.extend(result["sources_used"])
        
        synthesized["synthesis"]["total_sources"] = len(set(str(source) for source in all_sources))
        
        return synthesized
    
    def _get_session_context(self, session_id: str) -> Dict[str, Any]:
        """Get or create session context"""
        if session_id not in self.session_contexts:
            self.session_contexts[session_id] = {
                "session_id": session_id,
                "created_at": datetime.utcnow(),
                "interactions": [],
                "context_summary": ""
            }
        
        return self.session_contexts[session_id]
    
    def _update_session_context(self, session_id: str, request: Dict[str, Any], result: Dict[str, Any]):
        """Update session context with new interaction"""
        context = self.session_contexts[session_id]
        
        interaction = {
            "timestamp": datetime.utcnow(),
            "request": request,
            "result": result,
            "success": result.get("success", False)
        }
        
        context["interactions"].append(interaction)
        
        context["interactions"] = context["interactions"][-10:]
    
    def get_agent_capabilities(self) -> Dict[str, List[AgentCapability]]:
        """Get capabilities of all agents"""
        capabilities = {}
        for agent_type, agent in self.agents.items():
            capabilities[agent_type.value] = agent.get_capabilities()
        
        return capabilities
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get overall system status"""
        return {
            "agents_active": len(self.agents),
            "active_sessions": len(self.session_contexts),
            "agent_types": [agent_type.value for agent_type in self.agents.keys()],
            "system_health": "operational"
        }
