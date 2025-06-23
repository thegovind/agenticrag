from fastapi import APIRouter, HTTPException, Depends, Header
from typing import List, Optional
import logging
import uuid
import time
from datetime import datetime

from app.models.schemas import (
    QARequest,
    QAResponse,
    QuestionDecompositionRequest,
    QuestionDecompositionResponse,
    SourceVerificationRequest,
    SourceVerificationResponse,
    Citation
)
from app.core.observability import observability
# Temporarily disable evaluation due to package conflicts
# from app.core.evaluation import get_evaluation_framework
from app.services.multi_agent_orchestrator import MultiAgentOrchestrator, AgentType
from app.services.azure_ai_agent_service import AzureAIAgentService
from app.services.azure_services import AzureServiceManager

router = APIRouter()
logger = logging.getLogger(__name__)

@router.post("/ask", response_model=QAResponse)
async def ask_question(
    request: QARequest,
    x_session_id: Optional[str] = Header(None),
    x_user_id: Optional[str] = Header(None)
):
    """Main QA endpoint for complex financial question answering with source verification"""
    start_time = time.time()
    session_id = x_session_id or request.session_id or str(uuid.uuid4())
    
    try:
        async with observability.trace_operation(
            "qa_processing",
            session_id=session_id,
            model=request.chat_model,
            verification_level=request.verification_level
        ) as span:
            
            observability.track_request("qa_ask", session_id=session_id)
            logger.info(f"QA request received for session {session_id}, question: {request.question[:100]}...")
            
            logger.info("Initializing Azure services...")
            azure_manager = AzureServiceManager()
            await azure_manager.initialize()
            logger.info("Azure services initialized successfully")
            
            logger.info("Creating knowledge base manager...")
            from app.services.knowledge_base_manager import AdaptiveKnowledgeBaseManager
            kb_manager = AdaptiveKnowledgeBaseManager(azure_manager)
            
            logger.info("Creating multi-agent orchestrator...")
            from app.services.multi_agent_orchestrator import MultiAgentOrchestrator
            orchestrator = MultiAgentOrchestrator(azure_manager, kb_manager)
            
            logger.info("Getting Azure AI Agent Service...")
            azure_ai_agent_service = await orchestrator._get_azure_ai_agent_service()
            logger.info(f"Azure AI Agent Service type: {type(azure_ai_agent_service).__name__}")
            
            logger.info("Processing QA request with Azure AI Agent Service...")
            qa_result = await azure_ai_agent_service.process_qa_request(
                question=request.question,
                context={
                    **(request.context or {}),
                    'kb_manager': kb_manager
                },
                verification_level=request.verification_level,
                session_id=session_id,
                model_config={
                    "chat_model": request.chat_model,
                    "embedding_model": request.embedding_model,
                    "temperature": request.temperature
                }
            )
            logger.info(f"QA result received: {len(qa_result.get('answer', ''))} characters")
            
            answer = qa_result.get("answer", "I apologize, but I couldn't generate a comprehensive answer to your question at this time.")
            confidence_score = qa_result.get("confidence_score", 0.5)
            sub_questions = qa_result.get("sub_questions", [])
            verification_details = qa_result.get("verification_details", {})
            
            citations = []
            for source in qa_result.get("sources", []):
                citations.append(Citation(
                    id=source.get("id", str(uuid.uuid4())),
                    content=source.get("content", ""),
                    source=source.get("source", "Unknown Source"),
                    document_id=source.get("document_id", ""),
                    document_title=source.get("document_title", ""),
                    page_number=source.get("page_number"),
                    section_title=source.get("section_title"),
                    confidence=source.get("confidence", "medium"),
                    url=source.get("url", ""),
                    credibility_score=source.get("credibility_score", 0.5)
                ))
            
            prompt_tokens = len(request.question.split()) * 1.3
            completion_tokens = len(answer.split()) * 1.3
            total_tokens = int(prompt_tokens + completion_tokens)
            
            token_usage = {
                "prompt_tokens": int(prompt_tokens),
                "completion_tokens": int(completion_tokens),
                "total_tokens": total_tokens
            }
            
            observability.track_tokens(
                model=request.chat_model,
                prompt_tokens=token_usage["prompt_tokens"],
                completion_tokens=token_usage["completion_tokens"],
                session_id=session_id
            )
            
            response_time = time.time() - start_time
            
            sources = []
            for citation in citations:
                sources.append({
                    "content": citation.content,
                    "source": citation.source,
                    "title": citation.document_title,
                    "page_number": citation.page_number,
                    "section_title": citation.section_title,
                    "credibility_score": getattr(citation, 'credibility_score', 0.5)
                })
            
            evaluation_results = []
            # Temporarily disable evaluation due to package conflicts
            try:
                # eval_framework = get_evaluation_framework()
                # evaluation_results = await eval_framework.evaluate_response(
                #     query=request.question,
                #     response=answer,
                #     sources=sources,
                #     session_id=session_id,
                #     model_used=request.chat_model,
                #     response_time=response_time,
                #     financial_context={
                #         "user_id": x_user_id,
                #         "exercise_type": "qa",
                #         "embedding_model": request.embedding_model,
                #         "verification_level": request.verification_level
                #     }
                # )
                
                eval_data = []
                # eval_data = [
                #     {
                #         "metric": result.metric,
                #         "score": result.score,
                #         "model_used": result.model_used,
                #         "reasoning": result.reasoning
                #     }
                #     for result in evaluation_results
                # ]
                observability.track_evaluation_metrics(session_id, eval_data)
                
                span.set_attribute("evaluation.count", len(evaluation_results))
                span.set_attribute("evaluation.avg_score", 0)
                
            except Exception as e:
                logger.warning(f"Evaluation failed for QA session {session_id}: {e}")
            
            response = QAResponse(
                answer=answer,
                session_id=session_id,
                confidence_score=confidence_score,
                citations=citations,
                sub_questions=sub_questions,
                verification_details=verification_details,
                metadata={
                    "exercise_type": "qa",
                    "model_used": request.chat_model,
                    "embedding_model": request.embedding_model,
                    "temperature": request.temperature,
                    "verification_level": request.verification_level,
                    "evaluation_count": len(evaluation_results),
                    "avg_evaluation_score": sum(r.score for r in evaluation_results) / len(evaluation_results) if evaluation_results else 0,
                    "response_time": response_time
                },
                token_usage=token_usage
            )
            
            span.set_attribute("response.tokens", total_tokens)
            span.set_attribute("response.citations", len(citations))
            span.set_attribute("response.confidence", confidence_score)
            span.set_attribute("response.sub_questions", len(sub_questions))
            
            return response
        
    except Exception as e:
        observability.track_error(
            error_type=type(e).__name__,
            endpoint="/api/v1/qa/ask",
            error_message=str(e),
            session_id=session_id,
            user_id=x_user_id
        )
        
        logger.error(f"QA processing failed for session {session_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to process QA request: {str(e)}")

@router.post("/decompose", response_model=QuestionDecompositionResponse)
async def decompose_question(
    request: QuestionDecompositionRequest,
    x_session_id: Optional[str] = Header(None),
    x_user_id: Optional[str] = Header(None)
):
    """Decompose complex questions into researchable sub-questions"""
    start_time = time.time()
    session_id = x_session_id or str(uuid.uuid4())
    
    try:
        async with observability.trace_operation(
            "question_decomposition",
            session_id=session_id,
            model=request.chat_model.value
        ) as span:
            
            observability.track_request("qa_decompose", session_id=session_id)
            logger.info(f"Question decomposition request for session {session_id}")
            
            azure_manager = AzureServiceManager()
            await azure_manager.initialize()
            
            from app.services.knowledge_base_manager import AdaptiveKnowledgeBaseManager
            kb_manager = AdaptiveKnowledgeBaseManager(azure_manager)
            
            from app.services.multi_agent_orchestrator import MultiAgentOrchestrator
            orchestrator = MultiAgentOrchestrator(azure_manager, kb_manager)
            azure_ai_agent_service = await orchestrator._get_azure_ai_agent_service()
            
            # Use Azure AI Agent Service for question decomposition
            decomposition_result = await azure_ai_agent_service.decompose_complex_question(
                question=request.question,
                context=request.context or {},
                session_id=session_id,
                model_config={
                    "chat_model": request.chat_model.value
                }
            )
            
            sub_questions = decomposition_result.get("sub_questions", [])
            reasoning = decomposition_result.get("reasoning", "Question decomposition completed.")
            
            response_time = time.time() - start_time
            
            response = QuestionDecompositionResponse(
                original_question=request.question,
                sub_questions=sub_questions,
                reasoning=reasoning,
                session_id=session_id,
                metadata={
                    "model_used": request.chat_model.value,
                    "response_time": response_time
                }
            )
            
            span.set_attribute("response.sub_questions_count", len(sub_questions))
            
            return response
        
    except Exception as e:
        observability.track_error(
            error_type=type(e).__name__,
            endpoint="/api/v1/qa/decompose",
            error_message=str(e),
            session_id=session_id,
            user_id=x_user_id
        )
        
        logger.error(f"Question decomposition failed for session {session_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to decompose question: {str(e)}")

@router.post("/verify-sources")
async def verify_sources_flexible(
    request_data: dict,
    x_session_id: Optional[str] = Header(None),
    x_user_id: Optional[str] = Header(None)
):
    """Flexible verify sources endpoint that handles different payload formats"""
    start_time = time.time()
    session_id = request_data.get("session_id") or x_session_id or str(uuid.uuid4())
    
    try:
        async with observability.trace_operation(
            "flexible_source_verification",
            session_id=session_id
        ) as span:
            
            observability.track_request("qa_verify_sources_flexible", session_id=session_id)
            logger.info(f"Flexible source verification request for session {session_id}")
            logger.info(f"Request payload keys: {list(request_data.keys())}")
            
            azure_manager = AzureServiceManager()
            await azure_manager.initialize()
            
            from app.services.knowledge_base_manager import AdaptiveKnowledgeBaseManager
            kb_manager = AdaptiveKnowledgeBaseManager(azure_manager)
            
            from app.services.multi_agent_orchestrator import MultiAgentOrchestrator
            orchestrator = MultiAgentOrchestrator(azure_manager, kb_manager)
            azure_ai_agent_service = await orchestrator._get_azure_ai_agent_service()
            
            # Handle different payload formats
            sources_dict = []
            
            # Check if it's the single source format (from frontend)
            if "source_url" in request_data and "content" in request_data:
                logger.info("Processing single source format")
                source_dict = {
                    "id": str(uuid.uuid4()),
                    "url": request_data.get("source_url", ""),
                    "title": request_data.get("title", "Document Source"),
                    "content": request_data.get("content", ""),
                    "metadata": {
                        "source": request_data.get("source_url", ""),
                        "verification_timestamp": time.time()
                    }
                }
                sources_dict = [source_dict]
                
            # Check if it's the expected SourceVerificationRequest format
            elif "sources" in request_data:
                logger.info("Processing sources array format")
                for source in request_data["sources"]:
                    source_dict = {
                        "id": source.get("id", str(uuid.uuid4())),
                        "url": source.get("url", ""),
                        "title": source.get("title", ""),
                        "content": source.get("content", ""),
                        "metadata": source.get("metadata", {})
                    }
                    sources_dict.append(source_dict)
            else:
                raise HTTPException(status_code=422, detail="Invalid payload format. Expected either 'source_url' and 'content' or 'sources' array.")
            
            verification_result = await azure_ai_agent_service.verify_source_credibility(
                sources=sources_dict,
                context=request_data.get("context", {}),
                session_id=session_id
            )
            
            verified_sources = verification_result.get("verified_sources", [])
            overall_credibility = verification_result.get("overall_credibility_score", 0.5)
            
            response_time = time.time() - start_time
            
            # Ensure all credibility scores are proper numbers
            for source in verified_sources:
                score = source.get("credibility_score", 0.0)
                if score is None or str(score).lower() == 'nan':
                    source["credibility_score"] = 0.0
                    source["credibility_percentage"] = 0.0
                    source["verification_status"] = "unverified"
                
            overall_credibility = overall_credibility or 0.0
            if str(overall_credibility).lower() == 'nan':
                overall_credibility = 0.0
            
            response = {
                "verified_sources": verified_sources,
                "overall_credibility_score": round(overall_credibility, 3),  # Ensure it's a proper number
                "verification_summary": f"Verified {len(verified_sources)} source(s) with average credibility score of {overall_credibility:.1%}",
                "session_id": session_id,
                "metadata": {
                    "sources_count": len(sources_dict),
                    "verified_count": len([s for s in verified_sources if s["verification_status"] == "verified"]),
                    "questionable_count": len([s for s in verified_sources if s["verification_status"] == "questionable"]),
                    "unverified_count": len([s for s in verified_sources if s["verification_status"] == "unverified"]),
                    "response_time": round(response_time, 3),
                    "average_credibility_percentage": round(overall_credibility * 100, 1)
                }
            }
            
            span.set_attribute("response.sources_verified", len(verified_sources))
            span.set_attribute("response.overall_credibility", overall_credibility)
            
            return response
        
    except Exception as e:
        observability.track_error(
            error_type=type(e).__name__,
            endpoint="/api/v1/qa/verify-sources",
            error_message=str(e),
            session_id=session_id,
            user_id=x_user_id
        )
        
        logger.error(f"Flexible source verification failed for session {session_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to verify sources: {str(e)}")

@router.post("/verify-citations", response_model=SourceVerificationResponse)
async def verify_citations(
    citations: List[Citation],
    x_session_id: Optional[str] = Header(None),
    x_user_id: Optional[str] = Header(None)
):
    """Verify the credibility of Citation objects (alternative endpoint for frontend compatibility)"""
    start_time = time.time()
    session_id = x_session_id or str(uuid.uuid4())
    
    try:
        async with observability.trace_operation(
            "citation_verification",
            session_id=session_id
        ) as span:
            
            observability.track_request("qa_verify_citations", session_id=session_id)
            logger.info(f"Citation verification request for session {session_id}, {len(citations)} citations")
            
            azure_manager = AzureServiceManager()
            await azure_manager.initialize()
            
            from app.services.knowledge_base_manager import AdaptiveKnowledgeBaseManager
            kb_manager = AdaptiveKnowledgeBaseManager(azure_manager)
            
            from app.services.multi_agent_orchestrator import MultiAgentOrchestrator
            orchestrator = MultiAgentOrchestrator(azure_manager, kb_manager)
            azure_ai_agent_service = await orchestrator._get_azure_ai_agent_service()
            
            # Convert Citation objects to dictionaries for processing
            sources_dict = []
            for citation in citations:
                source_dict = {
                    "id": citation.id,
                    "url": citation.url or "",
                    "title": citation.document_title,
                    "content": citation.content,
                    "metadata": {
                        "document_id": citation.document_id,
                        "page_number": citation.page_number,
                        "section_title": citation.section_title,
                        "credibility_score": citation.credibility_score
                    }
                }
                sources_dict.append(source_dict)
            
            verification_result = await azure_ai_agent_service.verify_source_credibility(
                sources=sources_dict,
                context={},
                session_id=session_id
            )
            
            verified_sources = verification_result.get("verified_sources", [])
            overall_credibility = verification_result.get("overall_credibility_score", 0.5)
            
            response_time = time.time() - start_time
            
            response = SourceVerificationResponse(
                verified_sources=verified_sources,
                overall_credibility_score=overall_credibility,
                verification_summary=f"Verified {len(verified_sources)} citations with average credibility score of {overall_credibility:.2f}",
                session_id=session_id,
                metadata={
                    "sources_count": len(citations),
                    "verified_count": len([s for s in verified_sources if s["verification_status"] == "verified"]),
                    "response_time": response_time
                }
            )
            
            span.set_attribute("response.sources_verified", len(verified_sources))
            span.set_attribute("response.overall_credibility", overall_credibility)
            
            return response
        
    except Exception as e:
        observability.track_error(
            error_type=type(e).__name__,
            endpoint="/api/v1/qa/verify-citations",
            error_message=str(e),
            session_id=session_id,
            user_id=x_user_id
        )
        
        logger.error(f"Citation verification failed for session {session_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to verify citations: {str(e)}")

@router.post("/verify-source", response_model=SourceVerificationResponse)
async def verify_single_source(
    source_url: str,
    content: str,
    session_id: str,
    title: Optional[str] = None,
    x_session_id: Optional[str] = Header(None),
    x_user_id: Optional[str] = Header(None)
):
    """Verify a single source (matches the frontend payload format)"""
    start_time = time.time()
    session_id = session_id or x_session_id or str(uuid.uuid4())
    
    try:
        async with observability.trace_operation(
            "single_source_verification",
            session_id=session_id
        ) as span:
            
            observability.track_request("qa_verify_single_source", session_id=session_id)
            logger.info(f"Single source verification request for session {session_id}")
            
            azure_manager = AzureServiceManager()
            await azure_manager.initialize()
            
            from app.services.knowledge_base_manager import AdaptiveKnowledgeBaseManager
            kb_manager = AdaptiveKnowledgeBaseManager(azure_manager)
            
            from app.services.multi_agent_orchestrator import MultiAgentOrchestrator
            orchestrator = MultiAgentOrchestrator(azure_manager, kb_manager)
            azure_ai_agent_service = await orchestrator._get_azure_ai_agent_service()
            
            # Convert single source to the expected format
            source_dict = {
                "id": str(uuid.uuid4()),
                "url": source_url,
                "title": title or "Document Source",
                "content": content,
                "metadata": {
                    "source": source_url,
                    "verification_timestamp": time.time()
                }
            }
            
            verification_result = await azure_ai_agent_service.verify_source_credibility(
                sources=[source_dict],
                context={},
                session_id=session_id
            )
            
            verified_sources = verification_result.get("verified_sources", [])
            overall_credibility = verification_result.get("overall_credibility_score", 0.5)
            
            response_time = time.time() - start_time
            
            response = SourceVerificationResponse(
                verified_sources=verified_sources,
                overall_credibility_score=overall_credibility,
                verification_summary=f"Verified source with credibility score of {overall_credibility:.2f}",
                session_id=session_id,
                metadata={
                    "sources_count": 1,
                    "verified_count": len([s for s in verified_sources if s["verification_status"] == "verified"]),
                    "response_time": response_time,
                    "source_type": "single_source"
                }
            )
            
            span.set_attribute("response.sources_verified", len(verified_sources))
            span.set_attribute("response.overall_credibility", overall_credibility)
            
            return response
        
    except Exception as e:
        observability.track_error(
            error_type=type(e).__name__,
            endpoint="/api/v1/qa/verify-source",
            error_message=str(e),
            session_id=session_id,
            user_id=x_user_id
        )
        
        logger.error(f"Single source verification failed for session {session_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to verify source: {str(e)}")

@router.get("/capabilities")
async def get_qa_capabilities():
    """Get available QA agent capabilities"""
    try:
        observability.track_request("qa_capabilities")
        
        azure_manager = AzureServiceManager()
        await azure_manager.initialize()
        
        from app.services.knowledge_base_manager import AdaptiveKnowledgeBaseManager
        kb_manager = AdaptiveKnowledgeBaseManager(azure_manager)
        
        orchestrator = MultiAgentOrchestrator(azure_manager, kb_manager)
        all_capabilities = orchestrator.get_agent_capabilities()
        capabilities = all_capabilities.get("qa_agent", [])
        
        return {
            "capabilities": capabilities,
            "verification_levels": ["basic", "thorough", "comprehensive"],
            "supported_question_types": [
                "financial_analysis",
                "market_research", 
                "regulatory_compliance",
                "risk_assessment",
                "performance_evaluation"
            ]
        }
    except Exception as e:
        logger.error(f"Error getting QA capabilities: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve QA capabilities")
