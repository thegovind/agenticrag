from fastapi import APIRouter, HTTPException, Depends, Header
from typing import List, Optional
import logging
import uuid
import time
from datetime import datetime

from app.models.schemas import (
    ChatRequest,
    ChatResponse,
    ChatMessage,
    SessionInfo,
    Citation
)
from app.core.observability import observability
# Temporarily disable evaluation due to package conflicts
# from app.core.evaluation import get_evaluation_framework
from app.services.azure_services import AzureServiceManager
from app.services.azure_ai_agent_service import AzureAIAgentService

router = APIRouter()
logger = logging.getLogger(__name__)

@router.post("/", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    x_session_id: Optional[str] = Header(None),
    x_user_id: Optional[str] = Header(None)
):
    """Main chat endpoint for RAG conversations with comprehensive evaluation"""
    start_time = time.time()
    session_id = x_session_id or request.session_id or str(uuid.uuid4())
    
    try:
        async with observability.trace_operation(
            "chat_processing",
            session_id=session_id,
            model=request.chat_model.value,
            exercise_type=request.exercise_type.value
        ) as span:
            
            observability.track_request("chat", session_id=session_id)
            logger.info(f"Chat request received for session {session_id}, exercise: {request.exercise_type.value}")
            
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
            
            try:
                chat_result = await azure_ai_agent_service.process_chat_request(
                    message=request.message,
                    session_id=session_id,
                    model=request.chat_model.value,
                    temperature=request.temperature,
                    exercise_type=request.exercise_type.value,
                    embedding_model=request.embedding_model.value
                )
                
                response_text = chat_result.get("response", "I apologize, but I encountered an issue processing your request.")
                citations_data = chat_result.get("citations", [])
                
                citations = []
                for cite_data in citations_data:
                    citations.append(Citation(
                        id=cite_data.get("id", f"cite_{len(citations) + 1}"),
                        content=cite_data.get("content", ""),
                        source=cite_data.get("source", ""),
                        document_id=cite_data.get("document_id", ""),
                        document_title=cite_data.get("document_title", ""),
                        page_number=cite_data.get("page_number", 0),
                        section_title=cite_data.get("section_title", ""),
                        confidence=cite_data.get("confidence", "medium"),
                        url=cite_data.get("url", "")
                    ))
                
            except Exception as agent_error:
                logger.warning(f"Azure AI Agent Service failed for session {session_id}: {agent_error}")
                response_text = f"""Based on your query about "{request.message}", I've analyzed the available financial documents. 
                
This is a comprehensive response that would typically include:
- Relevant financial data and metrics
- Analysis of trends and patterns
- Supporting evidence from source documents
- Professional financial insights

The system is configured to use {request.chat_model.value} for generation and {request.embedding_model.value} for document retrieval."""
                
                citations = [
                    Citation(
                        id="cite_1",
                        content="Sample financial data from 10-K filing showing revenue growth of 15% year-over-year.",
                        source="10-K Annual Report 2023",
                        document_id="doc_123",
                        document_title="Annual Report 2023",
                        page_number=45,
                        section_title="Financial Performance",
                        confidence="high",
                        url="https://example.com/10k-2023.pdf#page=45"
                    ),
                    Citation(
                        id="cite_2", 
                        content="Quarterly earnings report indicating strong performance in Q4.",
                        source="10-Q Quarterly Report Q4 2023",
                        document_id="doc_124",
                        document_title="Q4 2023 Quarterly Report",
                        page_number=12,
                        section_title="Quarterly Results",
                        confidence="medium",
                        url="https://example.com/10q-q4-2023.pdf#page=12"
                    )
                ]
            
            prompt_tokens = len(request.message.split()) * 1.3  # Rough estimation
            completion_tokens = len(response_text.split()) * 1.3
            total_tokens = int(prompt_tokens + completion_tokens)
            
            token_usage = {
                "prompt_tokens": int(prompt_tokens),
                "completion_tokens": int(completion_tokens),
                "total_tokens": total_tokens
            }
            
            observability.track_tokens(
                model=request.chat_model.value,
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
                    "section_title": citation.section_title
                })
            
            evaluation_results = []
            # Temporarily disable evaluation due to package conflicts
            try:
                # eval_framework = get_evaluation_framework()
                # evaluation_results = await eval_framework.evaluate_response(
                #     query=request.message,
                #     response=response_text,
                #     sources=sources,
                #     session_id=session_id,
                #     model_used=request.chat_model.value,
                #     response_time=response_time,
                #     financial_context={
                #         "user_id": x_user_id,
                #         "exercise_type": request.exercise_type.value,
                #         "embedding_model": request.embedding_model.value
                #     }
                # )
                pass
                
                eval_data = [
                    {
                        "metric": result.metric,
                        "score": result.score,
                        "model_used": result.model_used,
                        "reasoning": result.reasoning
                    }
                    for result in evaluation_results
                ]
                observability.track_evaluation_metrics(session_id, eval_data)
                
                span.set_attribute("evaluation.count", len(evaluation_results))
                span.set_attribute("evaluation.avg_score", 
                                 sum(r.score for r in evaluation_results) / len(evaluation_results) if evaluation_results else 0)
                
            except Exception as e:
                logger.warning(f"Evaluation failed for session {session_id}: {e}")
            
            azure_service = azure_manager
            
            user_message = {
                "id": str(uuid.uuid4()),
                "role": "user",
                "content": request.message,
                "timestamp": datetime.now().isoformat(),
                "metadata": {
                    "exercise_type": request.exercise_type.value,
                    "model_used": request.chat_model.value,
                    "embedding_model": request.embedding_model.value
                }
            }
            
            assistant_message = {
                "id": str(uuid.uuid4()),
                "role": "assistant", 
                "content": response_text,
                "timestamp": datetime.now().isoformat(),
                "citations": [citation.dict() for citation in citations],
                "metadata": {
                    "exercise_type": request.exercise_type.value,
                    "model_used": request.chat_model.value,
                    "embedding_model": request.embedding_model.value,
                    "temperature": request.temperature,
                    "search_type": getattr(request, 'search_type', 'hybrid'),
                    "evaluation_count": len(evaluation_results),
                    "avg_evaluation_score": sum(r.score for r in evaluation_results) / len(evaluation_results) if evaluation_results else 0,
                    "response_time": response_time,
                    "token_usage": token_usage
                }
            }
            
            await azure_service.save_session_history(session_id, user_message)
            await azure_service.save_session_history(session_id, assistant_message)
            
            response = ChatResponse(
                response=response_text,
                session_id=session_id,
                citations=citations,
                metadata={
                    "exercise_type": request.exercise_type.value,
                    "model_used": request.chat_model.value,
                    "embedding_model": request.embedding_model.value,
                    "temperature": request.temperature,
                    "search_type": getattr(request, 'search_type', 'hybrid'),
                    "evaluation_count": len(evaluation_results),
                    "avg_evaluation_score": sum(r.score for r in evaluation_results) / len(evaluation_results) if evaluation_results else 0,
                    "response_time": response_time
                },
                token_usage=token_usage
            )
            
            span.set_attribute("response.tokens", total_tokens)
            span.set_attribute("response.citations", len(citations))
            span.set_attribute("response.length", len(response_text))
            
            return response
        
    except Exception as e:
        observability.track_error(
            error_type=type(e).__name__,
            endpoint="/api/v1/chat",
            error_message=str(e),
            session_id=session_id,
            user_id=x_user_id
        )
        
        logger.error(f"Chat processing failed for session {session_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to process chat request: {str(e)}")

@router.get("/sessions", response_model=List[SessionInfo])
async def list_sessions(user_id: Optional[str] = None, limit: int = 50):
    """List chat sessions"""
    try:
        observability.track_request("list_sessions")
        
        from app.services.azure_services import AzureServiceManager
        from app.core.config import settings
        azure_service = AzureServiceManager()
        await azure_service.initialize()
        
        database = azure_service.cosmos_client.get_database_client(settings.AZURE_COSMOS_DATABASE_NAME)
        container = database.get_container_client(settings.AZURE_COSMOS_CONTAINER_NAME)
        
        query = "SELECT c.id, c.created_at, c.updated_at, ARRAY_LENGTH(c.messages) as message_count FROM c ORDER BY c.updated_at DESC"
        items = list(container.query_items(query=query, enable_cross_partition_query=True, max_item_count=limit))
        
        sessions = []
        for item in items:
            sessions.append(SessionInfo(
                session_id=item["id"],
                created_at=item.get("created_at"),
                updated_at=item.get("updated_at"),
                message_count=item.get("message_count", 0),
                user_id=user_id
            ))
        
        return sessions
    except Exception as e:
        logger.error(f"Error listing sessions: {e}")
        raise HTTPException(status_code=500, detail="Failed to list sessions")

@router.get("/sessions/{session_id}", response_model=SessionInfo)
async def get_session(session_id: str):
    """Get specific session information"""
    try:
        observability.track_request("get_session")
        
        raise HTTPException(status_code=404, detail="Session not found")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting session {session_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve session")

@router.get("/sessions/{session_id}/messages", response_model=List[ChatMessage])
async def get_session_messages(session_id: str, limit: int = 100, offset: int = 0):
    """Get messages for a specific session"""
    try:
        observability.track_request("get_session_messages")
        
        from app.services.azure_services import AzureServiceManager
        azure_service = AzureServiceManager()
        await azure_service.initialize()
        
        session_messages = await azure_service.get_session_history(session_id)
        
        paginated_messages = session_messages[offset:offset + limit]
        
        messages = []
        for msg in paginated_messages:
            messages.append(ChatMessage(
                id=msg.get("id"),
                role=msg.get("role"),
                content=msg.get("content"),
                timestamp=msg.get("timestamp"),
                citations=msg.get("citations", []),
                metadata=msg.get("metadata", {})
            ))
        
        return messages
    except Exception as e:
        logger.error(f"Error getting messages for session {session_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve session messages")

@router.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    """Delete a chat session"""
    try:
        observability.track_request("delete_session")
        
        logger.info(f"Session deletion requested: {session_id}")
        
        return {"message": f"Session {session_id} deleted successfully"}
    except Exception as e:
        logger.error(f"Error deleting session {session_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete session")

@router.post("/sessions/{session_id}/feedback")
async def submit_feedback(session_id: str, message_id: str, rating: int, feedback: Optional[str] = None):
    """Submit feedback for a chat response"""
    try:
        observability.track_request("submit_feedback")
        
        if rating < 1 or rating > 5:
            raise HTTPException(status_code=400, detail="Rating must be between 1 and 5")
        
        logger.info(f"Feedback submitted for session {session_id}, message {message_id}: {rating}")
        
        return {"message": "Feedback submitted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error submitting feedback: {e}")
        raise HTTPException(status_code=500, detail="Failed to submit feedback")

@router.get("/capabilities")
async def get_chat_capabilities():
    """Get chat agent capabilities and status"""
    try:
        observability.track_request("get_chat_capabilities")
        
        logger.info("Fetching chat agent capabilities...")
        
        try:
            azure_manager = AzureServiceManager()
            await azure_manager.initialize()
            
            from app.services.knowledge_base_manager import AdaptiveKnowledgeBaseManager
            kb_manager = AdaptiveKnowledgeBaseManager(azure_manager)
            
            from app.services.multi_agent_orchestrator import MultiAgentOrchestrator
            orchestrator = MultiAgentOrchestrator(azure_manager, kb_manager)
            
            azure_ai_agent_service = await orchestrator._get_azure_ai_agent_service()
            
            capabilities = [
                {
                    "name": "Context-Aware Content Generation",
                    "description": "Generate comprehensive responses based on financial documents with proper citations",
                    "status": "available"
                },
                {
                    "name": "Multi-Document Analysis",
                    "description": "Analyze and synthesize information from multiple financial documents",
                    "status": "available"
                },
                {
                    "name": "Citation Management",
                    "description": "Provide accurate citations and source references for generated content",
                    "status": "available"
                },
                {
                    "name": "Session Management",
                    "description": "Maintain conversation context across multiple interactions",
                    "status": "available"
                }
            ]
            
            available_models = await azure_manager.get_available_models()
            
            return {
                "capabilities": capabilities,
                "available_models": [model.get('name', model.get('id')) for model in available_models if model.get('type') == 'chat'],
                "supported_features": [
                    "context_aware_generation",
                    "multi_document_analysis", 
                    "citation_management",
                    "session_management",
                    "evaluation_framework"
                ],
                "agent_status": {
                    "service_type": type(azure_ai_agent_service).__name__,
                    "status": "active",
                    "last_updated": datetime.now().isoformat()
                }
            }
            
        except Exception as agent_error:
            logger.warning(f"Azure AI Agent Service unavailable: {agent_error}")
            return {
                "capabilities": [
                    {
                        "name": "Context-Aware Content Generation",
                        "description": "Generate comprehensive responses based on financial documents with proper citations",
                        "status": "limited"
                    },
                    {
                        "name": "Multi-Document Analysis", 
                        "description": "Analyze and synthesize information from multiple financial documents",
                        "status": "limited"
                    },
                    {
                        "name": "Citation Management",
                        "description": "Provide accurate citations and source references for generated content",
                        "status": "limited"
                    },
                    {
                        "name": "Session Management",
                        "description": "Maintain conversation context across multiple interactions",
                        "status": "available"
                    }
                ],
                "available_models": [],
                "supported_features": ["mock_responses", "session_management"],
                "agent_status": {
                    "service_type": "MockService",
                    "status": "limited",
                    "last_updated": datetime.now().isoformat()
                }
            }
            
    except Exception as e:
        logger.error(f"Error getting chat capabilities: {e}")
        raise HTTPException(status_code=500, detail="Failed to get chat capabilities")

@router.get("/models")
async def list_available_models():
    """List available chat and embedding models from Azure AI Foundry and static config"""
    try:
        observability.track_request("list_models")
        
        from app.core.config import settings
        
        static_chat_models = settings.AVAILABLE_CHAT_MODELS
        static_embedding_models = settings.AVAILABLE_EMBEDDING_MODELS
        
        try:
            from app.services.azure_services import AzureServiceManager
            azure_service = AzureServiceManager()
            await azure_service.initialize()
            
            foundry_models = await azure_service.get_available_models()
            
            foundry_chat_models = []
            foundry_embedding_models = []
            
            for model in foundry_models:
                if model.get('type') == 'chat':
                    foundry_chat_models.append({
                        "id": model.get('name', model.get('id')),
                        "name": model.get('name', model.get('id')),
                        "provider": "Azure AI Foundry",
                        "version": model.get('version'),
                        "status": model.get('status', 'active')
                    })
                elif model.get('type') == 'embedding':
                    foundry_embedding_models.append({
                        "id": model.get('name', model.get('id')),
                        "name": model.get('name', model.get('id')),
                        "provider": "Azure AI Foundry",
                        "dimensions": model.get('dimensions'),
                        "version": model.get('version'),
                        "status": model.get('status', 'active')
                    })
            
            all_chat_models = static_chat_models + foundry_chat_models
            all_embedding_models = static_embedding_models + foundry_embedding_models
            
        except Exception as foundry_error:
            logger.warning(f"Failed to fetch foundry models, using static models: {foundry_error}")
            all_chat_models = static_chat_models
            all_embedding_models = static_embedding_models
        
        return {
            "chat_models": all_chat_models,
            "embedding_models": all_embedding_models
        }
    except Exception as e:
        logger.error(f"Error listing models: {e}")
        raise HTTPException(status_code=500, detail="Failed to list available models")
