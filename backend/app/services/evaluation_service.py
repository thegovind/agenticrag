"""
Comprehensive Evaluation Service for RAG QA System

This service provides both Azure AI Foundry and Custom evaluation capabilities
for all RAG methods, measuring various metrics like groundedness, relevance,
coherence, and fluency.

Azure AI Foundry evaluators are run in a subprocess to avoid HTTP client conflicts.
"""

import asyncio
import logging
import time
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import json
import os

# Import OpenAI for custom evaluation
from openai import AsyncAzureOpenAI

from ..core.config import settings
from ..models.schemas import (
    EvaluationRequest, EvaluationResult, EvaluationSummary, 
    EvaluatorType, EvaluationMetric
)

logger = logging.getLogger(__name__)

class EvaluationService:
    """
    Main evaluation service that orchestrates both Foundry and Custom evaluators
    """
    
    def __init__(self):
        self.foundry_evaluator = None
        self.custom_evaluator = None
        self._foundry_initialized = False
        self._custom_initialized = False
        self.azure_manager = None  # Will be set when needed for evaluation
    
    def _ensure_foundry_evaluator(self):
        """Lazy initialization of Foundry evaluator"""
        if not self._foundry_initialized:
            self._foundry_initialized = True
            try:
                if settings.AZURE_AI_PROJECT_CONNECTION_STRING:
                    self.foundry_evaluator = FoundryEvaluator()
                    logger.info("Azure AI Foundry evaluator initialized")
                else:
                    logger.warning("Azure AI Foundry configuration missing, foundry evaluator disabled")
            except Exception as e:
                logger.warning(f"Failed to initialize Foundry evaluator: {e}")
                self.foundry_evaluator = None
    
    def _ensure_custom_evaluator(self):
        """Lazy initialization of Custom evaluator"""
        if not self._custom_initialized:
            self._custom_initialized = True
            try:
                if settings.AZURE_EVALUATION_ENDPOINT or settings.AZURE_OPENAI_ENDPOINT:
                    logger.info(f"Initializing Custom evaluator with endpoint: {settings.AZURE_EVALUATION_ENDPOINT or settings.AZURE_OPENAI_ENDPOINT}")
                    logger.info(f"Using model deployment: {settings.AZURE_EVALUATION_MODEL_DEPLOYMENT}")
                    self.custom_evaluator = CustomEvaluator()
                    logger.info("Custom evaluator initialized successfully")
                else:
                    logger.warning("Custom evaluation configuration missing, custom evaluator disabled")
            except Exception as e:
                logger.error(f"Failed to initialize Custom evaluator: {e}")
                import traceback
                logger.error(f"Custom evaluator traceback:\n{traceback.format_exc()}")
                self.custom_evaluator = None
    
    def get_available_evaluators(self) -> Dict[str, Any]:
        """Get information about available evaluators"""
        # Ensure evaluators are initialized
        self._ensure_foundry_evaluator()
        self._ensure_custom_evaluator()
        
        return {
            "foundry_available": self.foundry_evaluator is not None and bool(getattr(self.foundry_evaluator, 'available_evaluators', {})),
            "custom_available": self.custom_evaluator is not None,
            "supported_evaluators": [
                {
                    "type": "foundry", 
                    "available": self.foundry_evaluator is not None and bool(getattr(self.foundry_evaluator, 'available_evaluators', {})),
                    "metrics": ["groundedness", "relevance", "coherence", "fluency"]
                },
                {
                    "type": "custom", 
                    "available": self.custom_evaluator is not None,
                    "metrics": ["groundedness", "relevance", "coherence", "fluency"]
                }
            ]
        }
    
    async def evaluate_answer(self, request: EvaluationRequest, azure_manager=None) -> EvaluationResult:
        """
        Main evaluation method that routes to appropriate evaluator
        """
        # Temporarily store azure_manager for this evaluation
        original_azure_manager = self.azure_manager
        if azure_manager:
            self.azure_manager = azure_manager
        
        logger.info(f"Starting evaluation for question_id: {request.question_id}, evaluator: {request.evaluator_type}")
        
        start_time = time.time()
        
        try:
            # Route to the appropriate evaluator based on request type - NO FALLBACKS
            if request.evaluator_type == EvaluatorType.FOUNDRY:
                self._ensure_foundry_evaluator()
                if self.foundry_evaluator and getattr(self.foundry_evaluator, 'available_evaluators', {}):
                    logger.info("Using Azure AI Foundry evaluator")
                    result = await self.foundry_evaluator.evaluate(request)
                else:
                    raise ValueError("Azure AI Foundry evaluator is not available or not properly initialized")
            else:  # Custom evaluator requested
                self._ensure_custom_evaluator()
                if self.custom_evaluator and getattr(self.custom_evaluator, 'client', None):
                    logger.info("Using Custom evaluator")
                    result = await self.custom_evaluator.evaluate(request)
                else:
                    raise ValueError("Custom evaluator is not available or not properly initialized")
            
            # Set timing information
            result.evaluation_duration_ms = int((time.time() - start_time) * 1000)
            result.evaluation_timestamp = datetime.utcnow()
            
            # Store result in Cosmos DB
            await self._store_result(result)
            
            logger.info(f"Evaluation completed for question_id: {request.question_id} in {result.evaluation_duration_ms}ms")
            return result
            
        except Exception as e:
            logger.error(f"Evaluation failed for question_id: {request.question_id}: {e}")
            # Return error result
            error_result = EvaluationResult(
                question_id=request.question_id,
                session_id=request.session_id,
                evaluator_type=request.evaluator_type,
                rag_method=request.rag_method,
                question=request.question,
                answer=request.answer,
                context=request.context,
                ground_truth=request.ground_truth,
                evaluation_model=request.evaluation_model or settings.AZURE_EVALUATION_MODEL_NAME,
                error_message=str(e),
                evaluation_duration_ms=int((time.time() - start_time) * 1000),
                evaluation_timestamp=datetime.utcnow()
            )
            return error_result
        finally:
            # Restore original azure_manager
            self.azure_manager = original_azure_manager
    
    async def evaluate_batch(self, requests: List[EvaluationRequest]) -> List[EvaluationResult]:
        """
        Evaluate multiple requests concurrently
        """
        logger.info(f"Starting batch evaluation for {len(requests)} requests")
        
        # Create evaluation tasks
        tasks = [self.evaluate_answer(request) for request in requests]
        
        # Execute concurrently with some limits
        semaphore = asyncio.Semaphore(5)  # Limit concurrent evaluations
        
        async def evaluate_with_semaphore(request):
            async with semaphore:
                return await self.evaluate_answer(request)
        
        results = await asyncio.gather(*[evaluate_with_semaphore(req) for req in requests])
        
        logger.info(f"Batch evaluation completed for {len(results)} requests")
        return results
    
    async def get_results_by_question(self, question_id: str) -> List[EvaluationResult]:
        """
        Get evaluation results for a specific question
        Note: This would typically query Cosmos DB, but for now returns empty list
        """
        try:
            # TODO: Implement Cosmos DB query to get results by question_id
            logger.info(f"Getting evaluation results for question_id: {question_id}")
            return []
        except Exception as e:
            logger.error(f"Error getting results by question: {e}")
            return []
    
    async def get_results_by_session(
        self, 
        session_id: str, 
        evaluator_type: Optional[EvaluatorType] = None,
        rag_method: Optional[str] = None,
        limit: int = 50
    ) -> List[EvaluationResult]:
        """
        Get evaluation results for a session with optional filtering
        Note: This would typically query Cosmos DB, but for now returns empty list
        """
        try:
            logger.info(f"Getting evaluation results for session_id: {session_id}")
            # TODO: Implement Cosmos DB query with filtering
            return []
        except Exception as e:
            logger.error(f"Error getting results by session: {e}")
            return []
    
    async def get_session_summary(
        self,
        session_id: str,
        evaluator_type: Optional[EvaluatorType] = None,
        rag_method: Optional[str] = None
    ) -> EvaluationSummary:
        """
        Get evaluation summary for a session
        """
        try:
            # Get results for the session
            results = await self.get_results_by_session(session_id, evaluator_type, rag_method, 1000)
            
            # Calculate summary
            summary = EvaluationSummary(
                session_id=session_id,
                total_evaluations=len(results),
                evaluator_type=evaluator_type or EvaluatorType.CUSTOM,
                rag_method=rag_method or "unknown",
                start_time=datetime.utcnow(),
                end_time=datetime.utcnow()
            )
            
            if results:
                # Calculate averages
                groundedness_scores = [r.groundedness_score for r in results if r.groundedness_score is not None]
                relevance_scores = [r.relevance_score for r in results if r.relevance_score is not None]
                coherence_scores = [r.coherence_score for r in results if r.coherence_score is not None]
                fluency_scores = [r.fluency_score for r in results if r.fluency_score is not None]
                overall_scores = [r.overall_score for r in results if r.overall_score is not None]
                
                summary.avg_groundedness = sum(groundedness_scores) / len(groundedness_scores) if groundedness_scores else None
                summary.avg_relevance = sum(relevance_scores) / len(relevance_scores) if relevance_scores else None
                summary.avg_coherence = sum(coherence_scores) / len(coherence_scores) if coherence_scores else None
                summary.avg_fluency = sum(fluency_scores) / len(fluency_scores) if fluency_scores else None
                summary.avg_overall = sum(overall_scores) / len(overall_scores) if overall_scores else None
                
                summary.start_time = min(r.evaluation_timestamp for r in results)
                summary.end_time = max(r.evaluation_timestamp for r in results)
                
                # Find best and worst performing questions
                sorted_by_overall = sorted(
                    [r for r in results if r.overall_score is not None],
                    key=lambda x: x.overall_score,
                    reverse=True
                )
                summary.best_performing_questions = [r.question_id for r in sorted_by_overall[:5]]
                summary.worst_performing_questions = [r.question_id for r in sorted_by_overall[-5:]]
            
            return summary
            
        except Exception as e:
            logger.error(f"Error getting session summary: {e}")
            # Return basic summary on error
            return EvaluationSummary(
                session_id=session_id,
                total_evaluations=0,
                evaluator_type=evaluator_type or EvaluatorType.CUSTOM,
                rag_method=rag_method or "unknown",
                start_time=datetime.utcnow(),
                end_time=datetime.utcnow()
            )
    
    async def get_analytics(
        self,
        days: int,
        evaluator_type: Optional[EvaluatorType] = None,
        rag_method: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get evaluation analytics across multiple sessions
        """
        try:
            logger.info(f"Getting evaluation analytics for {days} days")
            
            # TODO: Implement date range query and analytics calculation
            return {
                "total_evaluations": 0,
                "date_range": {
                    "start": (datetime.utcnow() - timedelta(days=days)).isoformat(),
                    "end": datetime.utcnow().isoformat()
                },
                "by_evaluator_type": {},
                "by_rag_method": {},
                "overall_metrics": {
                    "avg_groundedness": None,
                    "avg_relevance": None,
                    "avg_coherence": None,
                    "avg_fluency": None,
                    "avg_overall": None
                },
                "trends": {
                    "daily_averages": {},
                    "trend_direction": "stable"
                },
                "performance_distribution": {
                    "excellent": 0,
                    "good": 0,
                    "fair": 0,
                    "poor": 0
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting analytics: {e}")
            return {"error": str(e)}
    
    async def delete_session_results(self, session_id: str) -> int:
        """
        Delete all evaluation results for a session
        """
        try:
            logger.info(f"Deleting evaluation results for session_id: {session_id}")
            # TODO: Implement Cosmos DB deletion
            return 0
        except Exception as e:
            logger.error(f"Error deleting session results: {e}")
            return 0

    async def _store_result(self, result: EvaluationResult) -> bool:
        """Store evaluation result in Cosmos DB"""
        try:
            if not self.azure_manager:
                logger.error("Azure manager not available for storing evaluation result")
                return False
            
            # Convert result to dictionary for storage
            result_dict = {
                "id": result.id,
                "question_id": result.question_id,
                "session_id": result.session_id,
                "evaluator_type": result.evaluator_type.value if hasattr(result.evaluator_type, 'value') else str(result.evaluator_type),
                "rag_method": result.rag_method,
                "evaluation_model": result.evaluation_model,
                "question": result.question,
                "answer": result.answer,
                "context": result.context,
                "ground_truth": result.ground_truth,
                "groundedness_score": result.groundedness_score,
                "relevance_score": result.relevance_score,
                "coherence_score": result.coherence_score,
                "fluency_score": result.fluency_score,
                "similarity_score": result.similarity_score,
                "f1_score": result.f1_score,
                "bleu_score": result.bleu_score,
                "rouge_score": result.rouge_score,
                "overall_score": result.overall_score,
                "detailed_scores": result.detailed_scores,
                "reasoning": result.reasoning,
                "feedback": result.feedback,
                "recommendations": result.recommendations,
                "metadata": result.metadata,
                "error_message": result.error_message,
                "evaluation_timestamp": result.evaluation_timestamp,
                "evaluation_duration_ms": result.evaluation_duration_ms
            }
            
            success = await self.azure_manager.save_evaluation_result(result_dict)
            if success:
                logger.info(f"Evaluation result {result.id} stored in Cosmos DB successfully")
            else:
                logger.error(f"Failed to store evaluation result {result.id} in Cosmos DB")
            return success
            
        except Exception as e:
            logger.error(f"Error storing evaluation result: {e}")
            return False

    async def get_evaluation_result(self, evaluation_id: str) -> Optional[EvaluationResult]:
        """Retrieve evaluation result by ID"""
        try:
            if not self.azure_manager:
                logger.error("Azure manager not available for retrieving evaluation result")
                return None
            
            result_dict = await self.azure_manager.get_evaluation_result(evaluation_id)
            if not result_dict:
                return None
            
            # Convert dictionary back to EvaluationResult object
            result = EvaluationResult(
                id=result_dict.get("id"),
                question_id=result_dict.get("question_id"),
                session_id=result_dict.get("session_id"),
                evaluator_type=EvaluatorType(result_dict.get("evaluator_type")),
                rag_method=result_dict.get("rag_method"),
                evaluation_model=result_dict.get("evaluation_model"),
                question=result_dict.get("question"),
                answer=result_dict.get("answer"),
                context=result_dict.get("context", []),
                ground_truth=result_dict.get("ground_truth"),
                groundedness_score=result_dict.get("groundedness_score"),
                relevance_score=result_dict.get("relevance_score"),
                coherence_score=result_dict.get("coherence_score"),
                fluency_score=result_dict.get("fluency_score"),
                similarity_score=result_dict.get("similarity_score"),
                f1_score=result_dict.get("f1_score"),
                bleu_score=result_dict.get("bleu_score"),
                rouge_score=result_dict.get("rouge_score"),
                overall_score=result_dict.get("overall_score"),
                detailed_scores=result_dict.get("detailed_scores", {}),
                reasoning=result_dict.get("reasoning"),
                feedback=result_dict.get("feedback"),
                recommendations=result_dict.get("recommendations", []),
                metadata=result_dict.get("metadata", {}),
                error_message=result_dict.get("error_message"),
                evaluation_timestamp=result_dict.get("evaluation_timestamp"),
                evaluation_duration_ms=result_dict.get("evaluation_duration_ms")
            )
            
            # Handle datetime conversion if needed
            if isinstance(result_dict.get("evaluation_timestamp"), str):
                try:
                    result.evaluation_timestamp = datetime.fromisoformat(result_dict["evaluation_timestamp"].replace('Z', '+00:00'))
                except:
                    pass
            
            return result
            
        except Exception as e:
            logger.error(f"Error retrieving evaluation result {evaluation_id}: {e}")
            return None

    async def list_evaluation_results(self, session_id: Optional[str] = None, question_id: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
        """List evaluation results with optional filtering"""
        try:
            from app.services.azure_services import azure_manager
            return await azure_manager.list_evaluation_results(session_id, question_id, limit)
        except Exception as e:
            logger.error(f"Error listing evaluation results: {e}")
            return []

class FoundryEvaluator:
    """
    Azure AI Foundry evaluator that uses subprocess isolation to avoid HTTP client conflicts.
    
    This implementation runs all Azure AI Foundry evaluations in a separate subprocess to completely
    isolate the Azure AI Evaluation SDK from the OpenAI SDK's HTTP client, preventing conflicts
    that arise from incompatible HTTP client implementations.
    
    Supported evaluators:
    - GroundednessEvaluator: Measures if the answer is supported by the provided context
    - RelevanceEvaluator: Measures how well the answer addresses the question
    - CoherenceEvaluator: Measures logical flow and understandability  
    - FluencyEvaluator: Measures language quality and readability
    """
    
    def __init__(self):
        self.available_evaluators = {}
        self._initialize_evaluators()

    def _initialize_evaluators(self):
        """Initialize Foundry evaluators using subprocess approach to avoid HTTP client conflicts"""
        try:
            logger.info("Initializing Azure AI Foundry evaluators (subprocess approach)...")

            if not settings.AZURE_EVALUATION_ENDPOINT or not settings.AZURE_EVALUATION_API_KEY:
                raise ValueError("Azure evaluation endpoint and API key are required")

            # Validate model compatibility
            supported_models = ["gpt-4", "gpt-4-turbo", "gpt-4o", "gpt-4o-mini", "gpt-35-turbo"]
            model_name = settings.AZURE_EVALUATION_MODEL_DEPLOYMENT.lower()
            
            if not any(supported in model_name for supported in supported_models):
                logger.warning(f"Model '{settings.AZURE_EVALUATION_MODEL_DEPLOYMENT}' may not be compatible with Azure AI Evaluation SDK")
                logger.warning(f"Supported models: {supported_models}")
                logger.warning("Consider using gpt-4o-mini for best compatibility")

            # Since we're using subprocess approach, we don't need to initialize the actual evaluators
            # Just mark them as available if we have the required configuration
            self.available_evaluators = {
                'groundedness': 'GroundednessEvaluator',
                'relevance': 'RelevanceEvaluator',
                'coherence': 'CoherenceEvaluator',
                'fluency': 'FluencyEvaluator'
            }
            
            logger.info(f"Successfully configured {len(self.available_evaluators)} Azure AI Foundry evaluators for subprocess execution")

        except Exception as e:
            logger.error(f"Failed to initialize Foundry evaluators: {e}", exc_info=True)
            self.available_evaluators = {}

    @property
    def evaluators(self):
        """Compatibility property to maintain existing interface"""
        return self.available_evaluators

    async def evaluate(self, request: EvaluationRequest) -> EvaluationResult:
        if not self.evaluators:
            raise ValueError("Azure AI Foundry evaluators not available")
            
        result = EvaluationResult(
            question_id=request.question_id,
            session_id=request.session_id,
            evaluator_type=EvaluatorType.FOUNDRY,
            rag_method=request.rag_method,
            question=request.question,
            answer=request.answer,
            context=request.context,
            ground_truth=request.ground_truth,
            evaluation_model=request.evaluation_model or settings.AZURE_EVALUATION_MODEL_NAME
        )

        eval_data = {
            "query": request.question,
            "response": request.answer,
            "context": "\n".join(request.context) if request.context else ""
        }

        detailed_scores = {}
        scores = []

        # Run evaluations for requested metrics
        for metric in request.metrics:
            metric_key = metric.value.lower()
            if metric_key in self.evaluators:
                try:
                    score_result = await self._run_evaluator(metric_key, eval_data)
                    detailed_scores[metric_key] = score_result
                    
                    # Set specific metric scores
                    if metric_key == "groundedness":
                        result.groundedness_score = score_result.get("score", 0.0)
                        scores.append(result.groundedness_score)
                    elif metric_key == "relevance":
                        result.relevance_score = score_result.get("score", 0.0)
                        scores.append(result.relevance_score)
                    elif metric_key == "coherence":
                        result.coherence_score = score_result.get("score", 0.0)
                        scores.append(result.coherence_score)
                    elif metric_key == "fluency":
                        result.fluency_score = score_result.get("score", 0.0)
                        scores.append(result.fluency_score)
                        
                except Exception as e:
                    logger.error(f"Error evaluating {metric_key}: {e}")
                    detailed_scores[metric_key] = {"score": 0.0, "error": str(e)}

        # Calculate overall score
        result.overall_score = sum(scores) / len(scores) if scores else 0.0
        result.detailed_scores = detailed_scores
        result.reasoning = "Evaluated using Azure AI Foundry built-in evaluators"

        logger.info(f"Foundry evaluation completed for question_id: {request.question_id}")
        return result

    async def _run_evaluator(self, evaluator_name: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run Foundry evaluator using subprocess approach to avoid HTTP client conflicts.
        This method always uses the subprocess approach for reliability and consistency.
        """
        try:
            logger.info(f"Running {evaluator_name} evaluator using subprocess approach...")
            
            from .foundry_evaluator_subprocess import run_foundry_evaluation
            
            # Map evaluator names to class names
            evaluator_class_map = {
                "groundedness": "GroundednessEvaluator",
                "relevance": "RelevanceEvaluator", 
                "coherence": "CoherenceEvaluator",
                "fluency": "FluencyEvaluator"
            }
            
            evaluator_class = evaluator_class_map.get(evaluator_name)
            if not evaluator_class:
                raise ValueError(f"Unknown evaluator: {evaluator_name}")
            
            # Use evaluation-specific settings if available, otherwise fall back to main OpenAI settings
            azure_endpoint = settings.AZURE_EVALUATION_ENDPOINT or settings.AZURE_OPENAI_ENDPOINT
            api_key = settings.AZURE_EVALUATION_API_KEY or settings.AZURE_OPENAI_API_KEY
            deployment = settings.AZURE_EVALUATION_MODEL_DEPLOYMENT or settings.AZURE_OPENAI_DEPLOYMENT_NAME
            
            # Run evaluation in subprocess
            result = run_foundry_evaluation(
                evaluator_name=evaluator_class,
                query=data["query"],
                response=data["response"],
                context=data["context"],
                azure_endpoint=azure_endpoint,
                api_key=api_key,
                deployment=deployment,
                api_version=settings.AZURE_OPENAI_API_VERSION
            )
            
            logger.info(f"Subprocess evaluation successful for {evaluator_name}: {result}")
            return result
            
        except Exception as e:
            logger.error(f"Error running {evaluator_name} evaluator: {e}", exc_info=True)
            return {"score": 0.0, "error": str(e), "reasoning": f"Error in {evaluator_name} evaluation: {str(e)}"}

class CustomEvaluator:
    """
    Custom evaluator using Azure OpenAI for evaluation
    """
    
    def __init__(self):
        self.client = None
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize Azure OpenAI client for evaluation"""
        try:
            self.client = AsyncAzureOpenAI(
                azure_endpoint=settings.AZURE_EVALUATION_ENDPOINT or settings.AZURE_OPENAI_ENDPOINT,
                api_key=settings.AZURE_EVALUATION_API_KEY or settings.AZURE_OPENAI_API_KEY,
                api_version=settings.AZURE_OPENAI_API_VERSION
            )
            logger.info("Custom evaluator OpenAI client initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize custom evaluator: {e}")
            self.client = None
    
    async def evaluate(self, request: EvaluationRequest) -> EvaluationResult:
        """
        Evaluate using custom OpenAI-based evaluation
        """
        # Check if client is available
        if not self.client:
            raise ValueError("Custom evaluator OpenAI client not available")
        
        result = EvaluationResult(
            question_id=request.question_id,
            session_id=request.session_id,
            evaluator_type=EvaluatorType.CUSTOM,
            rag_method=request.rag_method,
            question=request.question,
            answer=request.answer,
            context=request.context,
            ground_truth=request.ground_truth,
            evaluation_model=request.evaluation_model or settings.AZURE_EVALUATION_MODEL_NAME
        )
        
        try:
            # Run evaluation for each requested metric
            evaluation_tasks = []
            
            for metric in request.metrics:
                if metric == EvaluationMetric.GROUNDEDNESS:
                    evaluation_tasks.append(self._evaluate_groundedness(request))
                elif metric == EvaluationMetric.RELEVANCE:
                    evaluation_tasks.append(self._evaluate_relevance(request))
                elif metric == EvaluationMetric.COHERENCE:
                    evaluation_tasks.append(self._evaluate_coherence(request))
                elif metric == EvaluationMetric.FLUENCY:
                    evaluation_tasks.append(self._evaluate_fluency(request))
            
            # Run evaluations concurrently
            evaluation_results = await asyncio.gather(*evaluation_tasks, return_exceptions=True)
            
            # Process results
            detailed_scores = {}
            scores = []
            
            for i, metric in enumerate(request.metrics):
                if i < len(evaluation_results) and not isinstance(evaluation_results[i], Exception):
                    eval_result = evaluation_results[i]
                    detailed_scores[metric.value] = eval_result
                    
                    if metric == EvaluationMetric.GROUNDEDNESS:
                        result.groundedness_score = eval_result.get("score", 0.0)
                        scores.append(result.groundedness_score)
                    elif metric == EvaluationMetric.RELEVANCE:
                        result.relevance_score = eval_result.get("score", 0.0)
                        scores.append(result.relevance_score)
                    elif metric == EvaluationMetric.COHERENCE:
                        result.coherence_score = eval_result.get("score", 0.0)
                        scores.append(result.coherence_score)
                    elif metric == EvaluationMetric.FLUENCY:
                        result.fluency_score = eval_result.get("score", 0.0)
                        scores.append(result.fluency_score)
            
            # Calculate overall score
            result.overall_score = sum(scores) / len(scores) if scores else 0.0
            result.detailed_scores = detailed_scores
            result.reasoning = "Evaluated using custom Azure OpenAI-based evaluators"
            
            logger.info(f"Custom evaluation completed for question_id: {request.question_id}")
            
        except Exception as e:
            logger.error(f"Custom evaluation failed: {e}")
            result.error_message = str(e)
        
        return result
    
    async def _evaluate_groundedness(self, request: EvaluationRequest) -> Dict[str, Any]:
        """Evaluate groundedness using custom prompt"""
        prompt = f"""
        You are an expert evaluator assessing the groundedness of an AI-generated answer.
        Groundedness measures whether the answer is supported by the provided context.

        Question: {request.question}
        
        Context: {chr(10).join(request.context)}
        
        Answer: {request.answer}

        Please evaluate the groundedness on a scale of 0.0 to 1.0 where:
        - 1.0: Fully grounded, every claim is supported by the context
        - 0.8: Mostly grounded, most claims supported
        - 0.6: Partially grounded, some claims supported
        - 0.4: Minimally grounded, few claims supported
        - 0.2: Poorly grounded, very few claims supported
        - 0.0: Not grounded, no claims supported

        Return your evaluation as JSON:
        {{
            "score": <float between 0.0 and 1.0>,
            "reasoning": "<detailed explanation>",
            "supported_claims": ["<list of supported claims>"],
            "unsupported_claims": ["<list of unsupported claims>"]
        }}
        """
        
        return await self._call_evaluation_model(prompt)
    
    async def _evaluate_relevance(self, request: EvaluationRequest) -> Dict[str, Any]:
        """Evaluate relevance using custom prompt"""
        prompt = f"""
        You are an expert evaluator assessing the relevance of an AI-generated answer.
        Relevance measures how well the answer addresses the specific question asked.

        Question: {request.question}
        
        Answer: {request.answer}

        Please evaluate the relevance on a scale of 0.0 to 1.0 where:
        - 1.0: Perfectly relevant, directly answers the question
        - 0.8: Highly relevant, answers most aspects of the question
        - 0.6: Moderately relevant, answers some aspects
        - 0.4: Somewhat relevant, tangentially related
        - 0.2: Minimally relevant, barely addresses the question
        - 0.0: Not relevant, does not address the question

        Return your evaluation as JSON:
        {{
            "score": <float between 0.0 and 1.0>,
            "reasoning": "<detailed explanation>",
            "addressed_aspects": ["<list of question aspects addressed>"],
            "missing_aspects": ["<list of question aspects not addressed>"]
        }}
        """
        
        return await self._call_evaluation_model(prompt)
    
    async def _evaluate_coherence(self, request: EvaluationRequest) -> Dict[str, Any]:
        """Evaluate coherence using custom prompt"""
        prompt = f"""
        You are an expert evaluator assessing the coherence of an AI-generated answer.
        Coherence measures how well the answer flows logically and is easy to understand.

        Question: {request.question}
        
        Answer: {request.answer}

        Please evaluate the coherence on a scale of 0.0 to 1.0 where:
        - 1.0: Highly coherent, logical flow, easy to follow
        - 0.8: Mostly coherent, generally easy to follow
        - 0.6: Moderately coherent, some confusing parts
        - 0.4: Somewhat coherent, several confusing parts
        - 0.2: Minimally coherent, difficult to follow
        - 0.0: Incoherent, no logical flow

        Return your evaluation as JSON:
        {{
            "score": <float between 0.0 and 1.0>,
            "reasoning": "<detailed explanation>",
            "strengths": ["<list of coherent aspects>"],
            "weaknesses": ["<list of incoherent aspects>"]
        }}
        """
        
        return await self._call_evaluation_model(prompt)
    
    async def _evaluate_fluency(self, request: EvaluationRequest) -> Dict[str, Any]:
        """Evaluate fluency using custom prompt"""
        prompt = f"""
        You are an expert evaluator assessing the fluency of an AI-generated answer.
        Fluency measures the quality of language, grammar, and readability.

        Answer: {request.answer}

        Please evaluate the fluency on a scale of 0.0 to 1.0 where:
        - 1.0: Excellent fluency, perfect grammar and style
        - 0.8: Good fluency, minor issues
        - 0.6: Adequate fluency, some grammatical issues
        - 0.4: Poor fluency, many grammatical issues
        - 0.2: Very poor fluency, difficult to read
        - 0.0: Incomprehensible

        Return your evaluation as JSON:
        {{
            "score": <float between 0.0 and 1.0>,
            "reasoning": "<detailed explanation>",
            "grammar_score": <float between 0.0 and 1.0>,
            "style_score": <float between 0.0 and 1.0>,
            "readability_score": <float between 0.0 and 1.0>
        }}
        """
        
        return await self._call_evaluation_model(prompt)
    
    async def _call_evaluation_model(self, prompt: str) -> Dict[str, Any]:
        """Make a call to the evaluation model"""
        try:
            response = await self.client.chat.completions.create(
                model=settings.AZURE_EVALUATION_MODEL_DEPLOYMENT,
                messages=[
                    {
                        "role": "system", 
                        "content": "You are a precise AI evaluator. Always return valid JSON."
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=1000
            )
            
            content = response.choices[0].message.content
            
            # Try to parse JSON response
            try:
                return json.loads(content)
            except json.JSONDecodeError:
                # Fallback if JSON parsing fails
                return {
                    "score": 0.5,
                    "reasoning": content,
                    "error": "Could not parse JSON response"
                }
                
        except Exception as e:
            logger.error(f"Error calling evaluation model: {e}")
            return {
                "score": 0.0,
                "reasoning": f"Evaluation failed: {str(e)}",
                "error": str(e)
            }


# Global evaluation service instance
evaluation_service = EvaluationService()
