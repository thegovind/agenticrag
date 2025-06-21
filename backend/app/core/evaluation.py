import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum


logger = logging.getLogger(__name__)

class EvaluationMetric(Enum):
    RELEVANCE = "relevance"
    GROUNDEDNESS = "groundedness"
    COHERENCE = "coherence"
    FLUENCY = "fluency"
    FINANCIAL_ACCURACY = "financial_accuracy"
    CITATION_QUALITY = "citation_quality"
    RESPONSE_TIME = "response_time"

@dataclass
class EvaluationResult:
    metric: str
    score: float
    reasoning: str
    timestamp: datetime
    session_id: str
    query: str
    response: str
    sources: List[str]
    model_used: str
    metadata: Dict[str, Any] = None

@dataclass
class FinancialEvaluationContext:
    query: str
    response: str
    sources: List[Dict[str, Any]]
    ground_truth: Optional[str] = None
    financial_context: Optional[Dict[str, Any]] = None

class FinancialAccuracyEvaluator:
    """Custom evaluator for financial document accuracy"""
    
    def __init__(self, azure_openai_client):
        self.client = azure_openai_client
        self.financial_keywords = [
            "revenue", "profit", "loss", "earnings", "EBITDA", "cash flow",
            "assets", "liabilities", "equity", "debt", "margin", "growth",
            "dividend", "share", "stock", "market cap", "valuation"
        ]
    
    async def evaluate(self, context: FinancialEvaluationContext) -> EvaluationResult:
        """Evaluate financial accuracy of the response"""
        
        prompt = f"""
        You are a financial expert evaluating the accuracy of AI-generated responses about financial documents.
        
        Query: {context.query}
        Response: {context.response}
        Sources: {json.dumps([s.get('content', '') for s in context.sources], indent=2)}
        
        Evaluate the financial accuracy on a scale of 1-5 where:
        1 = Completely inaccurate financial information
        2 = Mostly inaccurate with some correct elements
        3 = Partially accurate but with significant errors
        4 = Mostly accurate with minor errors
        5 = Completely accurate financial information
        
        Consider:
        - Numerical accuracy of financial figures
        - Correct use of financial terminology
        - Proper context and time periods
        - Alignment with source documents
        
        Respond in JSON format:
        {{
            "score": <1-5>,
            "reasoning": "<detailed explanation>",
            "financial_errors": ["<list of any errors found>"],
            "confidence": <0.0-1.0>
        }}
        """
        
        try:
            response = await self.client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1
            )
            
            result = json.loads(response.choices[0].message.content)
            
            return EvaluationResult(
                metric=EvaluationMetric.FINANCIAL_ACCURACY.value,
                score=result["score"] / 5.0,  # Normalize to 0-1
                reasoning=result["reasoning"],
                timestamp=datetime.utcnow(),
                session_id=context.financial_context.get("session_id", "") if context.financial_context else "",
                query=context.query,
                response=context.response,
                sources=[s.get("source", "") for s in context.sources],
                model_used="gpt-4",
                metadata={
                    "financial_errors": result.get("financial_errors", []),
                    "confidence": result.get("confidence", 0.0)
                }
            )
            
        except Exception as e:
            logger.error(f"Financial accuracy evaluation failed: {e}")
            return EvaluationResult(
                metric=EvaluationMetric.FINANCIAL_ACCURACY.value,
                score=0.0,
                reasoning=f"Evaluation failed: {str(e)}",
                timestamp=datetime.utcnow(),
                session_id="",
                query=context.query,
                response=context.response,
                sources=[],
                model_used="gpt-4"
            )

class CitationQualityEvaluator:
    """Custom evaluator for citation quality and accuracy"""
    
    def __init__(self, azure_openai_client):
        self.client = azure_openai_client
    
    async def evaluate(self, context: FinancialEvaluationContext) -> EvaluationResult:
        """Evaluate citation quality and accuracy"""
        
        prompt = f"""
        You are evaluating the quality and accuracy of citations in an AI response.
        
        Query: {context.query}
        Response: {context.response}
        Available Sources: {json.dumps([{
            'title': s.get('title', ''),
            'content': s.get('content', '')[:500] + '...' if len(s.get('content', '')) > 500 else s.get('content', ''),
            'page': s.get('page_number', ''),
            'section': s.get('section_title', '')
        } for s in context.sources], indent=2)}
        
        Evaluate citation quality on a scale of 1-5 where:
        1 = No citations or completely incorrect citations
        2 = Few citations, mostly incorrect or irrelevant
        3 = Some citations present but with accuracy issues
        4 = Good citations with minor issues
        5 = Excellent, accurate, and comprehensive citations
        
        Consider:
        - Are claims properly cited?
        - Do citations match the source content?
        - Are citation formats consistent?
        - Is the citation coverage comprehensive?
        
        Respond in JSON format:
        {{
            "score": <1-5>,
            "reasoning": "<detailed explanation>",
            "citation_issues": ["<list of citation problems>"],
            "missing_citations": ["<claims that need citations>"],
            "confidence": <0.0-1.0>
        }}
        """
        
        try:
            response = await self.client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1
            )
            
            result = json.loads(response.choices[0].message.content)
            
            return EvaluationResult(
                metric=EvaluationMetric.CITATION_QUALITY.value,
                score=result["score"] / 5.0,  # Normalize to 0-1
                reasoning=result["reasoning"],
                timestamp=datetime.utcnow(),
                session_id=context.financial_context.get("session_id", "") if context.financial_context else "",
                query=context.query,
                response=context.response,
                sources=[s.get("source", "") for s in context.sources],
                model_used="gpt-4",
                metadata={
                    "citation_issues": result.get("citation_issues", []),
                    "missing_citations": result.get("missing_citations", []),
                    "confidence": result.get("confidence", 0.0)
                }
            )
            
        except Exception as e:
            logger.error(f"Citation quality evaluation failed: {e}")
            return EvaluationResult(
                metric=EvaluationMetric.CITATION_QUALITY.value,
                score=0.0,
                reasoning=f"Evaluation failed: {str(e)}",
                timestamp=datetime.utcnow(),
                session_id="",
                query=context.query,
                response=context.response,
                sources=[],
                model_used="gpt-4"
            )

class EvaluationFramework:
    """Comprehensive evaluation framework for RAG system"""
    
    def __init__(self, azure_openai_client, cosmos_client=None):
        self.azure_client = azure_openai_client
        self.cosmos_client = cosmos_client
        
        self.relevance_evaluator = self._create_custom_relevance_evaluator(azure_openai_client)
        self.groundedness_evaluator = self._create_custom_groundedness_evaluator(azure_openai_client)
        self.coherence_evaluator = self._create_custom_coherence_evaluator(azure_openai_client)
        self.fluency_evaluator = self._create_custom_fluency_evaluator(azure_openai_client)
        
        self.financial_accuracy_evaluator = FinancialAccuracyEvaluator(azure_openai_client)
        self.citation_quality_evaluator = CitationQualityEvaluator(azure_openai_client)
        
        self.evaluation_results = []
    
    def _create_custom_relevance_evaluator(self, azure_openai_client):
        """Create custom relevance evaluator"""
        class CustomRelevanceEvaluator:
            def __init__(self, client):
                self.client = client
            
            async def evaluate(self, query: str, response: str, context: str = "", **kwargs):
                try:
                    evaluation_prompt = f"""
                    Evaluate how relevant the following response is to the given query.
                    
                    Query: {query}
                    Response: {response}
                    Context: {context}
                    
                    Rate the relevance on a scale of 0.0 to 1.0 where:
                    - 1.0 = Perfectly relevant and directly addresses the query
                    - 0.8 = Highly relevant with minor tangential content
                    - 0.6 = Generally relevant but some off-topic content
                    - 0.4 = Somewhat relevant but significant off-topic content
                    - 0.2 = Minimally relevant
                    - 0.0 = Completely irrelevant
                    
                    Provide your rating and reasoning in JSON format:
                    {{"relevance": 0.0-1.0, "reasoning": "explanation"}}
                    """
                    
                    response_obj = await self.client.chat.completions.create(
                        model="gpt-4",
                        messages=[{"role": "user", "content": evaluation_prompt}],
                        temperature=0.1,
                        max_tokens=500
                    )
                    
                    result = json.loads(response_obj.choices[0].message.content)
                    return result
                    
                except Exception as e:
                    logger.error(f"Relevance evaluation failed: {e}")
                    return {"relevance": 0.0, "reasoning": f"Evaluation failed: {str(e)}"}
        
        return CustomRelevanceEvaluator(azure_openai_client)
    
    def _create_custom_groundedness_evaluator(self, azure_openai_client):
        """Create custom groundedness evaluator"""
        class CustomGroundednessEvaluator:
            def __init__(self, client):
                self.client = client
            
            async def evaluate(self, response: str, context: str = "", **kwargs):
                try:
                    evaluation_prompt = f"""
                    Evaluate how well the following response is grounded in the provided context.
                    
                    Response: {response}
                    Context: {context}
                    
                    Rate the groundedness on a scale of 0.0 to 1.0 where:
                    - 1.0 = All claims are fully supported by context
                    - 0.8 = Most claims supported, minor unsupported details
                    - 0.6 = Generally supported but some unsupported claims
                    - 0.4 = Some claims supported, some unsupported
                    - 0.2 = Few claims supported by context
                    - 0.0 = No claims supported by context
                    
                    Provide your rating and reasoning in JSON format:
                    {{"groundedness": 0.0-1.0, "reasoning": "explanation"}}
                    """
                    
                    response_obj = await self.client.chat.completions.create(
                        model="gpt-4",
                        messages=[{"role": "user", "content": evaluation_prompt}],
                        temperature=0.1,
                        max_tokens=500
                    )
                    
                    result = json.loads(response_obj.choices[0].message.content)
                    return result
                    
                except Exception as e:
                    logger.error(f"Groundedness evaluation failed: {e}")
                    return {"groundedness": 0.0, "reasoning": f"Evaluation failed: {str(e)}"}
        
        return CustomGroundednessEvaluator(azure_openai_client)
    
    def _create_custom_coherence_evaluator(self, azure_openai_client):
        """Create custom coherence evaluator"""
        class CustomCoherenceEvaluator:
            def __init__(self, client):
                self.client = client
            
            async def evaluate(self, response: str, **kwargs):
                try:
                    evaluation_prompt = f"""
                    Evaluate the coherence and logical flow of the following response.
                    
                    Response: {response}
                    
                    Rate the coherence on a scale of 0.0 to 1.0 where:
                    - 1.0 = Perfectly coherent with excellent logical flow
                    - 0.8 = Highly coherent with minor flow issues
                    - 0.6 = Generally coherent but some confusing parts
                    - 0.4 = Somewhat coherent but significant flow problems
                    - 0.2 = Poor coherence with major logical issues
                    - 0.0 = Completely incoherent
                    
                    Provide your rating and reasoning in JSON format:
                    {{"coherence": 0.0-1.0, "reasoning": "explanation"}}
                    """
                    
                    response_obj = await self.client.chat.completions.create(
                        model="gpt-4",
                        messages=[{"role": "user", "content": evaluation_prompt}],
                        temperature=0.1,
                        max_tokens=500
                    )
                    
                    result = json.loads(response_obj.choices[0].message.content)
                    return result
                    
                except Exception as e:
                    logger.error(f"Coherence evaluation failed: {e}")
                    return {"coherence": 0.0, "reasoning": f"Evaluation failed: {str(e)}"}
        
        return CustomCoherenceEvaluator(azure_openai_client)
    
    def _create_custom_fluency_evaluator(self, azure_openai_client):
        """Create custom fluency evaluator"""
        class CustomFluencyEvaluator:
            def __init__(self, client):
                self.client = client
            
            async def evaluate(self, response: str, **kwargs):
                try:
                    evaluation_prompt = f"""
                    Evaluate the fluency and readability of the following response.
                    
                    Response: {response}
                    
                    Rate the fluency on a scale of 0.0 to 1.0 where:
                    - 1.0 = Perfectly fluent with excellent readability
                    - 0.8 = Highly fluent with minor language issues
                    - 0.6 = Generally fluent but some awkward phrasing
                    - 0.4 = Somewhat fluent but noticeable language problems
                    - 0.2 = Poor fluency with significant language issues
                    - 0.0 = Very poor fluency, difficult to understand
                    
                    Provide your rating and reasoning in JSON format:
                    {{"fluency": 0.0-1.0, "reasoning": "explanation"}}
                    """
                    
                    response_obj = await self.client.chat.completions.create(
                        model="gpt-4",
                        messages=[{"role": "user", "content": evaluation_prompt}],
                        temperature=0.1,
                        max_tokens=500
                    )
                    
                    result = json.loads(response_obj.choices[0].message.content)
                    return result
                    
                except Exception as e:
                    logger.error(f"Fluency evaluation failed: {e}")
                    return {"fluency": 0.0, "reasoning": f"Evaluation failed: {str(e)}"}
        
        return CustomFluencyEvaluator(azure_openai_client)
    
    async def evaluate_response(
        self,
        query: str,
        response: str,
        sources: List[Dict[str, Any]],
        session_id: str,
        model_used: str,
        response_time: float,
        ground_truth: Optional[str] = None,
        financial_context: Optional[Dict[str, Any]] = None
    ) -> List[EvaluationResult]:
        """Comprehensive evaluation of a RAG response"""
        
        results = []
        context = FinancialEvaluationContext(
            query=query,
            response=response,
            sources=sources,
            ground_truth=ground_truth,
            financial_context=financial_context or {"session_id": session_id}
        )
        
        response_time_result = EvaluationResult(
            metric=EvaluationMetric.RESPONSE_TIME.value,
            score=min(1.0, max(0.0, (5.0 - response_time) / 5.0)),  # Normalize: 0s=1.0, 5s+=0.0
            reasoning=f"Response generated in {response_time:.2f} seconds",
            timestamp=datetime.utcnow(),
            session_id=session_id,
            query=query,
            response=response,
            sources=[s.get("source", "") for s in sources],
            model_used=model_used,
            metadata={"response_time_seconds": response_time}
        )
        results.append(response_time_result)
        
        evaluation_tasks = [
            self._evaluate_relevance(context, session_id, model_used),
            self._evaluate_groundedness(context, session_id, model_used),
            self._evaluate_coherence(context, session_id, model_used),
            self._evaluate_fluency(context, session_id, model_used),
            self.financial_accuracy_evaluator.evaluate(context),
            self.citation_quality_evaluator.evaluate(context)
        ]
        
        try:
            evaluation_results = await asyncio.gather(*evaluation_tasks, return_exceptions=True)
            
            for result in evaluation_results:
                if isinstance(result, Exception):
                    logger.error(f"Evaluation failed: {result}")
                else:
                    results.append(result)
                    
        except Exception as e:
            logger.error(f"Batch evaluation failed: {e}")
        
        self.evaluation_results.extend(results)
        
        if self.cosmos_client:
            await self._store_evaluation_results(results)
        
        return results
    
    async def _evaluate_relevance(self, context: FinancialEvaluationContext, session_id: str, model_used: str) -> EvaluationResult:
        """Evaluate response relevance using Azure AI evaluator"""
        try:
            result = await self.relevance_evaluator.evaluate(
                query=context.query,
                response=context.response,
                context="\n".join([s.get("content", "") for s in context.sources])
            )
            
            return EvaluationResult(
                metric=EvaluationMetric.RELEVANCE.value,
                score=result.get("relevance", 0.0),
                reasoning=result.get("reasoning", "Azure AI relevance evaluation"),
                timestamp=datetime.utcnow(),
                session_id=session_id,
                query=context.query,
                response=context.response,
                sources=[s.get("source", "") for s in context.sources],
                model_used=model_used,
                metadata=result
            )
        except Exception as e:
            logger.error(f"Relevance evaluation failed: {e}")
            return self._create_error_result(EvaluationMetric.RELEVANCE.value, context, session_id, model_used, str(e))
    
    async def _evaluate_groundedness(self, context: FinancialEvaluationContext, session_id: str, model_used: str) -> EvaluationResult:
        """Evaluate response groundedness using Azure AI evaluator"""
        try:
            result = await self.groundedness_evaluator.evaluate(
                response=context.response,
                context="\n".join([s.get("content", "") for s in context.sources])
            )
            
            return EvaluationResult(
                metric=EvaluationMetric.GROUNDEDNESS.value,
                score=result.get("groundedness", 0.0),
                reasoning=result.get("reasoning", "Azure AI groundedness evaluation"),
                timestamp=datetime.utcnow(),
                session_id=session_id,
                query=context.query,
                response=context.response,
                sources=[s.get("source", "") for s in context.sources],
                model_used=model_used,
                metadata=result
            )
        except Exception as e:
            logger.error(f"Groundedness evaluation failed: {e}")
            return self._create_error_result(EvaluationMetric.GROUNDEDNESS.value, context, session_id, model_used, str(e))
    
    async def _evaluate_coherence(self, context: FinancialEvaluationContext, session_id: str, model_used: str) -> EvaluationResult:
        """Evaluate response coherence using Azure AI evaluator"""
        try:
            result = await self.coherence_evaluator.evaluate(
                response=context.response
            )
            
            return EvaluationResult(
                metric=EvaluationMetric.COHERENCE.value,
                score=result.get("coherence", 0.0),
                reasoning=result.get("reasoning", "Azure AI coherence evaluation"),
                timestamp=datetime.utcnow(),
                session_id=session_id,
                query=context.query,
                response=context.response,
                sources=[s.get("source", "") for s in context.sources],
                model_used=model_used,
                metadata=result
            )
        except Exception as e:
            logger.error(f"Coherence evaluation failed: {e}")
            return self._create_error_result(EvaluationMetric.COHERENCE.value, context, session_id, model_used, str(e))
    
    async def _evaluate_fluency(self, context: FinancialEvaluationContext, session_id: str, model_used: str) -> EvaluationResult:
        """Evaluate response fluency using Azure AI evaluator"""
        try:
            result = await self.fluency_evaluator.evaluate(
                response=context.response
            )
            
            return EvaluationResult(
                metric=EvaluationMetric.FLUENCY.value,
                score=result.get("fluency", 0.0),
                reasoning=result.get("reasoning", "Azure AI fluency evaluation"),
                timestamp=datetime.utcnow(),
                session_id=session_id,
                query=context.query,
                response=context.response,
                sources=[s.get("source", "") for s in context.sources],
                model_used=model_used,
                metadata=result
            )
        except Exception as e:
            logger.error(f"Fluency evaluation failed: {e}")
            return self._create_error_result(EvaluationMetric.FLUENCY.value, context, session_id, model_used, str(e))
    
    def _create_error_result(self, metric: str, context: FinancialEvaluationContext, session_id: str, model_used: str, error: str) -> EvaluationResult:
        """Create an error evaluation result"""
        return EvaluationResult(
            metric=metric,
            score=0.0,
            reasoning=f"Evaluation failed: {error}",
            timestamp=datetime.utcnow(),
            session_id=session_id,
            query=context.query,
            response=context.response,
            sources=[s.get("source", "") for s in context.sources],
            model_used=model_used,
            metadata={"error": error}
        )
    
    async def _store_evaluation_results(self, results: List[EvaluationResult]):
        """Store evaluation results in Cosmos DB"""
        if not self.cosmos_client:
            return
        
        try:
            for result in results:
                document = {
                    "id": f"{result.session_id}_{result.metric}_{int(result.timestamp.timestamp())}",
                    "type": "evaluation_result",
                    **asdict(result)
                }
                document["timestamp"] = result.timestamp.isoformat()
                
                await self.cosmos_client.create_item(
                    body=document,
                    partition_key=result.session_id
                )
        except Exception as e:
            logger.error(f"Failed to store evaluation results: {e}")
    
    def get_evaluation_summary(self, session_id: Optional[str] = None, hours: int = 24) -> Dict[str, Any]:
        """Get evaluation summary statistics"""
        
        filtered_results = self.evaluation_results
        if session_id:
            filtered_results = [r for r in filtered_results if r.session_id == session_id]
        
        cutoff_time = datetime.utcnow().timestamp() - (hours * 3600)
        filtered_results = [r for r in filtered_results if r.timestamp.timestamp() > cutoff_time]
        
        if not filtered_results:
            return {"total_evaluations": 0, "metrics": {}}
        
        metrics = {}
        for metric in EvaluationMetric:
            metric_results = [r for r in filtered_results if r.metric == metric.value]
            if metric_results:
                scores = [r.score for r in metric_results]
                metrics[metric.value] = {
                    "count": len(scores),
                    "average": sum(scores) / len(scores),
                    "min": min(scores),
                    "max": max(scores),
                    "latest": metric_results[-1].score
                }
        
        return {
            "total_evaluations": len(filtered_results),
            "unique_sessions": len(set(r.session_id for r in filtered_results)),
            "time_range_hours": hours,
            "metrics": metrics
        }

evaluation_framework = None

def get_evaluation_framework() -> EvaluationFramework:
    """Get the global evaluation framework instance"""
    global evaluation_framework
    if evaluation_framework is None:
        raise RuntimeError("Evaluation framework not initialized. Call setup_evaluation_framework first.")
    return evaluation_framework

def setup_evaluation_framework(azure_openai_client, cosmos_client=None) -> EvaluationFramework:
    """Setup the global evaluation framework"""
    global evaluation_framework
    evaluation_framework = EvaluationFramework(azure_openai_client, cosmos_client)
    return evaluation_framework
