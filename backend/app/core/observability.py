import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from contextlib import asynccontextmanager
from azure.monitor.opentelemetry import configure_azure_monitor
from opentelemetry import trace, metrics
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
from opentelemetry.trace import Status, StatusCode
import os

def setup_observability():
    """Setup Azure Monitor observability with OpenTelemetry"""
    
    if os.getenv("APPLICATIONINSIGHTS_CONNECTION_STRING"):
        configure_azure_monitor(
            connection_string=os.getenv("APPLICATIONINSIGHTS_CONNECTION_STRING")
        )
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    tracer = trace.get_tracer(__name__)
    meter = metrics.get_meter(__name__)
    
    request_counter = meter.create_counter(
        "rag_requests_total",
        description="Total number of RAG requests"
    )
    
    token_usage_counter = meter.create_counter(
        "token_usage_total", 
        description="Total token usage"
    )
    
    knowledge_base_updates = meter.create_counter(
        "knowledge_base_updates_total",
        description="Total knowledge base updates"
    )
    
    return tracer, meter

class ObservabilityManager:
    def __init__(self):
        self.tracer, self.meter = setup_observability()
        
        self.request_counter = self.meter.create_counter(
            "rag_requests_total",
            description="Total number of RAG requests"
        )
        self.token_counter = self.meter.create_counter(
            "token_usage_total",
            description="Total token usage"
        )
        self.kb_update_counter = self.meter.create_counter(
            "knowledge_base_updates_total", 
            description="Knowledge base updates"
        )
        self.error_counter = self.meter.create_counter(
            "rag_errors_total",
            description="Total number of errors"
        )
        
        self.response_time_histogram = self.meter.create_histogram(
            "rag_response_time_seconds",
            description="Response time in seconds"
        )
        self.evaluation_time_histogram = self.meter.create_histogram(
            "evaluation_time_seconds", 
            description="Evaluation time in seconds"
        )
        
        self.active_sessions_gauge = self.meter.create_up_down_counter(
            "active_sessions_current",
            description="Current number of active sessions"
        )
        
        self.metrics_storage = {
            "requests": [],
            "tokens": [],
            "evaluations": [],
            "errors": [],
            "response_times": [],
            "system_metrics": []
        }
        
    def track_request(self, endpoint: str, user_id: str = None, session_id: str = None):
        """Track API request with enhanced metadata"""
        attributes = {"endpoint": endpoint}
        if user_id:
            attributes["user_id"] = user_id
        if session_id:
            attributes["session_id"] = session_id
            
        self.request_counter.add(1, attributes)
        
        self.metrics_storage["requests"].append({
            "timestamp": datetime.utcnow(),
            "endpoint": endpoint,
            "user_id": user_id,
            "session_id": session_id
        })
        
    def track_tokens(self, model: str, prompt_tokens: int, completion_tokens: int, 
                    session_id: str = None, cost: float = None):
        """Track token usage with cost calculation"""
        self.token_counter.add(prompt_tokens, {"model": model, "type": "prompt"})
        self.token_counter.add(completion_tokens, {"model": model, "type": "completion"})
        
        token_data = {
            "timestamp": datetime.utcnow(),
            "model": model,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "session_id": session_id,
            "cost": cost or self._calculate_cost(model, prompt_tokens, completion_tokens)
        }
        self.metrics_storage["tokens"].append(token_data)
        
    def track_response_time(self, endpoint: str, duration: float, model: str = None, session_id: str = None):
        """Track response time"""
        attributes = {"endpoint": endpoint}
        if model:
            attributes["model"] = model
        if session_id:
            attributes["session_id"] = session_id
            
        self.response_time_histogram.record(duration, attributes)
        
        self.metrics_storage["response_times"].append({
            "timestamp": datetime.utcnow(),
            "endpoint": endpoint,
            "duration": duration,
            "model": model,
            "session_id": session_id
        })
        
    def track_evaluation_metrics(self, session_id: str, evaluation_results: List[Dict[str, Any]]):
        """Track evaluation metrics from the evaluation framework"""
        for result in evaluation_results:
            eval_data = {
                "timestamp": datetime.utcnow(),
                "session_id": session_id,
                "metric": result.get("metric"),
                "score": result.get("score", 0.0),
                "model": result.get("model_used"),
                "reasoning": result.get("reasoning", "")
            }
            self.metrics_storage["evaluations"].append(eval_data)
            
    def track_error(self, error_type: str, endpoint: str, error_message: str, 
                   session_id: str = None, user_id: str = None):
        """Track errors"""
        attributes = {
            "error_type": error_type,
            "endpoint": endpoint
        }
        if session_id:
            attributes["session_id"] = session_id
        if user_id:
            attributes["user_id"] = user_id
            
        self.error_counter.add(1, attributes)
        
        self.metrics_storage["errors"].append({
            "timestamp": datetime.utcnow(),
            "error_type": error_type,
            "endpoint": endpoint,
            "error_message": error_message,
            "session_id": session_id,
            "user_id": user_id
        })
        
    def track_kb_update(self, source: str, documents_added: int, documents_updated: int):
        """Track knowledge base updates"""
        self.kb_update_counter.add(documents_added, {"source": source, "type": "added"})
        self.kb_update_counter.add(documents_updated, {"source": source, "type": "updated"})
        
    def track_system_metrics(self, cpu_usage: float, memory_usage: float, disk_usage: float):
        """Track system performance metrics"""
        system_data = {
            "timestamp": datetime.utcnow(),
            "cpu_usage": cpu_usage,
            "memory_usage": memory_usage,
            "disk_usage": disk_usage
        }
        self.metrics_storage["system_metrics"].append(system_data)
        
    def _calculate_cost(self, model: str, prompt_tokens: int, completion_tokens: int) -> float:
        """Calculate cost based on model and token usage"""
        pricing = {
            "gpt-4": {"prompt": 0.03, "completion": 0.06},
            "gpt-4-turbo": {"prompt": 0.01, "completion": 0.03},
            "gpt-35-turbo": {"prompt": 0.0015, "completion": 0.002},
            "text-embedding-ada-002": {"prompt": 0.0001, "completion": 0.0001},
            "text-embedding-3-small": {"prompt": 0.00002, "completion": 0.00002},
            "text-embedding-3-large": {"prompt": 0.00013, "completion": 0.00013}
        }
        
        model_pricing = pricing.get(model, {"prompt": 0.001, "completion": 0.001})
        prompt_cost = (prompt_tokens / 1000) * model_pricing["prompt"]
        completion_cost = (completion_tokens / 1000) * model_pricing["completion"]
        
        return prompt_cost + completion_cost
        
    def get_metrics_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get comprehensive metrics summary for admin dashboard"""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        
        recent_requests = [r for r in self.metrics_storage["requests"] if r["timestamp"] > cutoff_time]
        recent_tokens = [t for t in self.metrics_storage["tokens"] if t["timestamp"] > cutoff_time]
        recent_evaluations = [e for e in self.metrics_storage["evaluations"] if e["timestamp"] > cutoff_time]
        recent_errors = [e for e in self.metrics_storage["errors"] if e["timestamp"] > cutoff_time]
        recent_response_times = [r for r in self.metrics_storage["response_times"] if r["timestamp"] > cutoff_time]
        
        total_tokens = sum(t["total_tokens"] for t in recent_tokens)
        total_cost = sum(t["cost"] for t in recent_tokens)
        total_requests = len(recent_requests)
        
        avg_response_time = 0
        if recent_response_times:
            avg_response_time = sum(r["duration"] for r in recent_response_times) / len(recent_response_times)
        
        token_by_model = {}
        for token_data in recent_tokens:
            model = token_data["model"]
            if model not in token_by_model:
                token_by_model[model] = {"prompt": 0, "completion": 0, "total": 0, "cost": 0}
            token_by_model[model]["prompt"] += token_data["prompt_tokens"]
            token_by_model[model]["completion"] += token_data["completion_tokens"]
            token_by_model[model]["total"] += token_data["total_tokens"]
            token_by_model[model]["cost"] += token_data["cost"]
        
        evaluation_summary = {}
        for eval_data in recent_evaluations:
            metric = eval_data["metric"]
            if metric not in evaluation_summary:
                evaluation_summary[metric] = {"scores": [], "count": 0}
            evaluation_summary[metric]["scores"].append(eval_data["score"])
            evaluation_summary[metric]["count"] += 1
        
        for metric, data in evaluation_summary.items():
            if data["scores"]:
                data["average"] = sum(data["scores"]) / len(data["scores"])
                data["min"] = min(data["scores"])
                data["max"] = max(data["scores"])
            else:
                data["average"] = 0
                data["min"] = 0
                data["max"] = 0
        
        system_health = "Healthy"
        if len(recent_errors) > 10:
            system_health = "Warning"
        if len(recent_errors) > 50:
            system_health = "Critical"
        
        return {
            "summary": {
                "total_tokens": total_tokens,
                "total_cost": round(total_cost, 4),
                "total_requests": total_requests,
                "total_errors": len(recent_errors),
                "avg_response_time": round(avg_response_time, 3),
                "system_health": system_health,
                "time_range_hours": hours
            },
            "token_usage": {
                "by_model": token_by_model,
                "recent_usage": recent_tokens[-10:] if recent_tokens else []
            },
            "evaluation_metrics": evaluation_summary,
            "response_times": {
                "recent": recent_response_times[-20:] if recent_response_times else [],
                "average": avg_response_time
            },
            "errors": {
                "recent": recent_errors[-10:] if recent_errors else [],
                "count": len(recent_errors)
            },
            "requests": {
                "recent": recent_requests[-10:] if recent_requests else [],
                "count": total_requests
            }
        }
        
    @asynccontextmanager
    async def trace_operation(self, operation_name: str, **attributes):
        """Context manager for distributed tracing"""
        with self.tracer.start_as_current_span(operation_name) as span:
            for key, value in attributes.items():
                span.set_attribute(key, str(value))
            
            start_time = time.time()
            try:
                yield span
                span.set_status(Status(StatusCode.OK))
            except Exception as e:
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
                raise
            finally:
                duration = time.time() - start_time
                span.set_attribute("duration_seconds", duration)

observability = ObservabilityManager()

def setup_fastapi_instrumentation(app):
    """Setup FastAPI instrumentation for distributed tracing"""
    FastAPIInstrumentor.instrument_app(app)
    HTTPXClientInstrumentor().instrument()
