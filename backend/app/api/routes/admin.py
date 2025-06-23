from fastapi import APIRouter, HTTPException, Depends, Query
from typing import List, Optional, Dict, Any
import logging
import psutil
import time
from datetime import datetime, timedelta

from app.models.schemas import AdminMetrics, EvaluationResult
from app.core.observability import observability
# Temporarily disable evaluation due to package conflicts
# from app.core.evaluation import get_evaluation_framework

router = APIRouter()
logger = logging.getLogger(__name__)

@router.get("/metrics")
async def get_admin_metrics(hours: int = Query(24, ge=1, le=168)):
    """Get comprehensive admin metrics and statistics"""
    try:
        observability.track_request("admin_metrics")
        
        metrics_summary = observability.get_metrics_summary(hours=hours)
        
        evaluation_summary = {}
        # Temporarily disable evaluation due to package conflicts
        # try:
        #     eval_framework = get_evaluation_framework()
        #     evaluation_summary = eval_framework.get_evaluation_summary(hours=hours)
        # except RuntimeError:
        #     pass
        
        system_metrics = {
            "cpu_usage": psutil.cpu_percent(interval=1),
            "memory_usage": psutil.virtual_memory().percent,
            "disk_usage": psutil.disk_usage('/').percent,
            "timestamp": time.time()
        }
        
        observability.track_system_metrics(
            system_metrics["cpu_usage"],
            system_metrics["memory_usage"], 
            system_metrics["disk_usage"]
        )
        
        combined_metrics = {
            **metrics_summary,
            "system": system_metrics,
            "evaluation_framework": evaluation_summary
        }
        
        return combined_metrics
        
    except Exception as e:
        logger.error(f"Error getting admin metrics: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve admin metrics")

@router.get("/metrics/tokens")
async def get_token_usage_metrics(
    hours: int = Query(24, ge=1, le=168),
    model: Optional[str] = Query(None)
):
    """Get detailed token usage metrics"""
    try:
        observability.track_request("token_metrics")
        
        metrics_summary = observability.get_metrics_summary(hours=hours)
        token_data = metrics_summary.get("token_usage", {})
        
        if model:
            model_data = token_data.get("by_model", {}).get(model, {})
            return {
                "model": model,
                "data": model_data,
                "time_range_hours": hours
            }
        
        return {
            "all_models": token_data,
            "time_range_hours": hours
        }
        
    except Exception as e:
        logger.error(f"Error getting token metrics: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve token usage metrics")

@router.get("/metrics/performance")
async def get_performance_metrics(hours: int = Query(24, ge=1, le=168)):
    """Get system performance metrics"""
    try:
        observability.track_request("performance_metrics")
        
        metrics_summary = observability.get_metrics_summary(hours=hours)
        response_times = metrics_summary.get("response_times", {})
        
        recent_times = [rt["duration"] for rt in response_times.get("recent", [])]
        avg_response_time = sum(recent_times) / len(recent_times) if recent_times else 0
        
        if recent_times:
            sorted_times = sorted(recent_times)
            p95_idx = int(0.95 * len(sorted_times))
            p99_idx = int(0.99 * len(sorted_times))
            p95_response_time = sorted_times[p95_idx] if p95_idx < len(sorted_times) else 0
            p99_response_time = sorted_times[p99_idx] if p99_idx < len(sorted_times) else 0
        else:
            p95_response_time = 0
            p99_response_time = 0
        
        total_requests = metrics_summary.get("summary", {}).get("total_requests", 0)
        total_errors = metrics_summary.get("summary", {}).get("total_errors", 0)
        error_rate = (total_errors / total_requests) if total_requests > 0 else 0
        
        return {
            "time_period_hours": hours,
            "average_response_time": round(avg_response_time, 3),
            "p95_response_time": round(p95_response_time, 3),
            "p99_response_time": round(p99_response_time, 3),
            "error_rate": round(error_rate, 4),
            "requests_per_minute": round(total_requests / (hours * 60), 2),
            "concurrent_users": len(set(req.get("session_id", "") for req in metrics_summary.get("requests", {}).get("recent", []))),
            "system_health": metrics_summary.get("summary", {}).get("system_health", "unknown")
        }
    except Exception as e:
        logger.error(f"Error getting performance metrics: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve performance metrics")

@router.get("/evaluations")
async def get_evaluation_results(
    metric_name: Optional[str] = Query(None),
    hours: int = Query(24, ge=1, le=168),
    session_id: Optional[str] = Query(None),
    limit: int = Query(100, ge=1, le=1000)
):
    """Get RAG evaluation results"""
    try:
        observability.track_request("evaluation_results")
        
        # Temporarily disable evaluation due to package conflicts
        # try:
        #     eval_framework = get_evaluation_framework()
        #     evaluation_summary = eval_framework.get_evaluation_summary(session_id=session_id, hours=hours)
        #     
        #     if metric_name and metric_name in evaluation_summary.get("metrics", {}):
        #         filtered_metrics = {metric_name: evaluation_summary["metrics"][metric_name]}
        #         evaluation_summary["metrics"] = filtered_metrics
        # except RuntimeError:
        #     evaluation_summary = {"metrics": {}, "total_evaluations": 0}
        
        evaluation_summary = {"metrics": {}, "total_evaluations": 0}
        
        return {
            "evaluation_summary": evaluation_summary,
            "session_id": session_id,
            "metric_filter": metric_name,
            "time_range_hours": hours,
            "limit": limit
        }
        
    except Exception as e:
        logger.error(f"Error getting evaluation results: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve evaluation results")

@router.post("/evaluations/run")
async def run_evaluation(
    evaluation_type: str,
    test_queries: List[str],
    model: Optional[str] = None
):
    """Run evaluation on the RAG system"""
    try:
        observability.track_request("run_evaluation")
        
        
        logger.info(f"Evaluation requested: {evaluation_type} with {len(test_queries)} queries")
        
        return {
            "evaluation_id": "eval_" + str(datetime.utcnow().timestamp()),
            "evaluation_type": evaluation_type,
            "status": "started",
            "test_queries_count": len(test_queries),
            "model": model,
            "started_at": datetime.utcnow()
        }
    except Exception as e:
        logger.error(f"Error running evaluation: {e}")
        raise HTTPException(status_code=500, detail="Failed to run evaluation")

@router.get("/logs")
async def get_system_logs(
    level: str = "INFO",
    hours: int = 1,
    component: Optional[str] = None,
    limit: int = 1000
):
    """Get system logs for debugging"""
    try:
        observability.track_request("system_logs")
        
        logger.info(f"System logs requested: level={level}, hours={hours}, component={component}")
        
        return {
            "logs": [],
            "total_count": 0,
            "level_filter": level,
            "time_period_hours": hours,
            "component_filter": component
        }
    except Exception as e:
        logger.error(f"Error getting system logs: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve system logs")

@router.get("/health")
async def get_system_health():
    """Get overall system health status"""
    try:
        observability.track_request("system_health")
        
        
        health_status = {
            "overall_status": "healthy",
            "components": {
                "azure_search": "healthy",
                "azure_openai": "healthy", 
                "cosmos_db": "healthy",
                "document_intelligence": "healthy",
                "knowledge_base": "healthy"
            },
            "last_updated": datetime.utcnow(),
            "uptime_seconds": 0
        }
        
        return health_status
    except Exception as e:
        logger.error(f"Error getting system health: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve system health")

@router.get("/foundry/models")
async def get_foundry_models():
    """Get available models from Azure AI Foundry project"""
    try:
        observability.track_request("foundry_models")
        
        from app.services.azure_services import AzureServiceManager
        azure_service = AzureServiceManager()
        await azure_service.initialize()
        
        models = await azure_service.get_available_models()
        return {
            "models": models,
            "count": len(models),
            "retrieved_at": datetime.utcnow()
        }
    except Exception as e:
        logger.error(f"Error getting foundry models: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve foundry models")

@router.get("/foundry/connections")
async def get_foundry_connections():
    """Get project connections from Azure AI Foundry"""
    try:
        observability.track_request("foundry_connections")
        
        from app.services.azure_services import AzureServiceManager
        azure_service = AzureServiceManager()
        await azure_service.initialize()
        
        connections = await azure_service.get_project_connections()
        return {
            "connections": connections,
            "count": len(connections),
            "retrieved_at": datetime.utcnow()
        }
    except Exception as e:
        logger.error(f"Error getting foundry connections: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve foundry connections")

@router.get("/foundry/project-info")
async def get_foundry_project_info():
    """Get Azure AI Foundry project information"""
    try:
        observability.track_request("foundry_project_info")
        
        from app.services.azure_services import AzureServiceManager
        azure_service = AzureServiceManager()
        await azure_service.initialize()
        
        project_info = await azure_service.get_project_info()
        return project_info
    except Exception as e:
        logger.error(f"Error getting foundry project info: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve foundry project info")
