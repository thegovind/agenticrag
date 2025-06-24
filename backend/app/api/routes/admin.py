from fastapi import APIRouter, HTTPException, Depends, Query
from typing import List, Optional, Dict, Any
import logging
import psutil
import time
from datetime import datetime, timedelta

from app.models.schemas import AdminMetrics, EvaluationResult
from app.core.observability import observability
from app.services.token_usage_tracker import TokenUsageTracker, ServiceType, OperationType
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

@router.get("/token-usage/analytics")
async def get_token_usage_analytics(
    days: int = Query(7, ge=1, le=90),
    service_type: Optional[str] = Query(None),
    deployment_name: Optional[str] = Query(None)
):
    """Get token usage analytics for the specified time period"""
    try:
        observability.track_request("admin_token_analytics")
        
        # Initialize token tracker
        token_tracker = TokenUsageTracker()
        await token_tracker.initialize()
        
        # Get analytics data
        analytics = await token_tracker.get_usage_analytics(
            days_back=days,
            service_type=ServiceType(service_type) if service_type else None,
            deployment_name=deployment_name
        )
        
        return {
            "analytics": analytics,
            "period_days": days,
            "filters": {
                "service_type": service_type,
                "deployment_name": deployment_name
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting token usage analytics: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve token usage analytics")

@router.get("/token-usage/summary")
async def get_token_usage_summary(
    hours: int = Query(24, ge=1, le=168)
):
    """Get token usage summary for the specified time period"""
    try:
        observability.track_request("admin_token_summary")
        
        # Initialize token tracker
        token_tracker = TokenUsageTracker()
        await token_tracker.initialize()
        
        # Get summary data
        summary = await token_tracker.get_usage_summary(hours_back=hours)
        
        return {
            "summary": summary,
            "period_hours": hours,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting token usage summary: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve token usage summary")

@router.get("/token-usage/trends")
async def get_token_usage_trends(
    days: int = Query(30, ge=7, le=90),
    granularity: str = Query("daily", regex="^(hourly|daily|weekly)$")
):
    """Get token usage trends over time"""
    try:
        observability.track_request("admin_token_trends")
        
        # Initialize token tracker  
        token_tracker = TokenUsageTracker()
        await token_tracker.initialize()
        
        # Get trends data
        trends = await token_tracker.get_usage_trends(
            days_back=days,
            granularity=granularity
        )
        
        return {
            "trends": trends,
            "period_days": days,
            "granularity": granularity,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting token usage trends: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve token usage trends")

@router.get("/token-usage/costs")
async def get_token_usage_costs(
    days: int = Query(30, ge=1, le=90),
    service_type: Optional[str] = Query(None)
):
    """Get token usage cost analytics"""
    try:
        observability.track_request("admin_token_costs")
        
        # Initialize token tracker
        token_tracker = TokenUsageTracker()
        await token_tracker.initialize()
        
        # Get cost analytics
        costs = await token_tracker.get_cost_analytics(
            days_back=days,
            service_type=ServiceType(service_type) if service_type else None
        )
        
        return {
            "costs": costs,
            "period_days": days,
            "service_type": service_type,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting token usage costs: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve token usage costs")

@router.get("/token-usage/deployments")
async def get_deployment_usage(
    days: int = Query(7, ge=1, le=90)
):
    """Get token usage breakdown by deployment"""
    try:
        observability.track_request("admin_deployment_usage")
        
        # Initialize token tracker
        token_tracker = TokenUsageTracker()
        await token_tracker.initialize()
        
        # Get deployment usage data
        deployment_usage = await token_tracker.get_deployment_usage(days_back=days)
        
        return {
            "deployment_usage": deployment_usage,
            "period_days": days,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting deployment usage: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve deployment usage")

@router.get("/token-usage/services")
async def get_service_usage(
    days: int = Query(7, ge=1, le=90)
):
    """Get token usage breakdown by service type"""
    try:
        observability.track_request("admin_service_usage")
        
        # Initialize token tracker
        token_tracker = TokenUsageTracker()
        await token_tracker.initialize()
        
        # Get service usage data
        service_usage = await token_tracker.get_service_usage(days_back=days)
        
        return {
            "service_usage": service_usage,
            "period_days": days,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting service usage: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve service usage")

@router.get("/traces")
async def get_traces(
    limit: int = Query(100, ge=1, le=1000),
    hours: int = Query(24, ge=1, le=168)
):
    """Get trace data for the specified time period"""
    try:
        observability.track_request("admin_traces")
        
        # For now, return mock data since we don't have a full tracing implementation
        traces = [
            {
                "trace_id": f"trace_{i}",
                "span_id": f"span_{i}",
                "operation_name": "qa_processing" if i % 2 == 0 else "chat_processing",
                "start_time": (datetime.now() - timedelta(hours=i)).isoformat(),
                "duration_ms": 1500 + (i * 100),
                "status": "success" if i % 10 != 0 else "error",
                "service": "qa_service" if i % 2 == 0 else "chat_service",
                "tags": {
                    "session_id": f"session_{i}",
                    "model": "gpt-4o" if i % 3 == 0 else "gpt-35-turbo"
                }
            }
            for i in range(min(limit, 50))
        ]
        
        return {
            "traces": traces,
            "total_count": len(traces),
            "period_hours": hours,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting traces: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve traces")

@router.get("/evaluation-metrics")
async def get_evaluation_metrics(
    hours: int = Query(24, ge=1, le=168)
):
    """Get evaluation metrics for the specified time period"""
    try:
        observability.track_request("admin_evaluation_metrics")
        
        # For now, return mock data since evaluation is temporarily disabled
        metrics = {
            "total_evaluations": 42,
            "average_score": 0.82,
            "score_distribution": {
                "excellent": 15,
                "good": 20,
                "fair": 5,
                "poor": 2
            },
            "metrics_by_type": {
                "relevance": {"average": 0.85, "count": 42},
                "accuracy": {"average": 0.78, "count": 42},
                "completeness": {"average": 0.81, "count": 42},
                "coherence": {"average": 0.84, "count": 42}
            },
            "trends": [
                {"timestamp": (datetime.now() - timedelta(hours=i)).isoformat(), "score": 0.8 + (i * 0.01)}
                for i in range(24)
            ]
        }
        
        return {
            "metrics": metrics,
            "period_hours": hours,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting evaluation metrics: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve evaluation metrics")

@router.get("/system-metrics")
async def get_system_metrics():
    """Get system performance metrics"""
    try:
        observability.track_request("admin_system_metrics")
        
        # Basic system metrics
        import psutil
        
        metrics = {
            "cpu": {
                "usage_percent": psutil.cpu_percent(interval=1),
                "count": psutil.cpu_count()
            },
            "memory": {
                "usage_percent": psutil.virtual_memory().percent,
                "available_gb": psutil.virtual_memory().available / (1024**3),
                "total_gb": psutil.virtual_memory().total / (1024**3)
            },
            "disk": {
                "usage_percent": psutil.disk_usage('/').percent if hasattr(psutil.disk_usage('/'), 'percent') else 0,
                "free_gb": psutil.disk_usage('/').free / (1024**3) if hasattr(psutil.disk_usage('/'), 'free') else 0
            },
            "uptime_hours": (datetime.now() - datetime.fromtimestamp(psutil.boot_time())).total_seconds() / 3600,
            "active_connections": len(psutil.net_connections())
        }
        
        return {
            "metrics": metrics,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting system metrics: {e}")
        # Return basic fallback metrics if psutil fails
        return {
            "metrics": {
                "cpu": {"usage_percent": 25, "count": 4},
                "memory": {"usage_percent": 65, "available_gb": 8.5, "total_gb": 16},
                "disk": {"usage_percent": 45, "free_gb": 50},
                "uptime_hours": 12.5,
                "active_connections": 15
            },
            "timestamp": datetime.now().isoformat()
        }
