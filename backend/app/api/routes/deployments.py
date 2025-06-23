"""
Azure OpenAI Deployment API Routes

Provides endpoints for fetching available Azure OpenAI deployments
for model configuration dropdown functionality.
"""

from fastapi import APIRouter, HTTPException, Depends
from typing import List, Dict, Any
import logging

from app.services.azure_openai_deployment_service import (
    AzureOpenAIDeploymentService, 
    MockAzureOpenAIDeploymentService,
    DeploymentInfo
)
from app.core.config import Settings

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/deployments", tags=["deployments"])

# Use mock service for development - switch to real service when ready
USE_MOCK_SERVICE = True

def get_deployment_service():
    """Dependency to get deployment service"""
    settings = Settings()
    if USE_MOCK_SERVICE:
        return MockAzureOpenAIDeploymentService(settings)
    else:
        return AzureOpenAIDeploymentService(settings)

@router.get("/", response_model=Dict[str, Any])
async def get_all_deployments():
    """
    Get all available Azure OpenAI deployments
    
    Returns:
        Dictionary containing deployment summary with chat and embedding models
    """
    try:
        async with get_deployment_service() as service:
            summary = await service.get_deployments_summary()
            return {
                "success": True,
                "data": summary
            }
    except Exception as e:
        logger.error(f"Failed to fetch deployments: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch deployments: {str(e)}"
        )

@router.get("/chat", response_model=Dict[str, Any])
async def get_chat_deployments():
    """
    Get available chat model deployments
    
    Returns:
        List of chat model deployments
    """
    try:
        async with get_deployment_service() as service:
            deployments = await service.get_chat_deployments()
            return {
                "success": True,
                "data": [
                    {
                        "deployment_name": d.deployment_name,
                        "model_name": d.model_name,
                        "model_version": d.model_version,
                        "sku": d.sku_name,
                        "capacity": d.capacity,
                        "state": d.provisioning_state,
                        "display_name": f"{d.model_name} ({d.deployment_name})"
                    }
                    for d in deployments
                ]
            }
    except Exception as e:
        logger.error(f"Failed to fetch chat deployments: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch chat deployments: {str(e)}"
        )

@router.get("/embedding", response_model=Dict[str, Any])
async def get_embedding_deployments():
    """
    Get available embedding model deployments
    
    Returns:
        List of embedding model deployments
    """
    try:
        async with get_deployment_service() as service:
            deployments = await service.get_embedding_deployments()
            return {
                "success": True,
                "data": [
                    {
                        "deployment_name": d.deployment_name,
                        "model_name": d.model_name,
                        "model_version": d.model_version,
                        "sku": d.sku_name,
                        "capacity": d.capacity,
                        "state": d.provisioning_state,
                        "display_name": f"{d.model_name} ({d.deployment_name})"
                    }
                    for d in deployments
                ]
            }
    except Exception as e:
        logger.error(f"Failed to fetch embedding deployments: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch embedding deployments: {str(e)}"
        )

@router.get("/health")
async def deployment_service_health():
    """
    Health check for deployment service
    
    Returns:
        Service health status
    """
    try:
        async with get_deployment_service() as service:
            # Try to fetch a simple summary to verify service is working
            summary = await service.get_deployments_summary()
            return {
                "success": True,
                "status": "healthy",
                "account_name": summary.get("account_name", ""),
                "total_deployments": summary.get("total_deployments", 0),
                "service_type": "mock" if USE_MOCK_SERVICE else "real"
            }
    except Exception as e:
        logger.error(f"Deployment service health check failed: {e}")
        raise HTTPException(
            status_code=503,
            detail=f"Deployment service unhealthy: {str(e)}"
        )
