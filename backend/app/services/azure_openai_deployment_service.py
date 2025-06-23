"""
Azure OpenAI Deployment Management Service

This service uses the Azure Management API to fetch available deployments
for model configuration dropdown functionality.
"""

import logging
import asyncio
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from urllib.parse import urlparse
import aiohttp
import json

from azure.identity.aio import DefaultAzureCredential
from azure.core.exceptions import AzureError
from app.core.config import Settings

logger = logging.getLogger(__name__)

@dataclass
class DeploymentInfo:
    """Information about an Azure OpenAI deployment"""
    deployment_name: str  # The deployment name used for API calls
    model_name: str      # The actual model name (e.g., gpt-4o-mini)
    model_version: str   # Model version
    model_type: str      # 'chat' or 'embedding'
    sku_name: str        # SKU information
    capacity: int        # Current capacity
    capabilities: Dict[str, Any]  # Model capabilities
    provisioning_state: str  # Deployment state

class AzureOpenAIDeploymentService:
    """Service for managing Azure OpenAI deployments via Management API"""
    
    def __init__(self, settings: Settings = None):
        self.settings = settings or Settings()
        self.credential = None
        self._session = None
        
        # Extract account name from endpoint
        self.account_name = self._extract_account_name(self.settings.AZURE_OPENAI_ENDPOINT)
        
    def _extract_account_name(self, endpoint: str) -> str:
        """Extract account name from Azure OpenAI endpoint"""
        try:
            parsed = urlparse(endpoint)
            # Extract from hostname like: astdnapubaoai.openai.azure.com
            hostname_parts = parsed.hostname.split('.')
            return hostname_parts[0] if hostname_parts else ""
        except Exception as e:
            logger.error(f"Failed to extract account name from endpoint {endpoint}: {e}")
            return ""
    
    async def __aenter__(self):
        """Async context manager entry"""
        self.credential = DefaultAzureCredential()
        self._session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self._session:
            await self._session.close()
        if self.credential:
            await self.credential.close()
    
    async def _get_access_token(self) -> str:
        """Get Azure access token for Management API"""
        try:
            token = await self.credential.get_token("https://management.azure.com/.default")
            return token.token
        except Exception as e:
            logger.error(f"Failed to get access token: {e}")
            raise
    
    async def get_deployments(self) -> List[DeploymentInfo]:
        """
        Fetch all deployments from Azure OpenAI account using Management API
        
        Returns:
            List of DeploymentInfo objects with deployment details
        """
        if not self.account_name:
            logger.error("Cannot fetch deployments: account name not available")
            return []
        
        try:
            access_token = await self._get_access_token()
            
            # Construct the management API URL
            base_url = "https://management.azure.com"
            url = (
                f"{base_url}/subscriptions/{self.settings.AZURE_SUBSCRIPTION_ID}"
                f"/resourceGroups/{self.settings.AZURE_AI_FOUNDRY_RESOURCE_GROUP}"
                f"/providers/Microsoft.CognitiveServices/accounts/{self.account_name}"
                f"/deployments"
            )
            
            headers = {
                "Authorization": f"Bearer {access_token}",
                "Content-Type": "application/json"
            }
            
            params = {
                "api-version": "2024-10-01"
            }
            
            logger.info(f"Fetching deployments from: {url}")
            
            async with self._session.get(url, headers=headers, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    deployments = self._parse_deployments(data.get("value", []))
                    logger.info(f"Successfully fetched {len(deployments)} deployments")
                    return deployments
                else:
                    error_text = await response.text()
                    logger.error(f"Failed to fetch deployments: {response.status} - {error_text}")
                    return []
                    
        except Exception as e:
            logger.error(f"Error fetching deployments: {e}")
            return []
    
    def _parse_deployments(self, deployment_data: List[Dict[str, Any]]) -> List[DeploymentInfo]:
        """Parse deployment data from Azure Management API response"""
        deployments = []
        
        for deployment in deployment_data:
            try:
                properties = deployment.get("properties", {})
                model_info = properties.get("model", {})
                sku = deployment.get("sku", {})
                
                deployment_name = deployment.get("name", "")
                model_name = model_info.get("name", "")
                model_version = model_info.get("version", "")
                sku_name = sku.get("name", "")
                capacity = sku.get("capacity", 0)
                capabilities = properties.get("capabilities", {})
                provisioning_state = properties.get("provisioningState", "")
                
                # Determine model type based on model name
                model_type = "embedding" if "embedding" in model_name.lower() else "chat"
                
                deployment_info = DeploymentInfo(
                    deployment_name=deployment_name,
                    model_name=model_name,
                    model_version=model_version,
                    model_type=model_type,
                    sku_name=sku_name,
                    capacity=capacity,
                    capabilities=capabilities,
                    provisioning_state=provisioning_state
                )
                
                deployments.append(deployment_info)
                logger.debug(f"Parsed deployment: {deployment_name} ({model_name}, {model_type})")
                
            except Exception as e:
                logger.warning(f"Failed to parse deployment: {e}")
                continue
        
        return deployments
    
    async def get_chat_deployments(self) -> List[DeploymentInfo]:
        """Get only chat model deployments"""
        all_deployments = await self.get_deployments()
        return [d for d in all_deployments if d.model_type == "chat"]
    
    async def get_embedding_deployments(self) -> List[DeploymentInfo]:
        """Get only embedding model deployments"""
        all_deployments = await self.get_deployments()
        return [d for d in all_deployments if d.model_type == "embedding"]
    
    async def get_deployments_summary(self) -> Dict[str, Any]:
        """Get a summary of available deployments"""
        deployments = await self.get_deployments()
        
        chat_models = [d for d in deployments if d.model_type == "chat"]
        embedding_models = [d for d in deployments if d.model_type == "embedding"]
        
        return {
            "total_deployments": len(deployments),
            "chat_models": [
                {
                    "deployment_name": d.deployment_name,
                    "model_name": d.model_name,
                    "model_version": d.model_version,
                    "sku": d.sku_name,
                    "capacity": d.capacity,
                    "state": d.provisioning_state
                }
                for d in chat_models
            ],
            "embedding_models": [
                {
                    "deployment_name": d.deployment_name,
                    "model_name": d.model_name,
                    "model_version": d.model_version,
                    "sku": d.sku_name,
                    "capacity": d.capacity,
                    "state": d.provisioning_state
                }
                for d in embedding_models
            ],
            "account_name": self.account_name,
            "resource_group": self.settings.AZURE_AI_FOUNDRY_RESOURCE_GROUP,
            "subscription_id": self.settings.AZURE_SUBSCRIPTION_ID
        }


# Mock service for development/testing
class MockAzureOpenAIDeploymentService(AzureOpenAIDeploymentService):
    """Mock implementation for testing and development"""
    
    def __init__(self, settings: Settings = None):
        super().__init__(settings)
        self._mock_deployments = [
            DeploymentInfo(
                deployment_name="chat4omini",
                model_name="gpt-4o-mini",
                model_version="2024-07-18",
                model_type="chat",
                sku_name="GlobalStandard",
                capacity=1101,
                capabilities={"chatCompletion": "true", "maxContextToken": "128000"},
                provisioning_state="Succeeded"
            ),
            DeploymentInfo(
                deployment_name="gpt4turbo",
                model_name="gpt-4-turbo",
                model_version="2024-04-09",
                model_type="chat",
                sku_name="Standard",
                capacity=20,
                capabilities={"chatCompletion": "true", "maxContextToken": "128000"},
                provisioning_state="Succeeded"
            ),
            DeploymentInfo(
                deployment_name="embedding",
                model_name="text-embedding-3-small",
                model_version="1",
                model_type="embedding",
                sku_name="Standard",
                capacity=120,
                capabilities={"embedding": "true"},
                provisioning_state="Succeeded"
            ),
            DeploymentInfo(
                deployment_name="embedding-large",
                model_name="text-embedding-3-large",
                model_version="1",
                model_type="embedding",
                sku_name="Standard",
                capacity=50,
                capabilities={"embedding": "true"},
                provisioning_state="Succeeded"
            )
        ]
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass
    
    async def get_deployments(self) -> List[DeploymentInfo]:
        """Return mock deployments"""
        logger.info(f"Returning {len(self._mock_deployments)} mock deployments")
        # Simulate async delay
        await asyncio.sleep(0.1)
        return self._mock_deployments
