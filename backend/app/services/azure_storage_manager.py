import logging
import asyncio
from typing import Dict, List, Optional, Any, BinaryIO
from datetime import datetime, timedelta
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient
from azure.storage.blob.aio import BlobServiceClient as AsyncBlobServiceClient, BlobClient as AsyncBlobClient, ContainerClient as AsyncContainerClient
from azure.identity.aio import DefaultAzureCredential as AsyncDefaultAzureCredential, ClientSecretCredential as AsyncClientSecretCredential
from azure.identity import DefaultAzureCredential, ClientSecretCredential
from azure.core.exceptions import ResourceNotFoundError, ResourceExistsError
import os
import hashlib
import mimetypes

from app.core.config import settings
from app.core.observability import observability

logger = logging.getLogger(__name__)

class AzureStorageManager:
    """Azure Storage manager for document uploads and management"""
    
    def __init__(self):
        self.async_blob_service_client = None
        self.credential = None
        self.account_url = None
        self.container_name = settings.AZURE_STORAGE_CONTAINER_NAME or "financial-documents"
        
    async def initialize(self):
        """Initialize Azure Storage clients"""
        try:
            if not settings.AZURE_STORAGE_ACCOUNT_NAME:
                logger.warning("Azure Storage account name not configured, using mock storage")
                return
                
            if settings.AZURE_CLIENT_SECRET and settings.AZURE_TENANT_ID and settings.AZURE_CLIENT_ID:
                self.credential = AsyncClientSecretCredential(
                    tenant_id=settings.AZURE_TENANT_ID,
                    client_id=settings.AZURE_CLIENT_ID,
                    client_secret=settings.AZURE_CLIENT_SECRET
                )
            else:
                self.credential = AsyncDefaultAzureCredential()
            
            self.account_url = f"https://{settings.AZURE_STORAGE_ACCOUNT_NAME}.blob.core.windows.net"
            
            self.async_blob_service_client = AsyncBlobServiceClient(
                account_url=self.account_url,
                credential=self.credential
            )
            
            await self._ensure_container_exists()
            
            logger.info("Azure Storage manager initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Azure Storage: {e}")
            raise
    
    async def _ensure_container_exists(self):
        """Ensure the storage container exists"""
        try:
            container_client = self.async_blob_service_client.get_container_client(self.container_name)
            
            try:
                await container_client.get_container_properties()
                logger.info(f"Container '{self.container_name}' already exists")
            except ResourceNotFoundError:
                await container_client.create_container()
                logger.info(f"Created container '{self.container_name}'")
                    
        except Exception as e:
            logger.error(f"Failed to ensure container exists: {e}")
            raise

    async def upload_document(
        self, 
        file_content: bytes, 
        filename: str, 
        content_type: str = None,
        metadata: Dict[str, str] = None
    ) -> Dict[str, Any]:
        """Upload a document to Azure Storage"""
        try:
            # with observability.trace_operation("azure_storage_upload_document") as span:
            # span.set_attribute("filename", filename)
            # span.set_attribute("content_size", len(file_content))
                
                file_hash = hashlib.md5(file_content).hexdigest()
                timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
                blob_name = f"{timestamp}_{file_hash}_{filename}"
                
                if not content_type:
                    content_type, _ = mimetypes.guess_type(filename)
                    content_type = content_type or "application/octet-stream"
                
                blob_metadata = {
                    "original_filename": filename,
                    "upload_timestamp": datetime.utcnow().isoformat(),
                    "content_hash": file_hash,
                    "content_size": str(len(file_content))
                }
                
                if metadata:
                    blob_metadata.update(metadata)
                
                # Use the initialized async client directly
                blob_client = self.async_blob_service_client.get_blob_client(
                    container=self.container_name,
                    blob=blob_name
                )
                
                await blob_client.upload_blob(
                    file_content,
                    content_type=content_type,
                    metadata=blob_metadata,
                    overwrite=True
                )
                
                blob_url = f"https://{settings.AZURE_STORAGE_ACCOUNT_NAME}.blob.core.windows.net/{self.container_name}/{blob_name}"
                
                result = {
                    "blob_name": blob_name,
                    "blob_url": blob_url,
                    "content_type": content_type,
                    "size": len(file_content),
                    "hash": file_hash,
                    "metadata": blob_metadata,
                    "upload_timestamp": datetime.utcnow().isoformat()
                }
                
            # span.set_attribute("blob_name", blob_name)
            # span.set_attribute("blob_url", blob_url)
            # span.set_attribute("success", True)
                
                logger.info(f"Successfully uploaded document: {filename} -> {blob_name}")
                
                return result
                
        except Exception as e:
            logger.error(f"Failed to upload document {filename}: {e}")
            observability.record_error("azure_storage_upload_error", str(e))
            raise

    async def download_document(self, blob_name: str) -> Dict[str, Any]:
        """Download a document from Azure Storage"""
        try:
            # with observability.trace_operation("azure_storage_download_document") as span:
            # span.set_attribute("blob_name", blob_name)
                
                # Use the initialized async client directly
                blob_client = self.async_blob_service_client.get_blob_client(
                    container=self.container_name,
                    blob=blob_name
                )
                
                properties = await blob_client.get_blob_properties()
                
                download_stream = await blob_client.download_blob()
                content = await download_stream.readall()
                
                result = {
                    "content": content,
                    "content_type": properties.content_settings.content_type,
                    "size": properties.size,
                    "metadata": properties.metadata,
                    "last_modified": properties.last_modified.isoformat() if properties.last_modified else None
                }
                
            # span.set_attribute("content_size", len(content))
            # span.set_attribute("success", True)
                
                logger.info(f"Successfully downloaded document: {blob_name}")
                
                return result
                
        except Exception as e:
            logger.error(f"Failed to download document {blob_name}: {e}")
            observability.record_error("azure_storage_download_error", str(e))
            raise
    
    async def list_documents(
        self, 
        prefix: str = None, 
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """List documents in Azure Storage"""
        try:
            # with observability.trace_operation("azure_storage_list_documents") as span:
            # span.set_attribute("prefix", prefix or "all")
            # span.set_attribute("limit", limit)
                
                documents = []
                
                # Use the initialized async client directly
                container_client = self.async_blob_service_client.get_container_client(self.container_name)
                
                async for blob in container_client.list_blobs(name_starts_with=prefix):
                    if len(documents) >= limit:
                        break
                        
                    documents.append({
                        "blob_name": blob.name,
                        "size": blob.size,
                        "content_type": blob.content_settings.content_type if blob.content_settings else None,
                        "last_modified": blob.last_modified.isoformat() if blob.last_modified else None,
                        "metadata": blob.metadata or {},
                        "blob_url": f"https://{settings.AZURE_STORAGE_ACCOUNT_NAME}.blob.core.windows.net/{self.container_name}/{blob.name}"
                    })
                
            # span.set_attribute("documents_found", len(documents))
            # span.set_attribute("success", True)
                
                logger.info(f"Listed {len(documents)} documents from Azure Storage")
                
                return documents
                
        except Exception as e:
            logger.error(f"Failed to list documents: {e}")
            observability.record_error("azure_storage_list_error", str(e))
            raise
    
    async def delete_document(self, blob_name: str) -> bool:
        """Delete a document from Azure Storage"""
        try:
            # with observability.trace_operation("azure_storage_delete_document") as span:
            # span.set_attribute("blob_name", blob_name)
                
                # Use the initialized async client directly
                blob_client = self.async_blob_service_client.get_blob_client(
                    container=self.container_name,
                    blob=blob_name
                )
                
                await blob_client.delete_blob()
                
            # span.set_attribute("success", True)
                
                logger.info(f"Successfully deleted document: {blob_name}")
                
                return True
                
        except ResourceNotFoundError:
            logger.warning(f"Document not found for deletion: {blob_name}")
            return False
        except Exception as e:
            logger.error(f"Failed to delete document {blob_name}: {e}")
            observability.record_error("azure_storage_delete_error", str(e))
            raise
    
    async def get_document_url(self, blob_name: str, expires_in_hours: int = 24) -> str:
        """Generate a SAS URL for document access"""
        try:
            from azure.storage.blob import generate_blob_sas, BlobSasPermissions
            
            sas_token = generate_blob_sas(
                account_name=settings.AZURE_STORAGE_ACCOUNT_NAME,
                container_name=self.container_name,
                blob_name=blob_name,
                account_key=settings.AZURE_STORAGE_ACCOUNT_KEY,  # Need to add this to config
                permission=BlobSasPermissions(read=True),
                expiry=datetime.utcnow() + timedelta(hours=expires_in_hours)
            )
            
            sas_url = f"https://{settings.AZURE_STORAGE_ACCOUNT_NAME}.blob.core.windows.net/{self.container_name}/{blob_name}?{sas_token}"
            
            logger.info(f"Generated SAS URL for {blob_name}, expires in {expires_in_hours} hours")
            
            return sas_url
            
        except Exception as e:
            logger.error(f"Failed to generate SAS URL for {blob_name}: {e}")
            return f"https://{settings.AZURE_STORAGE_ACCOUNT_NAME}.blob.core.windows.net/{self.container_name}/{blob_name}"
    
    async def cleanup(self):
        """Cleanup resources"""
        try:
            if self.async_blob_service_client:
                await self.async_blob_service_client.close()
            logger.info("Azure Storage manager cleaned up")
        except Exception as e:
            logger.error(f"Error during Azure Storage cleanup: {e}")


class MockStorageManager:
    """Mock storage manager for development/testing"""
    
    def __init__(self):
        self.documents = {}
        
    async def initialize(self):
        logger.info("Mock storage manager initialized")
        
    async def upload_document(self, file_content: bytes, filename: str, content_type: str = None, metadata: Dict[str, str] = None) -> Dict[str, Any]:
        file_hash = hashlib.md5(file_content).hexdigest()
        blob_name = f"mock_{file_hash}_{filename}"
        
        self.documents[blob_name] = {
            "content": file_content,
            "filename": filename,
            "content_type": content_type or "application/octet-stream",
            "metadata": metadata or {},
            "upload_timestamp": datetime.utcnow().isoformat()
        }
        
        return {
            "blob_name": blob_name,
            "blob_url": f"mock://storage/{blob_name}",
            "content_type": content_type,
            "size": len(file_content),
            "hash": file_hash,
            "metadata": metadata or {},
            "upload_timestamp": datetime.utcnow().isoformat()
        }
    
    async def download_document(self, blob_name: str) -> Dict[str, Any]:
        if blob_name not in self.documents:
            raise ResourceNotFoundError(f"Document {blob_name} not found")
            
        doc = self.documents[blob_name]
        return {
            "content": doc["content"],
            "content_type": doc["content_type"],
            "size": len(doc["content"]),
            "metadata": doc["metadata"],
            "last_modified": doc["upload_timestamp"]
        }
    
    async def list_documents(self, prefix: str = None, limit: int = 100) -> List[Dict[str, Any]]:
        documents = []
        for blob_name, doc in self.documents.items():
            if prefix and not blob_name.startswith(prefix):
                continue
            if len(documents) >= limit:
                break
                
            documents.append({
                "blob_name": blob_name,
                "size": len(doc["content"]),
                "content_type": doc["content_type"],
                "last_modified": doc["upload_timestamp"],
                "metadata": doc["metadata"],
                "blob_url": f"mock://storage/{blob_name}"
            })
        
        return documents
    
    async def delete_document(self, blob_name: str) -> bool:
        if blob_name in self.documents:
            del self.documents[blob_name]
            return True
        return False
    
    async def get_document_url(self, blob_name: str, expires_in_hours: int = 24) -> str:
        return f"mock://storage/{blob_name}?expires={expires_in_hours}h"
    
    async def cleanup(self):
        logger.info("Mock storage manager cleaned up")
