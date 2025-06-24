from azure.search.documents import SearchClient
from azure.search.documents.aio import SearchClient as AsyncSearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.models import VectorizedQuery
from azure.ai.formrecognizer import DocumentAnalysisClient
from azure.cosmos import CosmosClient
from azure.identity import DefaultAzureCredential, ClientSecretCredential
from openai import AzureOpenAI, AsyncAzureOpenAI
from azure.ai.projects import AIProjectClient
from azure.ai.projects.models import ConnectionType
import asyncio
import logging
import os
import platform
from typing import List, Dict, Any, Optional, Union
from datetime import datetime, date
import hashlib
import json
from dataclasses import dataclass
import time

# Configure Windows event loop policy for Azure SDK compatibility
if platform.system() == "Windows":
    try:
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    except Exception:
        pass

from app.core.config import settings
from app.core import observability
from app.services.azure_storage_manager import AzureStorageManager, MockStorageManager
from app.services.azure_openai_deployment_service import AzureOpenAIDeploymentService, MockAzureOpenAIDeploymentService

logger = logging.getLogger(__name__)

class MockSearchClient:
    def __init__(self):
        self.documents = []
    
    def upload_documents(self, documents):
        self.documents.extend(documents)
        return {"status": "success", "count": len(documents)}
    
    def search(self, search_text=None, vector_queries=None, **kwargs):
        return [
            {
                "id": "mock-doc-1",
                "content": "Sample financial content from 10-K report",
                "title": "Sample Financial Corporation 10-K",
                "document_type": "10-K",
                "company": "Sample Financial Corporation",
                "filing_date": "2023-12-31",
                "source_url": "mock://sample-10k.pdf",
                "credibility_score": 0.95
            }
        ]

class MockSearchIndexClient:
    def create_or_update_index(self, index):
        return {"status": "success", "name": index.name}

class MockDocumentAnalysisClient:
    def begin_analyze_document(self, model_id, document):
        class MockPoller:
            def result(self):
                class MockResult:
                    def __init__(self):
                        self.content = "Mock extracted content from financial document"
                        self.pages = [{"page_number": 1}]
                        self.tables = []
                        self.key_value_pairs = []
                return MockResult()
        return MockPoller()

class MockCosmosClient:
    def __init__(self):
        self.sessions = {}
    
    def get_database_client(self, database_name):
        return MockDatabaseClient(self.sessions)
    
    def close(self):
        pass

class MockDatabaseClient:
    def __init__(self, sessions):
        self.sessions = sessions
    
    def get_container_client(self, container_name):
        return MockContainerClient(self.sessions)

class MockContainerClient:
    def __init__(self, sessions):
        self.sessions = sessions
    
    def read_item(self, item, partition_key):
        if item in self.sessions:
            return self.sessions[item]
        raise Exception("Item not found")
    
    def upsert_item(self, item):
        self.sessions[item["id"]] = item
        return item

class MockOpenAIClient:
    def __init__(self):
        self.embeddings = MockEmbeddings()

class MockAIFoundryClient:
    def __init__(self):
        self.agents = MockAgents()
        self.evaluations = MockEvaluations()
        self.connections = MockConnections()
    
    def get_connection_by_name(self, name: str):
        return {"name": name, "type": "mock", "status": "connected"}

class MockAgent:
    def __init__(self, agent_id: str, name: str, status: str = "active"):
        self.id = agent_id
        self.name = name
        self.status = status

class MockThread:
    def __init__(self, thread_id: str):
        self.id = thread_id

class MockRun:
    def __init__(self, run_id: str, status: str = "completed"):
        self.id = run_id
        self.status = status

class MockMessage:
    def __init__(self, message_id: str, content: str, role: str = "assistant"):
        self.id = message_id
        self.role = role
        self.content = [{"text": {"value": content}}]

class MockAgents:
    def create_agent(self, **kwargs):
        return MockAgent("mock-agent-1", kwargs.get("name", "Mock Agent"))
    
    def create_thread(self):
        return MockThread("mock-thread-1")
    
    def create_message(self, thread_id: str, **kwargs):
        return MockMessage("mock-message-1", kwargs.get("content", "Mock message content"))
    
    def create_run(self, thread_id: str, **kwargs):
        return MockRun("mock-run-1")
    
    def get_run(self, thread_id: str, run_id: str):
        return MockRun(run_id, "completed")
    
    def list_messages(self, thread_id: str, **kwargs):
        class MockMessageList:
            def __init__(self):
                self.data = [MockMessage("mock-message-1", "Mock response from financial AI agent")]
        return MockMessageList()
    
    def list_agents(self):
        return [MockAgent("mock-agent-1", "Mock Agent")]

class MockEvaluations:
    def create_evaluation(self, **kwargs):
        return {"id": "mock-eval-1", "status": "completed", "score": 0.85}
    
    def get_evaluation(self, evaluation_id: str):
        return {"id": evaluation_id, "status": "completed", "score": 0.85}

class MockConnections:
    def list_connections(self):
        return [{"name": "mock-connection", "type": "azure_openai", "status": "connected"}]
    
    def list(self):
        return [
            {
                "name": "mock-azure-openai-connection",
                "type": "azure_openai", 
                "status": "connected",
                "endpoint": "https://mock-openai.openai.azure.com/",
                "resource_id": "/subscriptions/mock/resourceGroups/mock/providers/Microsoft.CognitiveServices/accounts/mock-openai"
            }
        ]

class MockEmbeddings:
    def create(self, input, model):
        class MockResponse:
            def __init__(self):
                self.data = [MockEmbeddingData()]
        return MockResponse()

class MockEmbeddingData:
    def __init__(self):
        import random
        self.embedding = [random.random() for _ in range(1536)]

class AzureServiceManager:
    def __init__(self):
        self.search_client = None
        self.search_index_client = None
        self.form_recognizer_client = None
        self.cosmos_client = None
        self.openai_client = None
        self.ai_foundry_client = None
        self.project_client = None
        self.credential = None
        self.storage_manager = None
        
    async def initialize(self):
        """Initialize all Azure services"""
        try:
            # Force real Azure services - don't use mock services
            logger.info("Initializing real Azure services...")
            
            if settings.AZURE_CLIENT_SECRET and settings.AZURE_TENANT_ID and settings.AZURE_CLIENT_ID:
                self.credential = ClientSecretCredential(
                    tenant_id=settings.AZURE_TENANT_ID,
                    client_id=settings.AZURE_CLIENT_ID,
                    client_secret=settings.AZURE_CLIENT_SECRET
                )
                logger.info("Using Service Principal (SPN) authentication")
            else:
                raise ValueError("SPN authentication required: AZURE_TENANT_ID, AZURE_CLIENT_ID, and AZURE_CLIENT_SECRET must be provided")
            
            search_endpoint = f"https://{settings.AZURE_SEARCH_SERVICE_NAME}.search.windows.net"
            
            # Use API key authentication if available, otherwise use Service Principal
            if settings.AZURE_SEARCH_API_KEY:
                from azure.core.credentials import AzureKeyCredential
                search_credential = AzureKeyCredential(settings.AZURE_SEARCH_API_KEY)
                logger.info("Using API key authentication for Azure Search")
            else:
                search_credential = self.credential
                logger.info("Using Service Principal authentication for Azure Search")
            
            self.search_client = AsyncSearchClient(
                endpoint=search_endpoint,
                index_name=settings.AZURE_SEARCH_INDEX_NAME,
                credential=search_credential
            )
            
            self.search_index_client = SearchIndexClient(
                endpoint=search_endpoint,
                credential=search_credential
            )
            
            self.form_recognizer_client = DocumentAnalysisClient(
                endpoint=settings.AZURE_FORM_RECOGNIZER_ENDPOINT,
                credential=self.credential
            )
            
            self.cosmos_client = CosmosClient(
                url=settings.AZURE_COSMOS_ENDPOINT,
                credential=self.credential
            )
            if hasattr(settings, 'AZURE_AI_PROJECT_ENDPOINT') and settings.AZURE_AI_PROJECT_ENDPOINT:
                self.project_client = AIProjectClient(
                    endpoint=settings.AZURE_AI_PROJECT_ENDPOINT,
                    credential=DefaultAzureCredential()
                    # Using default API version instead of specifying "latest"
                )
                self.ai_foundry_client = self.project_client  # For backward compatibility
                logger.info("Azure AI Foundry project client initialized successfully")
            elif all([
                settings.AZURE_AI_FOUNDRY_PROJECT_NAME,
                settings.AZURE_AI_FOUNDRY_RESOURCE_GROUP,
                settings.AZURE_SUBSCRIPTION_ID
            ]):
                # Azure AI Foundry project endpoint construction is complex and varies by region
                # For now, skip AI Foundry initialization if direct endpoint is not provided
                logger.info("Azure AI Foundry project settings found but direct endpoint not provided. Skipping AI Foundry initialization.")
                logger.info("To use Azure AI Foundry, please provide AZURE_AI_PROJECT_ENDPOINT in .env")
                self.project_client = None
                self.ai_foundry_client = None
            else:
                logger.info("Azure AI Foundry configuration not found, skipping AI Foundry client initialization")
                self.project_client = None
                self.ai_foundry_client = None
            
            self.openai_client = AsyncAzureOpenAI(
                azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
                api_key=settings.AZURE_OPENAI_API_KEY,
                api_version=settings.AZURE_OPENAI_API_VERSION
            )
            
            # Ensure search index exists
            await self.ensure_search_index_exists()
            
            # Initialize Azure Storage Manager - commented out due to Windows event loop issue
            # self.storage_manager = AzureStorageManager()
            # await self.storage_manager.initialize()
            # logger.info("Azure Storage Manager initialized")
            
            logger.info("Azure services initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Azure services: {e}")
            raise
    
    async def _initialize_mock_services(self):
        """Initialize mock services for local development"""
        self.search_client = MockSearchClient()
        self.search_index_client = MockSearchIndexClient()
        self.form_recognizer_client = MockDocumentAnalysisClient()
        self.cosmos_client = MockCosmosClient()
        self.openai_client = MockOpenAIClient()
        self.ai_foundry_client = MockAIFoundryClient()
        self.project_client = MockAIFoundryClient()  # Same mock for project client
        self.credential = None
        
        self.storage_manager = MockStorageManager()
        await self.storage_manager.initialize()
        
        logger.info("Mock Azure services initialized for local development")
    
    async def cleanup(self):
        """Cleanup resources"""
        try:
            # Close all Azure clients that may have internal HTTP sessions
            if hasattr(self, 'cosmos_client') and self.cosmos_client:
                if hasattr(self.cosmos_client, 'close'):
                    self.cosmos_client.close()
                    
            if hasattr(self, 'project_client') and self.project_client:
                if hasattr(self.project_client, 'close'):
                    await self.project_client.close()
                elif hasattr(self.project_client, '_client') and hasattr(self.project_client._client, 'close'):
                    await self.project_client._client.close()
                    
            if hasattr(self, 'ai_foundry_client') and self.ai_foundry_client:
                if hasattr(self.ai_foundry_client, 'close'):
                    await self.ai_foundry_client.close()
                elif hasattr(self.ai_foundry_client, '_client') and hasattr(self.ai_foundry_client._client, 'close'):
                    await self.ai_foundry_client._client.close()
                    
            if hasattr(self, 'openai_client') and self.openai_client:
                if hasattr(self.openai_client, 'close'):
                    await self.openai_client.close()
                    
            if hasattr(self, 'storage_manager') and self.storage_manager:
                if hasattr(self.storage_manager, 'cleanup'):
                    await self.storage_manager.cleanup()
                    
            logger.info("Azure services cleaned up")
        except Exception as e:
            logger.error(f"Error during Azure services cleanup: {e}")
    
    def _validate_azure_credentials(self) -> bool:
        """Validate that all required Azure credentials are present"""
        required_settings = [
            'AZURE_CLIENT_SECRET',
            'AZURE_TENANT_ID', 
            'AZURE_CLIENT_ID',
            'AZURE_SEARCH_SERVICE_NAME',
            'AZURE_OPENAI_ENDPOINT',
            'AZURE_COSMOS_ENDPOINT'        ]
        
        missing_settings = []
        for setting in required_settings:
            if not getattr(settings, setting, None):
                missing_settings.append(setting)
        
        if missing_settings:
            logger.warning(f"Missing Azure credentials: {', '.join(missing_settings)}")
            return False
            
        return True
    
    def get_project_client(self) -> Optional[AIProjectClient]:
        """Get the Azure AI Foundry project client for agent services"""
        return self.project_client
    
    async def create_search_index(self):
        """
        DEPRECATED: Use ensure_search_index_exists() instead.
        This method is kept for backward compatibility but will call ensure_search_index_exists().
        """
        logger.warning("create_search_index() is deprecated. Use ensure_search_index_exists() instead.")
        return await self.ensure_search_index_exists()

    async def ensure_search_index_exists(self) -> bool:
        """Ensure the search index exists, create it if it doesn't"""
        try:
            logger.info(f"Checking if search index '{settings.AZURE_SEARCH_INDEX_NAME}' exists")
            
            # Check if index exists
            try:
                index = self.search_index_client.get_index(settings.AZURE_SEARCH_INDEX_NAME)
                logger.info(f"Search index '{settings.AZURE_SEARCH_INDEX_NAME}' already exists with {len(index.fields)} fields")
                return True
            except Exception as e:
                logger.info(f"Search index '{settings.AZURE_SEARCH_INDEX_NAME}' does not exist, creating it. Error: {e}")
                
            # Create the index
            from azure.search.documents.indexes.models import (
                SearchIndex, SearchField, SearchFieldDataType, SimpleField, 
                SearchableField, VectorSearch, HnswAlgorithmConfiguration,
                VectorSearchProfile, SemanticConfiguration, SemanticPrioritizedFields,
                SemanticField, SemanticSearch
            )
            fields = [
                SimpleField(name="id", type=SearchFieldDataType.String, key=True),
                SearchableField(name="content", type=SearchFieldDataType.String),
                SearchableField(name="title", type=SearchFieldDataType.String),
                SimpleField(name="document_id", type=SearchFieldDataType.String, filterable=True),
                SimpleField(name="source", type=SearchFieldDataType.String, filterable=True),
                SimpleField(name="chunk_id", type=SearchFieldDataType.String, filterable=True),
                SimpleField(name="document_type", type=SearchFieldDataType.String, filterable=True),
                SimpleField(name="company", type=SearchFieldDataType.String, filterable=True),
                SimpleField(name="filing_date", type=SearchFieldDataType.String, filterable=True),
                SimpleField(name="section_type", type=SearchFieldDataType.String, filterable=True),
                SimpleField(name="page_number", type=SearchFieldDataType.Int32, filterable=True),
                SimpleField(name="credibility_score", type=SearchFieldDataType.Double, filterable=True),
                SimpleField(name="processed_at", type=SearchFieldDataType.String, filterable=True),
                SimpleField(name="citation_info", type=SearchFieldDataType.String),                # SEC-specific fields from Edgar tools
                SimpleField(name="ticker", type=SearchFieldDataType.String, filterable=True),
                SimpleField(name="cik", type=SearchFieldDataType.String, filterable=True),
                SimpleField(name="industry", type=SearchFieldDataType.String, filterable=True),
                SimpleField(name="sic", type=SearchFieldDataType.String, filterable=True),
                SimpleField(name="entity_type", type=SearchFieldDataType.String, filterable=True),
                SimpleField(name="form_type", type=SearchFieldDataType.String, filterable=True),
                SimpleField(name="accession_number", type=SearchFieldDataType.String, filterable=True),
                SimpleField(name="period_end_date", type=SearchFieldDataType.String, filterable=True),
                SimpleField(name="chunk_index", type=SearchFieldDataType.Int32, filterable=True),
                SimpleField(name="content_type", type=SearchFieldDataType.String, filterable=True),
                SimpleField(name="chunk_method", type=SearchFieldDataType.String, filterable=True),
                SimpleField(name="file_size", type=SearchFieldDataType.Int64, filterable=True),
                SimpleField(name="document_url", type=SearchFieldDataType.String),
                SearchField(
                    name="content_vector", 
                    type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                    searchable=True,
                    vector_search_dimensions=1536,
                    vector_search_profile_name="default-vector-profile"
                )
            ]
            
            # Configure vector search
            vector_search = VectorSearch(
                algorithms=[
                    HnswAlgorithmConfiguration(
                        name="default-hnsw",
                        parameters={
                            "m": 4,
                            "efConstruction": 400,
                            "efSearch": 500,
                            "metric": "cosine"
                        }
                    )
                ],
                profiles=[
                    VectorSearchProfile(
                        name="default-vector-profile",
                        algorithm_configuration_name="default-hnsw"
                    )
                ]
            )
              # Configure semantic search with SEC-specific fields
            semantic_config = SemanticConfiguration(
                name="default-semantic-config",
                prioritized_fields=SemanticPrioritizedFields(
                    title_field=SemanticField(field_name="title"),
                    content_fields=[
                        SemanticField(field_name="content"),
                        SemanticField(field_name="section_type")
                    ],                    keywords_fields=[
                        SemanticField(field_name="ticker"),
                        SemanticField(field_name="company"),
                        SemanticField(field_name="form_type"),
                        SemanticField(field_name="document_type"),
                        SemanticField(field_name="industry"),
                        SemanticField(field_name="entity_type")
                    ]
                )
            )
            
            semantic_search = SemanticSearch(configurations=[semantic_config])
              # Create the index
            index = SearchIndex(
                name=settings.AZURE_SEARCH_INDEX_NAME,
                fields=fields,
                vector_search=vector_search,
                semantic_search=semantic_search
            )
            result = self.search_index_client.create_index(index)
            logger.info(f"Successfully created search index '{settings.AZURE_SEARCH_INDEX_NAME}'")
            return True
        except Exception as e:
            logger.error(f"Failed to ensure search index exists: {e}")
            return False

    async def get_embedding(self, text: str, model: str = None, token_tracker=None, tracking_id: str = None) -> List[float]:
        """Get embedding for text using Azure OpenAI async client"""
        try:
            # Use deployment name from settings, not model name
            deployment_name = model or settings.AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME
            
            # Add timing and thread logging for debugging
            import threading
            import time
            thread_id = threading.get_ident()
            start_time = time.time()
            logger.debug(f"🔤 [Thread-{thread_id}] Starting embedding request for {len(text)} chars using {deployment_name}")
            
            # Use async client directly - no need for run_in_executor
            response = await self.openai_client.embeddings.create(
                input=text,
                model=deployment_name
            )
            
            elapsed_time = time.time() - start_time
            logger.debug(f"✅ [Thread-{thread_id}] Embedding completed in {elapsed_time:.2f}s")
            
            # Track token usage for embedding if tracker is provided
            if token_tracker and tracking_id and hasattr(response, 'usage'):
                try:
                    await token_tracker.update_usage(
                        tracking_id=tracking_id,
                        model_name=deployment_name,
                        deployment_name=deployment_name,
                        prompt_tokens=response.usage.prompt_tokens,
                        completion_tokens=0,  # Embeddings don't have completion tokens
                        total_tokens=response.usage.total_tokens,
                        input_text=text[:200] + "..." if len(text) > 200 else text,
                        output_text=f"Generated embedding vector of dimension {len(response.data[0].embedding)}"
                    )
                    logger.info(f"Token usage tracked for embedding: {response.usage.total_tokens} tokens")
                except Exception as tracking_error:
                    logger.error(f"Failed to track embedding token usage: {tracking_error}")
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Failed to get embedding: {e}")
            raise

    async def hybrid_search(self, query: str, top_k: int = 10, filters: str = None, min_score: float = 0.0, token_tracker=None, tracking_id: str = None) -> List[Dict]:
        """Perform hybrid search (vector + keyword) on the knowledge base"""
        try:
            logger.info(f"🔍 Hybrid search with token tracking: tracker={token_tracker is not None}, tracking_id={tracking_id}")
            # Add timing and thread logging for debugging
            import threading
            thread_id = threading.get_ident()
            start_time = time.time()
            logger.debug(f"🔍 [Thread-{thread_id}] Starting hybrid search for query: '{query[:50]}...' (top_k={top_k})")
            
            query_vector = await self.get_embedding(query, token_tracker=token_tracker, tracking_id=tracking_id)
            vector_query = VectorizedQuery(
                vector=query_vector,
                k_nearest_neighbors=top_k,
                fields="content_vector"
            )
            
            # Use async search client directly - no need for run_in_executor
            search_start = time.time()
            results = await self.search_client.search(
                search_text=query,
                vector_queries=[vector_query],
                select=["id", "content", "title", "document_id", "source", "chunk_id", 
                       "document_type", "company", "filing_date", "section_type", 
                       "page_number", "credibility_score", "processed_at", "citation_info",
                       "ticker", "cik", "form_type", "accession_number", "industry", 
                       "document_url", "sic", "entity_type", "period_end_date", 
                       "chunk_index", "content_type", "chunk_method", "file_size"],
                filter=filters,
                top=top_k,
                query_type="semantic",
                semantic_configuration_name="default-semantic-config"
            )
            search_time = time.time() - search_start
              # Filter results by minimum score if specified - use async iteration
            filtered_results = []
            async for result in results:
                result_dict = dict(result)
                score = getattr(result, '@search.score', 0.0)
                if score >= min_score:
                    result_dict['search_score'] = score
                    filtered_results.append(result_dict)
            
            elapsed_time = time.time() - start_time
            logger.debug(f"✅ [Thread-{thread_id}] Hybrid search completed in {elapsed_time:.2f}s (embedding: {search_start - start_time:.2f}s, search: {search_time:.2f}s, found: {len(filtered_results)} results)")
            
            return filtered_results            
        except Exception as e:
            logger.error(f"Hybrid search failed: {e}")
            raise    
            
    async def add_documents_to_index(self, documents: List[Dict]) -> bool:
        """Add or update documents in the search index"""
        try:
            logger.info(f"Starting add_documents_to_index with {len(documents)} documents")
            
            # with observability.trace_operation("azure_add_documents_to_index") as span:
            #     span.set_attribute("documents_count", len(documents))
            
            validated_documents = []
            for doc in documents:
                if self._validate_document_schema(doc):
                    validated_documents.append(doc)
                else:
                    logger.warning(f"Skipping invalid document: {doc.get('id', 'unknown')}")
            if not validated_documents:
                logger.error("No valid documents to upload after validation")
                return False
                
            logger.info(f"Validated {len(validated_documents)} documents, uploading to search index")
            result = self.search_client.upload_documents(validated_documents)
            logger.info(f"Search client upload_documents result: {result}")
            
            #     span.set_attribute("uploaded_count", len(validated_documents))
            #     span.set_attribute("success", True)            logger.info(f"Successfully uploaded {len(validated_documents)} documents to search index")
            # observability.track_kb_update("search_index", len(validated_documents), 0)  # Method not available
            
            return True
                
        except Exception as e:
            logger.error(f"Failed to add documents to index: {e}")
            logger.error(f"Exception type: {type(e).__name__}")
            logger.error(f"Exception details: {str(e)}")
            observability.record_error("azure_add_documents_error", str(e))
            return False
    
    def _validate_document_schema(self, document: Dict) -> bool:
        """Validate document schema before uploading to search index"""
        required_fields = ['id', 'content']
        for field in required_fields:
            if field not in document or not document[field]:
                logger.warning(f"Document missing required field: {field}")
                return False
        
        if len(document['content']) > 1000000:  # 1MB limit
            logger.warning(f"Document content too large: {len(document['content'])} characters")
            return False
            
        return True
        
    async def analyze_document(self, document_content: bytes, content_type: str, filename: str = None) -> Dict:
        """Analyze document using Azure Document Intelligence with enhanced financial document processing"""
        try:
            # with observability.trace_operation("azure_analyze_document") as span:
            #     span.set_attribute("content_type", content_type)
            #     span.set_attribute("content_size", len(document_content))
            #     span.set_attribute("filename", filename or "unknown")
            
            model_id = self._select_document_model(content_type, filename)
            #     span.set_attribute("model_id", model_id)
            
            logger.info(f"Analyzing document with model {model_id}, size: {len(document_content)} bytes")
            
            poller = self.form_recognizer_client.begin_analyze_document(
                model_id=model_id,
                document=document_content
            )
            result = poller.result()
            
            extracted_content = {
                "content": result.content,
                "tables": [],
                "key_value_pairs": {},
                "pages": len(result.pages) if result.pages else 0,
                "financial_sections": [],
                "metadata": {
                    "model_used": model_id,
                    "confidence_scores": {},
                    "processing_time": None
                }
            }
            
            if result.tables:
                for i, table in enumerate(result.tables):
                    table_data = {
                        "table_id": i,
                        "cells": [],
                        "financial_context": self._identify_financial_table_context(table)
                    }
                    
                    for cell in table.cells:
                        table_data["cells"].append({
                            "content": cell.content,
                            "row_index": cell.row_index,
                            "column_index": cell.column_index,
                            "confidence": getattr(cell, 'confidence', 0.0)
                        })
                    
                    extracted_content["tables"].append(table_data)
            
            if result.key_value_pairs:
                for kv_pair in result.key_value_pairs:
                    if kv_pair.key and kv_pair.value:
                        key_content = kv_pair.key.content
                        value_content = kv_pair.value.content
                        
                        extracted_content["key_value_pairs"][key_content] = {
                            "value": value_content,
                            "confidence": getattr(kv_pair, 'confidence', 0.0),
                            "financial_relevance": self._score_financial_relevance(key_content, value_content)
                        }
            
            extracted_content["financial_sections"] = self._identify_financial_sections(result.content, filename)
            
            #     span.set_attribute("pages_processed", extracted_content["pages"])
            #     span.set_attribute("tables_found", len(extracted_content["tables"]))
            #     span.set_attribute("kv_pairs_found", len(extracted_content["key_value_pairs"]))
            #     span.set_attribute("success", True)
            
            logger.info(f"Document analysis completed: {extracted_content['pages']} pages, {len(extracted_content['tables'])} tables")
            
            return extracted_content
                
        except Exception as e:
            logger.error(f"Document analysis failed: {e}")
            observability.record_error("azure_document_analysis_error", str(e))
            raise
    
    def _select_document_model(self, content_type: str, filename: str = None) -> str:
        """Select appropriate Document Intelligence model based on content type and filename"""
        if filename:
            filename_lower = filename.lower()
            if any(term in filename_lower for term in ['10-k', '10k', '10-q', '10q', 'annual', 'quarterly']):
                return "prebuilt-layout"  # Best for structured financial documents
        
        if content_type == "application/pdf":
            return "prebuilt-layout"
        elif content_type in ["application/vnd.openxmlformats-officedocument.wordprocessingml.document", "application/msword"]:
            return "prebuilt-document"
        else:
            return "prebuilt-document"
    
    def _identify_financial_table_context(self, table) -> str:
        """Identify the financial context of a table based on its content"""
        sample_content = ""
        cell_count = 0
        
        for cell in table.cells:
            if cell_count >= 10:  # Sample first 10 cells
                break
            sample_content += cell.content.lower() + " "
            cell_count += 1
        
        financial_keywords = {
            "income_statement": ["revenue", "income", "expense", "profit", "loss", "earnings"],
            "balance_sheet": ["assets", "liabilities", "equity", "cash", "inventory", "debt"],
            "cash_flow": ["cash flow", "operating", "investing", "financing", "cash"],
            "financial_ratios": ["ratio", "margin", "return", "percentage", "%"]
        }
        
        for context, keywords in financial_keywords.items():
            if any(keyword in sample_content for keyword in keywords):
                return context
        
        return "general"
    
    def _score_financial_relevance(self, key: str, value: str) -> float:
        """Score the financial relevance of a key-value pair"""
        financial_terms = [
            "revenue", "income", "profit", "loss", "assets", "liabilities", "equity",
            "cash", "debt", "earnings", "dividend", "share", "stock", "market",
            "financial", "fiscal", "quarter", "annual", "year", "period"
        ]
        
        combined_text = (key + " " + value).lower()
        matches = sum(1 for term in financial_terms if term in combined_text)
        
        return min(matches / len(financial_terms), 1.0)
    
    def _identify_financial_sections(self, content: str, filename: str = None) -> List[Dict]:
        """Identify financial document sections in the content"""
        sections = []
        content_lower = content.lower()
        
        section_patterns = {
            "executive_summary": ["executive summary", "management discussion", "md&a"],
            "financial_statements": ["financial statements", "consolidated statements"],
            "income_statement": ["income statement", "statement of operations", "profit and loss"],
            "balance_sheet": ["balance sheet", "statement of financial position"],
            "cash_flow": ["cash flow statement", "statement of cash flows"],
            "notes": ["notes to financial statements", "footnotes"],
            "risk_factors": ["risk factors", "risks and uncertainties"]
        }
        
        for section_type, patterns in section_patterns.items():
            for pattern in patterns:
                if pattern in content_lower:
                    position = content_lower.find(pattern)
                    sections.append({
                        "type": section_type,
                        "pattern": pattern,
                        "position": position,
                        "confidence": 0.8  # Base confidence for pattern matching
                    })
                    break
        
        return sections

    async def save_session_history(self, session_id: str, message: Dict) -> bool:
        """Save chat session history to CosmosDB"""
        try:
            database = self.cosmos_client.get_database_client(settings.AZURE_COSMOS_DATABASE_NAME)
            container = database.get_container_client(settings.AZURE_COSMOS_CONTAINER_NAME)
            
            try:
                session_doc = container.read_item(item=session_id, partition_key=session_id)
            except:
                session_doc = {
                    "id": session_id,
                    "messages": [],
                    "created_at": message.get("timestamp"),
                    "updated_at": message.get("timestamp")
                }
            session_doc["messages"].append(message)
            session_doc["updated_at"] = message.get("timestamp")
            
            container.upsert_item(session_doc)
            logger.info(f"Session {session_id} updated in CosmosDB")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save session history: {e}")
            return False

    async def get_session_history(self, session_id: str) -> List[Dict]:
        """Retrieve chat session history from CosmosDB"""
        try:
            database = self.cosmos_client.get_database_client(settings.AZURE_COSMOS_DATABASE_NAME)
            container = database.get_container_client(settings.AZURE_COSMOS_CONTAINER_NAME)
            
            try:
                session_doc = container.read_item(item=session_id, partition_key=session_id)
                return session_doc.get("messages", [])
            except Exception as e:
                # Session doesn't exist yet, return empty history
                if "NotFound" in str(e) or "does not exist" in str(e):
                    logger.info(f"Session {session_id} not found, returning empty history")
                    return []
                else:
                    # Some other error occurred
                    logger.error(f"Failed to retrieve session history: {e}")
                    return []
        except Exception as e:
            logger.error(f"Failed to retrieve session history: {e}")
            return []

    async def get_available_models(self) -> List[Dict[str, Any]]:
        """Retrieve available model deployments from Azure AI Foundry project"""
        try:
            # with observability.trace_operation("azure_get_available_models") as span:
            if not self.project_client:
                logger.warning("Project client not initialized, returning mock models")
                # span.set_attribute("using_mock", True)
                return self._get_mock_models()
            
            connections = await self._get_project_connections_internal()
            models = []
            
            for connection in connections:
                if connection.get("connection_type") == ConnectionType.AZURE_OPEN_AI or connection.get("type") == "azure_openai":
                    try:
                        connection_models = await self._get_models_from_connection(connection)
                        models.extend(connection_models)
                    except Exception as e:
                        logger.error(f"Failed to get models from connection {connection.get('name')}: {e}")
            
            if not models:
                logger.warning("No models found from project connections, using direct Azure OpenAI configuration")
                # Fallback to using direct Azure OpenAI configuration
                models = await self._get_models_from_direct_config()
            
            # Remove duplicates based on deployment name/id
            unique_models = []
            seen_ids = set()
            for model in models:
                model_id = model.get('id') or model.get('deployment_name')
                if model_id and model_id not in seen_ids:
                    unique_models.append(model)
                    seen_ids.add(model_id)
            
            logger.info(f"Returning {len(unique_models)} unique models (removed {len(models) - len(unique_models)} duplicates)")
            
            # span.set_attribute("models_count", len(unique_models))
            # span.set_attribute("success", True)
            return unique_models
                
        except Exception as e:
            logger.error(f"Failed to retrieve available models: {e}")
            observability.record_error("azure_get_models_error", str(e))
            return self._get_mock_models()
    
    def _get_mock_models(self) -> List[Dict[str, Any]]:
        """Get mock models for development/fallback"""
        return [
            {
                "id": "gpt-4",
                "name": "GPT-4",
                "type": "chat",
                "version": "0613",
                "status": "active",
                "provider": "Azure OpenAI",
                "capabilities": ["chat", "completion"]
            },
            {
                "id": "gpt-35-turbo",
                "name": "GPT-3.5 Turbo",
                "type": "chat",
                "version": "0613",
                "status": "active",
                "provider": "Azure OpenAI",
                "capabilities": ["chat", "completion"]
            },
            {
                "id": "text-embedding-ada-002",
                "name": "Text Embedding Ada 002",
                "type": "embedding",
                "version": "2",
                "status": "active",
                "provider": "Azure OpenAI",
                "capabilities": ["embeddings"],
                "dimensions": 1536
            }
        ]
    
    async def _get_models_from_connection(self, connection: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get model deployments from a specific Azure OpenAI connection"""
        try:
            connection_name = connection.get("name", "unknown")
            endpoint = connection.get("endpoint", "")
            
            return [
                {
                    "id": f"gpt-4-{connection_name}",
                    "name": "GPT-4",
                    "type": "chat",
                    "version": "0613",
                    "status": "active",
                    "provider": "Azure OpenAI",
                    "connection": connection_name,
                    "endpoint": endpoint,
                    "capabilities": ["chat", "completion"]
                },
                {
                    "id": f"gpt-35-turbo-{connection_name}",
                    "name": "GPT-3.5 Turbo",
                    "type": "chat",
                    "version": "0613",
                    "status": "active",
                    "provider": "Azure OpenAI",
                    "connection": connection_name,
                    "endpoint": endpoint,
                    "capabilities": ["chat", "completion"]
                },
                {
                    "id": f"text-embedding-ada-002-{connection_name}",
                    "name": "Text Embedding Ada 002",
                    "type": "embedding",
                    "version": "2",
                    "status": "active",
                    "provider": "Azure OpenAI",
                    "connection": connection_name,
                    "endpoint": endpoint,
                    "capabilities": ["embeddings"],
                    "dimensions": 1536
                }
            ]
        except Exception as e:
            logger.error(f"Error getting models from connection {connection.get('name')}: {e}")
            return []

    async def get_project_connections(self) -> List[Dict[str, Any]]:
        """Retrieve project connections from Azure AI Foundry"""
        try:
            # with observability.trace_operation("azure_get_project_connections") as span:
                connections = await self._get_project_connections_internal()
            # span.set_attribute("connections_count", len(connections))
            # span.set_attribute("success", True)
                return connections
        except Exception as e:
            logger.error(f"Failed to retrieve project connections: {e}")
            observability.record_error("azure_get_connections_error", str(e))
            return []
    
    async def _get_project_connections_internal(self) -> List[Dict[str, Any]]:
        """Internal method to get project connections"""
        try:
            if not self.project_client:
                logger.warning("Project client not initialized, returning mock connections")
                return [
                    {
                        "name": "mock-azure-openai",
                        "type": "azure_openai",
                        "connection_type": ConnectionType.AZURE_OPEN_AI,
                        "status": "connected",
                        "endpoint": "https://mock-openai.openai.azure.com/"
                    }
                ]
            
            # Use the project client to list connections
            connections = self.project_client.connections.list()
            
            formatted_connections = []
            for conn in connections:
                formatted_connections.append({
                    "name": conn.name if hasattr(conn, 'name') else conn.get("name"),
                    "type": conn.connection_type.value if hasattr(conn, 'connection_type') else conn.get("type"),
                    "connection_type": conn.connection_type if hasattr(conn, 'connection_type') else conn.get("connection_type"),
                    "status": "connected",  # Assume connected if listed
                    "endpoint": conn.target if hasattr(conn, 'target') else conn.get("endpoint"),
                    "resource_id": conn.id if hasattr(conn, 'id') else conn.get("resource_id")
                })
            
            return formatted_connections
            
        except Exception as e:
            logger.error(f"Error getting project connections: {e}")
            return [
                {
                    "name": "fallback-azure-openai",
                    "type": "azure_openai",
                    "connection_type": ConnectionType.AZURE_OPEN_AI,                    "status": "connected",
                    "endpoint": "https://fallback-openai.openai.azure.com/"
                }
            ]

    async def get_project_info(self) -> Dict[str, Any]:
        """Get Azure AI Foundry project information"""
        try:
            # with observability.trace_operation("azure_get_project_info") as span:
            if not self.project_client:
                # span.set_attribute("using_mock", True)
                return {
                    "project_name": "mock-project",
                    "resource_group": "mock-rg",
                    "subscription_id": "mock-subscription",
                    "endpoint": "https://mock-project.cognitiveservices.azure.com/",
                    "status": "mock",
                    "client_type": "mock"
                }
            
            project_info = {
                "project_name": getattr(settings, 'AZURE_AI_FOUNDRY_PROJECT_NAME', 'unknown'),
                "resource_group": getattr(settings, 'AZURE_AI_FOUNDRY_RESOURCE_GROUP', 'unknown'),
                "subscription_id": getattr(settings, 'AZURE_SUBSCRIPTION_ID', 'unknown'),
                "endpoint": getattr(settings, 'AZURE_AI_PROJECT_ENDPOINT', 'unknown'),
                "status": "active",
                "client_type": "project_based"
            }
            
            try:
                connections = await self.get_project_connections()
                models = await self.get_available_models()
                project_info.update({
                    "connections_count": len(connections),
                    "models_count": len(models),
                    "connections": [conn.get("name") for conn in connections[:5]]  # First 5 connection names
                })
            except Exception as e:
                logger.warning(f"Could not get connection/model counts: {e}")
                project_info.update({
                    "connections_count": 0,
                    "models_count": 0
                })
            
            # span.set_attribute("project_name", project_info["project_name"])
            # span.set_attribute("connections_count", project_info.get("connections_count", 0))
            # span.set_attribute("success", True)
            
            return project_info
            
        except Exception as e:
            logger.error(f"Failed to get project info: {e}")
            observability.record_error("azure_get_project_info_error", str(e))
            return {"error": str(e), "status": "error"}

    async def _get_models_from_direct_config(self) -> List[Dict[str, Any]]:
        """Get models from Azure OpenAI Management API or fallback to direct configuration"""
        models = []
        
        try:
            # First try to get models from Azure OpenAI Management API
            if (settings.AZURE_OPENAI_ENDPOINT and 
                settings.AZURE_SUBSCRIPTION_ID and 
                settings.AZURE_AI_FOUNDRY_RESOURCE_GROUP):
                
                logger.info("Attempting to fetch models from Azure OpenAI Management API")
                
                # Use mock service in development, real service in production
                use_mock = os.getenv("MOCK_AZURE_SERVICES", "false").lower() == "true"
                service_class = MockAzureOpenAIDeploymentService if use_mock else AzureOpenAIDeploymentService
                
                async with service_class(settings) as deployment_service:
                    deployments = await deployment_service.get_deployments()
                    
                    for deployment in deployments:
                        models.append({
                            "id": deployment.deployment_name,
                            "name": f"{deployment.model_name} ({deployment.deployment_name})",
                            "deployment_name": deployment.deployment_name,
                            "model_name": deployment.model_name,
                            "model_version": deployment.model_version,
                            "type": deployment.model_type,
                            "status": "active" if deployment.provisioning_state == "Succeeded" else "inactive",
                            "provider": "Azure OpenAI (Management API)",
                            "capabilities": ["chat", "completion"] if deployment.model_type == "chat" else ["embedding"],
                            "endpoint": settings.AZURE_OPENAI_ENDPOINT,
                            "sku": deployment.sku_name,
                            "capacity": deployment.capacity,
                            "provisioning_state": deployment.provisioning_state
                        })
                    
                    if models:
                        logger.info(f"Successfully fetched {len(models)} models from Azure OpenAI Management API")
                        return models
                    else:
                        logger.warning("No deployments found from Management API, falling back to direct config")
        
        except Exception as e:
            logger.warning(f"Failed to fetch models from Management API: {e}, falling back to direct config")
        
        # Fallback to direct configuration if Management API fails
        if settings.AZURE_OPENAI_ENDPOINT and settings.AZURE_OPENAI_API_KEY:
            logger.info("Using direct Azure OpenAI configuration as fallback")
            
            # Add the configured chat model
            if settings.AZURE_OPENAI_DEPLOYMENT_NAME:
                models.append({
                    "id": settings.AZURE_OPENAI_DEPLOYMENT_NAME,
                    "name": settings.AZURE_OPENAI_DEPLOYMENT_NAME,
                    "deployment_name": settings.AZURE_OPENAI_DEPLOYMENT_NAME,
                    "model_name": "Unknown",  # We don't know the actual model name in fallback
                    "type": "chat",
                    "status": "active",
                    "provider": "Azure OpenAI (Direct Config Fallback)",
                    "capabilities": ["chat", "completion"],
                    "endpoint": settings.AZURE_OPENAI_ENDPOINT
                })
            
            # Add the configured embedding model
            if settings.AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME:
                models.append({
                    "id": settings.AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME,
                    "name": settings.AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME,
                    "deployment_name": settings.AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME,
                    "model_name": "Unknown",  # We don't know the actual model name in fallback
                    "type": "embedding",
                    "status": "active",
                    "provider": "Azure OpenAI (Direct Config Fallback)",
                    "capabilities": ["embedding"],
                    "endpoint": settings.AZURE_OPENAI_ENDPOINT
                })
        
        if not models:
            logger.warning("No Azure OpenAI configuration found, returning mock models")
            return self._get_mock_models()
            
        logger.info(f"Found {len(models)} models from direct Azure OpenAI configuration")
        return models

    async def recreate_search_index(self, force: bool = False) -> bool:
        """
        Force recreate the search index with the latest schema.
        This will delete the existing index and create a new one.
        Use with caution as this will delete all existing data.
        
        Args:
            force: If True, will delete existing index without checking
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if not force:
                logger.warning("recreate_search_index() will DELETE all existing data. Call with force=True to proceed.")
                return False
                
            logger.info(f"Force recreating search index '{settings.AZURE_SEARCH_INDEX_NAME}'")
            
            # Delete existing index if it exists
            try:
                self.search_index_client.delete_index(settings.AZURE_SEARCH_INDEX_NAME)
                logger.info(f"Deleted existing index '{settings.AZURE_SEARCH_INDEX_NAME}'")
            except Exception as e:
                logger.info(f"No existing index to delete: {e}")
            
            # Create fresh index using ensure_search_index_exists
            return await self.ensure_search_index_exists()
            
        except Exception as e:
            logger.error(f"Failed to recreate search index: {e}")
            return False

    async def upload_document_to_storage(self, content: bytes, filename: str, document_id: str) -> str:
        """Upload document to Azure Storage and return the URL"""
        try:
            logger.info(f"Uploading document {document_id} ({filename}) to Azure Storage...")
              # Use the existing storage manager
            if hasattr(self, 'storage_manager') and self.storage_manager:
                blob_name = f"{document_id}/{filename}"
                storage_result = await self.storage_manager.upload_document(
                    file_content=content,
                    filename=blob_name,
                    content_type="application/pdf"  # Default to PDF, could be made dynamic
                )
                storage_url = storage_result.get('url', storage_result.get('blob_url'))
                logger.info(f"Document uploaded to storage successfully: {storage_url}")
                return storage_url
            else:
                logger.warning("Storage manager not available, skipping storage upload")
                return None
                
        except Exception as e:
            logger.error(f"Failed to upload document to storage: {e}")
            raise

    async def check_document_exists(self, accession_number: str) -> bool:
        """Check if a document with the given accession number already exists in the search index"""
        try:
            logger.info(f"Checking if document exists in index: {accession_number}")
            
            # Search for documents with the specific accession number
            results = self.search_client.search(
                search_text="*",
                filter=f"accession_number eq '{accession_number}'",
                select=["id", "accession_number"],
                top=1
            )
            
            # Convert results to list to check if any documents exist
            documents = list(results)
            exists = len(documents) > 0
            
            if exists:
                logger.info(f"Document with accession number {accession_number} already exists in index")
            else:
                logger.info(f"Document with accession number {accession_number} not found in index")
                
            return exists
            
        except Exception as e:
            logger.error(f"Error checking if document exists: {e}")
            # In case of error, assume document doesn't exist to allow processing
            return False
