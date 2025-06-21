from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.models import VectorizedQuery
from azure.ai.formrecognizer import DocumentAnalysisClient
from azure.cosmos import CosmosClient
from azure.identity import DefaultAzureCredential, ClientSecretCredential
from openai import AzureOpenAI
from azure.ai.projects import AIProjectClient
from azure.ai.projects.models import ConnectionType
import asyncio
import logging
import os
from typing import List, Dict, Any, Optional
from app.core.config import settings
from app.core import observability

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
        
    async def initialize(self):
        """Initialize all Azure services"""
        try:
            if settings.mock_azure_services or not all([
                settings.AZURE_CLIENT_SECRET, 
                settings.AZURE_TENANT_ID, 
                settings.AZURE_CLIENT_ID,
                settings.AZURE_SEARCH_SERVICE_NAME,
                settings.AZURE_OPENAI_ENDPOINT,
                settings.COSMOS_DB_ENDPOINT
            ]):
                logger.info("Using mock Azure services for local development")
                self._initialize_mock_services()
                return
            
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
            self.search_client = SearchClient(
                endpoint=search_endpoint,
                index_name=settings.AZURE_SEARCH_INDEX_NAME,
                credential=self.credential
            )
            
            self.search_index_client = SearchIndexClient(
                endpoint=search_endpoint,
                credential=self.credential
            )
            
            self.form_recognizer_client = DocumentAnalysisClient(
                endpoint=settings.AZURE_FORM_RECOGNIZER_ENDPOINT,
                credential=self.credential
            )
            
            self.cosmos_client = CosmosClient(
                url=settings.COSMOS_DB_ENDPOINT,
                credential=self.credential
            )
            
            if hasattr(settings, 'AZURE_AI_PROJECT_ENDPOINT') and settings.AZURE_AI_PROJECT_ENDPOINT:
                self.project_client = AIProjectClient(
                    endpoint=settings.AZURE_AI_PROJECT_ENDPOINT,
                    credential=self.credential,
                    api_version="2024-07-01-preview"
                )
                self.ai_foundry_client = self.project_client  # For backward compatibility
                logger.info("Azure AI Foundry project client initialized successfully")
            elif all([
                settings.AZURE_AI_FOUNDRY_PROJECT_NAME,
                settings.AZURE_AI_FOUNDRY_RESOURCE_GROUP,
                settings.AZURE_SUBSCRIPTION_ID
            ]):
                self.project_client = AIProjectClient(
                    credential=self.credential,
                    subscription_id=settings.AZURE_SUBSCRIPTION_ID,
                    resource_group_name=settings.AZURE_AI_FOUNDRY_RESOURCE_GROUP,
                    project_name=settings.AZURE_AI_FOUNDRY_PROJECT_NAME
                )
                self.ai_foundry_client = self.project_client  # For backward compatibility
                logger.info("Azure AI Foundry project client initialized via subscription")
            
            self.openai_client = AzureOpenAI(
                azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
                api_key=settings.AZURE_OPENAI_API_KEY,
                api_version=settings.AZURE_OPENAI_API_VERSION
            )
            
            logger.info("Azure services initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Azure services: {e}")
            raise
    
    def _initialize_mock_services(self):
        """Initialize mock services for local development"""
        self.search_client = MockSearchClient()
        self.search_index_client = MockSearchIndexClient()
        self.form_recognizer_client = MockDocumentAnalysisClient()
        self.cosmos_client = MockCosmosClient()
        self.openai_client = MockOpenAIClient()
        self.ai_foundry_client = MockAIFoundryClient()
        self.project_client = MockAIFoundryClient()  # Same mock for project client
        self.credential = None
        logger.info("Mock Azure services initialized for local development")
    
    async def cleanup(self):
        """Cleanup resources"""
        if self.cosmos_client:
            self.cosmos_client.close()
        logger.info("Azure services cleaned up")
    
    def get_project_client(self) -> Optional[AIProjectClient]:
        """Get the Azure AI Foundry project client for agent services"""
        return self.project_client

    async def create_search_index(self):
        """Create the search index for financial documents"""
        from azure.search.documents.indexes.models import (
            SearchIndex, SearchField, SearchFieldDataType, SimpleField,
            SearchableField, VectorSearch, HnswAlgorithmConfiguration,
            VectorSearchProfile, SemanticConfiguration, SemanticSearch,
            SemanticPrioritizedFields, SemanticField
        )
        
        fields = [
            SimpleField(name="id", type=SearchFieldDataType.String, key=True),
            SearchableField(name="content", type=SearchFieldDataType.String),
            SearchableField(name="title", type=SearchFieldDataType.String),
            SimpleField(name="document_type", type=SearchFieldDataType.String, filterable=True),
            SimpleField(name="company", type=SearchFieldDataType.String, filterable=True),
            SimpleField(name="filing_date", type=SearchFieldDataType.DateTimeOffset, filterable=True),
            SimpleField(name="chunk_index", type=SearchFieldDataType.Int32),
            SimpleField(name="source_url", type=SearchFieldDataType.String),
            SimpleField(name="credibility_score", type=SearchFieldDataType.Double, filterable=True),
            SearchField(
                name="content_vector",
                type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                searchable=True,
                vector_search_dimensions=1536,
                vector_search_profile_name="default-vector-profile"
            )
        ]
        
        vector_search = VectorSearch(
            algorithms=[
                HnswAlgorithmConfiguration(name="default-hnsw-algorithm")
            ],
            profiles=[
                VectorSearchProfile(
                    name="default-vector-profile",
                    algorithm_configuration_name="default-hnsw-algorithm"
                )
            ]
        )
        
        semantic_config = SemanticConfiguration(
            name="default-semantic-config",
            prioritized_fields=SemanticPrioritizedFields(
                title_field=SemanticField(field_name="title"),
                content_fields=[SemanticField(field_name="content")]
            )
        )
        
        semantic_search = SemanticSearch(configurations=[semantic_config])
        
        index = SearchIndex(
            name=settings.AZURE_SEARCH_INDEX_NAME,
            fields=fields,
            vector_search=vector_search,
            semantic_search=semantic_search
        )
        
        try:
            result = self.search_index_client.create_or_update_index(index)
            logger.info(f"Search index '{settings.AZURE_SEARCH_INDEX_NAME}' created/updated successfully")
            return result
        except Exception as e:
            logger.error(f"Failed to create search index: {e}")
            raise

    async def get_embedding(self, text: str, model: str = "text-embedding-ada-002") -> List[float]:
        """Get embedding for text using Azure OpenAI"""
        try:
            response = self.openai_client.embeddings.create(
                input=text,
                model=model
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Failed to get embedding: {e}")
            raise

    async def hybrid_search(self, query: str, top_k: int = 10, filters: str = None) -> List[Dict]:
        """Perform hybrid search (vector + keyword) on the knowledge base"""
        try:
            query_vector = await self.get_embedding(query)
            
            vector_query = VectorizedQuery(
                vector=query_vector,
                k_nearest_neighbors=top_k,
                fields="content_vector"
            )
            
            results = self.search_client.search(
                search_text=query,
                vector_queries=[vector_query],
                select=["id", "content", "title", "document_type", "company", 
                       "filing_date", "source_url", "credibility_score"],
                filter=filters,
                top=top_k,
                query_type="semantic",
                semantic_configuration_name="default-semantic-config"
            )
            
            return [dict(result) for result in results]
            
        except Exception as e:
            logger.error(f"Hybrid search failed: {e}")
            raise

    async def add_documents_to_index(self, documents: List[Dict]) -> bool:
        """Add or update documents in the search index"""
        try:
            result = self.search_client.upload_documents(documents)
            logger.info(f"Uploaded {len(documents)} documents to search index")
            return True
        except Exception as e:
            logger.error(f"Failed to add documents to index: {e}")
            return False

    async def analyze_document(self, document_content: bytes, content_type: str) -> Dict:
        """Analyze document using Azure Document Intelligence"""
        try:
            if content_type == "application/pdf":
                model_id = "prebuilt-layout"
            else:
                model_id = "prebuilt-document"
                
            poller = self.form_recognizer_client.begin_analyze_document(
                model_id=model_id,
                document=document_content
            )
            result = poller.result()
            
            extracted_content = {
                "content": result.content,
                "tables": [],
                "key_value_pairs": {},
                "pages": len(result.pages) if result.pages else 0
            }
            
            if result.tables:
                for table in result.tables:
                    table_data = []
                    for cell in table.cells:
                        table_data.append({
                            "content": cell.content,
                            "row_index": cell.row_index,
                            "column_index": cell.column_index
                        })
                    extracted_content["tables"].append(table_data)
            
            if result.key_value_pairs:
                for kv_pair in result.key_value_pairs:
                    if kv_pair.key and kv_pair.value:
                        extracted_content["key_value_pairs"][kv_pair.key.content] = kv_pair.value.content
            
            return extracted_content
            
        except Exception as e:
            logger.error(f"Document analysis failed: {e}")
            raise

    async def save_session_history(self, session_id: str, message: Dict) -> bool:
        """Save chat session history to CosmosDB"""
        try:
            database = self.cosmos_client.get_database_client(settings.COSMOS_DB_DATABASE_NAME)
            container = database.get_container_client(settings.COSMOS_DB_CONTAINER_NAME)
            
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
            database = self.cosmos_client.get_database_client(settings.COSMOS_DB_DATABASE_NAME)
            container = database.get_container_client(settings.COSMOS_DB_CONTAINER_NAME)
            
            session_doc = container.read_item(item=session_id, partition_key=session_id)
            return session_doc.get("messages", [])
            
        except Exception as e:
            logger.error(f"Failed to retrieve session history: {e}")
            return []

    async def get_available_models(self) -> List[Dict[str, Any]]:
        """Retrieve available model deployments from Azure AI Foundry project"""
        try:
            with observability.trace_operation("azure_get_available_models") as span:
                if not self.project_client:
                    logger.warning("Project client not initialized, returning mock models")
                    span.set_attribute("using_mock", True)
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
                    logger.warning("No models found from project connections, returning mock models")
                    models = self._get_mock_models()
                
                span.set_attribute("models_count", len(models))
                span.set_attribute("success", True)
                return models
                
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
            with observability.trace_operation("azure_get_project_connections") as span:
                connections = await self._get_project_connections_internal()
                span.set_attribute("connections_count", len(connections))
                span.set_attribute("success", True)
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
                    "connection_type": ConnectionType.AZURE_OPEN_AI,
                    "status": "connected",
                    "endpoint": "https://fallback-openai.openai.azure.com/"
                }
            ]

    async def get_project_info(self) -> Dict[str, Any]:
        """Get Azure AI Foundry project information"""
        try:
            with observability.trace_operation("azure_get_project_info") as span:
                if not self.project_client:
                    span.set_attribute("using_mock", True)
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
                
                span.set_attribute("project_name", project_info["project_name"])
                span.set_attribute("connections_count", project_info.get("connections_count", 0))
                span.set_attribute("success", True)
                
                return project_info
                
        except Exception as e:
            logger.error(f"Failed to get project info: {e}")
            observability.record_error("azure_get_project_info_error", str(e))
            return {"error": str(e), "status": "error"}
