from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.models import VectorizedQuery
from azure.ai.formrecognizer import DocumentAnalysisClient
from azure.cosmos import CosmosClient
from azure.identity import DefaultAzureCredential, ClientSecretCredential
from openai import AzureOpenAI
import asyncio
import logging
from typing import List, Dict, Any, Optional
from app.core.config import settings

logger = logging.getLogger(__name__)

class AzureServiceManager:
    def __init__(self):
        self.search_client = None
        self.search_index_client = None
        self.form_recognizer_client = None
        self.cosmos_client = None
        self.openai_client = None
        self.credential = None
        
    async def initialize(self):
        """Initialize all Azure services"""
        try:
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
            
            self.openai_client = AzureOpenAI(
                azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
                api_key=settings.AZURE_OPENAI_API_KEY,
                api_version=settings.AZURE_OPENAI_API_VERSION
            )
            
            logger.info("Azure services initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Azure services: {e}")
            raise
    
    async def cleanup(self):
        """Cleanup resources"""
        if self.cosmos_client:
            self.cosmos_client.close()
        logger.info("Azure services cleaned up")

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
