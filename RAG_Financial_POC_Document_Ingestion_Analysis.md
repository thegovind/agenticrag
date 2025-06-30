# RAG Financial POC - Repository and Document Ingestion Analysis

## Repository Overview

This repository implements a comprehensive **Retrieval Augmented Generation (RAG) system** specifically designed for financial document analysis. The system features adaptive knowledge base management, multi-agent orchestration, and advanced observability capabilities.

### Architecture Overview

The repository follows a **full-stack architecture** with:

- **Backend**: Python 3.11 + FastAPI with Azure AI services
- **Frontend**: React + TypeScript with Vite
- **Infrastructure**: Azure-based cloud services
- **Database**: Azure Cosmos DB for session storage, Azure AI Search for vector storage

### Key Components

```
├── backend/                    # Python FastAPI backend
│   ├── app/
│   │   ├── api/routes/        # API endpoints
│   │   ├── services/          # Core business logic
│   │   ├── core/              # Configuration and utilities
│   │   └── models/            # Data models and schemas
├── frontend/                   # React TypeScript frontend
└── docker-compose files       # Containerization configs
```

## Document Ingestion Architecture

The document ingestion system is designed to handle **financial documents** (primarily SEC filings like 10-K, 10-Q, 8-K reports) through a sophisticated multi-stage pipeline.

### Core Ingestion Services

#### 1. Document Processor (`document_processor.py`)
The main orchestrator for document ingestion with these capabilities:

**Key Features:**
- **Azure Document Intelligence Integration**: Uses Azure Form Recognizer for advanced document analysis
- **Financial Document Specialization**: Optimized for SEC filings with specialized section parsing
- **Intelligent Chunking**: Multiple chunking strategies (markdown-based, hierarchical, basic text)
- **Metadata Extraction**: Comprehensive financial metadata extraction (company info, filing dates, CIK, etc.)
- **Embedding Generation**: Automatic vector embedding creation for semantic search

**Processing Pipeline:**
```python
async def process_document(content: bytes, content_type: str, source: str, metadata: Dict) -> Dict:
    # 1. Azure Document Intelligence analysis
    extracted_content = await self.azure_manager.analyze_document(content, content_type)
    
    # 2. Generate unique document ID
    document_id = self._generate_document_id(source, extracted_content["content"])
    
    # 3. Extract comprehensive financial information
    financial_info = await self._extract_comprehensive_financial_info(extracted_content, source)
    
    # 4. Parse document structure (sections, tables, footnotes)
    document_structure = await self._parse_financial_document_structure(extracted_content, financial_info)
    
    # 5. Create intelligent chunks with multiple strategies
    chunks = await self._create_markdown_chunks(document_structure, document_id, metadata)
    
    # 6. Generate embeddings for all chunks
    for chunk in chunks:
        chunk.embedding = await self.azure_manager.get_embedding(chunk.content)
    
    # 7. Prepare and index in Azure Search
    search_documents = self._prepare_search_documents(chunks)
    await self.azure_manager.add_documents_to_index(search_documents)
    
    # 8. Extract key financial metrics
    key_metrics = await self._extract_key_financial_metrics(content, financial_info)
```

#### 2. SEC Document Service (`sec_document_service.py`)
Specialized service for SEC filing ingestion using the Edgar tools library:

**Key Features:**
- **Edgar Tools Integration**: Direct connection to SEC EDGAR API
- **Company Search**: Intelligent company name/ticker resolution
- **Filing Retrieval**: Automated download of specific SEC documents
- **Batch Processing**: Parallel processing of multiple documents
- **Progress Tracking**: Real-time progress updates for long-running operations

**SEC-Specific Workflow:**
```python
async def retrieve_and_process_document(ticker: str, accession_number: str) -> Dict:
    # 1. Connect to SEC EDGAR API
    company = Company(ticker.upper())
    
    # 2. Find specific filing by accession number
    filing = find_filing_by_accession(company, accession_number)
    
    # 3. Download document content
    content = filing.text() or filing.html()
    
    # 4. Extract SEC-specific metadata
    metadata = extract_sec_metadata(filing, company)
    
    # 5. Process with md2chunks for intelligent chunking
    chunks = await self._process_content_with_md2chunks(content, document_id, metadata)
    
    # 6. Generate embeddings and index
    await self._generate_embeddings_for_chunks(chunks)
    await self._prepare_search_documents(chunks)
```

#### 3. Azure Services Manager (`azure_services.py`)
Central hub for all Azure AI service integrations:

**Integrated Services:**
- **Azure AI Search**: Vector and hybrid search with semantic ranking
- **Azure OpenAI**: GPT-4 models and text-embedding-ada-002
- **Azure Document Intelligence**: Advanced document analysis and OCR
- **Azure Cosmos DB**: Session history and evaluation storage
- **Azure AI Foundry**: Multi-agent orchestration platform

**Search Index Schema:**
```python
# Core document fields
"id", "content", "title", "document_id", "source"

# Financial document specific fields  
"ticker", "company", "cik", "form_type", "filing_date"
"section_type", "accession_number", "industry"

# Processing metadata
"credibility_score", "chunk_index", "content_vector"
"processed_at", "citation_info"
```

#### 4. Knowledge Base Manager (`knowledge_base_manager.py`)
Implements adaptive knowledge base management for Exercise 3:

**Adaptive Features:**
- **Automatic Source Discovery**: RSS feeds, SEC filings, news sources
- **Credibility Assessment**: AI-powered content evaluation
- **Conflict Resolution**: Intelligent handling of contradictory information
- **Knowledge Organization**: Hierarchical structuring with semantic relationships

### Document Ingestion Flow

#### Standard Document Upload Flow

1. **Upload Endpoint** (`/api/documents/upload`)
   ```python
   POST /api/documents/upload
   # Accepts: PDF, DOCX, DOC, TXT files
   # Returns: Processing status and document ID
   ```

2. **Processing Pipeline**
   - File validation and temporary storage
   - Azure Document Intelligence analysis
   - Financial metadata extraction
   - Intelligent chunking (markdown → hierarchical → basic fallback)
   - Vector embedding generation
   - Azure Search indexing

3. **Async Processing**
   - Documents processed asynchronously to handle large files
   - Progress tracking and status updates
   - Error handling and retry mechanisms

#### SEC Document Ingestion Flow

1. **Company Search** (`/api/v1/sec/companies/search`)
   ```python
   POST /sec/companies/search
   {
     "query": "Apple" or "AAPL"
   }
   # Returns: Company info with available filings summary
   ```

2. **Filing Discovery** (`/api/v1/sec/filings`)
   ```python
   POST /sec/filings
   {
     "ticker": "AAPL",
     "form_types": ["10-K", "10-Q"],
     "limit": 20
   }
   # Returns: List of available filings with metadata
   ```

3. **Document Processing** (`/api/v1/sec/documents/process`)
   ```python
   POST /sec/documents/process
   {
     "ticker": "AAPL",
     "accession_number": "0000320193-23-000077"
   }
   # Triggers: Automated download and processing
   ```

4. **Batch Processing** (`/api/v1/sec/documents/process-multiple`)
   ```python
   POST /sec/documents/process-multiple
   {
     "filings": [...],
     "max_parallel": 3
   }
   # Enables: Parallel processing of multiple documents
   ```

### Chunking Strategies

The system implements multiple chunking strategies with intelligent fallbacks:

#### 1. Markdown-Based Chunking (Primary)
- Converts documents to markdown format
- Preserves document structure (headers, tables, lists)
- Intelligent section boundaries
- Table-aware chunking

#### 2. Hierarchical Financial Chunking (Secondary)
- SEC filing section recognition (PART I, ITEM 1, etc.)
- Financial statement identification
- Table and footnote extraction
- Metadata-rich chunk creation

#### 3. Basic Text Chunking (Fallback)
- Simple text splitting with overlap
- Used when structured approaches fail
- Configurable chunk size and overlap

### Financial Document Specialization

#### SEC Filing Recognition
The system recognizes and specially processes:
- **10-K Annual Reports**: Complete financial statements
- **10-Q Quarterly Reports**: Quarterly updates
- **8-K Current Reports**: Material events
- **Proxy Statements**: Governance information

#### Metadata Extraction
Comprehensive financial metadata extraction:
```python
financial_info = {
    "document_type": "10-K",
    "company_name": "Apple Inc.",
    "cik": "0000320193",
    "ticker_symbol": "AAPL", 
    "filing_date": "2023-11-03",
    "fiscal_year": "2023",
    "fiscal_quarter": "Q4",
    "sections": ["business", "risk_factors", "financial_statements"],
    "key_metrics": [...]
}
```

#### Financial Section Patterns
The system uses regex patterns to identify:
- PART I/II sections and ITEM subdivisions
- Financial statements (Balance Sheet, Income Statement, Cash Flow)
- Management Discussion & Analysis (MD&A)
- Risk factors and footnotes

### Search and Retrieval

#### Hybrid Search Implementation
The system implements sophisticated search capabilities:

1. **Vector Search**: Semantic similarity using embeddings
2. **Keyword Search**: Traditional text matching
3. **Hybrid Search**: Combined approach with semantic ranking
4. **Filtered Search**: Company, date, document type filters

#### Citation Management
- Comprehensive source tracking
- Inline citation generation
- Document chunk references
- Credibility scoring

### Observability and Monitoring

#### Token Usage Tracking
- Real-time monitoring across all Azure OpenAI models
- Cost tracking and budget alerts
- Performance metrics per operation

#### Evaluation Framework
- Custom metrics for relevance, groundedness, coherence
- Distributed tracing with OpenTelemetry
- Admin dashboard for system monitoring

### API Endpoints Summary

#### Document Management
- `POST /api/documents/upload` - Upload and process documents
- `GET /api/documents/` - List processed documents
- `GET /api/documents/{id}` - Get document details
- `DELETE /api/documents/{id}` - Remove document

#### SEC Document Operations
- `POST /api/v1/sec/companies/search` - Search companies
- `POST /api/v1/sec/filings` - Get company filings
- `POST /api/v1/sec/documents/process` - Process single document
- `POST /api/v1/sec/documents/process-multiple` - Batch processing
- `GET /api/v1/sec/batch/{id}/status` - Check processing status

#### Knowledge Base Management
- Automatic knowledge base updates
- Conflict resolution between sources
- Response adaptation based on new information

## Technology Stack

### Backend Technologies
- **Python 3.11** with FastAPI
- **Azure AI Services** (OpenAI, Search, Document Intelligence)
- **Edgar Tools** for SEC data access
- **Pydantic** for data validation
- **AsyncIO** for concurrent processing

### Frontend Technologies
- **React 18** with TypeScript
- **Vite** for fast development
- **Tailwind CSS** for styling
- **ChatGPT-like UI** for user interaction

### Infrastructure
- **Azure AI Search** for vector storage
- **Azure Cosmos DB** for session management
- **Azure OpenAI** for LLM and embeddings
- **Docker** for containerization

## Key Strengths

1. **Financial Document Expertise**: Specialized for SEC filings and financial reports
2. **Intelligent Processing**: Multiple chunking strategies with smart fallbacks
3. **Scalable Architecture**: Async processing with batch capabilities
4. **Comprehensive Observability**: Full tracking and monitoring
5. **Adaptive Knowledge Base**: Automatic updates and conflict resolution
6. **Production Ready**: Robust error handling and retry mechanisms

This RAG Financial POC represents a sophisticated approach to financial document analysis, combining modern AI capabilities with domain-specific expertise in financial document processing.