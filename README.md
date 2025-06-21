# RAG Financial POC - Adaptive Knowledge Base Management

A comprehensive Retrieval Augmented Generation (RAG) system designed for financial document analysis, featuring adaptive knowledge base management, multi-agent orchestration, and advanced observability.

## Overview

This project implements a complete RAG Financial POC with three interconnected use cases:

1. **Context-Aware Content Generation**: RAG-based content generation with source citation
2. **Agentic Question Answering**: Multi-source information retrieval with credibility verification
3. **Adaptive Knowledge Base Management**: Automated knowledge base updates and curation

## Key Features

### 🧠 Adaptive Knowledge Base Management (Exercise 3)
- **Information Acquisition**: Automated ingestion from multiple financial document sources
- **Relevance Assessment**: AI-powered content evaluation and credibility scoring
- **Knowledge Organization**: Hierarchical structuring with semantic relationships
- **Conflict Resolution**: Intelligent handling of contradictory financial information
- **Response Adaptation**: Dynamic updates based on new market data and reports

### 📊 Advanced Observability & Evaluation
- **Token Usage Tracking**: Real-time monitoring across all Azure OpenAI models
- **Evaluation Framework**: Custom metrics for relevance, groundedness, coherence, and fluency
- **Distributed Tracing**: OpenTelemetry integration for complete request tracking
- **Admin Dashboard**: Comprehensive metrics visualization with real-time updates
- **Cost Monitoring**: Detailed cost tracking and budget alerts

### 🤖 Multi-Agent Orchestration
- **Semantic Kernel Integration**: Coordinated agent workflows for document processing
- **MCP (Model Context Protocol)**: Standardized agent communication patterns
- **A2A (Agent-to-Agent)**: Inter-agent collaboration for complex financial analysis
- **Financial Document Specialists**: Dedicated agents for 10-K/10-Q report analysis

### 🔍 Hybrid Search & RAG Pipeline
- **Vector Search**: Semantic similarity using Azure AI Search
- **Keyword Search**: Traditional text matching for precise queries
- **Hybrid Search**: Combined vector and keyword search with semantic ranking
- **Citation Management**: Comprehensive source tracking and inline citations
- **Financial Context**: Industry-specific prompt engineering and chunking strategies

## Architecture

### Backend Services (Python 3.11)
- **FastAPI**: RESTful API with automatic OpenAPI documentation
- **Azure AI Search**: Vector store with hybrid search and semantic ranking
- **Azure OpenAI**: GPT-4, GPT-4-Turbo, and embedding models
- **Azure Cosmos DB**: Session history and evaluation results storage
- **Azure Document Intelligence**: Advanced document processing and extraction
- **Azure AI Foundry**: Project-based AI model deployment and management
- **Semantic Kernel**: Multi-agent orchestration framework

### Frontend Application
- **React + TypeScript**: Modern, responsive web interface
- **Vite**: Fast development server and optimized builds
- **ChatGPT-like UI**: Intuitive chat interface inspired by open-webui
- **Admin Dashboard**: Real-time observability metrics and system monitoring
- **Citation Preview**: Interactive document source navigation

## Quick Start

### Prerequisites
- Python 3.11+
- Node.js 18+ with npm/yarn/pnpm
- Azure subscription with AI services
- Docker (optional, for containerized deployment)

### 1. Backend Setup
```bash
cd backend
pip install -r requirements.txt
cp .env.example .env
# Configure Azure services in .env file
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### 2. Frontend Setup
```bash
cd frontend/rag-financial-frontend
npm install
cp .env.example .env
# Configure API endpoints in .env file
npm run dev
```

### 3. Access the Application
- **Frontend**: http://localhost:5173
- **Backend API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **Admin Dashboard**: http://localhost:5173 (Admin tab)

## Configuration

### Azure Services Setup
1. **Azure AI Foundry**: Create project with Foundry Project approach
2. **Azure OpenAI**: Deploy GPT-4, GPT-4-Turbo, and embedding models
3. **Azure AI Search**: Configure with vector search and semantic ranking
4. **Azure Cosmos DB**: Set up containers for sessions and evaluations
5. **Azure Document Intelligence**: Enable for financial document processing
6. **Azure Application Insights**: Configure for observability and tracing

### Environment Configuration
See detailed configuration in:
- `backend/.env.example` - Backend Azure services and API settings
- `frontend/.env.example` - Frontend configuration and feature flags
- `azure-deployment-config.json` - Complete Azure resource definitions

## API Documentation

The FastAPI backend provides comprehensive API documentation:
- **Interactive Swagger UI**: http://localhost:8000/docs
- **ReDoc Documentation**: http://localhost:8000/redoc
- **OpenAPI Schema**: http://localhost:8000/openapi.json

### Key API Endpoints
- `POST /api/v1/chat/completions` - Chat completions with RAG
- `GET /api/v1/chat/models` - Available chat and embedding models
- `POST /api/v1/documents/upload` - Document upload and processing
- `GET /api/v1/admin/metrics` - Observability metrics
- `GET /api/v1/knowledge-base/status` - Knowledge base health

## Testing & Validation

### Backend Integration Tests
```bash
cd backend
python test_backend_integration.py      # Complete system integration
python test_observability_evaluation.py # Observability framework
python test_rag_pipeline.py            # RAG pipeline functionality
```

### Frontend Testing
```bash
cd frontend/rag-financial-frontend
npm test                                # Unit tests
npm run test:e2e                       # End-to-end tests
```

### Manual Testing Checklist
- [ ] Chat interface responds to financial queries
- [ ] Model selection (GPT-4, embeddings) works
- [ ] Admin dashboard displays metrics
- [ ] Document upload and processing
- [ ] Citation links navigate to sources
- [ ] Real-time metrics refresh

## MCP & A2A Integration Patterns

### Model Context Protocol (MCP)
```python
# Example MCP server integration
from app.services.mcp_server import MCPServer

mcp_server = MCPServer(port=3001)
mcp_server.register_tool("financial_analysis", financial_analysis_tool)
mcp_server.register_resource("market_data", market_data_resource)
```

### Agent-to-Agent (A2A) Communication
```python
# Example A2A workflow
from app.services.multi_agent_orchestrator import MultiAgentOrchestrator

orchestrator = MultiAgentOrchestrator()
result = await orchestrator.coordinate_agents([
    "document_processor",
    "credibility_assessor", 
    "knowledge_curator"
])
```

## Financial Document Processing

### Supported Document Types
- **10-K Annual Reports**: Complete financial statements and analysis
- **10-Q Quarterly Reports**: Quarterly financial updates
- **8-K Current Reports**: Material events and corporate changes
- **Proxy Statements**: Governance and executive compensation
- **Earnings Transcripts**: Quarterly earnings call transcripts

### Chunking Strategy
- **Hierarchical Chunking**: Section-aware document segmentation
- **Financial Context**: Industry-specific chunk boundaries
- **Overlap Management**: Intelligent chunk overlap for context preservation
- **Metadata Enrichment**: Document type, section, and page number tracking

## Deployment

### Azure Deployment
See `DEPLOYMENT_GUIDE.md` for step-by-step Azure deployment instructions including:
- Resource group and service provisioning
- Container Apps deployment
- Networking and security configuration
- Monitoring and alerting setup

### Docker Deployment
```bash
# Backend
docker build -t rag-financial-backend ./backend
docker run -p 8000:8000 rag-financial-backend

# Frontend
docker build -t rag-financial-frontend ./frontend
docker run -p 3000:80 rag-financial-frontend
```

## Observability & Monitoring

### Metrics Tracked
- **Token Usage**: By model, user, and session
- **Response Times**: API endpoint performance
- **Evaluation Scores**: Relevance, groundedness, coherence
- **System Resources**: CPU, memory, and storage utilization
- **Error Rates**: Failed requests and error categorization

### Distributed Tracing
- **OpenTelemetry**: Complete request tracing across services
- **Azure Application Insights**: Centralized logging and monitoring
- **Custom Spans**: Financial document processing workflows
- **Performance Profiling**: Bottleneck identification and optimization

## Project Structure

```
rag-financial-poc/
├── backend/                           # Python FastAPI backend
│   ├── app/
│   │   ├── api/routes/               # API endpoint definitions
│   │   │   ├── admin.py             # Admin dashboard endpoints
│   │   │   ├── chat.py              # Chat completion endpoints
│   │   │   ├── documents.py         # Document processing endpoints
│   │   │   └── knowledge_base.py    # Knowledge base management
│   │   ├── core/                    # Core functionality
│   │   │   ├── config.py           # Configuration management
│   │   │   ├── evaluation.py       # Evaluation framework
│   │   │   └── observability.py    # Observability and metrics
│   │   ├── models/                  # Pydantic data models
│   │   │   └── schemas.py          # API request/response schemas
│   │   └── services/               # Business logic services
│   │       ├── azure_services.py   # Azure service integrations
│   │       ├── document_processor.py # Document processing pipeline
│   │       ├── knowledge_base_manager.py # Knowledge base operations
│   │       ├── mcp_server.py       # MCP server implementation
│   │       ├── multi_agent_orchestrator.py # Agent coordination
│   │       └── rag_pipeline.py     # RAG processing pipeline
│   ├── test_*.py                   # Integration and unit tests
│   ├── requirements.txt            # Python dependencies
│   └── .env.example               # Environment configuration template
├── frontend/                       # React TypeScript frontend
│   └── rag-financial-frontend/
│       ├── src/
│       │   ├── components/         # React components
│       │   │   ├── admin/         # Admin dashboard components
│       │   │   └── chat/          # Chat interface components
│       │   ├── services/          # API service clients
│       │   └── types/             # TypeScript type definitions
│       ├── package.json           # Node.js dependencies
│       └── .env.example          # Frontend environment template
├── DEPLOYMENT_GUIDE.md            # Comprehensive deployment instructions
├── azure-deployment-config.json   # Azure resource configuration
├── API_DOCUMENTATION.md           # Detailed API reference
├── MCP_A2A_INTEGRATION.md         # MCP and A2A patterns guide
└── README.md                      # This file
```

## Success Criteria & KPIs

### Exercise 3: Adaptive Knowledge Base Management
- ✅ **Information Acquisition**: Automated document ingestion and processing
- ✅ **Relevance Assessment**: AI-powered content evaluation with confidence scoring
- ✅ **Knowledge Organization**: Hierarchical document structure with semantic indexing
- ✅ **Conflict Resolution**: Intelligent handling of contradictory information
- ✅ **Response Adaptation**: Dynamic knowledge base updates and query adaptation

### Technical Requirements
- ✅ **Azure Services Integration**: AI Foundry, OpenAI, Search, Cosmos DB, Document Intelligence
- ✅ **Observability Framework**: Token tracking, evaluation metrics, distributed tracing
- ✅ **MCP/A2A Implementation**: Multi-agent coordination and communication protocols
- ✅ **Financial Document Focus**: 10-K/10-Q specialized processing and analysis
- ✅ **Citation Management**: Comprehensive source tracking and navigation

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes with proper tests
4. Ensure all tests pass (`python test_backend_integration.py`)
5. Update documentation as needed
6. Submit a pull request with detailed description

## Support & Documentation

- **Azure AI Foundry**: https://learn.microsoft.com/en-us/azure/ai-foundry/
- **Semantic Kernel**: https://learn.microsoft.com/en-us/semantic-kernel/
- **Azure OpenAI**: https://learn.microsoft.com/en-us/azure/ai-services/openai/
- **FastAPI Documentation**: https://fastapi.tiangolo.com/
- **React Documentation**: https://react.dev/

## License

This project is licensed under the MIT License - see the LICENSE file for details.

---

**Built for Ashish Talati (Microsoft) - RAG Financial POC**  
*Devin Session: https://app.devin.ai/sessions/00a44a41101e43aa823ad015cb1fdd70*
