# RAG Financial POC - API Documentation

## Overview

The RAG Financial Assistant provides a comprehensive RESTful API built with FastAPI, featuring automatic OpenAPI documentation, comprehensive observability, and specialized endpoints for financial document analysis.

## Base URL

- **Development**: `http://localhost:8000`
- **Production**: `https://your-deployment-url.com`

## Authentication

Currently, the API operates in development mode without authentication. For production deployment, implement Azure AD authentication as configured in the environment variables.

## API Endpoints

### Health Check

#### GET /health
Returns the health status of the API service.

**Response:**
```json
{
  "status": "healthy",
  "service": "rag-financial-backend",
  "version": "1.0.0",
  "metrics": {
    "requests_last_hour": 42,
    "errors_last_hour": 0,
    "avg_response_time": 0.245
  },
  "timestamp": 1750475156.621027
}
```

### Chat Endpoints

#### GET /api/v1/chat/models
Returns available chat and embedding models.

**Response:**
```json
{
  "chat_models": [
    "gpt-4",
    "gpt-4-turbo", 
    "gpt-35-turbo",
    "financial-llm",
    "grok-beta",
    "deepseek-chat"
  ],
  "embedding_models": [
    "text-embedding-ada-002",
    "text-embedding-3-small",
    "text-embedding-3-large"
  ]
}
```

#### POST /api/v1/chat/completions
Process chat completion with RAG pipeline integration.

**Request Body:**
```json
{
  "message": "What are the key revenue trends in the latest 10-K report?",
  "session_id": "session_123",
  "model": "gpt-4",
  "embedding_model": "text-embedding-ada-002",
  "search_type": "hybrid",
  "temperature": 0.7,
  "max_tokens": 1000,
  "include_citations": true
}
```

**Response:**
```json
{
  "response": "Based on the latest 10-K report analysis...",
  "citations": [
    {
      "id": "cite_1",
      "document_title": "Company XYZ 10-K 2024",
      "page_number": 15,
      "section": "Management's Discussion and Analysis",
      "confidence": 0.95,
      "url": "https://documents.example.com/10k-2024.pdf#page=15"
    }
  ],
  "evaluation_metrics": {
    "relevance": 0.89,
    "groundedness": 0.94,
    "coherence": 0.91,
    "fluency": 0.96
  },
  "metadata": {
    "tokens_used": 1250,
    "response_time": 2.34,
    "search_results_count": 8,
    "model_used": "gpt-4"
  }
}
```

### Document Management

#### POST /api/v1/documents/upload
Upload and process financial documents for knowledge base integration.

**Request:**
- **Content-Type**: `multipart/form-data`
- **file**: Document file (PDF, DOCX, XLSX, TXT)
- **document_type**: Type of document (10-K, 10-Q, 8-K, etc.)
- **metadata**: Additional document metadata (JSON string)

**Response:**
```json
{
  "document_id": "doc_12345",
  "filename": "company-10k-2024.pdf",
  "document_type": "10-K",
  "processing_status": "completed",
  "chunks_created": 45,
  "pages_processed": 120,
  "extraction_summary": {
    "tables_found": 12,
    "sections_identified": 8,
    "key_metrics_extracted": 25
  },
  "processing_time": 45.2
}
```

#### GET /api/v1/documents/{document_id}
Retrieve document processing status and metadata.

**Response:**
```json
{
  "document_id": "doc_12345",
  "filename": "company-10k-2024.pdf",
  "document_type": "10-K",
  "upload_date": "2024-01-15T10:30:00Z",
  "processing_status": "completed",
  "chunks": [
    {
      "chunk_id": "chunk_1",
      "content_preview": "Revenue for fiscal year 2024...",
      "page_number": 15,
      "section": "Financial Highlights"
    }
  ],
  "metadata": {
    "company_name": "Example Corp",
    "fiscal_year": "2024",
    "filing_date": "2024-03-15"
  }
}
```

#### DELETE /api/v1/documents/{document_id}
Remove document from knowledge base.

**Response:**
```json
{
  "message": "Document successfully removed from knowledge base",
  "document_id": "doc_12345",
  "chunks_removed": 45
}
```

### Knowledge Base Management

#### GET /api/v1/knowledge-base/status
Get knowledge base health and statistics.

**Response:**
```json
{
  "status": "healthy",
  "total_documents": 156,
  "total_chunks": 12450,
  "index_size_mb": 245.7,
  "last_update": "2024-01-15T14:22:00Z",
  "document_types": {
    "10-K": 45,
    "10-Q": 89,
    "8-K": 22
  },
  "search_performance": {
    "avg_query_time": 0.125,
    "index_health": "optimal"
  }
}
```

#### POST /api/v1/knowledge-base/update
Trigger knowledge base update and curation process.

**Request Body:**
```json
{
  "update_type": "incremental",
  "sources": ["sec_filings", "earnings_transcripts"],
  "date_range": {
    "start": "2024-01-01",
    "end": "2024-01-31"
  }
}
```

**Response:**
```json
{
  "update_id": "update_789",
  "status": "in_progress",
  "estimated_completion": "2024-01-15T15:30:00Z",
  "sources_processed": 0,
  "total_sources": 25
}
```

#### GET /api/v1/knowledge-base/updates/{update_id}
Get knowledge base update progress.

**Response:**
```json
{
  "update_id": "update_789",
  "status": "completed",
  "started_at": "2024-01-15T14:45:00Z",
  "completed_at": "2024-01-15T15:22:00Z",
  "summary": {
    "documents_added": 12,
    "documents_updated": 8,
    "conflicts_resolved": 3,
    "total_chunks_added": 456
  }
}
```

### Admin & Observability

#### GET /api/v1/admin/metrics
Comprehensive system metrics and observability data.

**Query Parameters:**
- `hours`: Time range for metrics (default: 24)
- `include_details`: Include detailed breakdowns (default: false)

**Response:**
```json
{
  "summary": {
    "total_tokens": 125000,
    "total_cost": 15.75,
    "total_requests": 342,
    "total_errors": 2,
    "avg_response_time": 1.234,
    "system_health": "Healthy",
    "time_range_hours": 24
  },
  "token_usage": {
    "by_model": {
      "gpt-4": 85000,
      "gpt-4-turbo": 25000,
      "text-embedding-ada-002": 15000
    },
    "recent_usage": [
      {
        "timestamp": "2024-01-15T14:30:00Z",
        "model": "gpt-4",
        "tokens": 1250,
        "cost": 0.025
      }
    ]
  },
  "evaluation_metrics": {
    "relevance": {
      "average": 0.89,
      "samples": 156,
      "distribution": {
        "excellent": 89,
        "good": 52,
        "fair": 15
      }
    },
    "groundedness": {
      "average": 0.94,
      "samples": 156
    }
  },
  "response_times": {
    "recent": [
      {
        "timestamp": "2024-01-15T14:30:00Z",
        "endpoint": "/api/v1/chat/completions",
        "duration": 2.34,
        "model": "gpt-4"
      }
    ],
    "average": 1.234
  },
  "system": {
    "cpu_usage": 45.2,
    "memory_usage": 67.8,
    "disk_usage": 23.1,
    "timestamp": "2024-01-15T14:35:00Z"
  }
}
```

#### GET /api/v1/admin/traces
Distributed tracing information for system operations.

**Query Parameters:**
- `session_id`: Filter by session ID
- `operation`: Filter by operation type
- `limit`: Number of traces to return (default: 50)

**Response:**
```json
{
  "traces": [
    {
      "trace_id": "trace_abc123",
      "operation": "chat_completion",
      "session_id": "session_456",
      "start_time": "2024-01-15T14:30:00Z",
      "duration": 2.34,
      "status": "success",
      "spans": [
        {
          "span_id": "span_1",
          "operation": "document_search",
          "duration": 0.45,
          "attributes": {
            "search_type": "hybrid",
            "results_count": 8
          }
        },
        {
          "span_id": "span_2", 
          "operation": "llm_completion",
          "duration": 1.89,
          "attributes": {
            "model": "gpt-4",
            "tokens": 1250
          }
        }
      ]
    }
  ],
  "total_traces": 1250,
  "page": 1
}
```

#### POST /api/v1/admin/evaluate
Trigger evaluation of system responses for quality assessment.

**Request Body:**
```json
{
  "session_ids": ["session_123", "session_456"],
  "evaluation_types": ["relevance", "groundedness", "coherence"],
  "sample_size": 50
}
```

**Response:**
```json
{
  "evaluation_id": "eval_789",
  "status": "in_progress",
  "estimated_completion": "2024-01-15T15:00:00Z",
  "sessions_to_evaluate": 2,
  "total_responses": 25
}
```

## Error Handling

### Standard Error Response Format
```json
{
  "error": {
    "code": "INVALID_REQUEST",
    "message": "The request parameters are invalid",
    "details": {
      "field": "model",
      "issue": "Model 'gpt-5' is not available"
    },
    "timestamp": "2024-01-15T14:30:00Z",
    "request_id": "req_12345"
  }
}
```

### Common Error Codes
- `INVALID_REQUEST` (400): Malformed request or invalid parameters
- `UNAUTHORIZED` (401): Authentication required or invalid
- `FORBIDDEN` (403): Insufficient permissions
- `NOT_FOUND` (404): Resource not found
- `RATE_LIMITED` (429): Too many requests
- `INTERNAL_ERROR` (500): Server-side error
- `SERVICE_UNAVAILABLE` (503): Azure services temporarily unavailable

## Rate Limiting

- **Default Limits**: 100 requests per minute per IP
- **Token Limits**: 50,000 tokens per minute per user
- **Headers**: Rate limit information included in response headers
  - `X-RateLimit-Limit`: Request limit
  - `X-RateLimit-Remaining`: Remaining requests
  - `X-RateLimit-Reset`: Reset timestamp

## WebSocket Support

### Real-time Chat Streaming
Connect to `/ws/chat/{session_id}` for real-time streaming responses.

**Message Format:**
```json
{
  "type": "chat_message",
  "data": {
    "message": "What are the revenue trends?",
    "model": "gpt-4",
    "stream": true
  }
}
```

**Response Stream:**
```json
{
  "type": "chat_chunk",
  "data": {
    "content": "Based on the analysis...",
    "finished": false
  }
}
```

## SDK Examples

### Python SDK Usage
```python
import requests

# Initialize client
base_url = "http://localhost:8000"
headers = {"Content-Type": "application/json"}

# Chat completion
response = requests.post(
    f"{base_url}/api/v1/chat/completions",
    json={
        "message": "Analyze the revenue trends in the 10-K report",
        "model": "gpt-4",
        "include_citations": True
    },
    headers=headers
)

result = response.json()
print(f"Response: {result['response']}")
print(f"Citations: {len(result['citations'])}")
```

### JavaScript/TypeScript SDK Usage
```typescript
interface ChatRequest {
  message: string;
  model: string;
  include_citations?: boolean;
}

async function chatCompletion(request: ChatRequest) {
  const response = await fetch('/api/v1/chat/completions', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(request)
  });
  
  return await response.json();
}

// Usage
const result = await chatCompletion({
  message: "What are the key financial metrics?",
  model: "gpt-4",
  include_citations: true
});
```

## Interactive Documentation

### Swagger UI
Access the interactive API documentation at:
- **Local Development**: http://localhost:8000/docs
- **Production**: https://your-deployment-url.com/docs

### ReDoc
Alternative documentation interface:
- **Local Development**: http://localhost:8000/redoc
- **Production**: https://your-deployment-url.com/redoc

### OpenAPI Schema
Raw OpenAPI 3.0 schema available at:
- **Local Development**: http://localhost:8000/openapi.json
- **Production**: https://your-deployment-url.com/openapi.json

## Testing the API

### Using curl
```bash
# Health check
curl -X GET "http://localhost:8000/health"

# Get available models
curl -X GET "http://localhost:8000/api/v1/chat/models"

# Chat completion
curl -X POST "http://localhost:8000/api/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "What are the key revenue trends?",
    "model": "gpt-4",
    "include_citations": true
  }'

# Get metrics
curl -X GET "http://localhost:8000/api/v1/admin/metrics?hours=1"
```

### Using Postman
Import the OpenAPI schema from `/openapi.json` to automatically generate a Postman collection with all endpoints and examples.

## Performance Considerations

### Response Times
- **Chat Completions**: 1-5 seconds (depending on model and complexity)
- **Document Upload**: 10-60 seconds (depending on document size)
- **Search Queries**: 100-500ms
- **Admin Metrics**: 50-200ms

### Optimization Tips
1. Use appropriate models for your use case (GPT-3.5 for simple queries)
2. Implement caching for frequently accessed documents
3. Use streaming for long responses
4. Batch document uploads when possible
5. Monitor token usage to optimize costs

## Support & Troubleshooting

### Common Issues
1. **Azure Service Errors**: Check service health and API keys
2. **Slow Responses**: Monitor system metrics and consider scaling
3. **High Token Usage**: Review query complexity and model selection
4. **Document Processing Failures**: Verify document format and size limits

### Getting Help
- Check the comprehensive logs in Azure Application Insights
- Review the admin dashboard for system health metrics
- Consult the deployment guide for configuration issues
- Monitor distributed traces for request flow analysis

---

**API Version**: 1.0.0  
**Last Updated**: January 2024  
**OpenAPI Specification**: 3.0.0
