# RAG Financial POC - Deployment Guide

This guide provides step-by-step instructions for deploying the RAG Financial POC solution to Azure.

## Prerequisites

- Azure subscription with appropriate permissions
- Azure CLI installed and configured
- Docker installed (for containerization)
- Node.js 18+ and Python 3.11+ for local development

## Azure Services Setup

### 1. Resource Group Creation

```bash
# Create resource group
az group create \
  --name rg-rag-financial-poc \
  --location "East US 2" \
  --tags project=rag-financial-poc environment=production
```

### 2. Azure AI Foundry Setup

```bash
# Create AI Foundry workspace
az ml workspace create \
  --name rag-financial-workspace \
  --resource-group rg-rag-financial-poc \
  --location "East US 2"

# Create AI Foundry project
az ml project create \
  --name rag-financial-project \
  --workspace-name rag-financial-workspace \
  --resource-group rg-rag-financial-poc
```

### 3. Azure OpenAI Service

```bash
# Create OpenAI service
az cognitiveservices account create \
  --name openai-rag-financial \
  --resource-group rg-rag-financial-poc \
  --location "East US 2" \
  --kind OpenAI \
  --sku S0

# Deploy models
az cognitiveservices account deployment create \
  --name openai-rag-financial \
  --resource-group rg-rag-financial-poc \
  --deployment-name gpt-4 \
  --model-name gpt-4 \
  --model-version "0613" \
  --model-format OpenAI \
  --scale-settings-scale-type "Standard" \
  --scale-settings-capacity 10

az cognitiveservices account deployment create \
  --name openai-rag-financial \
  --resource-group rg-rag-financial-poc \
  --deployment-name text-embedding-ada-002 \
  --model-name text-embedding-ada-002 \
  --model-version "2" \
  --model-format OpenAI \
  --scale-settings-scale-type "Standard" \
  --scale-settings-capacity 30
```

### 4. Azure AI Search

```bash
# Create search service
az search service create \
  --name search-rag-financial \
  --resource-group rg-rag-financial-poc \
  --location "East US 2" \
  --sku Standard \
  --replica-count 1 \
  --partition-count 1

# Create search index (use the REST API or Azure portal)
# Index configuration is provided in azure-deployment-config.json
```

### 5. Azure Cosmos DB

```bash
# Create Cosmos DB account
az cosmosdb create \
  --name cosmos-rag-financial \
  --resource-group rg-rag-financial-poc \
  --location "East US 2" \
  --default-consistency-level Session

# Create database
az cosmosdb sql database create \
  --account-name cosmos-rag-financial \
  --resource-group rg-rag-financial-poc \
  --name rag-financial-db

# Create containers
az cosmosdb sql container create \
  --account-name cosmos-rag-financial \
  --resource-group rg-rag-financial-poc \
  --database-name rag-financial-db \
  --name chat-sessions \
  --partition-key-path "/session_id" \
  --throughput 400

az cosmosdb sql container create \
  --account-name cosmos-rag-financial \
  --resource-group rg-rag-financial-poc \
  --database-name rag-financial-db \
  --name evaluation-results \
  --partition-key-path "/session_id" \
  --throughput 400
```

### 6. Azure Document Intelligence

```bash
# Create Document Intelligence service
az cognitiveservices account create \
  --name doc-intel-rag-financial \
  --resource-group rg-rag-financial-poc \
  --location "East US 2" \
  --kind FormRecognizer \
  --sku S0
```

### 7. Azure Key Vault

```bash
# Create Key Vault
az keyvault create \
  --name kv-rag-financial \
  --resource-group rg-rag-financial-poc \
  --location "East US 2" \
  --sku Standard

# Add secrets (replace with actual values)
az keyvault secret set \
  --vault-name kv-rag-financial \
  --name azure-openai-api-key \
  --value "your-openai-api-key"

az keyvault secret set \
  --vault-name kv-rag-financial \
  --name azure-search-api-key \
  --value "your-search-api-key"
```

### 8. Application Insights

```bash
# Create Application Insights
az monitor app-insights component create \
  --app ai-rag-financial \
  --location "East US 2" \
  --resource-group rg-rag-financial-poc \
  --application-type web
```

## Environment Configuration

### Backend Configuration

1. Copy `.env.example` to `.env` in the backend directory
2. Update all Azure service endpoints and keys
3. Configure the following critical settings:

```bash
# Azure OpenAI
AZURE_OPENAI_ENDPOINT=https://openai-rag-financial.openai.azure.com/
AZURE_OPENAI_API_KEY=your-api-key-from-keyvault

# Azure AI Search
AZURE_SEARCH_SERVICE_NAME=search-rag-financial
AZURE_SEARCH_API_KEY=your-search-api-key

# Azure Cosmos DB
AZURE_COSMOS_ENDPOINT=https://cosmos-rag-financial.documents.azure.com:443/
AZURE_COSMOS_KEY=your-cosmos-key

# Azure Document Intelligence
AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT=https://doc-intel-rag-financial.cognitiveservices.azure.com/
AZURE_DOCUMENT_INTELLIGENCE_API_KEY=your-doc-intel-key
```

### Frontend Configuration

1. Copy `.env.example` to `.env` in the frontend directory
2. Update API base URL and other settings:

```bash
VITE_API_BASE_URL=https://your-backend-url.azurecontainerapps.io/api/v1
VITE_AVAILABLE_CHAT_MODELS=gpt-4,gpt-4-turbo,gpt-35-turbo
VITE_AVAILABLE_EMBEDDING_MODELS=text-embedding-ada-002,text-embedding-3-small
```

## Application Deployment

### Backend Deployment

1. Build Docker image:
```bash
cd backend
docker build -t rag-financial-backend:latest .
```

2. Deploy to Azure Container Apps:
```bash
# Create Container Apps environment
az containerapp env create \
  --name cae-rag-financial \
  --resource-group rg-rag-financial-poc \
  --location "East US 2"

# Deploy backend
az containerapp create \
  --name rag-financial-backend \
  --resource-group rg-rag-financial-poc \
  --environment cae-rag-financial \
  --image rag-financial-backend:latest \
  --target-port 8000 \
  --ingress external \
  --cpu 1.0 \
  --memory 2Gi \
  --min-replicas 1 \
  --max-replicas 10
```

### Frontend Deployment

1. Build and deploy:
```bash
cd frontend/rag-financial-frontend
npm run build

# Deploy to Azure Static Web Apps or Container Apps
az staticwebapp create \
  --name rag-financial-frontend \
  --resource-group rg-rag-financial-poc \
  --location "East US 2" \
  --source https://github.com/your-repo/rag-financial-poc \
  --branch main \
  --app-location "/frontend/rag-financial-frontend" \
  --output-location "dist"
```

## Security Configuration

### Managed Identity Setup

```bash
# Create managed identity
az identity create \
  --name mi-rag-financial \
  --resource-group rg-rag-financial-poc

# Assign roles
az role assignment create \
  --assignee-object-id $(az identity show --name mi-rag-financial --resource-group rg-rag-financial-poc --query principalId -o tsv) \
  --role "Cognitive Services OpenAI User" \
  --scope /subscriptions/your-subscription-id/resourceGroups/rg-rag-financial-poc/providers/Microsoft.CognitiveServices/accounts/openai-rag-financial
```

### Network Security

1. Configure private endpoints for sensitive services
2. Set up network security groups
3. Enable Azure Firewall if required

## Monitoring and Observability

### Application Insights Configuration

1. Configure distributed tracing
2. Set up custom metrics
3. Create alert rules for:
   - High error rates
   - High response times
   - Token usage thresholds

### Log Analytics

1. Configure log collection
2. Set up custom queries for financial document analysis
3. Create dashboards for monitoring

## Testing and Validation

### Functional Testing

1. Test document upload and processing
2. Verify RAG pipeline functionality
3. Test evaluation framework
4. Validate observability metrics

### Performance Testing

1. Load test with sample financial documents
2. Verify auto-scaling behavior
3. Test token usage optimization

### Security Testing

1. Verify authentication and authorization
2. Test data encryption at rest and in transit
3. Validate network security configurations

## Maintenance and Updates

### Regular Tasks

1. Monitor token usage and costs
2. Update model deployments
3. Review and update evaluation metrics
4. Backup Cosmos DB data

### Scaling Considerations

1. Monitor search service performance
2. Adjust Cosmos DB throughput as needed
3. Scale Container Apps based on usage
4. Optimize embedding model usage

## Troubleshooting

### Common Issues

1. **Authentication failures**: Check managed identity permissions
2. **High latency**: Review search index configuration
3. **Token limits**: Monitor and optimize prompt engineering
4. **Memory issues**: Adjust container resource allocation

### Monitoring Tools

1. Application Insights for application metrics
2. Azure Monitor for infrastructure metrics
3. Cosmos DB metrics for database performance
4. Search service metrics for search performance

## Cost Optimization

### Recommendations

1. Use appropriate SKUs for each service
2. Implement auto-scaling for Container Apps
3. Monitor and optimize token usage
4. Use reserved capacity for predictable workloads
5. Implement caching strategies

### Cost Monitoring

1. Set up cost alerts
2. Review monthly usage reports
3. Optimize model deployment configurations
4. Monitor search service usage patterns

## Support and Documentation

- Azure AI Foundry documentation: https://learn.microsoft.com/en-us/azure/ai-foundry/
- Azure OpenAI documentation: https://learn.microsoft.com/en-us/azure/ai-services/openai/
- Azure AI Search documentation: https://learn.microsoft.com/en-us/azure/search/
- Semantic Kernel documentation: https://learn.microsoft.com/en-us/semantic-kernel/

For additional support, refer to the project documentation and Azure support channels.
