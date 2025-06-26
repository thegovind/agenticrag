# Docker Deployment Guide

This guide explains how to build and deploy the Agentic RAG application using Docker.

## Prerequisites

- Docker Engine 20.10.0+
- Docker Compose v2.0.0+
- 4GB+ available RAM
- 10GB+ available disk space

## Quick Start

### 1. Environment Setup

Copy the environment template and configure your values:

```bash
cp .env.example .env
```

Edit `.env` file with your Azure credentials and configuration:

```bash
# Required Azure Configuration
AZURE_SUBSCRIPTION_ID=your-subscription-id
AZURE_RESOURCE_GROUP=your-resource-group
AZURE_SEARCH_SERVICE_NAME=your-search-service
AZURE_SEARCH_ADMIN_KEY=your-search-admin-key
AZURE_SEARCH_ENDPOINT=https://your-search-service.search.windows.net
AZURE_OPENAI_ENDPOINT=https://your-openai-resource.openai.azure.com/
AZURE_OPENAI_API_KEY=your-openai-api-key

# Storage Configuration
AZURE_STORAGE_ACCOUNT_NAME=your-storage-account
AZURE_STORAGE_ACCOUNT_KEY=your-storage-key
AZURE_STORAGE_CONNECTION_STRING=DefaultEndpointsProtocol=https;AccountName=...

# Model Configuration
CHAT_MODEL_NAME=gpt-4o
EMBEDDING_MODEL_NAME=text-embedding-3-small
```

### 2. Development Deployment

Start the application stack for development:

```bash
docker-compose up -d
```

This will:
- Build and start the backend API server on port 8000
- Build and start the frontend web server on port 80
- Set up networking between services

Access the application:
- Frontend: http://localhost
- Backend API: http://localhost:8000
- API Documentation: http://localhost:8000/docs

### 3. Production Deployment

For production deployment with resource limits and SSL support:

```bash
docker-compose -f docker-compose.prod.yml up -d
```

To enable the reverse proxy with SSL termination:

```bash
# Set domain name and email for Let's Encrypt
export DOMAIN_NAME=yourdomain.com
export ACME_EMAIL=your-email@domain.com

# Start with proxy
docker-compose -f docker-compose.prod.yml --profile proxy up -d
```

## Service Details

### Backend Service

**Image**: Custom Python 3.11.9 application
**Port**: 8000
**Features**:
- FastAPI web framework
- Azure AI services integration
- Health checks
- Logging and monitoring
- Non-root user execution

**Health Check**: `curl http://localhost:8000/health`

### Frontend Service

**Image**: Custom Nginx-served React application  
**Port**: 80
**Features**:
- React/TypeScript application
- Nginx with compression and security headers
- API proxy to backend
- Client-side routing support

**Health Check**: `wget http://localhost/`

## Management Commands

### View logs
```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f backend
docker-compose logs -f frontend
```

### Stop services
```bash
docker-compose down
```

### Rebuild after code changes
```bash
# Rebuild specific service
docker-compose build backend
docker-compose build frontend

# Rebuild and restart
docker-compose up -d --build
```

### Scale services
```bash
# Scale backend (production only)
docker-compose -f docker-compose.prod.yml up -d --scale backend=3
```

## Development Workflow

### Local Development with Docker

1. Make code changes
2. Rebuild the affected service:
   ```bash
   docker-compose build backend  # for backend changes
   docker-compose build frontend # for frontend changes
   ```
3. Restart the service:
   ```bash
   docker-compose up -d
   ```

### Debugging

Enter a running container:
```bash
# Backend container
docker-compose exec backend bash

# Frontend container  
docker-compose exec frontend sh
```

View container resource usage:
```bash
docker stats
```

## Resource Requirements

### Minimum Requirements
- **CPU**: 2 cores
- **Memory**: 4GB RAM
- **Disk**: 10GB free space

### Recommended Production
- **CPU**: 4+ cores
- **Memory**: 8GB+ RAM
- **Disk**: 50GB+ SSD storage

## Security Considerations

1. **Environment Variables**: Never commit `.env` files with real credentials
2. **Non-root Execution**: Both containers run as non-root users
3. **Security Headers**: Frontend includes security headers
4. **Network Isolation**: Services communicate on isolated Docker network
5. **Resource Limits**: Production compose includes resource constraints

## Troubleshooting

### Common Issues

**Backend fails to start:**
```bash
# Check logs
docker-compose logs backend

# Common causes:
# - Missing Azure credentials in .env
# - Invalid Azure resource configuration  
# - Port 8000 already in use
```

**Frontend cannot reach backend:**
```bash
# Check network connectivity
docker-compose exec frontend wget -O- http://backend:8000/health

# Verify backend is healthy
docker-compose ps
```

**Out of memory errors:**
```bash
# Check resource usage
docker stats

# Increase Docker memory limits in Docker Desktop
# Or reduce resource requirements in compose files
```

### Monitoring

View real-time metrics:
```bash
# Resource usage
docker stats

# Service health
docker-compose ps

# Logs from all services
docker-compose logs -f --tail=100
```

## Production Deployment Notes

1. **SSL Certificates**: Use Traefik service for automatic SSL with Let's Encrypt
2. **Load Balancing**: Scale backend service for high availability
3. **Monitoring**: Consider adding Prometheus/Grafana stack
4. **Backup**: Ensure Azure resources have proper backup policies
5. **Updates**: Use blue-green deployment strategy for zero-downtime updates

## Azure Integration

The application integrates with several Azure services:

- **Azure OpenAI**: For language models and embeddings
- **Azure AI Search**: For vector and hybrid search
- **Azure Storage**: For document storage and caching  
- **Azure AI Foundry**: For model evaluation and monitoring
- **Azure Application Insights**: For telemetry and logging

Ensure all these services are properly configured and accessible from your Docker deployment environment.
