from fastapi import FastAPI, HTTPException, UploadFile, File, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import os
import time
import logging
from dotenv import load_dotenv
from contextlib import asynccontextmanager

from app.core.config import settings
from app.api.routes import knowledge_base, chat, admin, documents
from app.services.azure_services import AzureServiceManager
from app.core.observability import observability, setup_fastapi_instrumentation
from app.core.evaluation import setup_evaluation_framework

load_dotenv()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management with observability and evaluation setup"""
    logger = logging.getLogger(__name__)
    logger.info("Starting RAG Financial Assistant API")
    
    try:
        azure_manager = AzureServiceManager()
        await azure_manager.initialize()
        app.state.azure_manager = azure_manager
        
        if hasattr(azure_manager, 'openai_client'):
            setup_evaluation_framework(
                azure_openai_client=azure_manager.openai_client,
                cosmos_client=getattr(azure_manager, 'cosmos_client', None)
            )
            logger.info("Evaluation framework initialized")
        
        logger.info("Azure services initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize services: {e}")
    
    yield
    
    logger.info("Shutting down RAG Financial Assistant API")
    await azure_manager.cleanup()

application = FastAPI(
    title="RAG Financial POC - Adaptive Knowledge Base",
    description="Exercise 3: Adaptive Knowledge Base Management for Financial Documents with comprehensive observability",
    version="1.0.0",
    lifespan=lifespan
)

setup_fastapi_instrumentation(application)

application.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@application.middleware("http")
async def track_requests(request: Request, call_next):
    """Middleware to track all requests for observability"""
    start_time = time.time()
    
    session_id = request.headers.get("X-Session-ID")
    user_id = request.headers.get("X-User-ID")
    
    observability.track_request(
        endpoint=str(request.url.path),
        user_id=user_id,
        session_id=session_id
    )
    
    try:
        async with observability.trace_operation(
            "http_request",
            endpoint=str(request.url.path),
            method=request.method,
            session_id=session_id or "unknown"
        ) as span:
            response = await call_next(request)
            
            duration = time.time() - start_time
            observability.track_response_time(
                endpoint=str(request.url.path),
                duration=duration,
                session_id=session_id
            )
            
            span.set_attribute("response.status_code", response.status_code)
            span.set_attribute("response.duration", duration)
            
            return response
            
    except Exception as e:
        duration = time.time() - start_time
        observability.track_error(
            error_type=type(e).__name__,
            endpoint=str(request.url.path),
            error_message=str(e),
            session_id=session_id,
            user_id=user_id
        )
        
        logging.getLogger(__name__).error(f"Request failed: {request.url.path} - {str(e)}")
        raise

application.include_router(knowledge_base.router, prefix="/api/v1/knowledge-base", tags=["Knowledge Base"])
application.include_router(chat.router, prefix="/api/v1/chat", tags=["Chat"])
application.include_router(admin.router, prefix="/api/v1/admin", tags=["Admin"])
application.include_router(documents.router, prefix="/api/v1/documents", tags=["Documents"])

@application.get("/")
async def root():
    return {"message": "RAG Financial POC - Adaptive Knowledge Base Management"}

@application.get("/health")
async def health_check():
    """Enhanced health check with system metrics"""
    try:
        metrics_summary = observability.get_metrics_summary(hours=1)
        
        return {
            "status": "healthy",
            "service": "rag-financial-backend",
            "version": "1.0.0",
            "metrics": {
                "requests_last_hour": len(metrics_summary.get("requests", {}).get("recent", [])),
                "errors_last_hour": len(metrics_summary.get("errors", {}).get("recent", [])),
                "avg_response_time": metrics_summary.get("response_times", {}).get("average", 0)
            },
            "timestamp": time.time()
        }
    except Exception as e:
        logging.getLogger(__name__).error(f"Health check failed: {e}")
        return {
            "status": "degraded",
            "service": "rag-financial-backend",
            "error": str(e),
            "timestamp": time.time()
        }

app = application

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(application, host="0.0.0.0", port=8000)
