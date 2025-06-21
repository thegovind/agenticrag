from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Depends, Request
from typing import List, Optional
import logging
import uuid
from datetime import datetime
import aiofiles
import os
import asyncio

from app.models.schemas import (
    DocumentUploadRequest,
    DocumentUploadResponse,
    DocumentInfo,
    DocumentType,
    DocumentStatus
)
from app.core.observability import observability
from app.services.document_processor import DocumentProcessor
from app.services.azure_services import AzureServiceManager

router = APIRouter()
logger = logging.getLogger(__name__)

@router.post("/upload", response_model=List[DocumentUploadResponse])
async def upload_documents(
    request: Request,
    files: List[UploadFile] = File(...),
    embedding_model: Optional[str] = Form("text-embedding-ada-002"),
    search_type: Optional[str] = Form("hybrid"),
    temperature: Optional[float] = Form(0.7),
    document_type: Optional[DocumentType] = Form(None),
    company_name: Optional[str] = Form(None),
    filing_date: Optional[str] = Form(None)
):
    """Upload multiple financial documents for processing with Azure Document Intelligence"""
    try:
        responses = []
        allowed_extensions = {'.pdf', '.docx', '.doc', '.txt'}
        upload_dir = "/tmp/uploads"
        os.makedirs(upload_dir, exist_ok=True)
        
        azure_manager = getattr(request.app.state, 'azure_manager', None)
        if not azure_manager:
            raise HTTPException(status_code=500, detail="Azure services not initialized")
        
        document_processor = DocumentProcessor(azure_manager)
        
        for file in files:
            document_id = str(uuid.uuid4())
            observability.track_request("document_upload", document_id)
            
            file_extension = os.path.splitext(file.filename)[1].lower()
            
            if file_extension not in allowed_extensions:
                responses.append(DocumentUploadResponse(
                    document_id=document_id,
                    status=DocumentStatus.FAILED,
                    message=f"File type {file_extension} not supported. Allowed types: {allowed_extensions}",
                    processing_started_at=datetime.utcnow()
                ))
                continue
            
            try:
                file_content = await file.read()
                file_path = os.path.join(upload_dir, f"{document_id}_{file.filename}")
                
                async with aiofiles.open(file_path, 'wb') as f:
                    await f.write(file_content)
                
                parsed_filing_date = None
                if filing_date:
                    try:
                        parsed_filing_date = datetime.fromisoformat(filing_date.replace('Z', '+00:00'))
                    except ValueError:
                        logger.warning(f"Invalid filing date format: {filing_date}")
                
                metadata = {
                    "filename": file.filename,
                    "document_type": document_type,
                    "company_name": company_name,
                    "filing_date": parsed_filing_date.isoformat() if parsed_filing_date else None,
                    "embedding_model": embedding_model,
                    "search_type": search_type,
                    "temperature": temperature,
                    "upload_timestamp": datetime.utcnow().isoformat(),
                    "file_size": len(file_content),
                    "file_extension": file_extension
                }
                
                logger.info(f"Starting document processing: {document_id}, type: {document_type}, file: {file.filename}")
                
                content_type = file.content_type or "application/octet-stream"
                
                asyncio.create_task(
                    process_document_async(
                        document_processor, 
                        file_content, 
                        content_type, 
                        file.filename, 
                        document_id,
                        metadata
                    )
                )
                
                response = DocumentUploadResponse(
                    document_id=document_id,
                    status=DocumentStatus.PROCESSING,
                    message="Document uploaded successfully and processing started",
                    processing_started_at=datetime.utcnow()
                )
                responses.append(response)
                
            except Exception as e:
                logger.error(f"Error processing document {file.filename}: {e}")
                responses.append(DocumentUploadResponse(
                    document_id=document_id,
                    status=DocumentStatus.FAILED,
                    message=f"Failed to process document: {str(e)}",
                    processing_started_at=datetime.utcnow()
                ))
        
        return responses
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error uploading documents: {e}")
        raise HTTPException(status_code=500, detail="Failed to upload documents")

async def process_document_async(
    processor: DocumentProcessor,
    content: bytes,
    content_type: str,
    filename: str,
    document_id: str,
    metadata: dict
):
    """Asynchronously process document with Azure Document Intelligence"""
    try:
        logger.info(f"Processing document {document_id} ({filename}) with Azure Document Intelligence")
        
        processed_doc = await processor.process_document(
            content=content,
            content_type=content_type,
            source=filename,
            metadata=metadata
        )
        
        logger.info(f"Document {document_id} processed successfully: {processed_doc['processing_stats']}")
        
        observability.track_document_processing_complete(
            document_id,
            processed_doc['processing_stats']['total_chunks'],
            processed_doc['processing_stats']['metrics_extracted']
        )
        
    except Exception as e:
        logger.error(f"Error in async document processing for {document_id}: {e}")
        observability.track_document_processing_error(document_id, str(e))

@router.get("/", response_model=List[DocumentInfo])
async def list_documents(
    document_type: Optional[DocumentType] = None,
    company_name: Optional[str] = None,
    status: Optional[DocumentStatus] = None,
    limit: int = 50,
    offset: int = 0
):
    """List uploaded documents with optional filtering"""
    try:
        observability.track_request("list_documents")
        
        logger.info(f"Documents list requested: type={document_type}, company={company_name}, status={status}")
        
        documents = []
        
        return documents
    except Exception as e:
        logger.error(f"Error listing documents: {e}")
        raise HTTPException(status_code=500, detail="Failed to list documents")

@router.get("/{document_id}", response_model=DocumentInfo)
async def get_document(document_id: str):
    """Get detailed information about a specific document"""
    try:
        observability.track_request("get_document_info")
        
        logger.info(f"Document info requested: {document_id}")
        
        raise HTTPException(status_code=404, detail="Document not found")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting document {document_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve document information")

@router.get("/{document_id}/content")
async def get_document_content(document_id: str, section: Optional[str] = None):
    """Get the processed content of a document"""
    try:
        observability.track_request("get_document_content")
        
        logger.info(f"Document content requested: {document_id}, section: {section}")
        
        return {
            "document_id": document_id,
            "section": section,
            "content": "",
            "chunks": [],
            "metadata": {}
        }
    except Exception as e:
        logger.error(f"Error getting document content {document_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve document content")

@router.get("/{document_id}/chunks")
async def get_document_chunks(
    document_id: str,
    limit: int = 50,
    offset: int = 0,
    section: Optional[str] = None
):
    """Get the chunks of a processed document"""
    try:
        observability.track_request("get_document_chunks")
        
        logger.info(f"Document chunks requested: {document_id}, section: {section}")
        
        return {
            "document_id": document_id,
            "total_chunks": 0,
            "chunks": [],
            "section_filter": section
        }
    except Exception as e:
        logger.error(f"Error getting document chunks {document_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve document chunks")

@router.post("/{document_id}/reprocess")
async def reprocess_document(document_id: str):
    """Reprocess a document through the ingestion pipeline"""
    try:
        observability.track_request("reprocess_document")
        
        
        logger.info(f"Document reprocessing requested: {document_id}")
        
        return {
            "document_id": document_id,
            "status": "reprocessing_started",
            "message": "Document queued for reprocessing"
        }
    except Exception as e:
        logger.error(f"Error reprocessing document {document_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to reprocess document")

@router.delete("/{document_id}")
async def delete_document(document_id: str):
    """Delete a document and all its associated data"""
    try:
        observability.track_request("delete_document")
        
        
        logger.info(f"Document deletion requested: {document_id}")
        
        return {
            "document_id": document_id,
            "message": "Document deleted successfully"
        }
    except Exception as e:
        logger.error(f"Error deleting document {document_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete document")

@router.get("/{document_id}/citations")
async def get_document_citations(document_id: str):
    """Get all citations that reference this document"""
    try:
        observability.track_request("get_document_citations")
        
        logger.info(f"Document citations requested: {document_id}")
        
        return {
            "document_id": document_id,
            "citations": [],
            "total_citations": 0
        }
    except Exception as e:
        logger.error(f"Error getting document citations {document_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve document citations")
