from fastapi import APIRouter, HTTPException, Depends, Query
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
import logging

from app.services.sec_document_service import SECDocumentService, SECDocumentInfo
from app.services.azure_services import AzureServiceManager
from app.core.observability import observability

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/sec", tags=["SEC Documents"])

class CompanySearchRequest(BaseModel):
    query: str

class CompanySearchResponse(BaseModel):
    companies: List[Dict[str, Any]]

class FilingRequest(BaseModel):
    ticker: str
    form_types: Optional[List[str]] = ["10-K", "10-Q", "8-K"]
    limit: Optional[int] = 20

class SpecificFilingRequest(BaseModel):
    ticker: str
    form_types: List[str]  # Multiple form types
    years: Optional[List[int]] = None  # Changed from single year to multiple years
    limit: Optional[int] = 10

class ProcessDocumentRequest(BaseModel):
    ticker: str
    accession_number: str
    document_id: Optional[str] = None

class ProcessMultipleDocumentsRequest(BaseModel):
    filings: List[ProcessDocumentRequest]

class ProcessDocumentResponse(BaseModel):
    document_id: str
    chunks_created: int
    metadata: Dict[str, Any]
    filing_info: Dict[str, Any]
    skipped: Optional[bool] = False

class ProcessMultipleDocumentsResponse(BaseModel):
    total_requested: int
    processed: int
    skipped: int
    total_chunks_created: int
    results: List[ProcessDocumentResponse]

class SECDocumentLibraryResponse(BaseModel):
    documents: List[Dict[str, Any]]
    total_count: int
    total_chunks: int
    companies: List[str]
    form_types: List[str]

class SECAnalyticsResponse(BaseModel):
    total_documents: int
    total_chunks: int
    companies_count: int
    form_types_distribution: Dict[str, int]
    chunks_per_document_avg: float
    recent_activity: List[Dict[str, Any]]
    company_distribution: Dict[str, int]
    filing_date_range: Dict[str, str]

class ChunkVisualizationResponse(BaseModel):
    document_id: str
    document_info: Dict[str, Any]
    chunks: List[Dict[str, Any]]
    chunk_stats: Dict[str, Any]

async def get_sec_service() -> SECDocumentService:
    """Dependency to get SEC document service"""
    try:
        # Get Azure manager from app state (assumes it's initialized in main.py)
        from fastapi import Request
        # For now, create a new instance - in production, this should be injected
        from app.core.config import settings
        azure_manager = AzureServiceManager()
        await azure_manager.initialize()
        return SECDocumentService(azure_manager)
    except Exception as e:
        logger.error(f"Failed to initialize SEC service: {e}")
        raise HTTPException(status_code=500, detail="Failed to initialize SEC service")

@router.post("/companies/search", response_model=CompanySearchResponse)
async def search_companies(
    request: CompanySearchRequest,
    sec_service: SECDocumentService = Depends(get_sec_service)
):
    """
    Search for companies by name or ticker symbol
    """
    try:
        # async with observability.trace_operation("sec_company_search") as span:
        #     span.set_attribute("query", request.query)
        
        companies = await sec_service.search_companies(request.query)
        
        #     span.set_attribute("companies_found", len(companies))
        #     span.set_attribute("success", True)
        
        return CompanySearchResponse(companies=companies)
        
    except Exception as e:
        logger.error(f"Error searching companies: {e}")
        # observability.record_error("sec_company_search_error", str(e))
        raise HTTPException(status_code=500, detail=f"Failed to search companies: {str(e)}")

@router.post("/filings")
async def get_company_filings(
    request: FilingRequest,
    sec_service: SECDocumentService = Depends(get_sec_service)
):
    """    Get recent filings for a company
    """
    try:
        # async with observability.trace_operation("sec_get_filings") as span:
        #     span.set_attribute("ticker", request.ticker)
        #     span.set_attribute("form_types", str(request.form_types))
        
        filings = await sec_service.get_company_filings(
            ticker=request.ticker,
            form_types=request.form_types,
            limit=request.limit
        )
        
        # Convert to dict format for JSON response
        filings_data = []
        for filing in filings:
            filing_dict = {
                "ticker": filing.ticker,
                "company_name": filing.company_name,
                "cik": filing.cik,
                "form_type": filing.form_type,
                "filing_date": filing.filing_date.isoformat(),
                "period_end_date": filing.period_end_date.isoformat() if filing.period_end_date else None,
                "document_url": filing.document_url,
                "accession_number": filing.accession_number,
                "file_size": filing.file_size
            }
            filings_data.append(filing_dict)
        
        #     span.set_attribute("filings_found", len(filings_data))
        #     span.set_attribute("success", True)
        
        return {
            "ticker": request.ticker,
            "filings": filings_data,
            "total_count": len(filings_data)
        }
        
    except Exception as e:
        logger.error(f"Error getting filings for {request.ticker}: {e}")
        # observability.record_error("sec_get_filings_error", str(e))
        raise HTTPException(status_code=500, detail=f"Failed to get filings: {str(e)}")

@router.post("/documents/process", response_model=ProcessDocumentResponse)
async def process_sec_document(
    request: ProcessDocumentRequest,
    sec_service: SECDocumentService = Depends(get_sec_service)
):
    """
    Retrieve and process a SEC document for the knowledge base
    """
    try:
        # async with observability.trace_operation("sec_process_document") as span:
        #     span.set_attribute("ticker", request.ticker)
        #     span.set_attribute("accession_number", request.accession_number)
        
        result = await sec_service.retrieve_and_process_document(
            ticker=request.ticker,
            accession_number=request.accession_number,
            document_id=request.document_id
        )
        
        #     span.set_attribute("chunks_created", result["chunks_created"])
        #     span.set_attribute("success", True)
        
        return ProcessDocumentResponse(**result)
            
    except Exception as e:
        logger.error(f"Error processing SEC document: {e}")
        # observability.record_error("sec_process_document_error", str(e))
        raise HTTPException(status_code=500, detail=f"Failed to process document: {str(e)}")

@router.post("/documents/process-multiple", response_model=ProcessMultipleDocumentsResponse)
async def process_multiple_sec_documents(
    request: ProcessMultipleDocumentsRequest,
    sec_service: SECDocumentService = Depends(get_sec_service)
):
    """
    Retrieve and process multiple SEC documents for the knowledge base
    """
    try:
        logger.info(f"Processing {len(request.filings)} SEC documents")
        
        results = []
        processed_count = 0
        skipped_count = 0
        total_chunks = 0
        
        for filing_request in request.filings:
            try:
                result = await sec_service.retrieve_and_process_document(
                    ticker=filing_request.ticker,
                    accession_number=filing_request.accession_number,
                    document_id=filing_request.document_id
                )
                
                result_response = ProcessDocumentResponse(**result)
                results.append(result_response)
                
                if result.get('skipped', False):
                    skipped_count += 1
                else:
                    processed_count += 1
                    total_chunks += result.get('chunks_created', 0)
                    
            except Exception as e:
                logger.error(f"Error processing filing {filing_request.accession_number}: {e}")
                # Continue processing other filings even if one fails
                error_result = ProcessDocumentResponse(
                    document_id=f"{filing_request.ticker}_{filing_request.accession_number}",
                    chunks_created=0,
                    metadata={"status": "error", "message": str(e)},
                    filing_info={"ticker": filing_request.ticker, "accession_number": filing_request.accession_number},
                    skipped=False
                )
                results.append(error_result)
        
        return ProcessMultipleDocumentsResponse(
            total_requested=len(request.filings),
            processed=processed_count,
            skipped=skipped_count,
            total_chunks_created=total_chunks,
            results=results
        )
            
    except Exception as e:
        logger.error(f"Error processing multiple SEC documents: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to process documents: {str(e)}")

@router.get("/health")
async def health_check():
    """
    Health check for SEC document service
    """
    try:
        # Basic health check - could expand to test EDGAR API connectivity
        return {
            "status": "healthy",
            "service": "SEC Document Service",
            "timestamp": observability.get_current_timestamp()
        }
    except Exception as e:
        logger.error(f"SEC service health check failed: {e}")
        raise HTTPException(status_code=503, detail="SEC service unhealthy")

@router.get("/form-types")
async def get_supported_form_types():
    """
    Get list of supported SEC form types
    """
    return {
        "supported_forms": [
            {
                "form_type": "10-K",
                "description": "Annual Report",
                "frequency": "Annual"
            },
            {
                "form_type": "10-Q", 
                "description": "Quarterly Report",
                "frequency": "Quarterly"
            },
            {
                "form_type": "8-K",
                "description": "Current Report", 
                "frequency": "As needed"
            },
            {
                "form_type": "20-F",
                "description": "Foreign Company Annual Report",
                "frequency": "Annual"
            },
            {
                "form_type": "DEF 14A",
                "description": "Proxy Statement",
                "frequency": "Annual"
            }
        ]
    }

@router.post("/filings/specific")
async def get_specific_filings(
    request: SpecificFilingRequest,
    sec_service: SECDocumentService = Depends(get_sec_service)
):
    """
    Get specific filings by form types and years
    """
    try:
        logger.info(f"Getting specific filings for {request.ticker}, forms: {request.form_types}, years: {request.years}")
        
        # Use the new method that handles company name mapping and year filtering better
        filings = await sec_service.get_specific_filings(
            ticker=request.ticker,
            form_types=request.form_types,
            years=request.years,
            limit=request.limit
        )
        
        # Convert to dict format for JSON response
        filings_data = []
        for filing in filings:
            filing_dict = {
                "ticker": filing.ticker,
                "company_name": filing.company_name,
                "cik": filing.cik,
                "form_type": filing.form_type,
                "filing_date": filing.filing_date.isoformat(),
                "period_end_date": filing.period_end_date.isoformat() if filing.period_end_date else None,
                "document_url": filing.document_url,
                "accession_number": filing.accession_number,
                "file_size": filing.file_size,
                "year": filing.filing_date.year
            }
            filings_data.append(filing_dict)
        
        return {
            "ticker": request.ticker,
            "form_types": request.form_types,
            "years": request.years,
            "filings": filings_data,
            "total_count": len(filings_data)
        }
        
    except Exception as e:
        logger.error(f"Error getting specific filings for {request.ticker}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get specific filings: {str(e)}")

@router.get("/analytics", response_model=SECAnalyticsResponse)
async def get_sec_analytics(
    sec_service: SECDocumentService = Depends(get_sec_service)
):
    """
    Get analytics about SEC documents in the vector store
    """
    try:
        logger.info("Getting SEC analytics")
        
        # Get all SEC documents from Azure Search
        results = await sec_service.azure_manager.hybrid_search(
            query="*",
            top_k=1000  # Get a large sample for analytics
        )
        
        logger.info(f"Found {len(results)} chunks for analytics")
        
        if not results:
            return SECAnalyticsResponse(
                total_documents=0,
                total_chunks=0,
                companies_count=0,
                form_types_distribution={},
                chunks_per_document_avg=0,
                recent_activity=[],
                company_distribution={},
                filing_date_range={}
            )
        
        # Analyze the data
        unique_documents = set()
        form_types_distribution = {}
        companies = {}
        filing_dates = []
        
        for result in results:
            doc_id = result.get('document_id', '')
            if doc_id:
                unique_documents.add(doc_id)
            
            form_type = result.get('form_type', '')
            if form_type:
                form_types_distribution[form_type] = form_types_distribution.get(form_type, 0) + 1
            
            company = result.get('company', '')
            if company:
                companies[company] = companies.get(company, 0) + 1
            
            filing_date = result.get('filing_date', '')
            if filing_date:
                filing_dates.append(filing_date)
        
        # Recent activity (last 10 processed documents)
        recent_results = sorted(
            [r for r in results if r.get('processed_at')],
            key=lambda x: x.get('processed_at', ''),
            reverse=True
        )[:50]  # Get more results to find unique documents
        
        recent_activity = []
        seen_docs = set()
        for result in recent_results:
            doc_id = result.get('document_id', '')
            if doc_id not in seen_docs and len(recent_activity) < 10:
                recent_activity.append({
                    "document_id": doc_id,
                    "company": result.get('company', ''),
                    "form_type": result.get('form_type', ''),
                    "filing_date": result.get('filing_date', ''),
                    "processed_at": result.get('processed_at', ''),
                    "chunk_count": len([r for r in results if r.get('document_id') == doc_id])
                })
                seen_docs.add(doc_id)
        
        # Calculate date range
        filing_date_range = {}
        if filing_dates:
            filing_dates.sort()
            filing_date_range = {
                "earliest": filing_dates[0],
                "latest": filing_dates[-1]
            }
        
        total_documents = len(unique_documents)
        total_chunks = len(results)
        chunks_per_doc_avg = total_chunks / total_documents if total_documents > 0 else 0
        
        logger.info(f"Analytics: {total_documents} docs, {total_chunks} chunks, {len(companies)} companies")
        
        return SECAnalyticsResponse(
            total_documents=total_documents,
            total_chunks=total_chunks,
            companies_count=len(companies),
            form_types_distribution=form_types_distribution,
            chunks_per_document_avg=round(chunks_per_doc_avg, 2),
            recent_activity=recent_activity,
            company_distribution=companies,
            filing_date_range=filing_date_range
        )
        
    except Exception as e:
        logger.error(f"Error getting SEC analytics: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get analytics: {str(e)}")

@router.get("/documents/{document_id}/chunks", response_model=ChunkVisualizationResponse)
async def get_document_chunks(
    document_id: str,
    sec_service: SECDocumentService = Depends(get_sec_service)
):
    """
    Get chunking visualization for a specific SEC document
    """
    try:
        logger.info(f"Getting chunks for document: {document_id}")
        
        # Get all chunks for this document
        results = await sec_service.azure_manager.hybrid_search(
            query="*",
            filters=f"document_id eq '{document_id}'",
            top_k=1000
        )
        
        if not results:
            raise HTTPException(status_code=404, detail="Document not found")
        
        # Get document info from first chunk
        doc_info = {
            "document_id": document_id,
            "company": results[0].get('company', ''),
            "ticker": results[0].get('ticker', ''),
            "form_type": results[0].get('form_type', ''),
            "filing_date": results[0].get('filing_date', ''),
            "accession_number": results[0].get('accession_number', ''),
            "total_chunks": len(results),
            "cik": results[0].get('cik', ''),
            "processed_at": results[0].get('processed_at', '')
        }
        
        # Prepare chunk data
        chunks = []
        total_content_length = 0
        page_numbers = []
        section_types = []
        
        for i, result in enumerate(results):
            content = result.get('content', '')
            total_content_length += len(content)
            
            page_num = result.get('page_number')
            if page_num:
                page_numbers.append(page_num)
            
            section_type = result.get('section_type', '')
            if section_type:
                section_types.append(section_type)
            
            chunks.append({
                "chunk_id": result.get('chunk_id', f"chunk_{i}"),
                "content": content[:200] + "..." if len(content) > 200 else content,
                "content_length": len(content),
                "page_number": page_num,
                "section_type": section_type,
                "credibility_score": result.get('credibility_score', 0),
                "citation_info": result.get('citation_info', ''),
                "search_score": result.get('search_score', 0)
            })
        
        # Calculate chunk statistics
        chunk_stats = {
            "total_chunks": len(chunks),
            "avg_chunk_length": round(total_content_length / len(chunks), 2) if chunks else 0,
            "total_content_length": total_content_length,
            "page_range": {
                "min": min(page_numbers) if page_numbers else None,
                "max": max(page_numbers) if page_numbers else None
            },
            "section_types": list(set(section_types)),
            "avg_credibility_score": round(
                sum(c.get('credibility_score', 0) for c in chunks) / len(chunks), 3
            ) if chunks else 0
        }
        
        return ChunkVisualizationResponse(
            document_id=document_id,
            document_info=doc_info,
            chunks=chunks,
            chunk_stats=chunk_stats
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting document chunks: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get document chunks: {str(e)}")

@router.get("/library", response_model=SECDocumentLibraryResponse)
async def get_sec_document_library(
    company: Optional[str] = None,
    form_type: Optional[str] = None,
    limit: Optional[int] = 100,
    sec_service: SECDocumentService = Depends(get_sec_service)
):
    """
    Get the SEC document library from the vector store.
    First retrieves all unique document_ids, then extracts metadata from a representative chunk,
    then applies company/form_type filters.
    """
    try:
        logger.info(f"Getting SEC document library - company: {company}, form_type: {form_type}")
        
        # Step 1: Get ALL documents from Azure Search (no filters yet)
        # Use a large top_k to get all documents
        all_results = await sec_service.azure_manager.hybrid_search(
            query="*",
            top_k=5000,  # Large number to get all documents
            filters=None  # No filters initially
        )
        
        logger.info(f"Found {len(all_results)} total chunks in index")
        
        # Step 2: Group by document_id and extract metadata from representative chunk
        documents_by_id = {}
        total_chunks = len(all_results)
        
        for result in all_results:
            doc_id = result.get('document_id', '')
            if not doc_id:
                continue
                
            if doc_id not in documents_by_id:
                # Use first chunk as representative for this document
                documents_by_id[doc_id] = {
                    "document_id": doc_id,
                    "company": result.get('company', 'Unknown Company'),
                    "ticker": result.get('ticker', 'N/A'),
                    "form_type": result.get('form_type', 'Unknown'),
                    "filing_date": result.get('filing_date', ''),
                    "accession_number": result.get('accession_number', ''),
                    "chunk_count": 0,
                    "processed_at": result.get('processed_at', ''),
                    "source": result.get('source', ''),
                    "cik": result.get('cik', ''),
                    "industry": result.get('industry', ''),
                    "document_url": result.get('document_url', ''),
                    "section_type": result.get('section_type', '')
                }
            
            # Count chunks for this document
            documents_by_id[doc_id]["chunk_count"] += 1
        
        all_documents = list(documents_by_id.values())
        logger.info(f"Found {len(all_documents)} unique documents before filtering")
        
        # Step 3: Apply filters based on extracted metadata
        filtered_documents = all_documents
        
        if company:
            # Filter by company name (case-insensitive partial match)
            company_lower = company.lower()
            filtered_documents = [
                doc for doc in filtered_documents 
                if company_lower in doc.get("company", "").lower() or 
                   company_lower in doc.get("ticker", "").lower()
            ]
            logger.info(f"After company filter '{company}': {len(filtered_documents)} documents")
        
        if form_type:
            # Filter by form type (exact match)
            filtered_documents = [
                doc for doc in filtered_documents 
                if doc.get("form_type", "").upper() == form_type.upper()
            ]
            logger.info(f"After form_type filter '{form_type}': {len(filtered_documents)} documents")
        
        # Step 4: Apply limit and sort by filing_date (most recent first)
        try:
            # Sort by filing_date, handling empty dates
            filtered_documents = sorted(
                filtered_documents, 
                key=lambda x: x.get("filing_date", "1900-01-01"), 
                reverse=True
            )
        except Exception as e:
            logger.warning(f"Could not sort by filing_date: {e}")
        
        # Apply limit
        limited_documents = filtered_documents[:limit]
        
        # Step 5: Get unique companies and form types for filter dropdowns
        all_companies = list(set(
            doc["company"] for doc in all_documents 
            if doc["company"] and doc["company"] != "Unknown Company"
        ))
        all_form_types = list(set(
            doc["form_type"] for doc in all_documents 
            if doc["form_type"] and doc["form_type"] != "Unknown"
        ))
        
        logger.info(f"Returning {len(limited_documents)} documents after all filtering")
        
        return SECDocumentLibraryResponse(
            documents=limited_documents,
            total_count=len(filtered_documents),  # Total after filtering, before limit
            total_chunks=sum(doc["chunk_count"] for doc in limited_documents),
            companies=sorted(all_companies),
            form_types=sorted(all_form_types)
        )
        
    except Exception as e:
        logger.error(f"Error getting SEC document library: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get document library: {str(e)}")

@router.get("/analytics", response_model=SECAnalyticsResponse)
async def get_sec_analytics(
    sec_service: SECDocumentService = Depends(get_sec_service)
):
    """
    Get analytics about SEC documents in the vector store
    """
    try:
        logger.info("Getting SEC analytics")
        
        # Get all SEC documents from Azure Search
        results = await sec_service.azure_manager.hybrid_search(
            query="*",
            top_k=1000  # Get a large sample for analytics
        )
        
        logger.info(f"Found {len(results)} chunks for analytics")
        
        if not results:
            return SECAnalyticsResponse(
                total_documents=0,
                total_chunks=0,
                companies_count=0,
                form_types_distribution={},
                chunks_per_document_avg=0,
                recent_activity=[],
                company_distribution={},
                filing_date_range={}
            )
        
        # Analyze the data
        unique_documents = set()
        form_types_distribution = {}
        companies = {}
        filing_dates = []
        
        for result in results:
            doc_id = result.get('document_id', '')
            if doc_id:
                unique_documents.add(doc_id)
            
            form_type = result.get('form_type', '')
            if form_type:
                form_types_distribution[form_type] = form_types_distribution.get(form_type, 0) + 1
            
            company = result.get('company', '')
            if company:
                companies[company] = companies.get(company, 0) + 1
            
            filing_date = result.get('filing_date', '')
            if filing_date:
                filing_dates.append(filing_date)
        
        # Recent activity (last 10 processed documents)
        recent_results = sorted(
            [r for r in results if r.get('processed_at')],
            key=lambda x: x.get('processed_at', ''),
            reverse=True
        )[:50]  # Get more results to find unique documents
        
        recent_activity = []
        seen_docs = set()
        for result in recent_results:
            doc_id = result.get('document_id', '')
            if doc_id not in seen_docs and len(recent_activity) < 10:
                recent_activity.append({
                    "document_id": doc_id,
                    "company": result.get('company', ''),
                    "form_type": result.get('form_type', ''),
                    "filing_date": result.get('filing_date', ''),
                    "processed_at": result.get('processed_at', ''),
                    "chunk_count": len([r for r in results if r.get('document_id') == doc_id])
                })
                seen_docs.add(doc_id)
        
        # Calculate date range
        filing_date_range = {}
        if filing_dates:
            filing_dates.sort()
            filing_date_range = {
                "earliest": filing_dates[0],
                "latest": filing_dates[-1]
            }
        
        total_documents = len(unique_documents)
        total_chunks = len(results)
        chunks_per_doc_avg = total_chunks / total_documents if total_documents > 0 else 0
        
        logger.info(f"Analytics: {total_documents} docs, {total_chunks} chunks, {len(companies)} companies")
        
        return SECAnalyticsResponse(
            total_documents=total_documents,
            total_chunks=total_chunks,
            companies_count=len(companies),
            form_types_distribution=form_types_distribution,
            chunks_per_document_avg=round(chunks_per_doc_avg, 2),
            recent_activity=recent_activity,
            company_distribution=companies,
            filing_date_range=filing_date_range
        )
        
    except Exception as e:
        logger.error(f"Error getting SEC analytics: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get analytics: {str(e)}")

@router.get("/test-index")
async def test_search_index(
    sec_service: SECDocumentService = Depends(get_sec_service)
):
    """
    Test endpoint to check search index status
    """
    try:
        logger.info("Testing search index")
          # Try a simple search to see if the index has any data
        try:
            # First try to get all documents without any query
            results = sec_service.azure_manager.search_client.search(
                search_text="*",
                top=10,
                select=["id", "document_id", "content", "company", "form_type"]
            )
            
            result_list = list(results)
            logger.info(f"Direct search found {len(result_list)} results")
            
            return {
                "index_status": "accessible",
                "total_results": len(result_list),
                "sample_results": result_list[:3] if result_list else [],
                "message": f"Found {len(result_list)} documents in index"
            }
            
        except Exception as search_error:
            logger.error(f"Direct search failed: {search_error}")
            return {
                "index_status": "error",
                "error": str(search_error),
                "message": "Search index is not accessible"
            }
        
    except Exception as e:
        logger.error(f"Test index error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to test index: {str(e)}")
