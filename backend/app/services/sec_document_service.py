import asyncio
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, date
import hashlib
import json
from dataclasses import dataclass

# Configure edgartools
import edgar
edgar.set_identity("assistant@agenticrag.example.com")

from edgar import Company, Filing, get_company_facts
import re

from app.services.azure_services import AzureServiceManager
from app.core.config import settings
from app.core.observability import observability
from app.services.document_processor import DocumentChunk
from app.utils.chunker import DocumentChunker

logger = logging.getLogger(__name__)

@dataclass
class SECDocumentInfo:
    """Information about a SEC document"""
    ticker: str
    company_name: str
    cik: str
    form_type: str
    filing_date: date
    period_end_date: Optional[date]
    document_url: str
    accession_number: str
    file_size: Optional[int] = None

class SECDocumentService:
    """
    Service for retrieving and processing SEC documents using edgartools
    """
    def __init__(self, azure_manager: AzureServiceManager):
        self.azure_manager = azure_manager
        self.chunker = DocumentChunker(
            chunk_size=settings.chunk_size
        )
        
    async def search_companies(self, query: str) -> List[Dict[str, Any]]:
        """
        Search for companies by name or ticker and include available filings summary
        """
        try:
            logger.info(f"Searching for companies with query: {query}")
            
            # Always use find_company first to get the proper ticker, regardless of input
            try:
                from edgar import find_company
                
                logger.info(f"Using find_company to search for: {query}")
                search_results = find_company(query)
                
                if search_results and len(search_results) > 0:
                    # Get the best match (Company object)
                    company = search_results[0]
                    
                    # Extract ticker from the Company object
                    ticker = company.get_ticker()
                    
                    if ticker:
                        logger.info(f"Found ticker via find_company: {query} -> {ticker}")
                        # Use the Company object we already have to get info
                        return await self._get_company_info(company, ticker)
                    else:
                        logger.warning(f"Found company but no ticker: {company.name}")
                        
            except Exception as search_error:
                logger.warning(f"find_company failed for '{query}': {search_error}")
                  # Fallback to hardcoded mappings for well-known companies
            logger.info(f"Trying fallback mapping for: {query}")
            name_to_ticker = {
                'apple': 'AAPL',
                'aaple': 'AAPL',  # Common typo
                'apple inc': 'AAPL', 
                'apple computer': 'AAPL',
                'apple inc.': 'AAPL',
                'microsoft': 'MSFT',
                'microsoft corp': 'MSFT',
                'microsoft corporation': 'MSFT',
                'amazon': 'AMZN',
                'amazon.com': 'AMZN',
                'amazon inc': 'AMZN',
                'amazon.com inc': 'AMZN',
                'google': 'GOOGL',
                'alphabet': 'GOOGL',
                'alphabet inc': 'GOOGL',
                'alphabet inc.': 'GOOGL',
                'tesla': 'TSLA',
                'tesla inc': 'TSLA',
                'tesla motors': 'TSLA',
                'tesla inc.': 'TSLA',
                'facebook': 'META',
                'meta': 'META',
                'meta platforms': 'META',
                'meta platforms inc': 'META',
                'nvidia': 'NVDA',
                'nvidia corp': 'NVDA',
                'nvidia corporation': 'NVDA',
                'netflix': 'NFLX',
                'oracle': 'ORCL',
                'oracle corp': 'ORCL',
                'oracle corporation': 'ORCL',
                'salesforce': 'CRM',
                'salesforce.com': 'CRM',
                'adobe': 'ADBE',
                'adobe inc': 'ADBE',
                'adobe systems': 'ADBE',
                'intel': 'INTC',
                'intel corp': 'INTC',
                'intel corporation': 'INTC',
                'ibm': 'IBM',
                'international business machines': 'IBM',
                'cisco': 'CSCO',
                'cisco systems': 'CSCO',
                'paypal': 'PYPL',
                'paypal holdings': 'PYPL'
            }
            
            query_lower = query.lower().strip()
            if query_lower in name_to_ticker:
                logger.info(f"Found fallback mapping: {query} -> {name_to_ticker[query_lower]}")
                company = Company(name_to_ticker[query_lower])
                return await self._get_company_info(company, name_to_ticker[query_lower])
            
            logger.warning(f"Company not found for query: {query}")
            return []
                
        except Exception as e:
            logger.error(f"Error searching companies: {e}")
            return []
                
            # Try dynamic company search using Edgar's find_company functionality
            try:
                from edgar import find_company
                
                # Use Edgar's find_company to search by name
                logger.info(f"Searching for company name: {query}")
                search_results = find_company(query)
                
                if search_results and len(search_results) > 0:
                    # find_company returns CompanySearchResults - get the best match
                    best_match = search_results[0]  # Get first/best result (Company object)
                    
                    # Extract ticker from the Company object
                    ticker = None
                    try:
                        ticker = best_match.get_ticker()
                    except:
                        # Fallback to tickers attribute
                        tickers = getattr(best_match, 'tickers', None)
                        if tickers and len(tickers) > 0:
                            ticker = tickers[0]
                    
                    if ticker:
                        logger.info(f"Found ticker via search: {query} -> {ticker}")
                        # Use the Company object we already have
                        return await self._get_company_info(best_match, ticker)
                    else:
                        logger.warning(f"Found company but no ticker: {best_match.name}")
                        
            except Exception as search_error:
                logger.warning(f"Edgar find_company failed: {search_error}")
                
            # Fallback to common company name mappings for well-known companies
            logger.info(f"Trying fallback mapping for: {query}")
            name_to_ticker = {
                'apple': 'AAPL',
                'apple inc': 'AAPL', 
                'apple computer': 'AAPL',
                'apple inc.': 'AAPL',
                'microsoft': 'MSFT',
                'microsoft corp': 'MSFT',
                'microsoft corporation': 'MSFT',
                'amazon': 'AMZN',
                'amazon.com': 'AMZN',
                'amazon inc': 'AMZN',
                'amazon.com inc': 'AMZN',
                'google': 'GOOGL',
                'alphabet': 'GOOGL',
                'alphabet inc': 'GOOGL',
                'alphabet inc.': 'GOOGL',
                'tesla': 'TSLA',
                'tesla inc': 'TSLA',
                'tesla motors': 'TSLA',
                'tesla inc.': 'TSLA',
                'facebook': 'META',
                'meta': 'META',
                'meta platforms': 'META',
                'meta platforms inc': 'META',
                'nvidia': 'NVDA',
                'nvidia corp': 'NVDA',
                'nvidia corporation': 'NVDA',
                'netflix': 'NFLX',
                'oracle': 'ORCL',
                'oracle corp': 'ORCL',
                'oracle corporation': 'ORCL',
                'salesforce': 'CRM',
                'salesforce.com': 'CRM',
                'adobe': 'ADBE',
                'adobe inc': 'ADBE',
                'adobe systems': 'ADBE',
                'intel': 'INTC',
                'intel corp': 'INTC',
                'intel corporation': 'INTC',
                'ibm': 'IBM',
                'international business machines': 'IBM',
                'cisco': 'CSCO',
                'cisco systems': 'CSCO',
                'paypal': 'PYPL',
                'paypal holdings': 'PYPL'
            }
            
            query_lower = query.lower().strip()
            if query_lower in name_to_ticker:
                logger.info(f"Found fallback mapping: {query} -> {name_to_ticker[query_lower]}")
                company = Company(name_to_ticker[query_lower])
                return await self._get_company_info(company, name_to_ticker[query_lower])
            
            logger.warning(f"Company not found for query: {query}")
            return []
                
        except Exception as e:
            logger.error(f"Error searching companies: {e}")
            return []

    async def _get_company_info(self, company, ticker: str) -> List[Dict[str, Any]]:
        """
        Extract company information and filing summary
        """
        try:
            # Get recent filings to show available document types and years
            recent_filings = company.get_filings(form=['10-K', '10-Q', '8-K', '10-Q/A', '10-K/A']).head(50)
            
            # Organize filings by type and year
            filings_summary = {}
            available_years = set()
            
            for filing in recent_filings:
                form_type = filing.form
                filing_year = filing.filing_date.year
                available_years.add(filing_year)
                
                if form_type not in filings_summary:
                    filings_summary[form_type] = {
                        "count": 0,
                        "years": set(),
                        "latest_date": None
                    }
                
                filings_summary[form_type]["count"] += 1
                filings_summary[form_type]["years"].add(filing_year)
                
                if (filings_summary[form_type]["latest_date"] is None or 
                    filing.filing_date > filings_summary[form_type]["latest_date"]):
                    filings_summary[form_type]["latest_date"] = filing.filing_date
            
            # Convert sets to sorted lists for JSON serialization
            for form_type in filings_summary:
                filings_summary[form_type]["years"] = sorted(list(filings_summary[form_type]["years"]), reverse=True)
                if filings_summary[form_type]["latest_date"]:
                    filings_summary[form_type]["latest_date"] = filings_summary[form_type]["latest_date"].isoformat()
            
            company_info = {
                "ticker": ticker,
                "company_name": company.name,
                "cik": str(company.cik).zfill(10),  # Ensure CIK is 10 digits with leading zeros
                "industry": getattr(company, 'industry', 'Unknown'),
                "sic": str(getattr(company, 'sic', 'Unknown')),
                "entity_type": getattr(company, 'entity_type', 'Unknown'),
                "available_years": sorted(list(available_years), reverse=True),
                "available_forms": filings_summary,
                "total_filings": len(recent_filings)
            }
            
            logger.info(f"Found company: {company.name} with {len(recent_filings)} recent filings")
            return [company_info]
            
        except Exception as e:
            logger.error(f"Error getting company info: {e}")
            return []
    
    async def get_company_filings(
        self, 
        ticker: str, 
        form_types: List[str] = None, 
        limit: int = 20
    ) -> List[SECDocumentInfo]:
        """
        Get recent filings for a company
        """
        try:
            logger.info(f"Getting filings for {ticker}, forms: {form_types}")
            
            if form_types is None:
                form_types = ['10-K', '10-Q', '8-K']
            
            company = Company(ticker.upper())
            filings = company.get_filings(form=form_types).head(limit)
            
            documents = []
            for filing in filings:
                doc_info = SECDocumentInfo(
                    ticker=ticker.upper(),
                    company_name=company.name,
                    cik=str(company.cik),
                    form_type=filing.form,
                    filing_date=filing.filing_date,
                    period_end_date=getattr(filing, 'period_end_date', None),
                    document_url=getattr(filing, 'document_url', ''),
                    accession_number=filing.accession_number,
                    file_size=getattr(filing, 'size', None)
                )
                documents.append(doc_info)
            
            logger.info(f"Found {len(documents)} filings for {ticker}")
            return documents            
        except Exception as e:
            logger.error(f"Error getting filings for {ticker}: {e}")
            return []

    async def retrieve_and_process_document(
        self, 
        ticker: str, 
        accession_number: str,
        document_id: Optional[str] = None    ) -> Dict[str, Any]:
        """
        Retrieve SEC document and process it into chunks using md2chunks
        """
        try:
            logger.info(f"Retrieving SEC document: {ticker} - {accession_number}")
            
            # Check if document already exists in the index
            document_exists = await self.azure_manager.check_document_exists(accession_number)
            if document_exists:
                logger.info(f"Document {accession_number} already exists in index, skipping processing")
                return {
                    "document_id": f"{ticker}_{accession_number}",
                    "chunks_created": 0,
                    "metadata": {"status": "already_exists", "message": "Document already processed and indexed"},
                    "filing_info": {"ticker": ticker, "accession_number": accession_number},
                    "skipped": True
                }
                
            # Get the company and filing
            company = Company(ticker.upper())
            
            # Get all filings and find the one with matching accession number
            all_filings = company.get_filings()
            filing = None
            for f in all_filings:
                if f.accession_number == accession_number:
                    filing = f
                    break
            
            if not filing:
                raise ValueError(f"Filing not found: {accession_number}")
              # Extract document metadata
            metadata = {
                "ticker": ticker.upper(),
                "company_name": company.name,
                "cik": str(company.cik).zfill(10),  # Ensure CIK is 10 digits with leading zeros
                "industry": getattr(company, 'industry', 'Unknown'),
                "sic": str(getattr(company, 'sic', 'Unknown')),
                "entity_type": getattr(company, 'entity_type', 'Unknown'),
                "form_type": filing.form,
                "filing_date": filing.filing_date.isoformat(),
                "period_end_date": filing.period_end_date.isoformat() if hasattr(filing, 'period_end_date') and filing.period_end_date else None,
                "accession_number": accession_number,
                "document_type": "SEC Filing",
                "source": "SEC EDGAR",
                "file_size": getattr(filing, 'size', None),
                "document_url": getattr(filing.document, 'url', '') if hasattr(filing, 'document') and filing.document else ''
            }
            
            logger.info(f"Document metadata: {metadata}")
            
            # Get the document content
            # edgartools provides different ways to get content
            if hasattr(filing, 'text'):
                content = filing.text()
            elif hasattr(filing, 'html'):
                content = filing.html()
            else:
                content = str(filing)
            
            logger.info(f"Retrieved content length: {len(content)} characters")
            
            # Process the document into chunks
            document_id = document_id or f"{ticker}_{accession_number}_{filing.form}"
            chunks = await self._process_content_with_md2chunks(
                content, document_id, metadata
            )
            
            # Generate embeddings for chunks
            await self._generate_embeddings_for_chunks(chunks)
            
            # Store in Azure Search
            search_documents = await self._prepare_search_documents(chunks)
            if search_documents:
                await self.azure_manager.add_documents_to_index(search_documents)
                logger.info(f"Added {len(search_documents)} documents to search index")
            
            logger.info(f"Successfully processed document with {len(chunks)} chunks")
            
            return {
                "document_id": document_id,
                "chunks_created": len(chunks),
                "metadata": metadata,
                "filing_info": {
                    "form_type": filing.form,
                    "filing_date": filing.filing_date.isoformat(),
                    "accession_number": accession_number,
                    "company_name": company.name,
                    "ticker": ticker.upper()
                }
            }
            
        except Exception as e:
            logger.error(f"Error retrieving and processing SEC document: {e}")
            # observability.record_error("sec_document_processing_error", str(e))
            raise
    
    async def _process_content_with_md2chunks(
        self, 
        content: str, 
        document_id: str, 
        metadata: Dict[str, Any]
    ) -> List[DocumentChunk]:
        """
        Process content using md2chunks for intelligent chunking
        """
        try:
            logger.info(f"Processing content with md2chunks, length: {len(content)}")
            
            # Clean and prepare content for markdown processing
            cleaned_content = self._clean_sec_content(content)
            
            # Convert to markdown if it's HTML
            if content.strip().startswith('<'):
                markdown_content = self._html_to_markdown(cleaned_content)
            else:
                markdown_content = cleaned_content
            
            logger.info(f"Markdown content length: {len(markdown_content)}")
            
            # Use md2chunks to intelligently chunk the content
            try:                # Configure chunker for financial documents
                chunks_data = self.chunker.chunk(
                    markdown_content,
                    chunk_size=settings.chunk_size,  # Use configured chunk size
                    overlap=settings.chunk_overlap,  # Use configured overlap
                    separator="\n\n"  # Prefer paragraph breaks
                )
                logger.info(f"md2chunks created {len(chunks_data)} chunks")
            except Exception as e:
                logger.warning(f"md2chunks failed: {e}, falling back to simple chunking")
                chunks_data = self._fallback_chunking(markdown_content)
              # Convert to DocumentChunk objects
            document_chunks = []
            for i, chunk_data in enumerate(chunks_data):
                # Handle md2chunks Attachment objects
                if hasattr(chunk_data, 'content'):
                    # This is an Attachment object from md2chunks
                    chunk_content = chunk_data.content
                    chunk_metadata = getattr(chunk_data, 'metadata', {}) if hasattr(chunk_data, 'metadata') else {}
                elif isinstance(chunk_data, dict):
                    # This is a dictionary (from fallback chunking)
                    chunk_content = chunk_data.get('content', str(chunk_data))
                    chunk_metadata = chunk_data.get('metadata', {})
                else:
                    # This is a string (from fallback chunking)
                    chunk_content = str(chunk_data)
                    chunk_metadata = {}
                
                if len(chunk_content.strip()) < 50:  # Skip very short chunks
                    continue
                
                # Extract section information from chunk
                section_info = self._extract_section_info(chunk_content)
                
                # Combine metadata
                final_metadata = {
                    **metadata,
                    **chunk_metadata,
                    "chunk_index": i,
                    "section_type": section_info.get("section_type", "unknown"),
                    "content_type": "markdown",
                    "chunk_method": "md2chunks"
                }
                
                chunk = DocumentChunk(
                    chunk_id=f"{document_id}_chunk_{i}",
                    content=chunk_content,
                    metadata=final_metadata,
                    citation_info={
                        "section_type": section_info.get("section_type"),
                        "document_type": metadata.get("form_type"),
                        "company": metadata.get("company_name"),
                        "ticker": metadata.get("ticker"),
                        "filing_date": metadata.get("filing_date")
                    }
                )
                document_chunks.append(chunk)
            
            logger.info(f"Created {len(document_chunks)} DocumentChunk objects")
            return document_chunks
            
        except Exception as e:
            logger.error(f"Error processing content with md2chunks: {e}")
            raise
    
    def _clean_sec_content(self, content: str) -> str:
        """
        Clean SEC document content for better processing
        """
        # Remove excessive whitespace
        content = re.sub(r'\s+', ' ', content)
        
        # Remove common SEC formatting artifacts
        content = re.sub(r'Table of Contents', '', content, flags=re.IGNORECASE)
        content = re.sub(r'\n\s*\n\s*\n', '\n\n', content)  # Multiple newlines to double
        
        # Remove page numbers and headers/footers patterns
        content = re.sub(r'\n\s*\d+\s*\n', '\n', content)  # Standalone page numbers
        content = re.sub(r'\n\s*-\s*\d+\s*-\s*\n', '\n', content)  # -1- style page numbers
        
        return content.strip()
    
    def _html_to_markdown(self, html_content: str) -> str:
        """
        Convert HTML SEC content to markdown
        """
        try:
            from markdownify import markdownify as md
            return md(html_content, heading_style="ATX")
        except ImportError:
            logger.warning("markdownify not available, using basic HTML cleaning")
            # Basic HTML tag removal
            import re
            content = re.sub(r'<[^>]+>', '', html_content)
            content = re.sub(r'&nbsp;', ' ', content)
            content = re.sub(r'&[a-zA-Z]+;', '', content)
            return content
    
    def _fallback_chunking(self, content: str) -> List[str]:
        """
        Simple fallback chunking if md2chunks fails
        """
        chunks = []
        lines = content.split('\n')
        current_chunk = []
        current_size = 0
        max_size = settings.chunk_size  # Use setting instead of hardcoded value
        
        for line in lines:
            if current_size + len(line) > max_size and current_chunk:
                chunks.append('\n'.join(current_chunk))
                current_chunk = []
                current_size = 0
            
            current_chunk.append(line)
            current_size += len(line)
        
        if current_chunk:
            chunks.append('\n'.join(current_chunk))
        
        return chunks
    
    def _extract_section_info(self, chunk_content: str) -> Dict[str, str]:
        """
        Extract section information from chunk content
        """
        # SEC document section patterns
        section_patterns = {
            r'(?i)item\s+1[^0-9].*business': 'business_overview',
            r'(?i)item\s+1a.*risk\s+factors': 'risk_factors',
            r'(?i)item\s+2.*properties': 'properties',
            r'(?i)item\s+3.*legal': 'legal_proceedings',
            r'(?i)item\s+7.*management.*discussion': 'mda',
            r'(?i)item\s+8.*financial\s+statements': 'financial_statements',
            r'(?i)consolidated\s+balance\s+sheets?': 'balance_sheet',
            r'(?i)consolidated\s+income\s+statements?': 'income_statement',
            r'(?i)consolidated\s+cash\s+flows?': 'cash_flow',
            r'(?i)notes?\s+to.*financial\s+statements?': 'financial_notes'        }
        
        for pattern, section_type in section_patterns.items():
            if re.search(pattern, chunk_content[:200]):  # Check first 200 chars
                return {"section_type": section_type}
        
        return {"section_type": "general"}
    
    async def _generate_embeddings_for_chunks(self, chunks: List[DocumentChunk]):
        """
        Generate embeddings for document chunks
        """
        try:
            logger.info(f"Generating embeddings for {len(chunks)} chunks")
            
            for i, chunk in enumerate(chunks):
                try:
                    embedding = await self.azure_manager.get_embedding(chunk.content)
                    chunk.embedding = embedding
                    logger.info(f"Generated embedding for chunk {i+1}/{len(chunks)}")
                except Exception as e:
                    logger.warning(f"Failed to generate embedding for chunk {i}: {e}")
                    chunk.embedding = None
                    
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
    
    async def _prepare_search_documents(self, chunks: List[DocumentChunk]) -> List[Dict[str, Any]]:
        """
        Prepare documents for Azure Search indexing with proper schema mapping
        """
        search_documents = []
        
        for chunk in chunks:
            if chunk.embedding is None:
                continue  # Skip chunks without embeddings
                
            # Extract metadata
            metadata = chunk.metadata
            
            # Use accession number as document_id for grouping
            document_id = metadata.get('accession_number', chunk.chunk_id.split('_')[0] if '_' in chunk.chunk_id else chunk.chunk_id)
            
            search_doc = {
                "id": chunk.chunk_id,  # Primary key
                "content": chunk.content,
                "title": f"{metadata.get('company_name', 'Unknown')} - {metadata.get('form_type', 'Unknown')}",
                "document_id": document_id,  # Document grouping identifier
                "source": metadata.get('source', 'SEC EDGAR'),
                "chunk_id": chunk.chunk_id,  # Also include as chunk_id field for filtering
                "document_type": metadata.get('document_type', 'SEC Filing'),
                "company": metadata.get('company_name', ''),
                "filing_date": metadata.get('filing_date', ''),
                "section_type": metadata.get('section_type', 'general'),
                "page_number": metadata.get('page_number', 0),
                "credibility_score": metadata.get('credibility_score', 0.5),
                "processed_at": datetime.now().isoformat(),
                "citation_info": json.dumps({
                    "ticker": metadata.get('ticker', ''),
                    "company_name": metadata.get('company_name', ''),
                    "form_type": metadata.get('form_type', ''),
                    "filing_date": metadata.get('filing_date', ''),
                    "accession_number": metadata.get('accession_number', ''),
                    "chunk_index": metadata.get('chunk_index', 0)
                }),
                # SEC-specific fields from Edgar tools - ensure these are all populated
                "ticker": metadata.get('ticker', ''),
                "cik": metadata.get('cik', ''),
                "industry": metadata.get('industry', ''),
                "sic": metadata.get('sic', ''),
                "entity_type": metadata.get('entity_type', ''),
                "form_type": metadata.get('form_type', ''),
                "accession_number": metadata.get('accession_number', ''),
                "period_end_date": metadata.get('period_end_date', ''),
                "chunk_index": metadata.get('chunk_index', 0),
                "content_type": metadata.get('content_type', 'text'),
                "chunk_method": metadata.get('chunk_method', 'unknown'),
                "file_size": metadata.get('file_size', 0) if metadata.get('file_size') is not None else 0,
                "document_url": metadata.get('document_url', ''),
                "content_vector": chunk.embedding
            }
            search_documents.append(search_doc)
        
        return search_documents
    
    async def get_specific_filings(
        self, 
        ticker: str, 
        form_types: List[str] = None, 
        years: List[int] = None,
        limit: int = 50
    ) -> List[SECDocumentInfo]:
        """
        Get specific filings filtered by form types and years with improved company name handling
        """
        try:
            logger.info(f"Getting specific filings for {ticker}, forms: {form_types}, years: {years}")
            
            # First try to find the company (handles name-to-ticker mapping)
            companies = await self.search_companies(ticker)
            if not companies:
                logger.warning(f"No company found for query: {ticker}")
                return []
            
            # Use the first company's ticker
            actual_ticker = companies[0]['ticker']
            logger.info(f"Using ticker: {actual_ticker} for query: {ticker}")
            
            if form_types is None:
                form_types = ['10-K', '10-Q', '8-K']
            
            company = Company(actual_ticker)
            
            # Get more filings if we need to filter by years
            fetch_limit = limit * 3 if years else limit
            filings = company.get_filings(form=form_types).head(fetch_limit)
            
            documents = []
            for filing in filings:
                # Filter by years if specified
                if years and filing.filing_date.year not in years:
                    continue
                    
                doc_info = SECDocumentInfo(
                    ticker=actual_ticker,
                    company_name=company.name,
                    cik=str(company.cik),
                    form_type=filing.form,
                    filing_date=filing.filing_date,
                    period_end_date=getattr(filing, 'period_end_date', None),
                    document_url=getattr(filing, 'document_url', ''),
                    accession_number=filing.accession_number,
                    file_size=getattr(filing, 'size', None)
                )
                documents.append(doc_info)
                
                # Stop if we have enough results
                if len(documents) >= limit:
                    break
            
            logger.info(f"Found {len(documents)} filings for {actual_ticker}")
            return documents
            
        except Exception as e:
            logger.error(f"Error getting specific filings for {ticker}: {e}")
            return []
