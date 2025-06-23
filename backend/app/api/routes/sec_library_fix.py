"""
Improved SEC Document Library endpoint with better fallback handling
"""

def get_improved_sec_document_library_logic():
    """
    Improved logic for processing search results with better fallback handling
    """
    def process_search_results(results):
        documents_by_id = {}
        total_chunks = len(results)
        
        for result in results:
            doc_id = result.get('document_id', '')
            if not doc_id:
                # Fallback: use chunk_id prefix if document_id is missing
                chunk_id = result.get('chunk_id', result.get('id', ''))
                if '_chunk_' in chunk_id:
                    doc_id = chunk_id.split('_chunk_')[0]
                else:
                    doc_id = f"unknown_{hash(str(result))[:8]}"
            
            if doc_id and doc_id not in documents_by_id:
                # Extract metadata with fallbacks for missing/empty fields
                ticker = result.get('ticker', '')
                if not ticker and doc_id:
                    # Try to extract ticker from document_id or chunk_id
                    parts = doc_id.split('_')
                    if len(parts) > 0 and len(parts[0]) <= 10:  # Likely a ticker
                        ticker = parts[0]
                
                form_type = result.get('form_type', '')
                if not form_type and doc_id:
                    # Try to extract form type from document_id or title
                    if '10-K' in str(result.get('title', '')) or '10-K' in doc_id:
                        form_type = '10-K'
                    elif '10-Q' in str(result.get('title', '')) or '10-Q' in doc_id:
                        form_type = '10-Q'
                    elif '8-K' in str(result.get('title', '')) or '8-K' in doc_id:
                        form_type = '8-K'
                
                accession_number = result.get('accession_number', '')
                if not accession_number and doc_id:
                    # Try to extract accession number from document_id
                    if '-' in doc_id and len(doc_id) > 10:
                        accession_number = doc_id
                
                cik = result.get('cik', '')
                if not cik and accession_number:
                    # Try to extract CIK from accession number format
                    parts = accession_number.split('-')
                    if len(parts) > 0 and parts[0].isdigit():
                        cik = parts[0].zfill(10)
                
                documents_by_id[doc_id] = {
                    "document_id": doc_id,
                    "company": result.get('company', 'Unknown Company') or 'Unknown Company',
                    "ticker": ticker,
                    "form_type": form_type,
                    "filing_date": result.get('filing_date', ''),
                    "accession_number": accession_number,
                    "chunk_count": 0,
                    "processed_at": result.get('processed_at', ''),
                    "source": result.get('source', 'SEC EDGAR') or 'SEC EDGAR',
                    "cik": cik
                }
            if doc_id in documents_by_id:
                documents_by_id[doc_id]["chunk_count"] += 1
        
        return documents_by_id, total_chunks
    
    return process_search_results
