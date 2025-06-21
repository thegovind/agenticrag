import asyncio
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import hashlib
import re
from dataclasses import dataclass
import json

from app.services.azure_services import AzureServiceManager
from app.core.config import settings
from app.core.observability import observability

logger = logging.getLogger(__name__)

@dataclass
class DocumentChunk:
    chunk_id: str
    content: str
    metadata: Dict[str, Any]
    embedding: Optional[List[float]] = None
    citation_info: Optional[Dict[str, Any]] = None

@dataclass
class FinancialTable:
    table_id: str
    title: str
    content: str
    rows: List[List[str]]
    headers: List[str]
    page_number: int
    bounding_box: Optional[Dict] = None

@dataclass
class FinancialSection:
    section_id: str
    title: str
    content: str
    subsections: List['FinancialSection']
    tables: List[FinancialTable]
    page_range: Tuple[int, int]
    section_type: str  # 'business', 'risk_factors', 'financial_statements', etc.

class DocumentProcessor:
    """
    Enhanced document processing service for financial documents
    Handles extraction, chunking, and preparation for knowledge base
    Specialized for 10-K/10-Q and other SEC filings
    """
    
    def __init__(self, azure_manager: AzureServiceManager):
        self.azure_manager = azure_manager
        self.financial_section_patterns = self._initialize_section_patterns()
        self.financial_metrics_patterns = self._initialize_metrics_patterns()
        
    async def process_document(self, content: bytes, content_type: str, 
                             source: str, metadata: Dict = None) -> Dict:
        """
        Enhanced document processing pipeline for financial documents
        
        Args:
            content: Raw document bytes
            content_type: MIME type of the document
            source: Source identifier
            metadata: Additional metadata
            
        Returns:
            Dict containing processed document information with enhanced structure
        """
        try:
            logger.info(f"Processing financial document from source: {source}")
            observability.track_document_processing_start(source, content_type)
            
            extracted_content = await self.azure_manager.analyze_document(content, content_type)
            
            document_id = self._generate_document_id(source, extracted_content["content"])
            
            financial_info = await self._extract_comprehensive_financial_info(
                extracted_content, source
            )
            
            document_structure = await self._parse_financial_document_structure(
                extracted_content, financial_info
            )
            
            chunks = await self._create_hierarchical_financial_chunks(
                document_structure, 
                document_id,
                {**(metadata or {}), **financial_info}
            )
            
            for chunk in chunks:
                chunk.embedding = await self.azure_manager.get_embedding(chunk.content)
            
            key_metrics = await self._extract_key_financial_metrics(
                extracted_content["content"], financial_info
            )
            
            processed_doc = {
                "document_id": document_id,
                "source": source,
                "content_type": content_type,
                "extracted_content": extracted_content,
                "financial_info": financial_info,
                "document_structure": self._structure_to_dict(document_structure),
                "key_metrics": key_metrics,
                "chunks": [self._chunk_to_dict(chunk) for chunk in chunks],
                "processed_at": datetime.utcnow().isoformat(),
                "metadata": {**(metadata or {}), **financial_info},
                "processing_stats": {
                    "total_chunks": len(chunks),
                    "sections_found": len(document_structure.get("sections", [])),
                    "tables_found": len(document_structure.get("tables", [])),
                    "metrics_extracted": len(key_metrics)
                }
            }
            
            observability.track_document_processing_complete(
                document_id, len(chunks), len(key_metrics)
            )
            
            logger.info(f"Successfully processed document {document_id} with {len(chunks)} chunks, "
                       f"{len(document_structure.get('sections', []))} sections, "
                       f"{len(document_structure.get('tables', []))} tables")
            
            return processed_doc
            
        except Exception as e:
            logger.error(f"Error processing document from {source}: {e}")
            observability.track_document_processing_error(source, str(e))
            raise
    
    def _initialize_section_patterns(self) -> Dict[str, List[str]]:
        """Initialize patterns for identifying financial document sections"""
        return {
            "10k_sections": [
                r"PART\s+I\s*ITEM\s+1[.\s]*BUSINESS",
                r"PART\s+I\s*ITEM\s+1A[.\s]*RISK\s+FACTORS",
                r"PART\s+I\s*ITEM\s+1B[.\s]*UNRESOLVED\s+STAFF\s+COMMENTS",
                r"PART\s+I\s*ITEM\s+2[.\s]*PROPERTIES",
                r"PART\s+I\s*ITEM\s+3[.\s]*LEGAL\s+PROCEEDINGS",
                r"PART\s+I\s*ITEM\s+4[.\s]*MINE\s+SAFETY\s+DISCLOSURES",
                r"PART\s+II\s*ITEM\s+5[.\s]*MARKET\s+FOR\s+REGISTRANT'S\s+COMMON\s+EQUITY",
                r"PART\s+II\s*ITEM\s+6[.\s]*SELECTED\s+FINANCIAL\s+DATA",
                r"PART\s+II\s*ITEM\s+7[.\s]*MANAGEMENT'S\s+DISCUSSION\s+AND\s+ANALYSIS",
                r"PART\s+II\s*ITEM\s+7A[.\s]*QUANTITATIVE\s+AND\s+QUALITATIVE\s+DISCLOSURES",
                r"PART\s+II\s*ITEM\s+8[.\s]*FINANCIAL\s+STATEMENTS\s+AND\s+SUPPLEMENTARY\s+DATA",
                r"PART\s+II\s*ITEM\s+9[.\s]*CHANGES\s+IN\s+AND\s+DISAGREEMENTS",
                r"PART\s+II\s*ITEM\s+9A[.\s]*CONTROLS\s+AND\s+PROCEDURES",
                r"PART\s+II\s*ITEM\s+9B[.\s]*OTHER\s+INFORMATION"
            ],
            "10q_sections": [
                r"PART\s+I\s*ITEM\s+1[.\s]*FINANCIAL\s+STATEMENTS",
                r"PART\s+I\s*ITEM\s+2[.\s]*MANAGEMENT'S\s+DISCUSSION\s+AND\s+ANALYSIS",
                r"PART\s+I\s*ITEM\s+3[.\s]*QUANTITATIVE\s+AND\s+QUALITATIVE\s+DISCLOSURES",
                r"PART\s+I\s*ITEM\s+4[.\s]*CONTROLS\s+AND\s+PROCEDURES",
                r"PART\s+II\s*ITEM\s+1[.\s]*LEGAL\s+PROCEEDINGS",
                r"PART\s+II\s*ITEM\s+1A[.\s]*RISK\s+FACTORS",
                r"PART\s+II\s*ITEM\s+2[.\s]*UNREGISTERED\s+SALES\s+OF\s+EQUITY\s+SECURITIES",
                r"PART\s+II\s*ITEM\s+3[.\s]*DEFAULTS\s+UPON\s+SENIOR\s+SECURITIES",
                r"PART\s+II\s*ITEM\s+4[.\s]*MINE\s+SAFETY\s+DISCLOSURES",
                r"PART\s+II\s*ITEM\s+5[.\s]*OTHER\s+INFORMATION",
                r"PART\s+II\s*ITEM\s+6[.\s]*EXHIBITS"
            ],
            "financial_statements": [
                r"CONSOLIDATED\s+BALANCE\s+SHEETS?",
                r"CONSOLIDATED\s+STATEMENTS?\s+OF\s+OPERATIONS",
                r"CONSOLIDATED\s+STATEMENTS?\s+OF\s+INCOME",
                r"CONSOLIDATED\s+STATEMENTS?\s+OF\s+COMPREHENSIVE\s+INCOME",
                r"CONSOLIDATED\s+STATEMENTS?\s+OF\s+CASH\s+FLOWS?",
                r"CONSOLIDATED\s+STATEMENTS?\s+OF\s+STOCKHOLDERS?\s+EQUITY",
                r"NOTES?\s+TO\s+CONSOLIDATED\s+FINANCIAL\s+STATEMENTS"
            ]
        }
    
    def _initialize_metrics_patterns(self) -> Dict[str, List[str]]:
        """Initialize patterns for extracting financial metrics"""
        return {
            "revenue_patterns": [
                r"(?:TOTAL\s+)?(?:NET\s+)?(?:REVENUES?|SALES?)[:\s]*\$?\s*([\d,]+(?:\.\d+)?)",
                r"(?:OPERATING\s+)?REVENUES?[:\s]*\$?\s*([\d,]+(?:\.\d+)?)",
            ],
            "income_patterns": [
                r"NET\s+INCOME[:\s]*\$?\s*([\d,]+(?:\.\d+)?)",
                r"(?:OPERATING\s+)?INCOME[:\s]*\$?\s*([\d,]+(?:\.\d+)?)",
                r"EARNINGS?\s+(?:PER\s+SHARE)?[:\s]*\$?\s*([\d,]+(?:\.\d+)?)"
            ],
            "asset_patterns": [
                r"TOTAL\s+ASSETS[:\s]*\$?\s*([\d,]+(?:\.\d+)?)",
                r"CURRENT\s+ASSETS[:\s]*\$?\s*([\d,]+(?:\.\d+)?)",
                r"CASH\s+AND\s+(?:CASH\s+)?EQUIVALENTS[:\s]*\$?\s*([\d,]+(?:\.\d+)?)"
            ],
            "liability_patterns": [
                r"TOTAL\s+LIABILITIES[:\s]*\$?\s*([\d,]+(?:\.\d+)?)",
                r"CURRENT\s+LIABILITIES[:\s]*\$?\s*([\d,]+(?:\.\d+)?)",
                r"LONG[- ]TERM\s+DEBT[:\s]*\$?\s*([\d,]+(?:\.\d+)?)"
            ]
        }
    
    def _generate_document_id(self, source: str, content: str) -> str:
        """Generate unique document ID based on source and content"""
        content_hash = hashlib.md5(content.encode()).hexdigest()[:12]
        source_hash = hashlib.md5(source.encode()).hexdigest()[:8]
        return f"doc_{source_hash}_{content_hash}"
    
    async def _extract_comprehensive_financial_info(self, extracted_content: Dict, source: str) -> Dict[str, Any]:
        """Enhanced financial information extraction with comprehensive metadata"""
        content = extracted_content["content"]
        
        financial_info = {
            "document_type": "unknown",
            "company_name": None,
            "cik": None,
            "ticker_symbol": None,
            "filing_date": None,
            "period_end_date": None,
            "fiscal_year": None,
            "fiscal_quarter": None,
            "amendment": False,
            "key_metrics": [],
            "sections": [],
            "page_count": extracted_content.get("pages", 0),
            "has_tables": len(extracted_content.get("tables", [])) > 0,
            "has_key_value_pairs": len(extracted_content.get("key_value_pairs", {})) > 0
        }
        
        try:
            content_upper = content.upper()
            if "FORM 10-K" in content_upper or "10-K" in content_upper:
                financial_info["document_type"] = "10-K"
                if "/A" in content_upper or "AMENDMENT" in content_upper:
                    financial_info["amendment"] = True
            elif "FORM 10-Q" in content_upper or "10-Q" in content_upper:
                financial_info["document_type"] = "10-Q"
                if "/A" in content_upper or "AMENDMENT" in content_upper:
                    financial_info["amendment"] = True
            elif "FORM 8-K" in content_upper:
                financial_info["document_type"] = "8-K"
            elif "PROXY STATEMENT" in content_upper or "DEF 14A" in content_upper:
                financial_info["document_type"] = "proxy-statement"
            elif "ANNUAL REPORT" in content_upper:
                financial_info["document_type"] = "annual-report"
            elif "EARNINGS" in content_upper:
                financial_info["document_type"] = "earnings-report"
            
            company_patterns = [
                r"COMPANY\s+NAME[:\s]+([A-Z][A-Za-z\s&,\.Inc]+?)(?:\n|$)",
                r"REGISTRANT[:\s]+([A-Z][A-Za-z\s&,\.Inc]+?)(?:\n|$)",
                r"^([A-Z][A-Za-z\s&,\.Inc]+?)\s+(?:FORM|10-[KQ])",
                r"COMMISSION\s+FILE\s+NUMBER[:\s]+[\d-]+\s*\n\s*([A-Z][A-Za-z\s&,\.Inc]+)"
            ]
            
            for pattern in company_patterns:
                match = re.search(pattern, content[:3000], re.MULTILINE | re.IGNORECASE)
                if match:
                    company_name = match.group(1).strip()
                    company_name = re.sub(r'\s+(INC\.?|CORP\.?|LLC\.?|LTD\.?)$', '', company_name, flags=re.IGNORECASE)
                    financial_info["company_name"] = company_name
                    break
            
            cik_match = re.search(r"CENTRAL\s+INDEX\s+KEY[:\s]+(\d+)", content[:2000], re.IGNORECASE)
            if cik_match:
                financial_info["cik"] = cik_match.group(1)
            
            ticker_patterns = [
                r"TRADING\s+SYMBOL[:\s]+([A-Z]{1,5})",
                r"TICKER\s+SYMBOL[:\s]+([A-Z]{1,5})",
                r"NASDAQ[:\s]+([A-Z]{1,5})",
                r"NYSE[:\s]+([A-Z]{1,5})"
            ]
            
            for pattern in ticker_patterns:
                match = re.search(pattern, content[:2000], re.IGNORECASE)
                if match:
                    financial_info["ticker_symbol"] = match.group(1)
                    break
            
            date_patterns = [
                r"FILING\s+DATE[:\s]+(\d{4}-\d{2}-\d{2})",
                r"DATE\s+OF\s+REPORT[:\s]+(\d{1,2}/\d{1,2}/\d{4})",
                r"PERIOD\s+END\s+DATE[:\s]+(\d{1,2}/\d{1,2}/\d{4})",
                r"(\d{4}-\d{2}-\d{2})"
            ]
            
            for pattern in date_patterns:
                match = re.search(pattern, content[:2000], re.IGNORECASE)
                if match:
                    financial_info["filing_date"] = match.group(1)
                    break
            
            period_patterns = [
                r"(?:FOR\s+THE\s+)?(?:QUARTER|PERIOD)\s+ENDED?\s+([A-Z]+\s+\d{1,2},\s+\d{4})",
                r"(?:FOR\s+THE\s+)?(?:YEAR)\s+ENDED?\s+([A-Z]+\s+\d{1,2},\s+\d{4})",
                r"PERIOD\s+END\s+DATE[:\s]+(\d{1,2}/\d{1,2}/\d{4})"
            ]
            
            for pattern in period_patterns:
                match = re.search(pattern, content[:2000], re.IGNORECASE)
                if match:
                    financial_info["period_end_date"] = match.group(1)
                    break
            
            fiscal_year_match = re.search(r"FISCAL\s+YEAR[:\s]+(\d{4})", content[:3000], re.IGNORECASE)
            if fiscal_year_match:
                financial_info["fiscal_year"] = fiscal_year_match.group(1)
            
            quarter_match = re.search(r"(?:FIRST|SECOND|THIRD|FOURTH|Q[1-4])\s+QUARTER", content[:2000], re.IGNORECASE)
            if quarter_match:
                quarter_text = quarter_match.group(0).upper()
                if "FIRST" in quarter_text or "Q1" in quarter_text:
                    financial_info["fiscal_quarter"] = "Q1"
                elif "SECOND" in quarter_text or "Q2" in quarter_text:
                    financial_info["fiscal_quarter"] = "Q2"
                elif "THIRD" in quarter_text or "Q3" in quarter_text:
                    financial_info["fiscal_quarter"] = "Q3"
                elif "FOURTH" in quarter_text or "Q4" in quarter_text:
                    financial_info["fiscal_quarter"] = "Q4"
            
            doc_type = financial_info["document_type"]
            if doc_type in ["10-K", "10-Q"]:
                section_patterns = self.financial_section_patterns.get(f"{doc_type.lower()}_sections", [])
                found_sections = []
                
                for pattern in section_patterns:
                    if re.search(pattern, content_upper):
                        section_name = pattern.split(r'\s+')[-1].lower().replace(r'\s+', '_')
                        found_sections.append(section_name)
                
                financial_info["sections"] = found_sections
            
            fs_patterns = self.financial_section_patterns.get("financial_statements", [])
            financial_statements = []
            for pattern in fs_patterns:
                if re.search(pattern, content_upper):
                    stmt_name = pattern.replace(r'\s+', '_').replace(r'\?', '').lower()
                    financial_statements.append(stmt_name)
            
            financial_info["financial_statements"] = financial_statements
            
        except Exception as e:
            logger.error(f"Error extracting comprehensive financial information: {e}")
        
        return financial_info
    
    async def _create_financial_chunks(self, content: str, document_id: str, 
                                     metadata: Dict) -> List[DocumentChunk]:
        """Create chunks optimized for financial document content"""
        chunks = []
        
        sections = self._split_into_sections(content)
        
        chunk_index = 0
        for section_name, section_content in sections.items():
            section_chunks = self._split_section_into_chunks(
                section_content, 
                settings.MAX_CHUNK_SIZE,
                settings.CHUNK_OVERLAP
            )
            
            for i, chunk_content in enumerate(section_chunks):
                chunk_id = f"{document_id}_chunk_{chunk_index}"
                
                chunk_metadata = {
                    **metadata,
                    "chunk_index": chunk_index,
                    "section": section_name,
                    "section_chunk_index": i,
                    "total_section_chunks": len(section_chunks)
                }
                
                chunk = DocumentChunk(
                    chunk_id=chunk_id,
                    content=chunk_content.strip(),
                    metadata=chunk_metadata
                )
                
                chunks.append(chunk)
                chunk_index += 1
        
        logger.info(f"Created {len(chunks)} chunks for document {document_id}")
        return chunks
    
    def _split_into_sections(self, content: str) -> Dict[str, str]:
        """Split document content into logical sections"""
        sections = {"main": content}  # Default fallback
        
        section_headers = [
            r"PART\s+[IVX]+[.\s]*([A-Z][A-Z\s,&]+)",
            r"ITEM\s+\d+[A-Z]*[.\s]*([A-Z][A-Z\s,&]+)",
            r"^([A-Z][A-Z\s,&]{10,})\s*$",  # All caps headers
            r"^\d+\.\s*([A-Z][A-Za-z\s,&]+)$"  # Numbered sections
        ]
        
        try:
            current_section = "introduction"
            current_content = []
            sections = {}
            
            lines = content.split('\n')
            
            for line in lines:
                line = line.strip()
                if not line:
                    current_content.append(line)
                    continue
                
                is_header = False
                for pattern in section_headers:
                    match = re.match(pattern, line, re.IGNORECASE)
                    if match and len(line) < 100:  # Reasonable header length
                        if current_content:
                            sections[current_section] = '\n'.join(current_content)
                        
                        current_section = match.group(1).lower().replace(' ', '_')
                        current_content = []
                        is_header = True
                        break
                
                if not is_header:
                    current_content.append(line)
            
            if current_content:
                sections[current_section] = '\n'.join(current_content)
            
            if len(sections) <= 1:
                sections = {"main": content}
                
        except Exception as e:
            logger.error(f"Error splitting into sections: {e}")
            sections = {"main": content}
        
        return sections
    
    def _split_section_into_chunks(self, content: str, max_size: int, overlap: int) -> List[str]:
        """Split a section into smaller chunks with overlap"""
        if len(content) <= max_size:
            return [content]
        
        chunks = []
        start = 0
        
        while start < len(content):
            end = start + max_size
            
            if end < len(content):
                sentence_end = content.rfind('.', start + max_size - 200, end)
                if sentence_end > start:
                    end = sentence_end + 1
                else:
                    para_break = content.rfind('\n\n', start, end)
                    if para_break > start:
                        end = para_break
            
            chunk = content[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            start = max(start + 1, end - overlap)
            
            if start >= len(content):
                break
        
        return chunks
    
    async def _parse_financial_document_structure(self, extracted_content: Dict, financial_info: Dict) -> Dict:
        """Parse the hierarchical structure of financial documents"""
        content = extracted_content["content"]
        tables = extracted_content.get("tables", [])
        
        structure = {
            "sections": [],
            "tables": [],
            "footnotes": [],
            "key_value_pairs": extracted_content.get("key_value_pairs", {})
        }
        
        try:
            doc_type = financial_info.get("document_type", "unknown")
            if doc_type in ["10-K", "10-Q"]:
                structure["sections"] = await self._parse_sec_filing_sections(content, doc_type)
            else:
                structure["sections"] = await self._parse_generic_sections(content)
            
            structure["tables"] = await self._parse_financial_tables(tables, content)
            
            structure["footnotes"] = await self._extract_footnotes(content)
            
        except Exception as e:
            logger.error(f"Error parsing document structure: {e}")
        
        return structure
    
    async def _parse_sec_filing_sections(self, content: str, doc_type: str) -> List[Dict]:
        """Parse SEC filing sections with proper hierarchy"""
        sections = []
        section_patterns = self.financial_section_patterns.get(f"{doc_type.lower()}_sections", [])
        
        current_section = None
        current_content = []
        current_section_start = 0
        lines = content.split('\n')
        
        for i, line in enumerate(lines):
            line_upper = line.strip().upper()
            
            section_found = False
            for pattern in section_patterns:
                if re.search(pattern, line_upper):
                    if current_section:
                        sections.append({
                            "title": current_section,
                            "content": '\n'.join(current_content),
                            "start_line": current_section_start,
                            "end_line": i,
                            "section_type": self._classify_section_type(current_section)
                        })
                    
                    current_section = line.strip()
                    current_section_start = i
                    current_content = []
                    section_found = True
                    break
            
            if not section_found and current_section:
                current_content.append(line)
        
        if current_section and current_content:
            sections.append({
                "title": current_section,
                "content": '\n'.join(current_content),
                "start_line": current_section_start,
                "end_line": len(lines),
                "section_type": self._classify_section_type(current_section)
            })
        
        return sections
    
    async def _parse_generic_sections(self, content: str) -> List[Dict]:
        """Parse generic document sections"""
        sections = []
        
        # Simple section detection for non-SEC documents
        section_patterns = [
            r"^([A-Z][A-Z\s,&]{10,})\s*$",  # All caps headers
            r"^\d+\.\s*([A-Z][A-Za-z\s,&]+)$"  # Numbered sections
        ]
        
        current_section = "Introduction"
        current_content = []
        lines = content.split('\n')
        
        for i, line in enumerate(lines):
            line_stripped = line.strip()
            
            is_header = False
            for pattern in section_patterns:
                match = re.match(pattern, line_stripped)
                if match and len(line_stripped) < 100:
                    if current_content:
                        sections.append({
                            "title": current_section,
                            "content": '\n'.join(current_content),
                            "start_line": 0 if not sections else sections[-1]["end_line"],
                            "end_line": i,
                            "section_type": "generic"
                        })
                    
                    current_section = match.group(1) if match.groups() else line_stripped
                    current_content = []
                    is_header = True
                    break
            
            if not is_header:
                current_content.append(line)
        
        if current_content:
            sections.append({
                "title": current_section,
                "content": '\n'.join(current_content),
                "start_line": 0 if not sections else sections[-1]["end_line"],
                "end_line": len(lines),
                "section_type": "generic"
            })
        
        return sections
    
    def _classify_section_type(self, section_title: str) -> str:
        """Classify the type of section based on title"""
        title_upper = section_title.upper()
        
        if "BUSINESS" in title_upper:
            return "business_overview"
        elif "RISK" in title_upper:
            return "risk_factors"
        elif "FINANCIAL STATEMENTS" in title_upper:
            return "financial_statements"
        elif "MANAGEMENT" in title_upper and "DISCUSSION" in title_upper:
            return "md_and_a"
        elif "LEGAL" in title_upper:
            return "legal_proceedings"
        elif "CONTROLS" in title_upper:
            return "controls_procedures"
        elif "PROPERTIES" in title_upper:
            return "properties"
        else:
            return "other"
    
    async def _parse_financial_tables(self, tables: List, content: str) -> List[Dict]:
        """Parse financial tables with enhanced metadata"""
        parsed_tables = []
        
        for i, table in enumerate(tables):
            try:
                table_data = {
                    "table_id": f"table_{i}",
                    "title": self._extract_table_title(table, content),
                    "rows": [],
                    "headers": [],
                    "financial_type": self._classify_table_type(table),
                    "page_number": getattr(table, 'page_number', 0),
                    "cell_count": len(getattr(table, 'cells', [])),
                    "bounding_box": getattr(table, 'bounding_box', None)
                }
                
                # Extract table structure
                if hasattr(table, 'cells'):
                    cells_by_row = {}
                    headers = []
                    
                    for cell in table.cells:
                        row_idx = cell.row_index
                        col_idx = cell.column_index
                        
                        if row_idx not in cells_by_row:
                            cells_by_row[row_idx] = {}
                        
                        cells_by_row[row_idx][col_idx] = cell.content
                        
                        if row_idx == 0:
                            headers.append(cell.content)
                    
                    table_data["headers"] = headers
                    
                    for row_idx in sorted(cells_by_row.keys()):
                        row_cells = cells_by_row[row_idx]
                        row_data = [row_cells.get(col_idx, "") for col_idx in sorted(row_cells.keys())]
                        table_data["rows"].append(row_data)
                
                parsed_tables.append(table_data)
                
            except Exception as e:
                logger.error(f"Error parsing table {i}: {e}")
        
        return parsed_tables
    
    def _extract_table_title(self, table, content: str) -> str:
        """Extract title for a financial table"""
        return f"Financial Table"
    
    def _classify_table_type(self, table) -> str:
        """Classify the type of financial table"""
        if hasattr(table, 'cells'):
            table_text = ' '.join([cell.content for cell in table.cells]).upper()
            
            if "BALANCE SHEET" in table_text:
                return "balance_sheet"
            elif "INCOME" in table_text or "OPERATIONS" in table_text:
                return "income_statement"
            elif "CASH FLOW" in table_text:
                return "cash_flow"
            elif "EQUITY" in table_text:
                return "equity_statement"
            elif any(term in table_text for term in ["REVENUE", "SALES", "INCOME"]):
                return "financial_performance"
            else:
                return "other_financial"
        
        return "unknown"
    
    async def _extract_footnotes(self, content: str) -> List[Dict]:
        """Extract footnotes with enhanced metadata"""
        footnotes = []
        
        footnote_patterns = [
            r"^\(\d+\)\s+(.+?)(?=^\(\d+\)|$)",  # (1) footnote format
            r"^\d+\.\s+(.+?)(?=^\d+\.|$)",      # 1. footnote format
            r"^\*\s+(.+?)(?=^\*|$)",            # * footnote format
        ]
        
        for pattern_idx, pattern in enumerate(footnote_patterns):
            matches = re.findall(pattern, content, re.MULTILINE | re.DOTALL)
            
            for match_idx, match in enumerate(matches):
                footnote = {
                    "footnote_id": f"footnote_{pattern_idx}_{match_idx}",
                    "content": match.strip(),
                    "pattern_type": pattern_idx,
                    "length": len(match.strip()),
                    "contains_financial_data": self._contains_financial_data(match)
                }
                footnotes.append(footnote)
        
        return footnotes[:50]  # Limit to first 50 footnotes
    
    def _contains_financial_data(self, text: str) -> bool:
        """Check if text contains financial data"""
        financial_indicators = [
            r'\$[\d,]+', r'[\d,]+\s*million', r'[\d,]+\s*billion',
            r'[\d.]+%', r'basis\s+points', r'revenue', r'income',
            r'assets', r'liabilities', r'equity'
        ]
        
        text_lower = text.lower()
        return any(re.search(pattern, text_lower) for pattern in financial_indicators)
    
    async def _create_hierarchical_financial_chunks(self, document_structure: Dict, 
                                                  document_id: str, metadata: Dict) -> List[DocumentChunk]:
        """Create hierarchical chunks with proper citation tracking"""
        chunks = []
        chunk_index = 0
        
        for section in document_structure.get("sections", []):
            section_chunks = await self._chunk_section_hierarchically(
                section, document_id, chunk_index, metadata
            )
            chunks.extend(section_chunks)
            chunk_index += len(section_chunks)
        
        for table in document_structure.get("tables", []):
            table_chunk = await self._create_table_chunk(
                table, document_id, chunk_index, metadata
            )
            chunks.append(table_chunk)
            chunk_index += 1
        
        for footnote in document_structure.get("footnotes", []):
            footnote_chunk = await self._create_footnote_chunk(
                footnote, document_id, chunk_index, metadata
            )
            chunks.append(footnote_chunk)
            chunk_index += 1
        
        logger.info(f"Created {len(chunks)} hierarchical chunks for document {document_id}")
        return chunks
    
    async def _chunk_section_hierarchically(self, section: Dict, document_id: str, 
                                          start_index: int, metadata: Dict) -> List[DocumentChunk]:
        """Chunk a section hierarchically with proper overlap"""
        chunks = []
        section_content = section["content"]
        
        if len(section_content) <= settings.MAX_CHUNK_SIZE:
            chunk = DocumentChunk(
                chunk_id=f"{document_id}_chunk_{start_index}",
                content=section_content,
                metadata={
                    **metadata,
                    "chunk_index": start_index,
                    "section_title": section["title"],
                    "section_type": section["section_type"],
                    "hierarchical_level": 1,
                    "start_line": section.get("start_line", 0),
                    "end_line": section.get("end_line", 0)
                },
                citation_info={
                    "section": section["title"],
                    "page_range": f"{section.get('start_line', 0)}-{section.get('end_line', 0)}",
                    "document_type": metadata.get("document_type", "unknown")
                }
            )
            chunks.append(chunk)
        else:
            section_chunks = self._split_with_overlap(
                section_content, settings.MAX_CHUNK_SIZE, settings.CHUNK_OVERLAP
            )
            
            for i, chunk_content in enumerate(section_chunks):
                chunk = DocumentChunk(
                    chunk_id=f"{document_id}_chunk_{start_index + i}",
                    content=chunk_content,
                    metadata={
                        **metadata,
                        "chunk_index": start_index + i,
                        "section_title": section["title"],
                        "section_type": section["section_type"],
                        "section_chunk_index": i,
                        "total_section_chunks": len(section_chunks),
                        "hierarchical_level": 2,
                        "start_line": section.get("start_line", 0),
                        "end_line": section.get("end_line", 0)
                    },
                    citation_info={
                        "section": section["title"],
                        "subsection": f"Part {i+1} of {len(section_chunks)}",
                        "page_range": f"{section.get('start_line', 0)}-{section.get('end_line', 0)}",
                        "document_type": metadata.get("document_type", "unknown")
                    }
                )
                chunks.append(chunk)
        
        return chunks
    
    async def _create_table_chunk(self, table: Dict, document_id: str, 
                                chunk_index: int, metadata: Dict) -> DocumentChunk:
        """Create a specialized chunk for financial tables"""
        table_content = self._format_table_content(table)
        
        return DocumentChunk(
            chunk_id=f"{document_id}_table_{chunk_index}",
            content=table_content,
            metadata={
                **metadata,
                "chunk_index": chunk_index,
                "chunk_type": "table",
                "table_id": table["table_id"],
                "table_title": table.get("title", ""),
                "financial_type": table.get("financial_type", "unknown"),
                "hierarchical_level": 0,  # Tables are top-level important
                "page_number": table.get("page_number", 0),
                "cell_count": table.get("cell_count", 0)
            },
            citation_info={
                "table_title": table.get("title", "Financial Table"),
                "table_type": table.get("financial_type", "unknown"),
                "page_number": table.get("page_number", 0),
                "document_type": metadata.get("document_type", "unknown")
            }
        )
    
    async def _create_footnote_chunk(self, footnote: Dict, document_id: str, 
                                   chunk_index: int, metadata: Dict) -> DocumentChunk:
        """Create a specialized chunk for footnotes"""
        return DocumentChunk(
            chunk_id=f"{document_id}_footnote_{chunk_index}",
            content=footnote["content"],
            metadata={
                **metadata,
                "chunk_index": chunk_index,
                "chunk_type": "footnote",
                "footnote_id": footnote["footnote_id"],
                "hierarchical_level": 3,  # Footnotes are detail-level
                "contains_financial_data": footnote.get("contains_financial_data", False),
                "footnote_length": footnote.get("length", 0)
            },
            citation_info={
                "footnote_id": footnote["footnote_id"],
                "citation_context": "footnote",
                "document_type": metadata.get("document_type", "unknown")
            }
        )
    
    def _format_table_content(self, table: Dict) -> str:
        """Format table content for text processing"""
        content_parts = []
        
        if table.get("title"):
            content_parts.append(f"Table: {table['title']}")
        
        headers = table.get("headers", [])
        if headers:
            content_parts.append("Headers: " + " | ".join(headers))
        
        rows = table.get("rows", [])
        for i, row in enumerate(rows[:10]):  # Limit to first 10 rows
            if i == 0 and headers:  # Skip header row if already processed
                continue
            row_text = " | ".join(str(cell) for cell in row)
            content_parts.append(f"Row {i}: {row_text}")
        
        if len(rows) > 10:
            content_parts.append(f"... and {len(rows) - 10} more rows")
        
        return "\n".join(content_parts)
    
    def _split_with_overlap(self, text: str, max_size: int, overlap: int) -> List[str]:
        """Split text with specified overlap, respecting sentence boundaries"""
        if len(text) <= max_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + max_size
            
            if end < len(text):
                sentence_end = text.rfind('.', start + max_size - 200, end)
                if sentence_end > start:
                    end = sentence_end + 1
                else:
                    para_end = text.rfind('\n\n', start, end)
                    if para_end > start:
                        end = para_end
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            start = max(start + 1, end - overlap)
            
            if start >= len(text):
                break
        
        return chunks
    
    async def _extract_key_financial_metrics(self, content: str, financial_info: Dict) -> List[Dict]:
        """Extract key financial metrics from document content"""
        metrics = []
        
        try:
            for pattern in self.financial_metrics_patterns["revenue_patterns"]:
                matches = re.finditer(pattern, content, re.IGNORECASE)
                for match in matches:
                    metrics.append({
                        "metric_type": "revenue",
                        "value": match.group(1),
                        "context": content[max(0, match.start()-100):match.end()+100],
                        "pattern_used": pattern
                    })
            
            for pattern in self.financial_metrics_patterns["income_patterns"]:
                matches = re.finditer(pattern, content, re.IGNORECASE)
                for match in matches:
                    metrics.append({
                        "metric_type": "income",
                        "value": match.group(1),
                        "context": content[max(0, match.start()-100):match.end()+100],
                        "pattern_used": pattern
                    })
            
            for pattern in self.financial_metrics_patterns["asset_patterns"]:
                matches = re.finditer(pattern, content, re.IGNORECASE)
                for match in matches:
                    metrics.append({
                        "metric_type": "assets",
                        "value": match.group(1),
                        "context": content[max(0, match.start()-100):match.end()+100],
                        "pattern_used": pattern
                    })
            
            for pattern in self.financial_metrics_patterns["liability_patterns"]:
                matches = re.finditer(pattern, content, re.IGNORECASE)
                for match in matches:
                    metrics.append({
                        "metric_type": "liabilities",
                        "value": match.group(1),
                        "context": content[max(0, match.start()-100):match.end()+100],
                        "pattern_used": pattern
                    })
            
        except Exception as e:
            logger.error(f"Error extracting financial metrics: {e}")
        
        return metrics[:100]  # Limit to first 100 metrics
    
    def _structure_to_dict(self, structure: Dict) -> Dict:
        """Convert document structure to dictionary format"""
        return {
            "sections": structure.get("sections", []),
            "tables": structure.get("tables", []),
            "footnotes": structure.get("footnotes", []),
            "key_value_pairs": structure.get("key_value_pairs", {}),
            "structure_stats": {
                "total_sections": len(structure.get("sections", [])),
                "total_tables": len(structure.get("tables", [])),
                "total_footnotes": len(structure.get("footnotes", [])),
                "has_key_value_pairs": len(structure.get("key_value_pairs", {})) > 0
            }
        }
    
    def _chunk_to_dict(self, chunk: DocumentChunk) -> Dict:
        """Convert DocumentChunk to dictionary with enhanced structure"""
        return {
            "chunk_id": chunk.chunk_id,
            "content": chunk.content,
            "metadata": chunk.metadata,
            "embedding": chunk.embedding,
            "citation_info": chunk.citation_info
        }
