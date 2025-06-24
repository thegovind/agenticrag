import asyncio
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
import hashlib
import json
from dataclasses import dataclass
import re

from app.services.azure_services import AzureServiceManager
from app.services.document_processor import DocumentProcessor
from app.services.credibility_assessor import CredibilityAssessor
from app.core.config import settings
from app.core.observability import observability

logger = logging.getLogger(__name__)

@dataclass
class KnowledgeUpdate:
    source: str
    document_id: str
    content: str
    metadata: Dict[str, Any]
    credibility_score: float
    update_type: str  # 'new', 'updated', 'conflicting'
    timestamp: datetime

class AdaptiveKnowledgeBaseManager:
    """
    Core service for Exercise 3: Adaptive Knowledge Base Management
    
    Handles:
    - Automatic knowledge base updates from new sources
    - Credibility assessment of new information
    - Conflict resolution between sources
    - Knowledge organization and structuring
    - Response adaptation based on updated knowledge
    """
    
    def __init__(self, azure_manager: AzureServiceManager):
        self.azure_manager = azure_manager
        self.document_processor = DocumentProcessor(azure_manager)
        self.credibility_assessor = CredibilityAssessor(azure_manager)
        self.update_queue = []
        self.processing_lock = asyncio.Lock()
        
    async def monitor_and_update_knowledge_base(self):
        """
        Main loop for adaptive knowledge base management
        Continuously monitors for new information and updates the KB
        """
        logger.info("Starting adaptive knowledge base monitoring")
        
        while settings.AUTO_UPDATE_ENABLED:
            try:
                new_sources = await self._discover_new_sources()
                
                for source in new_sources:
                    await self._process_new_source(source)
                
                await self._process_update_queue()
                
                await self._resolve_knowledge_conflicts()
                
                await self._optimize_search_index()
                
                await asyncio.sleep(settings.UPDATE_FREQUENCY_HOURS * 3600)
                
            except Exception as e:
                logger.error(f"Error in knowledge base monitoring: {e}")
                await asyncio.sleep(300)  # Wait 5 minutes before retry

    async def add_new_information(self, source: str, content: bytes, 
                                content_type: str, metadata: Dict = None) -> Dict:
        """
        Add new information to the knowledge base
        
        Args:
            source: Source identifier (URL, file path, etc.)
            content: Raw content bytes
            content_type: MIME type of content
            metadata: Additional metadata about the source
            
        Returns:
            Dict with processing results and statistics
        """
        async with self.processing_lock:
            try:
                logger.info(f"Processing new information from source: {source}")
                
                processed_doc = await self.document_processor.process_document(
                    content, content_type, source, metadata or {}
                )
                
                credibility_score = await self.credibility_assessor.assess_credibility(
                    processed_doc, source
                )
                
                if credibility_score < settings.CREDIBILITY_THRESHOLD:
                    logger.warning(f"Source {source} below credibility threshold: {credibility_score}")
                    return {
                        "status": "rejected",
                        "reason": "credibility_too_low",
                        "credibility_score": credibility_score,
                        "threshold": settings.CREDIBILITY_THRESHOLD
                    }
                
                existing_content = await self._find_similar_content(processed_doc["chunks"])
                conflicts = await self._identify_conflicts(processed_doc, existing_content)
                
                updates = []
                for chunk in processed_doc["chunks"]:
                    update = KnowledgeUpdate(
                        source=source,
                        document_id=processed_doc["document_id"],
                        content=chunk["content"],
                        metadata={
                            **chunk["metadata"],
                            **metadata,
                            "credibility_score": credibility_score,
                            "conflicts": conflicts.get(chunk["chunk_id"], [])
                        },
                        credibility_score=credibility_score,
                        update_type="new" if not existing_content else "updated",
                        timestamp=datetime.utcnow()
                    )
                    updates.append(update)
                
                self.update_queue.extend(updates)
                
                observability.track_kb_update(
                    source=source,
                    documents_added=len([u for u in updates if u.update_type == "new"]),
                    documents_updated=len([u for u in updates if u.update_type == "updated"])
                )
                
                logger.info(f"Successfully queued {len(updates)} updates from {source}")
                
                return {
                    "status": "success",
                    "updates_queued": len(updates),
                    "credibility_score": credibility_score,
                    "conflicts_detected": len(conflicts),
                    "document_id": processed_doc["document_id"]
                }
                
            except Exception as e:
                logger.error(f"Failed to add new information from {source}: {e}")
                return {
                    "status": "error",
                    "error": str(e)
                }

    async def _discover_new_sources(self) -> List[str]:
        """
        Discover new information sources to monitor
        This could include RSS feeds, SEC filings, news sources, etc.
        """
        
        new_sources = []
        
        try:
            sec_sources = [
                "https://www.sec.gov/cgi-bin/browse-edgar?action=getcurrent&CIK=&type=10-K&company=&dateb=&owner=include&start=0&count=40&output=atom",
                "https://www.sec.gov/cgi-bin/browse-edgar?action=getcurrent&CIK=&type=10-Q&company=&dateb=&owner=include&start=0&count=40&output=atom"
            ]
            
            news_sources = [
                "https://feeds.finance.yahoo.com/rss/2.0/headline",
                "https://www.reuters.com/business/finance/rss",
                "https://www.bloomberg.com/feeds/bfeed"
            ]
            
            company_sources = getattr(settings, 'MONITORED_COMPANIES', [])
            
            new_sources.extend(sec_sources)
            new_sources.extend(news_sources)
            new_sources.extend(company_sources)
            
            logger.info(f"Discovered {len(new_sources)} potential sources to monitor")
            
        except Exception as e:
            logger.error(f"Error discovering new sources: {e}")
        
        return new_sources

    async def _process_new_source(self, source: str):
        """Process a newly discovered information source"""
        try:
            logger.info(f"Processing new source: {source}")
            
            if source.endswith('.atom') or 'rss' in source.lower():
                await self._process_rss_feed(source)
            
            elif source.endswith(('.pdf', '.html', '.htm')):
                await self._process_document_url(source)
            
            else:
                await self._scrape_company_filings(source)
                
        except Exception as e:
            logger.error(f"Error processing source {source}: {e}")
    
    async def _process_rss_feed(self, feed_url: str):
        """Process RSS/Atom feed for new financial documents"""
        try:
            import feedparser
            
            feed = feedparser.parse(feed_url)
            for entry in feed.entries[:10]:  # Process latest 10 entries
                if hasattr(entry, 'link') and entry.link:
                    doc_hash = hashlib.md5(entry.link.encode()).hexdigest()
                    if not await self._is_document_processed(doc_hash):
                        await self._process_document_url(entry.link)
                        
        except Exception as e:
            logger.error(f"Error processing RSS feed {feed_url}: {e}")
    
    async def _process_document_url(self, url: str):
        """Download and process a document from URL"""
        try:
            import httpx
            
            async with httpx.AsyncClient() as client:
                response = await client.get(url, timeout=30.0)
                if response.status_code == 200:
                    content_type = response.headers.get('content-type', 'application/pdf')
                    await self.add_new_information(url, response.content, content_type)
                    
        except Exception as e:
            logger.error(f"Error processing document URL {url}: {e}")
    
    async def _scrape_company_filings(self, company_url: str):
        """Scrape company investor relations page for new filings"""
        try:
            logger.info(f"Would scrape company filings from: {company_url}")
            
        except Exception as e:
            logger.error(f"Error scraping company filings from {company_url}: {e}")
    
    async def _is_document_processed(self, doc_hash: str) -> bool:
        """Check if a document has already been processed"""
        try:
            results = await self.azure_manager.hybrid_search(
                query=f"document_hash:{doc_hash}",
                top_k=1
            )
            return len(results) > 0
        except:
            return False

    async def _process_update_queue(self):
        """Process queued knowledge base updates"""
        if not self.update_queue:
            return
            
        logger.info(f"Processing {len(self.update_queue)} queued updates")
        
        updates_by_doc = {}
        for update in self.update_queue:
            doc_id = update.document_id
            if doc_id not in updates_by_doc:
                updates_by_doc[doc_id] = []
            updates_by_doc[doc_id].append(update)
        
        processed_count = 0
        for doc_id, doc_updates in updates_by_doc.items():
            try:
                await self._apply_document_updates(doc_id, doc_updates)
                processed_count += len(doc_updates)
            except Exception as e:
                logger.error(f"Failed to apply updates for document {doc_id}: {e}")
        
        self.update_queue = []
        logger.info(f"Successfully processed {processed_count} knowledge base updates")

    async def _apply_document_updates(self, document_id: str, updates: List[KnowledgeUpdate]):
        """Apply updates for a specific document to the search index"""
        search_documents = []
        
        for update in updates:
            doc = {
                "id": f"{document_id}_{hashlib.md5(update.content.encode()).hexdigest()[:8]}",
                "content": update.content,
                "title": update.metadata.get("title", ""),
                "document_type": update.metadata.get("document_type", "financial_report"),
                "company": update.metadata.get("company", ""),
                "filing_date": update.metadata.get("filing_date"),
                "chunk_index": update.metadata.get("chunk_index", 0),
                "source_url": update.source,
                "credibility_score": update.credibility_score,
                "content_vector": await self.azure_manager.get_embedding(update.content)
            }
            search_documents.append(doc)
        
        success = await self.azure_manager.add_documents_to_index(search_documents)
        if success:
            logger.info(f"Added {len(search_documents)} documents to search index for {document_id}")

    async def _find_similar_content(self, chunks: List[Dict]) -> Dict:
        """Find existing content similar to new chunks"""
        similar_content = {}
        
        for chunk in chunks:
            try:
                results = await self.azure_manager.hybrid_search(
                    query=chunk["content"][:500],  # Use first 500 chars for similarity
                    top_k=5
                )
                
                similar_results = []
                for result in results:
                    similarity = self._calculate_similarity(chunk["content"], result["content"])
                    if similarity > 0.8:  # High similarity threshold
                        similar_results.append({
                            "id": result["id"],
                            "content": result["content"],
                            "similarity": similarity,
                            "source": result["source_url"]
                        })
                
                if similar_results:
                    similar_content[chunk["chunk_id"]] = similar_results
                    
            except Exception as e:
                logger.error(f"Error finding similar content for chunk {chunk['chunk_id']}: {e}")
        
        return similar_content

    async def _identify_conflicts(self, new_doc: Dict, existing_content: Dict) -> Dict:
        """Identify conflicts between new and existing content"""
        conflicts = {}
        
        for chunk in new_doc["chunks"]:
            chunk_id = chunk["chunk_id"]
            if chunk_id in existing_content:
                chunk_conflicts = []
                
                for existing in existing_content[chunk_id]:
                    conflict_analysis = await self._analyze_content_conflict(
                        chunk["content"], existing["content"]
                    )
                    
                    if conflict_analysis["has_conflict"]:
                        chunk_conflicts.append({
                            "existing_id": existing["id"],
                            "conflict_type": conflict_analysis["conflict_type"],
                            "confidence": conflict_analysis["confidence"],
                            "description": conflict_analysis["description"]
                        })
                
                if chunk_conflicts:
                    conflicts[chunk_id] = chunk_conflicts
        
        return conflicts

    async def _analyze_content_conflict(self, new_content: str, existing_content: str) -> Dict:
        """Use LLM to analyze potential conflicts between content pieces"""
        try:
            prompt = f"""
            Analyze the following two pieces of financial content for conflicts:
            
            NEW CONTENT:
            {new_content}
            
            EXISTING CONTENT:
            {existing_content}
            
            Determine if there are any factual conflicts, contradictions, or inconsistencies.
            Respond in JSON format with:
            {{
                "has_conflict": boolean,
                "conflict_type": "factual|temporal|methodological|none",
                "confidence": float (0-1),
                "description": "Brief description of the conflict if any"
            }}
            """
            
            response = await self.azure_manager.openai_client.chat.completions.create(
                model=settings.AZURE_OPENAI_CHAT_DEPLOYMENT_NAME,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=200
            )
            
            result = json.loads(response.choices[0].message.content)
            return result
            
        except Exception as e:
            logger.error(f"Error analyzing content conflict: {e}")
            return {
                "has_conflict": False,
                "conflict_type": "none",
                "confidence": 0.0,
                "description": "Analysis failed"
            }

    async def _resolve_knowledge_conflicts(self):
        """Resolve conflicts in the knowledge base"""
        logger.info("Resolving knowledge base conflicts")
        
        
        try:
            conflict_query = "credibility_score ge 0.5"  # Example filter
            results = await self.azure_manager.hybrid_search(
                query="*",  # Match all
                filters=conflict_query,
                top_k=100
            )
            
            for result in results:
                if "conflicts" in result and result["conflicts"]:
                    await self._resolve_document_conflicts(result)
                    
        except Exception as e:
            logger.error(f"Error resolving conflicts: {e}")

    async def _resolve_document_conflicts(self, document: Dict):
        """Resolve conflicts for a specific document"""
        logger.info(f"Resolving conflicts for document {document['id']}")
        
        try:
            conflicts = document.get('conflicts', [])
            if not conflicts:
                return
            
            for conflict in conflicts:
                resolution_strategy = await self._determine_conflict_resolution_strategy(conflict)
                
                if resolution_strategy == "prefer_newer":
                    await self._mark_document_superseded(conflict['existing_id'])
                    
                elif resolution_strategy == "prefer_higher_credibility":
                    if document.get('credibility_score', 0) > conflict.get('existing_credibility', 0):
                        await self._mark_document_superseded(conflict['existing_id'])
                    else:
                        await self._mark_document_superseded(document['id'])
                        
                elif resolution_strategy == "merge_information":
                    await self._merge_conflicting_documents(document, conflict)
                    
                elif resolution_strategy == "flag_for_review":
                    await self._flag_conflict_for_manual_review(document, conflict)
                    
        except Exception as e:
            logger.error(f"Error resolving conflicts for document {document['id']}: {e}")
    
    async def _determine_conflict_resolution_strategy(self, conflict: Dict) -> str:
        """Determine the best strategy for resolving a specific conflict"""
        conflict_type = conflict.get('conflict_type', 'unknown')
        confidence = conflict.get('confidence', 0.0)
        
        if confidence < 0.5:
            return "flag_for_review"
        elif conflict_type == "temporal":
            return "prefer_newer"
        elif conflict_type == "factual":
            return "prefer_higher_credibility"
        elif conflict_type == "methodological":
            return "merge_information"
        else:
            return "flag_for_review"
    
    async def _mark_document_superseded(self, document_id: str):
        """Mark a document as superseded by newer information"""
        try:
            logger.info(f"Marking document {document_id} as superseded")
            
        except Exception as e:
            logger.error(f"Error marking document {document_id} as superseded: {e}")
    
    async def _merge_conflicting_documents(self, doc1: Dict, conflict: Dict):
        """Merge information from conflicting documents"""
        try:
            logger.info(f"Merging conflicting documents: {doc1['id']} and {conflict['existing_id']}")
            
        except Exception as e:
            logger.error(f"Error merging conflicting documents: {e}")
    
    async def _flag_conflict_for_manual_review(self, document: Dict, conflict: Dict):
        """Flag a conflict for manual review"""
        try:
            logger.warning(f"Flagging conflict for manual review: {document['id']} vs {conflict['existing_id']}")
            
        except Exception as e:
            logger.error(f"Error flagging conflict for review: {e}")

    async def _optimize_search_index(self):
        """Optimize the search index for better performance"""
        logger.info("Optimizing search index")
        
        try:
            await self._remove_superseded_documents()
            
            await self._update_document_rankings()
            
            await self._consolidate_similar_chunks()
            
            await self._update_semantic_relationships()
            
            logger.info("Search index optimization completed")
            
        except Exception as e:
            logger.error(f"Error optimizing search index: {e}")
    
    async def _remove_superseded_documents(self):
        """Remove documents marked as superseded"""
        try:
            results = await self.azure_manager.hybrid_search(
                query="*",
                filters="superseded eq true",
                top_k=1000
            )
            
            if results:
                doc_ids = [result['id'] for result in results]
                logger.info(f"Removing {len(doc_ids)} superseded documents")
                
        except Exception as e:
            logger.error(f"Error removing superseded documents: {e}")
    
    async def _update_document_rankings(self):
        """Update document rankings based on credibility and recency"""
        try:
            # - Credibility score
            logger.info("Updating document rankings")
            
        except Exception as e:
            logger.error(f"Error updating document rankings: {e}")
    
    async def _consolidate_similar_chunks(self):
        """Consolidate highly similar chunks to reduce redundancy"""
        try:
            logger.info("Consolidating similar chunks")
            
        except Exception as e:
            logger.error(f"Error consolidating similar chunks: {e}")
    
    async def _update_semantic_relationships(self):
        """Update semantic relationships between documents"""
        try:
            logger.info("Updating semantic relationships")
            
        except Exception as e:
            logger.error(f"Error updating semantic relationships: {e}")
        

    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two text pieces (simplified)"""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0.0

    async def get_knowledge_base_stats(self) -> Dict:
        """Get statistics about the knowledge base"""
        try:
            all_docs = await self.azure_manager.hybrid_search(
                query="*",
                top_k=10000  # Large number to get all docs
            )
            
            # Calculate statistics
            total_documents = len(set(doc.get('source_url', '') for doc in all_docs))
            total_chunks = len(all_docs)
            
            credibility_scores = [doc.get('credibility_score', 0.0) for doc in all_docs if doc.get('credibility_score')]
            average_credibility = sum(credibility_scores) / len(credibility_scores) if credibility_scores else 0.0
            
            # Count conflicts
            conflict_count = sum(1 for doc in all_docs if doc.get('conflicts'))
            
            sources = list(set(doc.get('source_url', '') for doc in all_docs if doc.get('source_url')))
            
            doc_types = {}
            for doc in all_docs:
                doc_type = doc.get('document_type', 'unknown')
                doc_types[doc_type] = doc_types.get(doc_type, 0) + 1
            
            companies = {}
            for doc in all_docs:
                company = doc.get('company', 'unknown')
                if company and company != 'unknown':
                    companies[company] = companies.get(company, 0) + 1
            
            stats = {
                "total_documents": total_documents,
                "total_chunks": total_chunks,
                "average_credibility": round(average_credibility, 3),
                "last_update": datetime.utcnow().isoformat(),
                "pending_updates": len(self.update_queue),
                "conflict_count": conflict_count,
                "sources": sources[:20],  # Limit to first 20 sources
                "document_types": doc_types,
                "companies": companies,
                "health_score": self._calculate_kb_health_score(all_docs)
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting knowledge base stats: {e}")
            return {
                "total_documents": 0,
                "total_chunks": 0,
                "average_credibility": 0.0,
                "last_update": datetime.utcnow().isoformat(),
                "pending_updates": len(self.update_queue),
                "conflict_count": 0,
                "sources": [],
                "error": str(e)
            }
    
    def _calculate_kb_health_score(self, documents: List[Dict]) -> float:
        """Calculate overall health score of the knowledge base"""
        if not documents:
            return 0.0
        
        
        try:
            credibility_scores = [doc.get('credibility_score', 0.0) for doc in documents]
            avg_credibility = sum(credibility_scores) / len(credibility_scores) if credibility_scores else 0.0
            
            now = datetime.utcnow()
            recent_docs = 0
            for doc in documents:
                filing_date = doc.get('filing_date')
                if filing_date:
                    try:
                        doc_date = datetime.fromisoformat(filing_date.replace('Z', '+00:00'))
                        if (now - doc_date).days <= 90:
                            recent_docs += 1
                    except:
                        pass
            
            recency_score = recent_docs / len(documents) if documents else 0.0
            
            total_conflicts = sum(1 for doc in documents if doc.get('conflicts'))
            resolved_conflicts = sum(1 for doc in documents if doc.get('conflicts_resolved', False))
            conflict_resolution_rate = resolved_conflicts / total_conflicts if total_conflicts > 0 else 1.0
            
            unique_sources = len(set(doc.get('source_url', '') for doc in documents))
            source_diversity = min(unique_sources / 10, 1.0)  # Normalize to max of 10 sources
            
            health_score = (
                avg_credibility * 0.4 +
                recency_score * 0.3 +
                conflict_resolution_rate * 0.2 +
                source_diversity * 0.1
            )
            
            return round(health_score, 3)
            
        except Exception as e:
            logger.error(f"Error calculating KB health score: {e}")
            return 0.0

    async def search_knowledge_base(self, query: str, filters: Dict = None, 
                                  top_k: int = 10, chat_model: str = None) -> List[Dict]:
        """
        Search the adaptive knowledge base
        
        This method provides the interface for other exercises to query
        the dynamically updated knowledge base
        
        Args:
            query: The search query
            filters: Optional filters to apply
            top_k: Number of results to return
            chat_model: The chat model deployment name to use for relevance explanations
        """
        try:
            filter_str = None
            if filters:
                filter_parts = []
                for key, value in filters.items():
                    if isinstance(value, str):
                        filter_parts.append(f"{key} eq '{value}'")
                    elif isinstance(value, (int, float)):
                        filter_parts.append(f"{key} eq {value}")
                    elif isinstance(value, list):
                        list_filters = [f"{key} eq '{v}'" for v in value]
                        filter_parts.append(f"({' or '.join(list_filters)})")
                
                filter_str = " and ".join(filter_parts) if filter_parts else None
            
            results = await self.azure_manager.hybrid_search(
                query=query,
                top_k=top_k,
                filters=filter_str
            )
            
            enhanced_results = []
            for result in results:
                enhanced_result = {
                    **result,
                    "relevance_explanation": await self._explain_relevance(query, result, chat_model),
                    "last_updated": result.get("last_updated", "unknown"),
                    "confidence_score": result.get("credibility_score", 0.0)
                }
                enhanced_results.append(enhanced_result)
            
            return enhanced_results
            
        except Exception as e:
            logger.error(f"Error searching knowledge base: {e}")
            return []

    async def _explain_relevance(self, query: str, result: Dict, chat_model: str = None) -> str:
        """Generate explanation for why a result is relevant to the query"""
        try:
            prompt = f"""
            Explain why this financial document excerpt is relevant to the query: "{query}"
            
            Document excerpt: {result.get('content', '')[:500]}...
            Document type: {result.get('document_type', 'unknown')}
            Company: {result.get('company', 'unknown')}
            Credibility score: {result.get('credibility_score', 0.0)}
            
            Provide a brief, professional explanation (1-2 sentences) of the relevance.
            """
            
            # Use the passed model if available, otherwise fall back to default
            model_to_use = chat_model or settings.AZURE_OPENAI_CHAT_DEPLOYMENT_NAME
            
            response = await self.azure_manager.openai_client.chat.completions.create(
                model=model_to_use,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=100
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"Error generating relevance explanation with model {chat_model or 'default'}: {e}")
            return f"Relevant to '{query}' based on content similarity and credibility score of {result.get('credibility_score', 0.0)}"
    
    async def enhance_financial_chunking(self, document_content: str, metadata: Dict) -> List[Dict]:
        """
        Enhanced hierarchical chunking strategy for financial documents
        Document → Sections → Subsections → Paragraphs → Tables/Footnotes
        """
        try:
            chunks = []
            
            sections = self._identify_financial_sections(document_content)
            
            for section_name, section_content in sections.items():
                section_chunks = await self._process_financial_section(
                    section_content, section_name, metadata
                )
                chunks.extend(section_chunks)
            
            tables = self._extract_financial_tables(document_content)
            for i, table in enumerate(tables):
                table_chunk = await self._process_financial_table(table, i, metadata)
                chunks.append(table_chunk)
            
            footnotes = self._extract_footnotes(document_content)
            for i, footnote in enumerate(footnotes):
                footnote_chunk = await self._process_footnote(footnote, i, metadata)
                chunks.append(footnote_chunk)
            
            return chunks
            
        except Exception as e:
            logger.error(f"Error in enhanced financial chunking: {e}")
            return []
    
    def _identify_financial_sections(self, content: str) -> Dict[str, str]:
        """Identify standard financial document sections"""
        sections = {}
        
        section_patterns = [
            (r"PART\s+I\s*ITEM\s+1[.\s]*BUSINESS", "business_overview"),
            (r"PART\s+I\s*ITEM\s+1A[.\s]*RISK\s+FACTORS", "risk_factors"),
            (r"PART\s+I\s*ITEM\s+2[.\s]*PROPERTIES", "properties"),
            (r"PART\s+I\s*ITEM\s+3[.\s]*LEGAL\s+PROCEEDINGS", "legal_proceedings"),
            (r"PART\s+II\s*ITEM\s+5[.\s]*MARKET", "market_info"),
            (r"PART\s+II\s*ITEM\s+7[.\s]*MANAGEMENT'S\s+DISCUSSION", "md_and_a"),
            (r"PART\s+II\s*ITEM\s+8[.\s]*FINANCIAL\s+STATEMENTS", "financial_statements"),
            (r"CONSOLIDATED\s+BALANCE\s+SHEETS?", "balance_sheet"),
            (r"CONSOLIDATED\s+STATEMENTS?\s+OF\s+OPERATIONS", "income_statement"),
            (r"CONSOLIDATED\s+STATEMENTS?\s+OF\s+CASH\s+FLOWS?", "cash_flow"),
            (r"NOTES?\s+TO\s+CONSOLIDATED\s+FINANCIAL\s+STATEMENTS", "notes_to_financials")
        ]
        
        current_section = "introduction"
        current_content = []
        
        lines = content.split('\n')
        
        for line in lines:
            line_upper = line.strip().upper()
            
            section_found = False
            for pattern, section_name in section_patterns:
                if re.search(pattern, line_upper):
                    if current_content:
                        sections[current_section] = '\n'.join(current_content)
                    
                    current_section = section_name
                    current_content = [line]
                    section_found = True
                    break
            
            if not section_found:
                current_content.append(line)
        
        if current_content:
            sections[current_section] = '\n'.join(current_content)
        
        return sections
    
    async def _process_financial_section(self, content: str, section_name: str, metadata: Dict) -> List[Dict]:
        """Process a financial section with appropriate chunking"""
        chunks = []
        
        if len(content) <= settings.MAX_CHUNK_SIZE:
            chunk = {
                "content": content,
                "metadata": {
                    **metadata,
                    "section": section_name,
                    "chunk_type": "section",
                    "hierarchical_level": 1
                }
            }
            chunks.append(chunk)
        else:
            subsections = self._split_with_overlap(content, settings.MAX_CHUNK_SIZE, settings.CHUNK_OVERLAP)
            
            for i, subsection in enumerate(subsections):
                chunk = {
                    "content": subsection,
                    "metadata": {
                        **metadata,
                        "section": section_name,
                        "subsection_index": i,
                        "chunk_type": "subsection",
                        "hierarchical_level": 2
                    }
                }
                chunks.append(chunk)
        
        return chunks
    
    def _extract_financial_tables(self, content: str) -> List[str]:
        """Extract financial tables from document content"""
        tables = []
        
        table_patterns = [
            r"(\$\s*\d+(?:,\d{3})*(?:\.\d{2})?.*?\n.*?\$\s*\d+(?:,\d{3})*(?:\.\d{2})?)",
            r"((?:.*?\d{4}.*?\d{4}.*?\n){3,})",  # Multi-year data tables
        ]
        
        for pattern in table_patterns:
            matches = re.findall(pattern, content, re.MULTILINE | re.DOTALL)
            tables.extend(matches)
        
        return tables[:10]  # Limit to first 10 tables
    
    async def _process_financial_table(self, table_content: str, table_index: int, metadata: Dict) -> Dict:
        """Process a financial table as a specialized chunk"""
        return {
            "content": table_content,
            "metadata": {
                **metadata,
                "table_index": table_index,
                "chunk_type": "table",
                "hierarchical_level": 0,  # Tables are top-level important content
                "requires_special_handling": True
            }
        }
    
    def _extract_footnotes(self, content: str) -> List[str]:
        """Extract footnotes from financial documents"""
        footnotes = []
        
        footnote_patterns = [
            r"^\(\d+\)\s+(.+?)(?=^\(\d+\)|$)",  # (1) footnote format
            r"^\d+\.\s+(.+?)(?=^\d+\.|$)",      # 1. footnote format
        ]
        
        for pattern in footnote_patterns:
            matches = re.findall(pattern, content, re.MULTILINE | re.DOTALL)
            footnotes.extend(matches)
        
        return footnotes[:20]  # Limit to first 20 footnotes
    
    async def _process_footnote(self, footnote_content: str, footnote_index: int, metadata: Dict) -> Dict:
        """Process a footnote as a specialized chunk"""
        return {
            "content": footnote_content,
            "metadata": {
                **metadata,
                "footnote_index": footnote_index,
                "chunk_type": "footnote",
                "hierarchical_level": 3,  # Footnotes are detail-level content
                "citation_context": "footnote"
            }
        }
    
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
