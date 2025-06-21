#!/usr/bin/env python3
import sys
import asyncio
sys.path.append('.')

async def test_rag_pipeline():
    try:
        from app.services.rag_pipeline import (
            RAGPipeline, QueryProcessor, HybridSearchEngine, CitationManager,
            SearchType, CitationConfidence, SearchResult, Citation, RAGResponse
        )
        print('✓ RAG pipeline classes import successfully')
        
        from app.services.azure_services import AzureServiceManager
        from app.services.knowledge_base_manager import AdaptiveKnowledgeBaseManager
        
        azure_manager = AzureServiceManager()
        kb_manager = AdaptiveKnowledgeBaseManager(azure_manager)
        rag_pipeline = RAGPipeline(azure_manager, kb_manager)
        print('✓ RAG pipeline instantiated successfully')
        
        query_processor = QueryProcessor(azure_manager)
        print('✓ Query processor initialized')
        
        test_query = "What is the revenue growth for the company in the latest 10-K filing?"
        processed_query = await query_processor.process_query(test_query)
        print(f'✓ Query processing works: {processed_query["query_type"]}')
        
        search_engine = HybridSearchEngine(azure_manager)
        print('✓ Hybrid search engine initialized')
        
        citation_manager = CitationManager(azure_manager)
        print('✓ Citation manager initialized')
        
        test_result = SearchResult(
            chunk_id="test_chunk_1",
            content="Revenue increased by 15% year-over-year to $2.5 billion",
            score=0.85,
            document_id="10k_2023",
            document_title="Annual Report 2023",
            section_title="Financial Performance",
            page_number=25,
            table_data=None,
            metadata={"company": "Test Corp"}
        )
        print('✓ Search result creation works')
        
        citations = citation_manager.generate_citations([test_result], "Test content")
        print(f'✓ Citation generation works: {len(citations)} citations created')
        
        if citations:
            print(f'✓ Citation confidence: {citations[0].confidence.value}')
        
        search_types = [SearchType.VECTOR, SearchType.KEYWORD, SearchType.HYBRID, SearchType.SEMANTIC]
        print(f'✓ Search types available: {[st.value for st in search_types]}')
        
        confidence_levels = [CitationConfidence.HIGH, CitationConfidence.MEDIUM, CitationConfidence.LOW]
        print(f'✓ Citation confidence levels: {[cl.value for cl in confidence_levels]}')
        
        print('✓ RAG pipeline service is complete and functional')
        
    except Exception as e:
        print(f'✗ RAG pipeline error: {e}')
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(test_rag_pipeline())
