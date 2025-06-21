import asyncio
import json
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
from dataclasses import dataclass, asdict

from app.services.multi_agent_orchestrator import MultiAgentOrchestrator, AgentType
from app.services.azure_services import AzureServiceManager
from app.services.knowledge_base_manager import AdaptiveKnowledgeBaseManager
from app.core.config import settings

logger = logging.getLogger(__name__)

@dataclass
class MCPTool:
    name: str
    description: str
    input_schema: Dict[str, Any]

@dataclass
class MCPResource:
    uri: str
    name: str
    description: str
    mime_type: str

class FinancialMCPServer:
    """MCP Server for Financial RAG System"""
    
    def __init__(self):
        self.azure_manager = AzureServiceManager()
        self.kb_manager = AdaptiveKnowledgeBaseManager(self.azure_manager)
        self.orchestrator = MultiAgentOrchestrator(self.azure_manager, self.kb_manager)
        self.tools = self._initialize_tools()
        self.resources = self._initialize_resources()
        
    def _initialize_tools(self) -> List[MCPTool]:
        """Initialize MCP tools for financial analysis"""
        return [
            MCPTool(
                name="generate_financial_content",
                description="Generate high-quality financial content based on prompts and knowledge base",
                input_schema={
                    "type": "object",
                    "properties": {
                        "prompt": {"type": "string", "description": "Content generation prompt"},
                        "content_type": {"type": "string", "enum": ["report", "summary", "analysis"], "default": "analysis"},
                        "tone": {"type": "string", "enum": ["professional", "technical", "executive"], "default": "professional"},
                        "max_length": {"type": "integer", "default": 2000, "description": "Maximum content length in words"}
                    },
                    "required": ["prompt"]
                }
            ),
            MCPTool(
                name="answer_financial_question",
                description="Answer complex financial questions with source verification",
                input_schema={
                    "type": "object",
                    "properties": {
                        "question": {"type": "string", "description": "Financial question to answer"},
                        "context": {"type": "string", "description": "Additional context for the question"},
                        "verification_level": {"type": "string", "enum": ["basic", "thorough"], "default": "thorough"}
                    },
                    "required": ["question"]
                }
            ),
            MCPTool(
                name="update_knowledge_base",
                description="Update knowledge base with new financial information",
                input_schema={
                    "type": "object",
                    "properties": {
                        "source_url": {"type": "string", "description": "URL of the information source"},
                        "content": {"type": "string", "description": "Content to add to knowledge base"},
                        "metadata": {"type": "object", "description": "Additional metadata for the content"}
                    },
                    "required": ["content"]
                }
            ),
            MCPTool(
                name="assess_knowledge_health",
                description="Assess the health and quality of the knowledge base",
                input_schema={
                    "type": "object",
                    "properties": {
                        "assessment_type": {"type": "string", "enum": ["full", "incremental"], "default": "incremental"}
                    }
                }
            ),
            MCPTool(
                name="search_financial_documents",
                description="Search through financial documents in the knowledge base",
                input_schema={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query"},
                        "document_types": {"type": "array", "items": {"type": "string"}, "description": "Filter by document types"},
                        "top_k": {"type": "integer", "default": 10, "description": "Number of results to return"}
                    },
                    "required": ["query"]
                }
            ),
            MCPTool(
                name="verify_source_credibility",
                description="Verify the credibility of financial information sources",
                input_schema={
                    "type": "object",
                    "properties": {
                        "sources": {"type": "array", "items": {"type": "object"}, "description": "Sources to verify"}
                    },
                    "required": ["sources"]
                }
            ),
            MCPTool(
                name="coordinate_multi_agent_analysis",
                description="Coordinate multiple agents for comprehensive financial analysis",
                input_schema={
                    "type": "object",
                    "properties": {
                        "request_type": {"type": "string", "description": "Type of analysis request"},
                        "content": {"type": "string", "description": "Content or question to analyze"},
                        "requirements": {"type": "object", "description": "Specific requirements for the analysis"}
                    },
                    "required": ["request_type", "content"]
                }
            )
        ]
    
    def _initialize_resources(self) -> List[MCPResource]:
        """Initialize MCP resources for financial data"""
        return [
            MCPResource(
                uri="financial://knowledge-base/statistics",
                name="Knowledge Base Statistics",
                description="Current statistics and health metrics of the financial knowledge base",
                mime_type="application/json"
            ),
            MCPResource(
                uri="financial://agents/capabilities",
                name="Agent Capabilities",
                description="List of all available agent capabilities and their schemas",
                mime_type="application/json"
            ),
            MCPResource(
                uri="financial://documents/types",
                name="Document Types",
                description="Available financial document types in the knowledge base",
                mime_type="application/json"
            ),
            MCPResource(
                uri="financial://system/status",
                name="System Status",
                description="Current status of the financial RAG system",
                mime_type="application/json"
            )
        ]
    
    async def handle_tool_call(self, tool_name: str, arguments: Dict[str, Any], session_id: str = None) -> Dict[str, Any]:
        """Handle MCP tool calls"""
        try:
            session_id = session_id or f"mcp_session_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
            
            if tool_name == "generate_financial_content":
                return await self._handle_content_generation(arguments, session_id)
            elif tool_name == "answer_financial_question":
                return await self._handle_question_answering(arguments, session_id)
            elif tool_name == "update_knowledge_base":
                return await self._handle_knowledge_update(arguments, session_id)
            elif tool_name == "assess_knowledge_health":
                return await self._handle_health_assessment(arguments, session_id)
            elif tool_name == "search_financial_documents":
                return await self._handle_document_search(arguments, session_id)
            elif tool_name == "verify_source_credibility":
                return await self._handle_credibility_verification(arguments, session_id)
            elif tool_name == "coordinate_multi_agent_analysis":
                return await self._handle_multi_agent_coordination(arguments, session_id)
            else:
                return {"error": f"Unknown tool: {tool_name}", "success": False}
                
        except Exception as e:
            logger.error(f"Error handling tool call {tool_name}: {e}")
            return {"error": str(e), "success": False}
    
    async def _handle_content_generation(self, arguments: Dict[str, Any], session_id: str) -> Dict[str, Any]:
        """Handle content generation requests"""
        request = {
            "agent_type": AgentType.CONTENT_GENERATOR.value,
            "capability": "generate_financial_content",
            **arguments
        }
        
        return await self.orchestrator.process_request(request, session_id)
    
    async def _handle_question_answering(self, arguments: Dict[str, Any], session_id: str) -> Dict[str, Any]:
        """Handle question answering requests"""
        request = {
            "agent_type": AgentType.QA_AGENT.value,
            "capability": "answer_financial_question",
            **arguments
        }
        
        return await self.orchestrator.process_request(request, session_id)
    
    async def _handle_knowledge_update(self, arguments: Dict[str, Any], session_id: str) -> Dict[str, Any]:
        """Handle knowledge base update requests"""
        request = {
            "agent_type": AgentType.KNOWLEDGE_MANAGER.value,
            "capability": "update_knowledge_base",
            **arguments
        }
        
        return await self.orchestrator.process_request(request, session_id)
    
    async def _handle_health_assessment(self, arguments: Dict[str, Any], session_id: str) -> Dict[str, Any]:
        """Handle knowledge base health assessment"""
        request = {
            "agent_type": AgentType.KNOWLEDGE_MANAGER.value,
            "capability": "assess_knowledge_health",
            **arguments
        }
        
        return await self.orchestrator.process_request(request, session_id)
    
    async def _handle_document_search(self, arguments: Dict[str, Any], session_id: str) -> Dict[str, Any]:
        """Handle document search requests"""
        query = arguments["query"]
        document_types = arguments.get("document_types", [])
        top_k = arguments.get("top_k", 10)
        
        filters = {}
        if document_types:
            filters["document_type"] = document_types
        
        results = await self.kb_manager.search_knowledge_base(
            query=query,
            top_k=top_k,
            filters=filters
        )
        
        return {
            "results": results,
            "total_found": len(results),
            "query": query,
            "success": True
        }
    
    async def _handle_credibility_verification(self, arguments: Dict[str, Any], session_id: str) -> Dict[str, Any]:
        """Handle source credibility verification"""
        request = {
            "agent_type": AgentType.QA_AGENT.value,
            "capability": "verify_source_credibility",
            **arguments
        }
        
        return await self.orchestrator.process_request(request, session_id)
    
    async def _handle_multi_agent_coordination(self, arguments: Dict[str, Any], session_id: str) -> Dict[str, Any]:
        """Handle multi-agent coordination requests"""
        complex_request = {
            "type": arguments["request_type"],
            "content": arguments["content"],
            "requirements": arguments.get("requirements", {})
        }
        
        return await self.orchestrator.coordinate_agents(complex_request, session_id)
    
    async def handle_resource_read(self, resource_uri: str) -> Dict[str, Any]:
        """Handle MCP resource read requests"""
        try:
            if resource_uri == "financial://knowledge-base/statistics":
                return await self._get_knowledge_base_statistics()
            elif resource_uri == "financial://agents/capabilities":
                return await self._get_agent_capabilities()
            elif resource_uri == "financial://documents/types":
                return await self._get_document_types()
            elif resource_uri == "financial://system/status":
                return await self._get_system_status()
            else:
                return {"error": f"Unknown resource: {resource_uri}", "success": False}
                
        except Exception as e:
            logger.error(f"Error reading resource {resource_uri}: {e}")
            return {"error": str(e), "success": False}
    
    async def _get_knowledge_base_statistics(self) -> Dict[str, Any]:
        """Get knowledge base statistics"""
        stats = await self.kb_manager.get_knowledge_base_statistics()
        return {
            "content": stats,
            "mime_type": "application/json",
            "success": True
        }
    
    async def _get_agent_capabilities(self) -> Dict[str, Any]:
        """Get agent capabilities"""
        capabilities = self.orchestrator.get_agent_capabilities()
        return {
            "content": capabilities,
            "mime_type": "application/json",
            "success": True
        }
    
    async def _get_document_types(self) -> Dict[str, Any]:
        """Get available document types"""
        document_types = [
            "10-K", "10-Q", "8-K", "proxy-statement", 
            "annual-report", "earnings-report"
        ]
        return {
            "content": {"document_types": document_types},
            "mime_type": "application/json",
            "success": True
        }
    
    async def _get_system_status(self) -> Dict[str, Any]:
        """Get system status"""
        status = await self.orchestrator.get_system_status()
        return {
            "content": status,
            "mime_type": "application/json",
            "success": True
        }
    
    def get_tools(self) -> List[Dict[str, Any]]:
        """Get list of available MCP tools"""
        return [asdict(tool) for tool in self.tools]
    
    def get_resources(self) -> List[Dict[str, Any]]:
        """Get list of available MCP resources"""
        return [asdict(resource) for resource in self.resources]
    
    async def initialize(self):
        """Initialize the MCP server"""
        try:
            await self.kb_manager.initialize()
            logger.info("Financial MCP Server initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing MCP server: {e}")
            raise

class MCPServerManager:
    """Manager for multiple MCP servers"""
    
    def __init__(self):
        self.servers = {}
        self._initialize_servers()
    
    def _initialize_servers(self):
        """Initialize all MCP servers"""
        self.servers["financial_rag"] = FinancialMCPServer()
        logger.info(f"Initialized {len(self.servers)} MCP servers")
    
    async def initialize_all(self):
        """Initialize all MCP servers"""
        for name, server in self.servers.items():
            try:
                await server.initialize()
                logger.info(f"MCP server '{name}' initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize MCP server '{name}': {e}")
    
    def get_server(self, server_name: str) -> Optional[FinancialMCPServer]:
        """Get MCP server by name"""
        return self.servers.get(server_name)
    
    def list_servers(self) -> List[str]:
        """List all available MCP servers"""
        return list(self.servers.keys())
    
    async def handle_tool_call(self, server_name: str, tool_name: str, 
                             arguments: Dict[str, Any], session_id: str = None) -> Dict[str, Any]:
        """Handle tool call on specific server"""
        server = self.get_server(server_name)
        if not server:
            return {"error": f"Server '{server_name}' not found", "success": False}
        
        return await server.handle_tool_call(tool_name, arguments, session_id)
    
    async def handle_resource_read(self, server_name: str, resource_uri: str) -> Dict[str, Any]:
        """Handle resource read on specific server"""
        server = self.get_server(server_name)
        if not server:
            return {"error": f"Server '{server_name}' not found", "success": False}
        
        return await server.handle_resource_read(resource_uri)
    
    def get_all_tools(self) -> Dict[str, List[Dict[str, Any]]]:
        """Get tools from all servers"""
        all_tools = {}
        for name, server in self.servers.items():
            all_tools[name] = server.get_tools()
        return all_tools
    
    def get_all_resources(self) -> Dict[str, List[Dict[str, Any]]]:
        """Get resources from all servers"""
        all_resources = {}
        for name, server in self.servers.items():
            all_resources[name] = server.get_resources()
        return all_resources
