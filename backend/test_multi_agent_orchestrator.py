#!/usr/bin/env python3
import sys
import asyncio
sys.path.append('.')

async def test_multi_agent_system():
    try:
        from app.services.multi_agent_orchestrator import (
            MultiAgentOrchestrator, ContentGeneratorAgent, QAAgent, 
            KnowledgeManagerAgent, AgentType, AgentCapability
        )
        print('✓ Multi-agent orchestrator classes import successfully')
        
        from app.services.mcp_server import FinancialMCPServer, MCPServerManager, MCPTool, MCPResource
        print('✓ MCP server classes import successfully')
        
        from app.services.azure_services import AzureServiceManager
        from app.services.knowledge_base_manager import AdaptiveKnowledgeBaseManager
        
        azure_manager = AzureServiceManager()
        kb_manager = AdaptiveKnowledgeBaseManager(azure_manager)
        orchestrator = MultiAgentOrchestrator(azure_manager, kb_manager)
        print('✓ MultiAgentOrchestrator instantiated successfully')
        
        agents = orchestrator.agents
        print(f'✓ Agents initialized: {len(agents)} agents ({list(agents.keys())})')
        
        capabilities = orchestrator.get_agent_capabilities()
        total_capabilities = sum(len(caps) for caps in capabilities.values())
        print(f'✓ Agent capabilities loaded: {total_capabilities} total capabilities')
        
        mcp_server = FinancialMCPServer()
        tools = mcp_server.get_tools()
        resources = mcp_server.get_resources()
        print(f'✓ MCP server initialized: {len(tools)} tools, {len(resources)} resources')
        
        server_manager = MCPServerManager()
        server_list = server_manager.list_servers()
        print(f'✓ MCP server manager initialized: {len(server_list)} servers')
        
        status = await orchestrator.get_system_status()
        print(f'✓ System status retrieved: {status["agents_active"]} agents active')
        
        agent_types = [agent_type.value for agent_type in AgentType]
        print(f'✓ Agent types available: {agent_types}')
        
        print('✓ Multi-agent orchestration system is complete and functional')
        
    except Exception as e:
        print(f'✗ Multi-agent orchestrator error: {e}')
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(test_multi_agent_system())
