import asyncio
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

try:
    from azure.ai.projects import AIProjectClient
    from azure.ai.projects.models import (
        AgentThread, 
        AgentThreadMessage, 
        AgentThreadRun,
        MessageRole,
        RunStatus
    )
    from azure.identity import DefaultAzureCredential
    AZURE_AI_PROJECTS_AVAILABLE = True
    
    class Agent:
        def __init__(self, id: str = None, name: str = None, **kwargs):
            self.id = id or "mock-agent"
            self.name = name or "Mock Agent"
            
    class CodeInterpreterTool:
        def __init__(self):
            self.type = "code_interpreter"
            
    class FileSearchTool:
        def __init__(self):
            self.type = "file_search"
            
except ImportError:
    AZURE_AI_PROJECTS_AVAILABLE = False
    AIProjectClient = None
    AgentThread = None
    AgentThreadMessage = None
    AgentThreadRun = None
    MessageRole = None
    RunStatus = None
    DefaultAzureCredential = None
    
    class Agent:
        def __init__(self, id: str = None, name: str = None, **kwargs):
            self.id = id or "mock-agent"
            self.name = name or "Mock Agent"
            
    class CodeInterpreterTool:
        def __init__(self):
            self.type = "code_interpreter"
            
    class FileSearchTool:
        def __init__(self):
            self.type = "file_search"

from app.core.config import settings
from app.core.observability import observability

logger = logging.getLogger(__name__)

class AgentStatus(Enum):
    CREATED = "created"
    ACTIVE = "active"
    RUNNING = "running"
    COMPLETED = "completed"
    ERROR = "error"

@dataclass
class AgentConversation:
    agent_id: str
    thread_id: str
    messages: List[Dict[str, Any]]
    status: AgentStatus
    created_at: datetime
    updated_at: datetime

@dataclass
class AgentRunResult:
    run_id: str
    agent_id: str
    thread_id: str
    status: str
    response: str
    sources: List[Dict[str, Any]]
    metadata: Dict[str, Any]
    created_at: datetime

class AzureAIAgentService:
    """Azure AI Agent Service integration for MCP client functionality"""
    
    def __init__(self, project_client: AIProjectClient):
        self.client = project_client
        self.agents: Dict[str, Agent] = {}
        self.conversations: Dict[str, AgentConversation] = {}
        self.tools = self._initialize_tools()
        
    def _initialize_tools(self) -> List[Any]:
        """Initialize tools for agents"""
        tools = []
        
        code_interpreter = CodeInterpreterTool()
        tools.append(code_interpreter)
        
        file_search = FileSearchTool()
        tools.append(file_search)
        
        return tools
    
    async def create_qa_agent(self, name: str, instructions: str, model_deployment: str = None) -> Agent:
        """Create a QA agent for Exercise 2 functionality"""
        try:
            async with observability.trace_operation("create_qa_agent") as span:
                span.set_attribute("agent_name", name)
                
                if not model_deployment:
                    model_deployment = settings.AZURE_OPENAI_CHAT_DEPLOYMENT_NAME
                
                enhanced_instructions = f"""
                {instructions}
                
                You are a financial analysis expert specializing in question answering with source verification.
                
                Key capabilities:
                1. Answer complex financial questions comprehensively
                2. Decompose complex questions into researchable sub-questions
                3. Verify source credibility and trustworthiness
                4. Synthesize information from multiple sources
                5. Provide clear reasoning and justifications
                
                Guidelines:
                - Always cite sources with specific references
                - Evaluate source credibility (SEC filings are highly credible)
                - Acknowledge limitations and uncertainties
                - Use financial terminology appropriately
                - Provide quantitative analysis when possible
                """
                
                agent = self.client.agents.create_agent(
                    model=model_deployment,
                    name=name,
                    instructions=enhanced_instructions,
                    tools=self.tools
                )
                
                self.agents[agent.id] = agent
                
                span.set_attribute("agent_id", agent.id)
                span.set_attribute("success", True)
                
                logger.info(f"Created QA agent: {agent.id} with name: {name}")
                return agent
                
        except Exception as e:
            logger.error(f"Error creating QA agent: {e}")
            observability.record_error("create_qa_agent_error", str(e))
            raise
    
    async def create_content_generator_agent(self, name: str, instructions: str, model_deployment: str = None) -> Agent:
        """Create a content generator agent for Exercise 1 functionality"""
        try:
            async with observability.trace_operation("create_content_generator_agent") as span:
                span.set_attribute("agent_name", name)
                
                if not model_deployment:
                    model_deployment = settings.AZURE_OPENAI_CHAT_DEPLOYMENT_NAME
                
                enhanced_instructions = f"""
                {instructions}
                
                You are a financial content generation expert specializing in creating high-quality financial content.
                
                Key capabilities:
                1. Generate comprehensive financial reports and analyses
                2. Create executive summaries and technical documentation
                3. Enhance content with proper citations
                4. Adapt writing style and tone to context
                5. Ensure factual accuracy and professional standards
                
                Guidelines:
                - Use appropriate financial terminology
                - Include specific metrics and data points
                - Maintain professional tone and style
                - Cite all sources appropriately
                - Focus on accuracy and relevance
                """
                
                agent = self.client.agents.create_agent(
                    model=model_deployment,
                    name=name,
                    instructions=enhanced_instructions,
                    tools=self.tools
                )
                
                self.agents[agent.id] = agent
                
                span.set_attribute("agent_id", agent.id)
                span.set_attribute("success", True)
                
                logger.info(f"Created content generator agent: {agent.id} with name: {name}")
                return agent
                
        except Exception as e:
            logger.error(f"Error creating content generator agent: {e}")
            observability.record_error("create_content_generator_agent_error", str(e))
            raise
    
    async def create_thread(self, agent_id: str) -> str:
        """Create a new conversation thread for an agent"""
        try:
            async with observability.trace_operation("create_thread") as span:
                span.set_attribute("agent_id", agent_id)
                
                thread = self.client.agents.create_thread()
                thread_id = thread.id
                
                self.conversations[thread_id] = AgentConversation(
                    agent_id=agent_id,
                    thread_id=thread_id,
                    messages=[],
                    status=AgentStatus.CREATED,
                    created_at=datetime.utcnow(),
                    updated_at=datetime.utcnow()
                )
                
                span.set_attribute("thread_id", thread_id)
                span.set_attribute("success", True)
                
                logger.info(f"Created thread: {thread_id} for agent: {agent_id}")
                return thread_id
                
        except Exception as e:
            logger.error(f"Error creating thread: {e}")
            observability.record_error("create_thread_error", str(e))
            raise
    
    async def run_agent_conversation(self, agent_id: str, thread_id: str, message: str, context: Dict[str, Any] = None) -> AgentRunResult:
        """Execute agent conversation and return results"""
        try:
            async with observability.trace_operation("run_agent_conversation") as span:
                span.set_attribute("agent_id", agent_id)
                span.set_attribute("thread_id", thread_id)
                span.set_attribute("message_length", len(message))
                
                if context:
                    enhanced_message = f"""
                    Context: {context.get('knowledge_context', '')}
                    
                    Question: {message}
                    """
                else:
                    enhanced_message = message
                
                self.client.agents.create_message(
                    thread_id=thread_id,
                    role="user",
                    content=enhanced_message
                )
                
                if thread_id in self.conversations:
                    self.conversations[thread_id].messages.append({
                        "role": "user",
                        "content": message,
                        "timestamp": datetime.utcnow()
                    })
                    self.conversations[thread_id].status = AgentStatus.RUNNING
                    self.conversations[thread_id].updated_at = datetime.utcnow()
                
                run = self.client.agents.create_run(
                    thread_id=thread_id,
                    agent_id=agent_id
                )
                
                completed_run = await self._wait_for_run_completion(thread_id, run.id)
                
                messages = self.client.agents.list_messages(thread_id=thread_id)
                
                response_content = ""
                sources = []
                
                for message in messages.data:
                    if message.role == "assistant":
                        for content in message.content:
                            if hasattr(content, 'text'):
                                response_content = content.text.value
                                if hasattr(content.text, 'annotations'):
                                    for annotation in content.text.annotations:
                                        if hasattr(annotation, 'file_citation'):
                                            sources.append({
                                                "type": "file_citation",
                                                "file_id": annotation.file_citation.file_id,
                                                "quote": annotation.file_citation.quote
                                            })
                        break
                
                if thread_id in self.conversations:
                    self.conversations[thread_id].messages.append({
                        "role": "assistant",
                        "content": response_content,
                        "timestamp": datetime.utcnow()
                    })
                    self.conversations[thread_id].status = AgentStatus.COMPLETED
                    self.conversations[thread_id].updated_at = datetime.utcnow()
                
                result = AgentRunResult(
                    run_id=run.id,
                    agent_id=agent_id,
                    thread_id=thread_id,
                    status=completed_run.status,
                    response=response_content,
                    sources=sources,
                    metadata={
                        "context": context,
                        "run_steps": completed_run.steps if hasattr(completed_run, 'steps') else []
                    },
                    created_at=datetime.utcnow()
                )
                
                span.set_attribute("response_length", len(response_content))
                span.set_attribute("sources_count", len(sources))
                span.set_attribute("success", True)
                
                logger.info(f"Completed agent conversation: {run.id}")
                return result
                
        except Exception as e:
            logger.error(f"Error running agent conversation: {e}")
            observability.record_error("run_agent_conversation_error", str(e))
            
            if thread_id in self.conversations:
                self.conversations[thread_id].status = AgentStatus.ERROR
                self.conversations[thread_id].updated_at = datetime.utcnow()
            
            raise
    
    async def _wait_for_run_completion(self, thread_id: str, run_id: str, timeout: int = 300) -> Any:
        """Wait for agent run to complete with timeout"""
        start_time = datetime.utcnow()
        
        while (datetime.utcnow() - start_time).seconds < timeout:
            run = self.client.agents.get_run(thread_id=thread_id, run_id=run_id)
            
            if run.status in ["completed", "failed", "cancelled", "expired"]:
                return run
            
            await asyncio.sleep(2)  # Poll every 2 seconds
        
        raise TimeoutError(f"Agent run {run_id} did not complete within {timeout} seconds")
    
    async def get_agent_capabilities(self, agent_id: str) -> Dict[str, Any]:
        """Get agent capabilities and status"""
        try:
            if agent_id not in self.agents:
                raise ValueError(f"Agent {agent_id} not found")
            
            agent = self.agents[agent_id]
            
            return {
                "agent_id": agent.id,
                "name": agent.name,
                "model": agent.model,
                "tools": [tool.__class__.__name__ for tool in self.tools],
                "instructions": agent.instructions,
                "created_at": agent.created_at if hasattr(agent, 'created_at') else None
            }
            
        except Exception as e:
            logger.error(f"Error getting agent capabilities: {e}")
            raise
    
    async def list_agents(self) -> List[Dict[str, Any]]:
        """List all created agents"""
        try:
            agents_list = []
            
            for agent_id, agent in self.agents.items():
                agents_list.append({
                    "agent_id": agent.id,
                    "name": agent.name,
                    "model": agent.model,
                    "status": "active",
                    "created_at": agent.created_at if hasattr(agent, 'created_at') else None
                })
            
            return agents_list
            
        except Exception as e:
            logger.error(f"Error listing agents: {e}")
            raise
    
    async def get_conversation_history(self, thread_id: str) -> Optional[AgentConversation]:
        """Get conversation history for a thread"""
        return self.conversations.get(thread_id)
    
    async def delete_agent(self, agent_id: str) -> bool:
        """Delete an agent"""
        try:
            if agent_id in self.agents:
                self.client.agents.delete_agent(agent_id)
                del self.agents[agent_id]
                logger.info(f"Deleted agent: {agent_id}")
                return True
            return False
            
        except Exception as e:
            logger.error(f"Error deleting agent: {e}")
            raise
    
    async def cleanup_conversations(self, max_age_hours: int = 24) -> int:
        """Clean up old conversations"""
        try:
            cutoff_time = datetime.utcnow() - timedelta(hours=max_age_hours)
            cleaned_count = 0
            
            threads_to_remove = []
            for thread_id, conversation in self.conversations.items():
                if conversation.updated_at < cutoff_time:
                    threads_to_remove.append(thread_id)
            
            for thread_id in threads_to_remove:
                try:
                    self.client.agents.delete_thread(thread_id)
                    del self.conversations[thread_id]
                    cleaned_count += 1
                except Exception as e:
                    logger.warning(f"Error cleaning up thread {thread_id}: {e}")
            
            logger.info(f"Cleaned up {cleaned_count} old conversations")
            return cleaned_count
            
        except Exception as e:
            logger.error(f"Error cleaning up conversations: {e}")
            return 0
    
    async def process_qa_request(self, question: str, context: Dict[str, Any], verification_level: str, session_id: str, model_config: Dict[str, Any]) -> Dict[str, Any]:
        """Process QA request using Azure AI Agent Service"""
        try:
            async with observability.trace_operation("azure_ai_agent_qa_processing") as span:
                span.set_attribute("question_length", len(question))
                span.set_attribute("verification_level", verification_level)
                span.set_attribute("session_id", session_id)
                
                qa_agent = await self.create_qa_agent(
                    name=f"QA_Agent_{session_id}",
                    instructions=f"""
                    You are a financial QA expert specializing in comprehensive question answering with source verification.
                    
                    Verification Level: {verification_level}
                    
                    Your task is to:
                    1. Analyze the financial question thoroughly
                    2. Retrieve relevant information from available sources
                    3. Provide a comprehensive, accurate answer
                    4. Include proper citations and source references
                    5. Assess confidence level based on source quality
                    
                    Always maintain high standards for financial accuracy and cite your sources appropriately.
                    """,
                    model_deployment=model_config.get("chat_model", "gpt-4")
                )
                
                thread_id = await self.create_thread(qa_agent.id)
                
                enhanced_context = {
                    **context,
                    "verification_level": verification_level,
                    "model_config": model_config
                }
                
                result = await self.run_agent_conversation(
                    agent_id=qa_agent.id,
                    thread_id=thread_id,
                    message=question,
                    context=enhanced_context
                )
                
                return {
                    "answer": result.response,
                    "confidence_score": 0.8,  # Default confidence, could be enhanced
                    "sources": result.sources,
                    "sub_questions": [],  # Could be enhanced with decomposition
                    "verification_details": {
                        "verification_level": verification_level,
                        "agent_id": qa_agent.id,
                        "thread_id": thread_id
                    }
                }
                
        except Exception as e:
            logger.error(f"Error processing QA request: {e}")
            observability.record_error("azure_ai_agent_qa_error", str(e))
            raise
    
    async def decompose_complex_question(self, question: str, context: Dict[str, Any], session_id: str, model_config: Dict[str, Any]) -> Dict[str, Any]:
        """Decompose complex questions into sub-questions using Azure AI Agent Service"""
        try:
            async with observability.trace_operation("azure_ai_agent_question_decomposition") as span:
                span.set_attribute("question_length", len(question))
                span.set_attribute("session_id", session_id)
                
                decomposition_agent = await self.create_qa_agent(
                    name=f"Decomposition_Agent_{session_id}",
                    instructions="""
                    You are a financial question decomposition expert. Your task is to break down complex financial questions into smaller, researchable sub-questions.
                    
                    Guidelines:
                    1. Identify the main components of the complex question
                    2. Break it down into 3-7 specific, actionable sub-questions
                    3. Ensure each sub-question can be researched independently
                    4. Maintain logical flow and dependencies between sub-questions
                    5. Focus on financial analysis, data requirements, and research needs
                    
                    Format your response as:
                    Sub-questions:
                    1. [First sub-question]
                    2. [Second sub-question]
                    ...
                    
                    Reasoning: [Explain your decomposition approach]
                    """,
                    model_deployment=model_config.get("chat_model", "gpt-4")
                )
                
                thread_id = await self.create_thread(decomposition_agent.id)
                
                result = await self.run_agent_conversation(
                    agent_id=decomposition_agent.id,
                    thread_id=thread_id,
                    message=f"Decompose this complex financial question: {question}",
                    context=context
                )
                
                # Parse the response to extract sub-questions
                response_text = result.response
                sub_questions = []
                reasoning = "Question decomposition completed."
                
                if "Sub-questions:" in response_text:
                    lines = response_text.split('\n')
                    in_questions = False
                    in_reasoning = False
                    
                    for line in lines:
                        line = line.strip()
                        if line.startswith("Sub-questions:"):
                            in_questions = True
                            continue
                        elif line.startswith("Reasoning:"):
                            in_questions = False
                            in_reasoning = True
                            reasoning = line.replace("Reasoning:", "").strip()
                            continue
                        elif in_questions and line and (line[0].isdigit() or line.startswith("-")):
                            question_text = line
                            if '. ' in question_text:
                                question_text = question_text.split('. ', 1)[1]
                            elif '- ' in question_text:
                                question_text = question_text.split('- ', 1)[1]
                            sub_questions.append(question_text.strip())
                        elif in_reasoning:
                            reasoning += " " + line
                
                return {
                    "sub_questions": sub_questions,
                    "reasoning": reasoning.strip(),
                    "agent_id": decomposition_agent.id,
                    "thread_id": thread_id
                }
                
        except Exception as e:
            logger.error(f"Error decomposing question: {e}")
            observability.record_error("azure_ai_agent_decomposition_error", str(e))
            raise
    
    async def verify_source_credibility(self, sources: List[Dict[str, Any]], context: Dict[str, Any], session_id: str) -> Dict[str, Any]:
        """Verify source credibility using Azure AI Agent Service"""
        try:
            async with observability.trace_operation("azure_ai_agent_source_verification") as span:
                span.set_attribute("sources_count", len(sources))
                span.set_attribute("session_id", session_id)
                
                verification_agent = await self.create_qa_agent(
                    name=f"Verification_Agent_{session_id}",
                    instructions="""
                    You are a financial source credibility expert. Your task is to assess the trustworthiness and reliability of financial information sources.
                    
                    For each source, evaluate:
                    1. Source authority and reputation
                    2. Content accuracy and consistency
                    3. Publication date and relevance
                    4. Potential bias or conflicts of interest
                    5. Supporting evidence and citations
                    
                    Provide a credibility score (0.0-1.0) and explanation for each source.
                    Identify trust indicators and red flags.
                    
                    Format your response as:
                    Source 1: [URL/Title]
                    Credibility Score: [0.0-1.0]
                    Explanation: [Detailed assessment]
                    Trust Indicators: [List positive factors]
                    Red Flags: [List concerning factors]
                    
                    [Repeat for each source]
                    
                    Overall Assessment: [Summary of findings]
                    """,
                    model_deployment="gpt-4"
                )
                
                thread_id = await self.create_thread(verification_agent.id)
                
                sources_text = ""
                for i, source in enumerate(sources, 1):
                    sources_text += f"""
                    Source {i}:
                    URL: {source.url if hasattr(source, 'url') else 'N/A'}
                    Title: {source.title if hasattr(source, 'title') else 'N/A'}
                    Content: {source.content[:500] if hasattr(source, 'content') and source.content else ''}...
                    Metadata: {getattr(source, 'metadata', {})}
                    
                    """
                
                result = await self.run_agent_conversation(
                    agent_id=verification_agent.id,
                    thread_id=thread_id,
                    message=f"Verify the credibility of these financial sources:\n{sources_text}",
                    context=context
                )
                
                verified_sources = []
                overall_credibility = 0.0
                
                for i, source in enumerate(sources):
                    credibility_score = 0.5
                    explanation = "Credibility assessment completed"
                    trust_indicators = []
                    red_flags = []
                    
                    if f"Source {i+1}:" in result.response:
                        source_section = result.response.split(f"Source {i+1}:")[1]
                        if len(sources) > i+1:
                            source_section = source_section.split(f"Source {i+2}:")[0]
                        
                        if "Credibility Score:" in source_section:
                            score_line = source_section.split("Credibility Score:")[1].split('\n')[0]
                            try:
                                credibility_score = float(score_line.strip())
                            except:
                                pass
                    
                    verified_source = {
                        "source_id": getattr(source, "id", f"source_{i+1}"),
                        "url": getattr(source, "url", ""),
                        "title": getattr(source, "title", ""),
                        "content": getattr(source, "content", ""),
                        "credibility_score": credibility_score,
                        "credibility_explanation": explanation,
                        "trust_indicators": trust_indicators,
                        "red_flags": red_flags,
                        "verification_status": "verified" if credibility_score > 0.7 else "questionable"
                    }
                    
                    verified_sources.append(verified_source)
                    overall_credibility += credibility_score
                
                if verified_sources:
                    overall_credibility = overall_credibility / len(verified_sources)
                
                return {
                    "verified_sources": verified_sources,
                    "overall_credibility_score": overall_credibility,
                    "verification_summary": f"Verified {len(verified_sources)} sources with average credibility score of {overall_credibility:.2f}",
                    "agent_id": verification_agent.id,
                    "thread_id": thread_id
                }
                
        except Exception as e:
            logger.error(f"Error verifying source credibility: {e}")
            observability.record_error("azure_ai_agent_verification_error", str(e))
            raise

class MockAzureAIAgentService(AzureAIAgentService):
    """Mock implementation for testing and development"""
    
    def __init__(self):
        self.agents = {}
        self.conversations = {}
        self.mock_agent_counter = 0
        self.mock_thread_counter = 0
        
    async def create_qa_agent(self, name: str, instructions: str, model_deployment: str = None) -> Agent:
        """Mock QA agent creation"""
        self.mock_agent_counter += 1
        agent_id = f"mock_qa_agent_{self.mock_agent_counter}"
        
        mock_agent = type('MockAgent', (), {
            'id': agent_id,
            'name': name,
            'instructions': instructions,
            'model': model_deployment or "mock-gpt-4",
            'created_at': datetime.utcnow()
        })()
        
        self.agents[agent_id] = mock_agent
        logger.info(f"Created mock QA agent: {agent_id}")
        return mock_agent
    
    async def create_content_generator_agent(self, name: str, instructions: str, model_deployment: str = None) -> Agent:
        """Mock content generator agent creation"""
        self.mock_agent_counter += 1
        agent_id = f"mock_content_agent_{self.mock_agent_counter}"
        
        mock_agent = type('MockAgent', (), {
            'id': agent_id,
            'name': name,
            'instructions': instructions,
            'model': model_deployment or "mock-gpt-4",
            'created_at': datetime.utcnow()
        })()
        
        self.agents[agent_id] = mock_agent
        logger.info(f"Created mock content generator agent: {agent_id}")
        return mock_agent
    
    async def create_thread(self, agent_id: str) -> str:
        """Mock thread creation"""
        self.mock_thread_counter += 1
        thread_id = f"mock_thread_{self.mock_thread_counter}"
        
        self.conversations[thread_id] = AgentConversation(
            agent_id=agent_id,
            thread_id=thread_id,
            messages=[],
            status=AgentStatus.CREATED,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
        
        logger.info(f"Created mock thread: {thread_id}")
        return thread_id
    
    async def run_agent_conversation(self, agent_id: str, thread_id: str, message: str, context: Dict[str, Any] = None) -> AgentRunResult:
        """Mock agent conversation execution"""
        
        await asyncio.sleep(0.1)
        
        if "question" in message.lower() or "what" in message.lower():
            mock_response = f"Based on the financial analysis, here is a comprehensive answer to your question: {message[:100]}... [Mock response with financial insights and data]"
        else:
            mock_response = f"Mock financial analysis response for: {message[:50]}..."
        
        mock_sources = [
            {
                "type": "file_citation",
                "file_id": "mock_10k_filing.pdf",
                "quote": "Sample financial data from 10-K filing"
            }
        ]
        
        if thread_id in self.conversations:
            self.conversations[thread_id].messages.extend([
                {
                    "role": "user",
                    "content": message,
                    "timestamp": datetime.utcnow()
                },
                {
                    "role": "assistant", 
                    "content": mock_response,
                    "timestamp": datetime.utcnow()
                }
            ])
            self.conversations[thread_id].status = AgentStatus.COMPLETED
            self.conversations[thread_id].updated_at = datetime.utcnow()
        
        result = AgentRunResult(
            run_id=f"mock_run_{datetime.utcnow().timestamp()}",
            agent_id=agent_id,
            thread_id=thread_id,
            status="completed",
            response=mock_response,
            sources=mock_sources,
            metadata={"context": context, "mock": True},
            created_at=datetime.utcnow()
        )
        
        logger.info(f"Completed mock agent conversation")
        return result
    
    async def process_qa_request(self, question: str, context: Dict[str, Any], verification_level: str, session_id: str, model_config: Dict[str, Any]) -> Dict[str, Any]:
        """Mock QA request processing"""
        await asyncio.sleep(0.2)  # Simulate processing time
        
        mock_answer = f"""Based on comprehensive financial analysis, here is the answer to your question: {question[:100]}...

Key findings:
- Financial metric analysis shows positive trends
- Market conditions indicate stable performance
- Risk assessment suggests moderate exposure
- Regulatory compliance appears satisfactory

This analysis is based on the latest available financial data and market research."""
        
        mock_sources = [
            {
                "type": "file_citation",
                "file_id": "mock_10k_filing.pdf",
                "quote": "Revenue increased by 15% year-over-year"
            },
            {
                "type": "file_citation", 
                "file_id": "mock_earnings_report.pdf",
                "quote": "EBITDA margin improved to 22.5%"
            }
        ]
        
        return {
            "answer": mock_answer,
            "confidence_score": 0.85,
            "sources": mock_sources,
            "sub_questions": [
                "What are the key financial metrics?",
                "How do market conditions affect performance?",
                "What are the primary risk factors?"
            ],
            "verification_details": {
                "verification_level": verification_level,
                "agent_id": f"mock_qa_agent_{session_id}",
                "thread_id": f"mock_thread_{session_id}"
            }
        }
    
    async def decompose_complex_question(self, question: str, context: Dict[str, Any], session_id: str, model_config: Dict[str, Any]) -> Dict[str, Any]:
        """Mock question decomposition"""
        await asyncio.sleep(0.1)
        
        sub_questions = []
        if "financial" in question.lower():
            sub_questions.extend([
                "What are the current financial metrics and KPIs?",
                "How do these metrics compare to industry benchmarks?",
                "What are the historical trends for these metrics?"
            ])
        
        if "market" in question.lower():
            sub_questions.extend([
                "What is the current market position?",
                "Who are the main competitors?",
                "What are the market growth projections?"
            ])
        
        if "risk" in question.lower():
            sub_questions.extend([
                "What are the primary risk factors?",
                "How is risk being mitigated?",
                "What is the risk tolerance level?"
            ])
        
        if not sub_questions:
            sub_questions = [
                "What is the primary focus of this analysis?",
                "What data sources are most relevant?",
                "What are the key success metrics?"
            ]
        
        return {
            "sub_questions": sub_questions[:5],  # Limit to 5 sub-questions
            "reasoning": f"The question '{question[:50]}...' has been decomposed into specific, researchable components focusing on financial analysis, market conditions, and risk assessment.",
            "agent_id": f"mock_decomposition_agent_{session_id}",
            "thread_id": f"mock_thread_{session_id}"
        }
    
    async def verify_source_credibility(self, sources: List[Dict[str, Any]], context: Dict[str, Any], session_id: str) -> Dict[str, Any]:
        """Mock source credibility verification"""
        await asyncio.sleep(0.15)
        
        verified_sources = []
        total_credibility = 0.0
        
        for i, source in enumerate(sources):
            base_score = 0.7
            
            if source.get("url", "").endswith(".gov"):
                base_score += 0.2
            elif source.get("url", "").endswith(".edu"):
                base_score += 0.15
            elif "sec.gov" in source.get("url", ""):
                base_score += 0.25
            
            if len(source.get("content", "")) > 1000:
                base_score += 0.05
            
            credibility_score = min(base_score, 1.0)
            
            verified_source = {
                "source_id": source.get("id", f"mock_source_{i+1}"),
                "url": source.get("url", ""),
                "title": source.get("title", f"Mock Financial Document {i+1}"),
                "content": source.get("content", ""),
                "credibility_score": credibility_score,
                "credibility_explanation": f"Source assessed with credibility score of {credibility_score:.2f} based on authority, content quality, and relevance.",
                "trust_indicators": [
                    "Authoritative domain",
                    "Comprehensive content",
                    "Recent publication date",
                    "Professional formatting"
                ],
                "red_flags": [] if credibility_score > 0.7 else ["Limited source information"],
                "verification_status": "verified" if credibility_score > 0.7 else "questionable"
            }
            
            verified_sources.append(verified_source)
            total_credibility += credibility_score
        
        overall_credibility = total_credibility / len(sources) if sources else 0.5
        
        return {
            "verified_sources": verified_sources,
            "overall_credibility_score": overall_credibility,
            "verification_summary": f"Verified {len(verified_sources)} sources with average credibility score of {overall_credibility:.2f}",
            "agent_id": f"mock_verification_agent_{session_id}",
            "thread_id": f"mock_thread_{session_id}"
        }
