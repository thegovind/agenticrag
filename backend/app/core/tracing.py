"""
Azure AI Foundry Tracing Setup with Application Insights

This module sets up OpenTelemetry tracing for Azure AI Foundry projects 
using Application Insights as the backend. It instruments OpenAI SDK calls
and Azure AI Projects SDK to provide detailed traces visible in the 
Azure AI Foundry portal.

Based on: https://learn.microsoft.com/en-us/azure/ai-foundry/how-to/develop/trace-application
"""

import os
import logging
from typing import Optional
from opentelemetry.instrumentation.openai_v2 import OpenAIInstrumentor
from azure.monitor.opentelemetry import configure_azure_monitor
from opentelemetry import trace
from azure.ai.projects import AIProjectClient
from azure.identity import DefaultAzureCredential

logger = logging.getLogger(__name__)

# Global tracer instance for custom spans
tracer = None


def setup_ai_foundry_tracing(
    connection_string: Optional[str] = None,
    project_endpoint: Optional[str] = None,
    capture_content: bool = True
) -> bool:
    """
    Set up Azure AI Foundry tracing with Application Insights.
    
    Args:
        connection_string: Application Insights connection string. 
                          If None, will try to get from environment or Azure AI project.
        project_endpoint: Azure AI project endpoint for retrieving connection string.
                         If None, will use AZURE_AI_PROJECT_ENDPOINT from environment.
        capture_content: Whether to capture input/output content in traces.
    
    Returns:
        bool: True if tracing was set up successfully, False otherwise.
    """
    global tracer
    
    try:
        # Step 1: Set environment variable for capturing message content
        if capture_content:
            os.environ["OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT"] = "true"
            logger.info("✅ Enabled OpenTelemetry message content capture")
        
        # Step 2: Get connection string
        if not connection_string:
            # First try environment variable
            connection_string = os.getenv("AZURE_MONITOR_CONNECTION_STRING") or os.getenv("APPLICATIONINSIGHTS_CONNECTION_STRING")
            
            # If not found, try to get from Azure AI project
            if not connection_string and project_endpoint:
                try:
                    project_client = AIProjectClient(
                        credential=DefaultAzureCredential(),
                        endpoint=project_endpoint
                    )
                    connection_string = project_client.telemetry.get_connection_string()
                    logger.info("✅ Retrieved connection string from Azure AI project")
                except Exception as e:
                    logger.warning(f"Failed to get connection string from Azure AI project: {e}")
            
            # If still not found, try from environment endpoint
            if not connection_string:
                project_endpoint = project_endpoint or os.getenv("AZURE_AI_PROJECT_ENDPOINT")
                if project_endpoint:
                    try:
                        project_client = AIProjectClient(
                            credential=DefaultAzureCredential(),
                            endpoint=project_endpoint
                        )
                        connection_string = project_client.telemetry.get_connection_string()
                        logger.info("✅ Retrieved connection string from Azure AI project endpoint")
                    except Exception as e:
                        logger.warning(f"Failed to get connection string from project endpoint: {e}")
        
        if not connection_string:
            logger.error("❌ No Application Insights connection string found. Please set AZURE_MONITOR_CONNECTION_STRING or APPLICATIONINSIGHTS_CONNECTION_STRING environment variable.")
            return False
        
        # Step 3: Instrument OpenAI SDK
        OpenAIInstrumentor().instrument()
        logger.info("✅ OpenAI SDK instrumented for tracing")
        
        # Step 4: Configure Azure Monitor
        configure_azure_monitor(connection_string=connection_string)
        logger.info("✅ Azure Monitor configured for tracing")
        
        # Step 5: Set up global tracer for custom spans
        tracer = trace.get_tracer(__name__)
        logger.info("✅ Global tracer configured for custom spans")
        
        logger.info("🎯 Azure AI Foundry tracing setup completed successfully!")
        logger.info("📊 Traces will be visible in Azure AI Foundry portal under Tracing section")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to set up Azure AI Foundry tracing: {e}")
        return False


def get_tracer():
    """
    Get the global tracer instance for creating custom spans.
    
    Returns:
        Tracer instance or None if tracing is not set up.
    """
    return tracer


def trace_operation(operation_name: str):
    """
    Decorator for tracing operations with custom spans.
    
    Args:
        operation_name: Name of the operation to trace.
    
    Usage:
        @trace_operation("process_documents")
        def process_documents(docs):
            # Your code here
            pass
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            if tracer:
                with tracer.start_as_current_span(operation_name) as span:
                    # Add operation attributes
                    span.set_attribute("operation.name", operation_name)
                    span.set_attribute("operation.function", func.__name__)
                    
                    try:
                        result = func(*args, **kwargs)
                        span.set_attribute("operation.status", "success")
                        return result
                    except Exception as e:
                        span.set_attribute("operation.status", "error")
                        span.set_attribute("operation.error", str(e))
                        raise
            else:
                return func(*args, **kwargs)
        return wrapper
    return decorator


def add_trace_attributes(**attributes):
    """
    Add custom attributes to the current span.
    
    Args:
        **attributes: Key-value pairs to add as span attributes.
    """
    if tracer:
        current_span = trace.get_current_span()
        if current_span.is_recording():
            for key, value in attributes.items():
                current_span.set_attribute(key, value)


def setup_console_tracing():
    """
    Set up tracing to console for local development and testing.
    Useful for CI/CD pipelines and debugging.
    """
    try:
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import SimpleSpanProcessor, ConsoleSpanExporter
        
        # Instrument OpenAI SDK
        OpenAIInstrumentor().instrument()
        
        # Configure console tracing
        span_exporter = ConsoleSpanExporter()
        tracer_provider = TracerProvider()
        tracer_provider.add_span_processor(SimpleSpanProcessor(span_exporter))
        trace.set_tracer_provider(tracer_provider)
        
        global tracer
        tracer = trace.get_tracer(__name__)
        
        logger.info("✅ Console tracing configured for local development")
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to set up console tracing: {e}")
        return False
