import asyncio
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.core.observability import observability
from app.core.evaluation import setup_evaluation_framework, FinancialEvaluationContext
import time
import json

async def test_observability_and_evaluation():
    """Test observability and evaluation framework integration"""
    
    print("Testing Observability and Evaluation Framework Integration")
    print("=" * 60)
    
    print("\n1. Testing basic observability tracking...")
    
    observability.track_request("test_endpoint", user_id="test_user", session_id="test_session")
    observability.track_tokens("gpt-4", 100, 50, session_id="test_session")
    observability.track_response_time("test_endpoint", 1.5, model="gpt-4", session_id="test_session")
    
    observability.track_error("TestError", "test_endpoint", "Test error message", session_id="test_session")
    
    print("✓ Basic tracking completed")
    
    print("\n2. Testing metrics summary generation...")
    
    metrics_summary = observability.get_metrics_summary(hours=24)
    print(f"✓ Generated metrics summary with {len(metrics_summary)} top-level keys")
    print(f"  - Total requests: {metrics_summary.get('summary', {}).get('total_requests', 0)}")
    print(f"  - Total tokens: {metrics_summary.get('summary', {}).get('total_tokens', 0)}")
    print(f"  - Total cost: ${metrics_summary.get('summary', {}).get('total_cost', 0)}")
    
    print("\n3. Testing distributed tracing...")
    
    async with observability.trace_operation(
        "test_operation",
        session_id="test_session",
        model="gpt-4"
    ) as span:
        await asyncio.sleep(0.1)
        span.set_attribute("test.result", "success")
    
    print("✓ Distributed tracing test completed")
    
    print("\n4. Testing system metrics tracking...")
    
    observability.track_system_metrics(
        cpu_usage=45.2,
        memory_usage=67.8,
        disk_usage=23.1
    )
    
    print("✓ System metrics tracking completed")
    
    print("\n5. Testing evaluation metrics tracking...")
    
    sample_evaluation_results = [
        {
            "metric": "relevance",
            "score": 0.85,
            "model_used": "gpt-4",
            "reasoning": "Response is highly relevant to the query"
        },
        {
            "metric": "groundedness", 
            "score": 0.92,
            "model_used": "gpt-4",
            "reasoning": "Response is well-grounded in source documents"
        },
        {
            "metric": "financial_accuracy",
            "score": 0.88,
            "model_used": "gpt-4", 
            "reasoning": "Financial information is accurate"
        }
    ]
    
    observability.track_evaluation_metrics("test_session", sample_evaluation_results)
    print("✓ Evaluation metrics tracking completed")
    
    print("\n6. Testing comprehensive metrics summary...")
    
    final_summary = observability.get_metrics_summary(hours=1)
    
    print("✓ Final metrics summary:")
    print(f"  - Requests: {final_summary.get('summary', {}).get('total_requests', 0)}")
    print(f"  - Errors: {final_summary.get('summary', {}).get('total_errors', 0)}")
    print(f"  - Avg Response Time: {final_summary.get('summary', {}).get('avg_response_time', 0):.3f}s")
    print(f"  - System Health: {final_summary.get('summary', {}).get('system_health', 'unknown')}")
    print(f"  - Evaluation Metrics: {len(final_summary.get('evaluation_metrics', {}))}")
    
    print("\n" + "=" * 60)
    print("✅ All observability and evaluation tests completed successfully!")
    
    return final_summary

if __name__ == "__main__":
    result = asyncio.run(test_observability_and_evaluation())
    print(f"\nTest completed. Final summary has {len(result)} sections.")
