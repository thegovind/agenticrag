import asyncio
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

async def test_backend_integration():
    """Test backend API integration with observability and evaluation"""
    
    print("Testing Backend API Integration with Observability and Evaluation")
    print("=" * 70)
    
    try:
        print("\n1. Testing core imports...")
        from app.core.observability import observability
        from app.core.evaluation import setup_evaluation_framework, get_evaluation_framework
        from app.api.routes.admin import router as admin_router
        from app.api.routes.chat import router as chat_router
        print("✓ All core imports successful")
        
        print("\n2. Testing observability integration...")
        
        observability.track_request('test_api', session_id='integration_test')
        observability.track_tokens('gpt-4', 150, 75, session_id='integration_test')
        observability.track_response_time('test_api', 2.3, session_id='integration_test')
        
        observability.track_error('TestError', 'test_endpoint', 'Integration test error', 
                                session_id='integration_test')
        
        observability.track_system_metrics(cpu_usage=55.2, memory_usage=72.1, disk_usage=28.5)
        
        sample_eval_results = [
            {
                "metric": "relevance",
                "score": 0.89,
                "model_used": "gpt-4",
                "reasoning": "Highly relevant response"
            },
            {
                "metric": "groundedness",
                "score": 0.94,
                "model_used": "gpt-4",
                "reasoning": "Well-grounded in source material"
            }
        ]
        observability.track_evaluation_metrics('integration_test', sample_eval_results)
        
        print("✓ Observability tracking completed")
        
        print("\n3. Testing metrics summary generation...")
        
        metrics = observability.get_metrics_summary(hours=1)
        print(f"✓ Generated metrics summary with {len(metrics)} sections:")
        for key, value in metrics.items():
            if isinstance(value, dict) and 'total_requests' in value:
                print(f"  - {key}: {value.get('total_requests', 0)} requests")
            else:
                print(f"  - {key}: {type(value).__name__}")
        
        print("\n4. Testing evaluation framework setup...")
        
        class MockOpenAIClient:
            class chat:
                class completions:
                    @staticmethod
                    async def create(**kwargs):
                        class MockResponse:
                            class choices:
                                class message:
                                    content = '{"relevance": 0.85, "reasoning": "Mock evaluation result"}'
                            choices = [choices()]
                        return MockResponse()
        
        mock_client = MockOpenAIClient()
        eval_framework = setup_evaluation_framework(mock_client)
        print("✓ Evaluation framework setup successful")
        
        retrieved_framework = get_evaluation_framework()
        print("✓ Evaluation framework retrieval successful")
        
        print("\n5. Testing admin API routes...")
        
        admin_routes = [route.path for route in admin_router.routes if hasattr(route, 'path')]
        print(f"✓ Admin routes available ({len(admin_routes)}):")
        for route in admin_routes:
            print(f"  - {route}")
        
        print("\n6. Testing chat API routes...")
        
        chat_routes = [route.path for route in chat_router.routes if hasattr(route, 'path')]
        print(f"✓ Chat routes available ({len(chat_routes)}):")
        for route in chat_routes:
            print(f"  - {route}")
        
        print("\n7. Testing distributed tracing...")
        
        async with observability.trace_operation(
            "integration_test_operation",
            session_id="integration_test",
            model="gpt-4"
        ) as span:
            await asyncio.sleep(0.1)  # Simulate work
            span.set_attribute("test.result", "success")
            span.set_attribute("test.type", "integration")
        
        print("✓ Distributed tracing test completed")
        
        print("\n8. Final metrics verification...")
        
        final_metrics = observability.get_metrics_summary(hours=1)
        summary = final_metrics.get('summary', {})
        
        print("✓ Final integration test metrics:")
        print(f"  - Total requests: {summary.get('total_requests', 0)}")
        print(f"  - Total errors: {summary.get('total_errors', 0)}")
        print(f"  - Total tokens: {summary.get('total_tokens', 0)}")
        print(f"  - Total cost: ${summary.get('total_cost', 0)}")
        print(f"  - Avg response time: {summary.get('avg_response_time', 0):.3f}s")
        print(f"  - System health: {summary.get('system_health', 'unknown')}")
        
        evaluation_metrics = final_metrics.get('evaluation_metrics', {})
        print(f"  - Evaluation metrics tracked: {len(evaluation_metrics)}")
        
        print("\n" + "=" * 70)
        print("✅ All backend integration tests completed successfully!")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Backend integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    result = asyncio.run(test_backend_integration())
    print(f"\nIntegration test result: {'PASSED' if result else 'FAILED'}")
