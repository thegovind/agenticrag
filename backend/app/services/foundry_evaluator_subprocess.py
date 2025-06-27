"""
Subprocess-based Azure AI Foundry Evaluator to avoid HTTP client conflicts
"""

import asyncio
import json
import tempfile
import os
import sys
import platform
import subprocess
import concurrent.futures
from typing import Dict, Any, List
import logging

logger = logging.getLogger(__name__)

async def run_foundry_evaluation_async(
    evaluator_name: str,
    query: str,
    response: str,
    context: str = "",
    azure_endpoint: str = "",
    api_key: str = "",
    deployment: str = "",
    api_version: str = ""
) -> Dict[str, Any]:
    """
    Run Azure AI Foundry evaluation in a separate subprocess asynchronously to avoid HTTP client conflicts
    """
    import time
    eval_start_time = time.time()
    logger.info(f"🔄 Starting async evaluation for {evaluator_name}")
    
    # Properly escape strings for Python script
    def escape_string(s: str) -> str:
        """Escape a string for safe inclusion in Python code"""
        import json
        return json.dumps(s)
    
    escaped_query = escape_string(query)
    escaped_response = escape_string(response)
    escaped_context = escape_string(context)
    escaped_endpoint = escape_string(azure_endpoint)
    escaped_api_key = escape_string(api_key)
    escaped_deployment = escape_string(deployment)
    escaped_api_version = escape_string(api_version)
    
    # Create a temporary Python script for evaluation
    eval_script = f'''
import os
import sys
import json
from azure.ai.evaluation import {evaluator_name}, AzureOpenAIModelConfiguration

def main():
    # Define evaluator_key early for error handling
    evaluator_key = "{evaluator_name}".replace("Evaluator", "").lower()
    
    try:
        # Set environment variables
        os.environ["AZURE_OPENAI_ENDPOINT"] = {escaped_endpoint}
        os.environ["AZURE_OPENAI_API_KEY"] = {escaped_api_key}
        os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"] = {escaped_deployment}
        os.environ["AZURE_OPENAI_API_VERSION"] = {escaped_api_version}
        
        # Create model configuration
        model_config = AzureOpenAIModelConfiguration(
            azure_endpoint={escaped_endpoint},
            api_key={escaped_api_key},
            azure_deployment={escaped_deployment},
            api_version={escaped_api_version}
        )
        
        # Create evaluator
        evaluator = {evaluator_name}(model_config)
        
        # Run evaluation based on evaluator type
        if "{evaluator_name}" == "GroundednessEvaluator":
            result = evaluator(
                response={escaped_response},
                context={escaped_context}
            )
        elif "{evaluator_name}" == "RelevanceEvaluator":
            result = evaluator(
                query={escaped_query},
                response={escaped_response},
                context={escaped_context}
            )
        elif "{evaluator_name}" in ["CoherenceEvaluator", "FluencyEvaluator"]:
            result = evaluator(
                query={escaped_query},
                response={escaped_response}
            )
        else:
            raise ValueError(f"Unknown evaluator: {evaluator_name}")
        
        # Extract score and reasoning
        score = 0.0
        reasoning = ""
        
        # Try accessing the score directly from the result object
        if hasattr(result, evaluator_key):
            score = float(getattr(result, evaluator_key))
        elif hasattr(result, 'score'):
            score = float(result.score)
        elif isinstance(result, dict) and evaluator_key in result:
            score = float(result[evaluator_key])
        elif isinstance(result, dict) and 'score' in result:
            score = float(result['score'])
        
        # Try getting reasoning
        reason_attr = f"{{evaluator_key}}_reason"
        if hasattr(result, reason_attr):
            reasoning = str(getattr(result, reason_attr))
        elif hasattr(result, 'reasoning'):
            reasoning = str(result.reasoning)
        elif isinstance(result, dict) and reason_attr in result:
            reasoning = str(result[reason_attr])
        elif isinstance(result, dict) and 'reasoning' in result:
            reasoning = str(result['reasoning'])
        
        # Output result as JSON
        output = {{
            "score": score,
            "reasoning": reasoning or f"{{evaluator_key}} evaluation completed",
            "raw_result": str(result)
        }}
        
        print(json.dumps(output))
        
    except Exception as e:
        error_output = {{
            "score": 0.0,
            "error": str(e),
            "reasoning": f"Error in {{evaluator_key}} evaluation: {{str(e)}}"
        }}
        print(json.dumps(error_output))
        sys.exit(1)

if __name__ == "__main__":
    main()
'''
    
    try:
        # Write script to temporary file with UTF-8 encoding
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8') as temp_file:
            temp_file.write(eval_script)
            temp_script_path = temp_file.name
        
        # Debug: Print the generated script for troubleshooting
        logger.info(f"📝 Generated script saved to: {temp_script_path}")
        logger.info(f"🐍 Python executable: {sys.executable}")
        logger.debug(f"Script content:\n{eval_script}")
        
        # Run the script in a subprocess with UTF-8 encoding
        # Use async subprocess for true parallel execution
        logger.info(f"🔧 Starting async subprocess for {evaluator_name} with script: {temp_script_path}")
        try:
            # On Windows, we need to use ProactorEventLoop for subprocess support
            import platform
            if platform.system() == "Windows":
                # For Windows, use run_in_executor with a thread pool for subprocess
                import subprocess
                import concurrent.futures
                
                def run_subprocess():
                    return subprocess.run(
                        [sys.executable, temp_script_path],
                        capture_output=True,
                        text=True,
                        timeout=60
                    )
                
                logger.info(f"⏳ Running {evaluator_name} subprocess on Windows using thread executor...")
                loop = asyncio.get_event_loop()
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    completed_process = await loop.run_in_executor(executor, run_subprocess)
                
                # Convert subprocess.CompletedProcess to match our expected interface
                class MockProcess:
                    def __init__(self, completed_process):
                        self.returncode = completed_process.returncode
                        self.stdout = completed_process.stdout.encode('utf-8')
                        self.stderr = completed_process.stderr.encode('utf-8')
                
                process = MockProcess(completed_process)
                stdout, stderr = process.stdout, process.stderr
                
            else:
                # For Unix-like systems, use the standard async subprocess
                process = await asyncio.create_subprocess_exec(
                    sys.executable, temp_script_path,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                
                logger.info(f"⏳ Waiting for {evaluator_name} subprocess to complete...")
                # Wait for process to complete with timeout
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(), 
                    timeout=60.0  # 60 second timeout
                )
            
            # Debug: Log subprocess output
            stdout_text = stdout.decode('utf-8')
            stderr_text = stderr.decode('utf-8')
            
            logger.debug(f"Async subprocess return code: {process.returncode}")
            logger.debug(f"Async subprocess stdout: {stdout_text}")
            logger.debug(f"Async subprocess stderr: {stderr_text}")
            
            # Clean up temporary file
            os.unlink(temp_script_path)
            
            if process.returncode == 0:
                # Parse JSON output
                try:
                    eval_duration = (time.time() - eval_start_time) * 1000
                    result = json.loads(stdout_text.strip())
                    logger.info(f"✅ {evaluator_name} completed in {eval_duration:.1f}ms with score: {result.get('score', 'N/A')}")
                    return result
                except json.JSONDecodeError:
                    logger.error(f"❌ {evaluator_name} failed - JSON parse error")
                    return {
                        "score": 0.0,
                        "error": "Failed to parse evaluation result",
                        "reasoning": f"Subprocess output: {stdout_text}"
                    }
            else:
                eval_duration = (time.time() - eval_start_time) * 1000
                logger.error(f"❌ {evaluator_name} failed in {eval_duration:.1f}ms - return code {process.returncode}")
                return {
                    "score": 0.0,
                    "error": f"Subprocess failed with return code {process.returncode}",
                    "reasoning": f"Error: {stderr_text}"
                }
                
        except (asyncio.TimeoutError, subprocess.TimeoutExpired):
            logger.error(f"❌ {evaluator_name} subprocess timed out after 60 seconds")
            return {
                "score": 0.0,
                "error": "Evaluation timeout",
                "reasoning": "Evaluation process timed out after 60 seconds"
            }
    except Exception as e:
        eval_duration = (time.time() - eval_start_time) * 1000
        logger.error(f"❌ {evaluator_name} async subprocess failed in {eval_duration:.1f}ms: {str(e)}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        return {
            "score": 0.0,
            "error": str(e),
            "reasoning": f"Failed to run subprocess evaluation: {str(e)}"
        }

def run_foundry_evaluation(
    evaluator_name: str,
    query: str,
    response: str,
    context: str = "",
    azure_endpoint: str = "",
    api_key: str = "",
    deployment: str = "",
    api_version: str = ""
) -> Dict[str, Any]:
    """
    Synchronous wrapper for run_foundry_evaluation_async for backward compatibility
    """
    return asyncio.run(run_foundry_evaluation_async(
        evaluator_name, query, response, context,
        azure_endpoint, api_key, deployment, api_version
    ))
