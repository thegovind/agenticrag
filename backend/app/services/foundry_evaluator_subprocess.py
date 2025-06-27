"""
Subprocess-based Azure AI Foundry Evaluator to avoid HTTP client conflicts
"""

import subprocess
import json
import tempfile
import os
import sys
from typing import Dict, Any, List
import logging

logger = logging.getLogger(__name__)

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
    Run Azure AI Foundry evaluation in a separate subprocess to avoid HTTP client conflicts
    """
    
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
        logger.debug(f"Generated script saved to: {temp_script_path}")
        logger.debug(f"Script content:\n{eval_script}")
        
        # Run the script in a subprocess with UTF-8 encoding
        result = subprocess.run(
            [sys.executable, temp_script_path],
            capture_output=True,
            text=True,
            encoding='utf-8',
            timeout=60  # 60 second timeout
        )
        
        # Debug: Log subprocess output
        logger.debug(f"Subprocess return code: {result.returncode}")
        logger.debug(f"Subprocess stdout: {result.stdout}")
        logger.debug(f"Subprocess stderr: {result.stderr}")
        
        # Clean up temporary file
        os.unlink(temp_script_path)
        
        if result.returncode == 0:
            # Parse JSON output
            try:
                return json.loads(result.stdout.strip())
            except json.JSONDecodeError:
                return {
                    "score": 0.0,
                    "error": "Failed to parse evaluation result",
                    "reasoning": f"Subprocess output: {result.stdout}"
                }
        else:
            return {
                "score": 0.0,
                "error": f"Subprocess failed with return code {result.returncode}",
                "reasoning": f"Error: {result.stderr}"
            }
            
    except subprocess.TimeoutExpired:
        return {
            "score": 0.0,
            "error": "Evaluation timeout",
            "reasoning": "Evaluation process timed out after 60 seconds"
        }
    except Exception as e:
        return {
            "score": 0.0,
            "error": str(e),
            "reasoning": f"Failed to run subprocess evaluation: {str(e)}"
        }
