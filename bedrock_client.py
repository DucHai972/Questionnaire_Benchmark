"""
AWS Bedrock Client using Bearer Token Authentication
Uses direct HTTP API calls instead of boto3
"""

import os
import json
import time
import requests
from typing import Dict, Any
from dotenv import load_dotenv

load_dotenv()


class BedrockClient:
    """Bedrock API Client using Bearer Token"""

    def __init__(self, api_key: str, model_name: str, region: str = "us-east-1"):
        self.api_key = api_key
        self.model_name = model_name
        self.region = region
        self.provider = "bedrock"

        # Construct the endpoint URL
        self.base_url = f"https://bedrock-runtime.{region}.amazonaws.com"

    def generate(self, prompt: str, max_tokens: int = 4000) -> Dict[str, Any]:
        """Generate response using Bedrock API"""
        start_time = time.time()

        try:
            # Construct the full URL
            url = f"{self.base_url}/model/{self.model_name}/invoke"

            # Prepare the request body based on model type
            if "anthropic.claude" in self.model_name or "us.anthropic.claude" in self.model_name:
                # Claude models use Messages API format
                payload = {
                    "anthropic_version": "bedrock-2023-05-31",
                    "max_tokens": max_tokens,
                    "messages": [
                        {
                            "role": "user",
                            "content": [{"type": "text", "text": prompt}]
                        }
                    ]
                }
            elif "qwen" in self.model_name.lower():
                # Qwen models use messages format
                payload = {
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": max_tokens,
                    "temperature": 0.0
                }
            elif "deepseek" in self.model_name.lower():
                # DeepSeek R1 vs V3 handling
                if "r1" in self.model_name.lower():
                    # DeepSeek R1 requires special format with reasoning
                    formatted_prompt = f"<｜begin▁of▁sentence｜><｜User｜>{prompt}<｜Assistant｜><think>\n"
                    payload = {
                        "prompt": formatted_prompt,
                        "max_tokens": max_tokens,
                        "temperature": 0.0,
                        "top_p": 1.0
                    }
                else:
                    # DeepSeek V3 uses standard prompt format
                    payload = {
                        "prompt": prompt,
                        "max_tokens": max_tokens,
                        "temperature": 0.0
                    }
            elif "amazon.nova" in self.model_name:
                # Amazon Nova models use messages with inferenceConfig
                payload = {
                    "messages": [{"role": "user", "content": [{"text": prompt}]}],
                    "inferenceConfig": {
                        "max_new_tokens": max_tokens,
                        "temperature": 0.0
                    }
                }
            elif "meta.llama" in self.model_name or "us.meta.llama" in self.model_name:
                # Llama models require chat template format
                formatted_prompt = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
                payload = {
                    "prompt": formatted_prompt,
                    "max_gen_len": max_tokens,
                    "temperature": 0.0
                }
            elif "amazon.titan" in self.model_name:
                # Amazon Titan models
                payload = {
                    "inputText": prompt,
                    "textGenerationConfig": {
                        "maxTokenCount": max_tokens,
                        "temperature": 0.0
                    }
                }
            elif "mistral" in self.model_name:
                # Mistral models
                payload = {
                    "prompt": prompt,
                    "max_tokens": max_tokens,
                    "temperature": 0.0
                }
            else:
                # Generic format
                payload = {
                    "prompt": prompt,
                    "max_tokens": max_tokens
                }

            # Set headers with Bearer token
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }

            # Make the request
            response = requests.post(url, json=payload, headers=headers, timeout=120)

            # Check for errors
            response.raise_for_status()

            # Parse response
            result = response.json()

            # Extract text based on model type
            if "anthropic.claude" in self.model_name or "us.anthropic.claude" in self.model_name:
                response_text = result['content'][0]['text']
            elif "qwen" in self.model_name.lower():
                # Qwen returns choices array like ChatGPT
                if 'choices' in result and len(result['choices']) > 0:
                    response_text = result['choices'][0].get('message', {}).get('content', '')
                elif 'output' in result:
                    response_text = result['output']
                else:
                    response_text = str(result)
            elif "deepseek" in self.model_name.lower():
                # DeepSeek returns choices array like OpenAI
                if 'choices' in result and len(result['choices']) > 0:
                    response_text = result['choices'][0].get('text', '')
                else:
                    response_text = result.get('generation', result.get('output', ''))
            elif "amazon.nova" in self.model_name:
                # Nova returns output with message structure
                response_text = result['output']['message']['content'][0]['text']
            elif "meta.llama" in self.model_name or "us.meta.llama" in self.model_name:
                response_text = result.get('generation', '')
            elif "amazon.titan" in self.model_name:
                response_text = result['results'][0]['outputText']
            elif "mistral" in self.model_name:
                response_text = result.get('outputs', [{}])[0].get('text', '')
            else:
                response_text = str(result)

            end_time = time.time()

            result_dict = {
                "response": response_text.strip(),
                "success": True,
                "error": None,
                "response_time": end_time - start_time
            }


            return result_dict

        except requests.exceptions.RequestException as e:
            end_time = time.time()
            error_msg = str(e)

            # Try to get more details from response
            if hasattr(e, 'response') and e.response is not None:
                try:
                    error_details = e.response.json()
                    error_msg = f"{error_msg} - {error_details}"
                except:
                    error_msg = f"{error_msg} - {e.response.text}"

            return {
                "response": "",
                "success": False,
                "error": error_msg,
                "response_time": end_time - start_time
            }
        except Exception as e:
            end_time = time.time()
            return {
                "response": "",
                "success": False,
                "error": str(e),
                "response_time": end_time - start_time
            }
