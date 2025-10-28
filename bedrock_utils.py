#!/usr/bin/env python3
"""
AWS Bedrock Utilities
- List available models
- Check usage and costs
- Test model connectivity
"""

import os
import json
from datetime import datetime, timedelta
from dotenv import load_dotenv
import boto3
from botocore.exceptions import ClientError

load_dotenv()

class BedrockUtils:
    """Utilities for AWS Bedrock"""

    def __init__(self):
        """Initialize Bedrock client from environment variables"""
        self.aws_access_key = os.getenv("AWS_ACCESS_KEY_ID")
        self.aws_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
        self.aws_region = os.getenv("AWS_REGION", "us-east-1")

        if not self.aws_access_key or not self.aws_secret_key:
            raise ValueError("AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY must be set in .env file")

        # Initialize Bedrock Runtime client (for inference)
        self.bedrock_runtime = boto3.client(
            service_name='bedrock-runtime',
            region_name=self.aws_region,
            aws_access_key_id=self.aws_access_key,
            aws_secret_access_key=self.aws_secret_key
        )

        # Initialize Bedrock client (for listing models)
        self.bedrock = boto3.client(
            service_name='bedrock',
            region_name=self.aws_region,
            aws_access_key_id=self.aws_access_key,
            aws_secret_access_key=self.aws_secret_key
        )

        # Initialize CloudWatch client (for usage metrics)
        self.cloudwatch = boto3.client(
            service_name='cloudwatch',
            region_name=self.aws_region,
            aws_access_key_id=self.aws_access_key,
            aws_secret_access_key=self.aws_secret_key
        )

    def list_available_models(self):
        """List all available foundation models in Bedrock"""
        print("\n" + "="*80)
        print("AVAILABLE BEDROCK MODELS")
        print("="*80)

        try:
            response = self.bedrock.list_foundation_models()

            # Group by provider
            models_by_provider = {}
            for model in response.get('modelSummaries', []):
                provider = model.get('providerName', 'Unknown')
                if provider not in models_by_provider:
                    models_by_provider[provider] = []
                models_by_provider[provider].append(model)

            # Display models by provider
            for provider, models in sorted(models_by_provider.items()):
                print(f"\n{provider}:")
                for model in models:
                    model_id = model.get('modelId')
                    model_name = model.get('modelName')
                    input_modalities = model.get('inputModalities', [])
                    output_modalities = model.get('outputModalities', [])

                    print(f"  • {model_name}")
                    print(f"    ID: {model_id}")
                    print(f"    Input: {', '.join(input_modalities)}")
                    print(f"    Output: {', '.join(output_modalities)}")

            print("\n" + "="*80)
            print(f"Total Models: {len(response.get('modelSummaries', []))}")
            print("="*80)

            return response.get('modelSummaries', [])

        except ClientError as e:
            print(f"Error listing models: {e}")
            return []

    def test_model(self, model_id, test_prompt="Hello, how are you?"):
        """Test a specific model with a simple prompt"""
        print(f"\nTesting model: {model_id}")
        print(f"Prompt: {test_prompt}")
        print("-" * 60)

        try:
            # Determine request format based on model provider
            if "anthropic" in model_id.lower():
                # Claude models use Messages API
                body = json.dumps({
                    "anthropic_version": "bedrock-2023-05-31",
                    "max_tokens": 1000,
                    "messages": [
                        {"role": "user", "content": test_prompt}
                    ]
                })
            elif "meta" in model_id.lower():
                # Llama models
                body = json.dumps({
                    "prompt": test_prompt,
                    "max_gen_len": 1000,
                    "temperature": 0.0
                })
            elif "amazon" in model_id.lower():
                # Amazon Titan models
                body = json.dumps({
                    "inputText": test_prompt,
                    "textGenerationConfig": {
                        "maxTokenCount": 1000,
                        "temperature": 0.0
                    }
                })
            elif "mistral" in model_id.lower():
                # Mistral models
                body = json.dumps({
                    "prompt": test_prompt,
                    "max_tokens": 1000,
                    "temperature": 0.0
                })
            else:
                # Generic format
                body = json.dumps({
                    "prompt": test_prompt,
                    "max_tokens": 1000
                })

            # Invoke model
            start_time = datetime.now()
            response = self.bedrock_runtime.invoke_model(
                modelId=model_id,
                body=body
            )
            end_time = datetime.now()

            response_time = (end_time - start_time).total_seconds()

            # Parse response
            response_body = json.loads(response['body'].read())

            # Extract text based on model type
            if "anthropic" in model_id.lower():
                response_text = response_body['content'][0]['text']
            elif "meta" in model_id.lower():
                response_text = response_body.get('generation', '')
            elif "amazon" in model_id.lower():
                response_text = response_body['results'][0]['outputText']
            else:
                response_text = str(response_body)

            print(f"Response ({response_time:.2f}s):")
            print(response_text)
            print("-" * 60)
            print("✅ Model test successful!")

            return True

        except ClientError as e:
            print(f"❌ Error testing model: {e}")
            return False

    def check_usage(self, days=7):
        """Check Bedrock usage for the past N days"""
        print("\n" + "="*80)
        print(f"BEDROCK USAGE (Last {days} days)")
        print("="*80)

        end_time = datetime.utcnow()
        start_time = end_time - timedelta(days=days)

        try:
            # Get invocation metrics
            response = self.cloudwatch.get_metric_statistics(
                Namespace='AWS/Bedrock',
                MetricName='Invocations',
                Dimensions=[],
                StartTime=start_time,
                EndTime=end_time,
                Period=86400,  # 1 day
                Statistics=['Sum']
            )

            print("\nInvocations by Day:")
            if response['Datapoints']:
                for datapoint in sorted(response['Datapoints'], key=lambda x: x['Timestamp']):
                    date = datapoint['Timestamp'].strftime('%Y-%m-%d')
                    count = int(datapoint['Sum'])
                    print(f"  {date}: {count:,} invocations")

                total = sum(int(d['Sum']) for d in response['Datapoints'])
                print(f"\nTotal Invocations: {total:,}")
            else:
                print("  No invocations found in this period")

            # Get input/output token metrics
            for metric_name in ['InputTokens', 'OutputTokens']:
                response = self.cloudwatch.get_metric_statistics(
                    Namespace='AWS/Bedrock',
                    MetricName=metric_name,
                    Dimensions=[],
                    StartTime=start_time,
                    EndTime=end_time,
                    Period=86400 * days,  # Aggregate over entire period
                    Statistics=['Sum']
                )

                if response['Datapoints']:
                    total = sum(int(d['Sum']) for d in response['Datapoints'])
                    print(f"\nTotal {metric_name}: {total:,}")

            print("\n" + "="*80)
            print("Note: For detailed cost information, check AWS Cost Explorer in AWS Console")
            print("URL: https://console.aws.amazon.com/cost-management/home")
            print("="*80)

        except ClientError as e:
            print(f"Error checking usage: {e}")
            print("\nNote: Usage metrics may take up to 15 minutes to appear in CloudWatch")

    def recommend_models_for_benchmark(self):
        """Recommend models for the benchmark based on speed and capability"""
        print("\n" + "="*80)
        print("RECOMMENDED MODELS FOR BENCHMARK")
        print("="*80)

        recommendations = {
            "Fast & Cost-Effective": [
                "anthropic.claude-3-haiku-20240307-v1:0",
                "anthropic.claude-3-5-haiku-20241022-v1:0",
                "meta.llama3-1-8b-instruct-v1:0",
                "amazon.titan-text-express-v1"
            ],
            "Balanced (Speed + Quality)": [
                "anthropic.claude-3-5-sonnet-20241022-v2:0",
                "meta.llama3-1-70b-instruct-v1:0",
                "mistral.mistral-large-2402-v1:0"
            ],
            "Highest Quality": [
                "anthropic.claude-3-opus-20240229-v1:0",
                "anthropic.claude-3-5-sonnet-20241022-v2:0",
                "meta.llama3-3-70b-instruct-v1:0"
            ]
        }

        for category, models in recommendations.items():
            print(f"\n{category}:")
            for model_id in models:
                print(f"  • {model_id}")

        print("\n" + "="*80)
        print("For your 7-8 hour timeline with 8,820 cases:")
        print("  • Use Fast & Cost-Effective models (2-3s/case)")
        print("  • Or use Balanced models (3-5s/case)")
        print("  • All can complete in your timeframe")
        print("="*80)


def main():
    """Main function"""
    import sys

    try:
        utils = BedrockUtils()

        if len(sys.argv) > 1:
            command = sys.argv[1]

            if command == "list":
                utils.list_available_models()
            elif command == "usage":
                days = int(sys.argv[2]) if len(sys.argv) > 2 else 7
                utils.check_usage(days)
            elif command == "test":
                if len(sys.argv) < 3:
                    print("Usage: python bedrock_utils.py test <model-id>")
                    sys.exit(1)
                model_id = sys.argv[2]
                utils.test_model(model_id)
            elif command == "recommend":
                utils.recommend_models_for_benchmark()
            else:
                print("Unknown command. Available: list, usage, test, recommend")
        else:
            # Default: show everything
            print("AWS Bedrock configured successfully!")
            print(f"Region: {utils.aws_region}")
            utils.recommend_models_for_benchmark()
            utils.list_available_models()
            utils.check_usage()

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
