#!/usr/bin/env python3
"""
Example: Single Bedrock Query

This script demonstrates how to make a single query to an AWS Bedrock model using LLMRouter.

Prerequisites:
1. Install dependencies: pip install llmrouter-lib boto3
2. Configure AWS credentials (see docs/AWS_BEDROCK_CREDENTIALS.md)

Usage:
    python examples/bedrock_single_query.py
"""

from llmrouter.utils.api_calling import call_api


def main():
    """Make a single query to a Bedrock model."""
    
    # Define the request for a Bedrock model
    request = {
        "api_endpoint": "",  # Not used for Bedrock (determined by AWS region)
        "query": "Explain the concept of machine learning in simple terms.",
        "model_name": "claude-3-haiku-bedrock",
        "api_name": "anthropic.claude-3-haiku-20240307-v1:0",
        "service": "Bedrock",
        "aws_region": "us-west-2",  # Optional: defaults to AWS_DEFAULT_REGION if not specified
        "system_prompt": "You are a helpful AI assistant that explains technical concepts clearly."
    }
    
    print("Making request to Bedrock model...")
    print(f"Model: {request['model_name']}")
    print(f"Region: {request['aws_region']}")
    print(f"Query: {request['query']}\n")
    
    # Make the API call
    result = call_api(
        request,
        max_tokens=512,
        temperature=0.7,
        top_p=0.9,
        timeout=30
    )
    
    # Display results
    if 'error' in result:
        print(f"❌ Error: {result['error']}")
    else:
        print(f"✅ Response: {result['response']}\n")
        print(f"📊 Token Usage:")
        print(f"   - Input tokens: {result['prompt_tokens']}")
        print(f"   - Output tokens: {result['completion_tokens']}")
        print(f"   - Total tokens: {result['token_num']}")
        print(f"   - Response time: {result['response_time']:.2f}s")
        
        # Calculate cost (Claude 3 Haiku pricing: $0.25/$1.25 per million tokens)
        input_cost = (result['prompt_tokens'] / 1_000_000) * 0.25
        output_cost = (result['completion_tokens'] / 1_000_000) * 1.25
        total_cost = input_cost + output_cost
        print(f"   - Estimated cost: ${total_cost:.6f}")


if __name__ == "__main__":
    main()
