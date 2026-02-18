#!/usr/bin/env python3
"""
Example: Batch Bedrock Queries

This script demonstrates how to make multiple queries to AWS Bedrock models in batch using LLMRouter.
Batch processing is efficient for processing multiple queries at once.

Prerequisites:
1. Install dependencies: pip install llmrouter-lib boto3
2. Configure AWS credentials (see docs/AWS_BEDROCK_CREDENTIALS.md)

Usage:
    python examples/bedrock_batch_queries.py
"""

from llmrouter.utils.api_calling import call_api


def main():
    """Make batch queries to Bedrock models."""
    
    # Define multiple requests for different Bedrock models
    requests = [
        {
            "api_endpoint": "",
            "query": "What is the capital of France?",
            "model_name": "claude-3-haiku-bedrock",
            "api_name": "anthropic.claude-3-haiku-20240307-v1:0",
            "service": "Bedrock",
            "aws_region": "us-west-2"
        },
        {
            "api_endpoint": "",
            "query": "Write a haiku about artificial intelligence.",
            "model_name": "titan-text-express-bedrock",
            "api_name": "amazon.titan-text-express-v1",
            "service": "Bedrock",
            "aws_region": "us-east-1"
        },
        {
            "api_endpoint": "",
            "query": "Explain quantum computing in one sentence.",
            "model_name": "claude-3-sonnet-bedrock",
            "api_name": "anthropic.claude-3-sonnet-20240229-v1:0",
            "service": "Bedrock",
            "aws_region": "us-east-1",
            "system_prompt": "You are a concise technical writer."
        },
        {
            "api_endpoint": "",
            "query": "What are the benefits of cloud computing?",
            "model_name": "llama-3-70b-bedrock",
            "api_name": "meta.llama3-70b-instruct-v1:0",
            "service": "Bedrock",
            "aws_region": "us-west-2"
        }
    ]
    
    print(f"Making {len(requests)} batch requests to Bedrock models...\n")
    
    # Make batch API calls
    results = call_api(
        requests,
        max_tokens=256,
        temperature=0.7,
        top_p=0.9,
        timeout=30
    )
    
    # Display results for each request
    total_tokens = 0
    total_time = 0
    successful = 0
    failed = 0
    
    for i, result in enumerate(results, 1):
        print(f"{'='*80}")
        print(f"Request {i}/{len(results)}")
        print(f"{'='*80}")
        print(f"Model: {result['model_name']}")
        print(f"Region: {result.get('aws_region', 'default')}")
        print(f"Query: {result['query']}\n")
        
        if 'error' in result:
            print(f"❌ Error: {result['error']}\n")
            failed += 1
        else:
            print(f"✅ Response: {result['response']}\n")
            print(f"📊 Stats:")
            print(f"   - Tokens: {result['token_num']} (in: {result['prompt_tokens']}, out: {result['completion_tokens']})")
            print(f"   - Time: {result['response_time']:.2f}s\n")
            
            total_tokens += result['token_num']
            total_time += result['response_time']
            successful += 1
    
    # Summary
    print(f"{'='*80}")
    print(f"SUMMARY")
    print(f"{'='*80}")
    print(f"✅ Successful: {successful}/{len(results)}")
    print(f"❌ Failed: {failed}/{len(results)}")
    print(f"📊 Total tokens: {total_tokens}")
    print(f"⏱️  Total time: {total_time:.2f}s")
    if successful > 0:
        print(f"📈 Average time per request: {total_time/successful:.2f}s")


if __name__ == "__main__":
    main()
