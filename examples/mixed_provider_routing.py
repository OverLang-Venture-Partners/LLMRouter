#!/usr/bin/env python3
"""
Example: Mixed Provider Routing

This script demonstrates how to route queries across multiple providers (Bedrock, NVIDIA, OpenAI)
using LLMRouter. This showcases the flexibility of using different providers based on your needs.

Prerequisites:
1. Install dependencies: pip install llmrouter-lib boto3
2. Configure AWS credentials for Bedrock (see docs/AWS_BEDROCK_CREDENTIALS.md)
3. Set API keys for other providers: export API_KEYS='{"NVIDIA": "your-key", "OpenAI": "your-key"}'

Usage:
    python examples/mixed_provider_routing.py
"""

from llmrouter.utils.api_calling import call_api


def route_query_by_complexity(query: str, complexity: str = "simple") -> dict:
    """
    Route a query to an appropriate model based on complexity.
    
    Args:
        query: The query to process
        complexity: "simple", "medium", or "complex"
    
    Returns:
        Request dictionary for the selected model
    """
    if complexity == "simple":
        # Use fast, cost-effective model for simple queries
        return {
            "api_endpoint": "",
            "query": query,
            "model_name": "titan-text-express-bedrock",
            "api_name": "amazon.titan-text-express-v1",
            "service": "Bedrock",
            "aws_region": "us-east-1"
        }
    elif complexity == "medium":
        # Use balanced model for medium complexity
        return {
            "api_endpoint": "",
            "query": query,
            "model_name": "claude-3-haiku-bedrock",
            "api_name": "anthropic.claude-3-haiku-20240307-v1:0",
            "service": "Bedrock",
            "aws_region": "us-west-2"
        }
    else:  # complex
        # Use powerful model for complex queries
        return {
            "api_endpoint": "",
            "query": query,
            "model_name": "claude-3-sonnet-bedrock",
            "api_name": "anthropic.claude-3-sonnet-20240229-v1:0",
            "service": "Bedrock",
            "aws_region": "us-east-1"
        }


def route_query_by_region(query: str, user_region: str = "us-east") -> dict:
    """
    Route a query to a model in the nearest AWS region for lower latency.
    
    Args:
        query: The query to process
        user_region: User's region ("us-east", "us-west", "eu", "asia")
    
    Returns:
        Request dictionary for the selected model
    """
    if user_region == "us-west":
        return {
            "api_endpoint": "",
            "query": query,
            "model_name": "claude-3-haiku-bedrock",
            "api_name": "anthropic.claude-3-haiku-20240307-v1:0",
            "service": "Bedrock",
            "aws_region": "us-west-2"
        }
    else:  # Default to us-east
        return {
            "api_endpoint": "",
            "query": query,
            "model_name": "claude-3-haiku-bedrock",
            "api_name": "anthropic.claude-3-haiku-20240307-v1:0",
            "service": "Bedrock",
            "aws_region": "us-east-1"
        }


def main():
    """Demonstrate mixed provider routing strategies."""
    
    print("="*80)
    print("EXAMPLE 1: Routing by Query Complexity")
    print("="*80)
    print()
    
    # Define queries with different complexity levels
    queries = [
        ("What is 2+2?", "simple"),
        ("Explain the difference between machine learning and deep learning.", "medium"),
        ("Design a distributed system architecture for a real-time analytics platform.", "complex")
    ]
    
    requests = []
    for query, complexity in queries:
        request = route_query_by_complexity(query, complexity)
        requests.append(request)
        print(f"Query: {query}")
        print(f"Complexity: {complexity} → Model: {request['model_name']}")
        print()
    
    # Make batch API calls
    print("Making API calls...\n")
    results = call_api(requests, max_tokens=512, temperature=0.7)
    
    # Display results
    for i, (result, (query, complexity)) in enumerate(zip(results, queries), 1):
        print(f"{'-'*80}")
        print(f"Result {i}: {complexity.upper()} query")
        print(f"{'-'*80}")
        
        if 'error' in result:
            print(f"❌ Error: {result['error']}\n")
        else:
            print(f"✅ Response: {result['response'][:200]}...")
            print(f"📊 Tokens: {result['token_num']}, Time: {result['response_time']:.2f}s\n")
    
    print("\n" + "="*80)
    print("EXAMPLE 2: Routing by Geographic Region")
    print("="*80)
    print()
    
    # Demonstrate regional routing
    query = "What are the best practices for cloud security?"
    
    regions = ["us-east", "us-west"]
    regional_requests = []
    
    for region in regions:
        request = route_query_by_region(query, region)
        regional_requests.append(request)
        print(f"User Region: {region} → AWS Region: {request['aws_region']}")
    
    print("\nMaking API calls...\n")
    regional_results = call_api(regional_requests, max_tokens=256, temperature=0.7)
    
    # Display regional results
    for i, (result, region) in enumerate(zip(regional_results, regions), 1):
        print(f"{'-'*80}")
        print(f"Result {i}: User in {region}")
        print(f"{'-'*80}")
        
        if 'error' in result:
            print(f"❌ Error: {result['error']}\n")
        else:
            print(f"✅ Response: {result['response'][:200]}...")
            print(f"📊 Time: {result['response_time']:.2f}s (lower latency for nearby regions)\n")
    
    print("\n" + "="*80)
    print("EXAMPLE 3: Mixed Provider Configuration")
    print("="*80)
    print()
    
    # Demonstrate using multiple providers in one batch
    # Note: This requires API keys for NVIDIA/OpenAI to be configured
    mixed_requests = [
        {
            "api_endpoint": "",
            "query": "What is AWS Bedrock?",
            "model_name": "claude-3-haiku-bedrock",
            "api_name": "anthropic.claude-3-haiku-20240307-v1:0",
            "service": "Bedrock",
            "aws_region": "us-east-1"
        },
        {
            "api_endpoint": "https://integrate.api.nvidia.com/v1",
            "query": "What is NVIDIA NIM?",
            "model_name": "llama-3.1-8b-instruct",
            "api_name": "meta/llama-3.1-8b-instruct",
            "service": "NVIDIA"
        }
    ]
    
    print("Routing queries to different providers:")
    print(f"1. Bedrock (Claude 3 Haiku)")
    print(f"2. NVIDIA (Llama 3.1 8B)")
    print()
    
    print("Making API calls...\n")
    mixed_results = call_api(mixed_requests, max_tokens=256, temperature=0.7)
    
    # Display mixed provider results
    for i, result in enumerate(mixed_results, 1):
        print(f"{'-'*80}")
        print(f"Result {i}: {result['service']} - {result['model_name']}")
        print(f"{'-'*80}")
        
        if 'error' in result:
            print(f"❌ Error: {result['error']}")
            if result['service'] == 'NVIDIA':
                print("   (Make sure API_KEYS environment variable is set for NVIDIA)")
            print()
        else:
            print(f"✅ Response: {result['response'][:200]}...")
            print(f"📊 Tokens: {result['token_num']}, Time: {result['response_time']:.2f}s\n")
    
    print("="*80)
    print("KEY TAKEAWAYS")
    print("="*80)
    print("1. Route by complexity: Use cost-effective models for simple queries")
    print("2. Route by region: Use nearby AWS regions for lower latency")
    print("3. Mix providers: Combine Bedrock, NVIDIA, OpenAI in one application")
    print("4. Same API: All providers use the same call_api() interface")


if __name__ == "__main__":
    main()
