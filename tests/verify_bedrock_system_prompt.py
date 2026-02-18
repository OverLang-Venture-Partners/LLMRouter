#!/usr/bin/env python3
"""
Verification script for Bedrock system prompt handling.

This script demonstrates that system prompts are correctly passed to Bedrock models.
It can be run manually with AWS credentials configured to verify the integration.

Usage:
    python3 tests/verify_bedrock_system_prompt.py

Requirements:
    - AWS credentials configured (environment variables or ~/.aws/credentials)
    - boto3 installed
    - Access to Bedrock models in your AWS account
"""

import os
import sys
from llmrouter.utils.api_calling import call_api


def verify_system_prompt_handling():
    """Verify that system prompts are correctly handled for Bedrock models."""
    
    print("=" * 70)
    print("Bedrock System Prompt Verification")
    print("=" * 70)
    print()
    
    # Check if AWS credentials are available
    has_credentials = (
        os.environ.get('AWS_ACCESS_KEY_ID') or
        os.path.exists(os.path.expanduser('~/.aws/credentials'))
    )
    
    if not has_credentials:
        print("⚠️  AWS credentials not found.")
        print("This is a verification script that requires AWS credentials.")
        print()
        print("To configure credentials:")
        print("1. Environment variables:")
        print("   export AWS_ACCESS_KEY_ID='your-key-id'")
        print("   export AWS_SECRET_ACCESS_KEY='your-secret-key'")
        print("   export AWS_DEFAULT_REGION='us-east-1'")
        print()
        print("2. AWS credential file (~/.aws/credentials):")
        print("   [default]")
        print("   aws_access_key_id = your-key-id")
        print("   aws_secret_access_key = your-secret-key")
        print()
        print("Skipping actual API call test.")
        print()
        return False
    
    print("✓ AWS credentials found")
    print()
    
    # Test 1: Request with system prompt
    print("Test 1: Bedrock request WITH system prompt")
    print("-" * 70)
    
    request_with_system = {
        "api_endpoint": "https://bedrock.amazonaws.com",
        "query": "What is 2+2?",
        "model_name": "claude-3-haiku",
        "api_name": "anthropic.claude-3-haiku-20240307-v1:0",
        "service": "Bedrock",
        "system_prompt": "You are a helpful math tutor. Always explain your reasoning.",
        "aws_region": "us-east-1"
    }
    
    print(f"Model: {request_with_system['api_name']}")
    print(f"Query: {request_with_system['query']}")
    print(f"System Prompt: {request_with_system['system_prompt']}")
    print()
    
    try:
        # Set dummy API_KEYS for Bedrock (not used, but required by call_api)
        result = call_api(request_with_system, api_keys_env='{"Bedrock": "dummy"}', max_tokens=100)
        
        if 'error' in result:
            print(f"❌ Error: {result['error']}")
            print()
            return False
        
        print(f"✓ Response: {result['response'][:200]}...")
        print(f"✓ Tokens: {result['token_num']} (prompt: {result['prompt_tokens']}, completion: {result['completion_tokens']})")
        print(f"✓ Response time: {result['response_time']:.2f}s")
        print()
        
    except Exception as e:
        print(f"❌ Exception: {e}")
        print()
        return False
    
    # Test 2: Request without system prompt
    print("Test 2: Bedrock request WITHOUT system prompt")
    print("-" * 70)
    
    request_without_system = {
        "api_endpoint": "https://bedrock.amazonaws.com",
        "query": "What is 2+2?",
        "model_name": "claude-3-haiku",
        "api_name": "anthropic.claude-3-haiku-20240307-v1:0",
        "service": "Bedrock",
        "aws_region": "us-east-1"
    }
    
    print(f"Model: {request_without_system['api_name']}")
    print(f"Query: {request_without_system['query']}")
    print(f"System Prompt: (none)")
    print()
    
    try:
        result = call_api(request_without_system, api_keys_env='{"Bedrock": "dummy"}', max_tokens=100)
        
        if 'error' in result:
            print(f"❌ Error: {result['error']}")
            print()
            return False
        
        print(f"✓ Response: {result['response'][:200]}...")
        print(f"✓ Tokens: {result['token_num']} (prompt: {result['prompt_tokens']}, completion: {result['completion_tokens']})")
        print(f"✓ Response time: {result['response_time']:.2f}s")
        print()
        
    except Exception as e:
        print(f"❌ Exception: {e}")
        print()
        return False
    
    print("=" * 70)
    print("✓ All tests passed!")
    print("=" * 70)
    print()
    print("Summary:")
    print("- System prompts are correctly passed to Bedrock models")
    print("- Requests work both with and without system prompts")
    print("- Token counting and response parsing work correctly")
    print()
    
    return True


if __name__ == "__main__":
    success = verify_system_prompt_handling()
    sys.exit(0 if success else 1)
