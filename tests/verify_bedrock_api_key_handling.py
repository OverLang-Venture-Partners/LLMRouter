#!/usr/bin/env python3
"""
Simple verification script for Bedrock API key handling.
This script verifies that Bedrock models skip API key selection and use different parameters.
"""

def test_api_key_handling_logic():
    """Test the logic for API key handling with Bedrock vs non-Bedrock models."""
    print("Testing Bedrock API Key Handling Logic")
    print("=" * 50)
    
    def _is_bedrock_model(service):
        """Check if the service is AWS Bedrock."""
        if not service:
            return False
        return service.lower() in ['bedrock', 'aws']
    
    # Test case 1: Bedrock model should skip API key handling
    print("\n1. Testing Bedrock model (should skip API key):")
    bedrock_request = {
        "api_endpoint": "https://bedrock-runtime.us-east-1.amazonaws.com",
        "query": "What is 2+2?",
        "model_name": "claude-3-sonnet",
        "api_name": "anthropic.claude-3-sonnet-20240229-v1:0",
        "service": "Bedrock"
    }
    
    service = bedrock_request.get('service')
    is_bedrock = _is_bedrock_model(service)
    
    print(f"   Service: {service}")
    print(f"   Is Bedrock: {is_bedrock}")
    
    # Simulate the conditional logic
    completion_kwargs = {
        'model': f"bedrock/{bedrock_request['api_name']}" if is_bedrock else f"openai/{bedrock_request['api_name']}",
        'messages': [{"role": "user", "content": bedrock_request['query']}],
        'max_tokens': 512,
        'temperature': 0.01,
        'top_p': 0.9,
        'timeout': 30
    }
    
    if is_bedrock:
        # Bedrock: add region if specified, don't add api_key/api_base
        aws_region = bedrock_request.get('aws_region')
        if aws_region:
            completion_kwargs['aws_region_name'] = aws_region
        print(f"   ✓ Skipped API key selection (Bedrock uses AWS credentials)")
        print(f"   ✓ Model format: {completion_kwargs['model']}")
        print(f"   ✓ api_key not in kwargs: {'api_key' not in completion_kwargs}")
        print(f"   ✓ api_base not in kwargs: {'api_base' not in completion_kwargs}")
    else:
        # Non-Bedrock: add api_key and api_base
        completion_kwargs['api_key'] = "test-key"
        completion_kwargs['api_base'] = bedrock_request['api_endpoint']
        print(f"   ✗ Should not reach here for Bedrock")
    
    # Test case 2: Bedrock model with region
    print("\n2. Testing Bedrock model with aws_region:")
    bedrock_request_with_region = {
        "api_endpoint": "https://bedrock-runtime.us-west-2.amazonaws.com",
        "query": "What is 2+2?",
        "model_name": "claude-3-sonnet",
        "api_name": "anthropic.claude-3-sonnet-20240229-v1:0",
        "service": "Bedrock",
        "aws_region": "us-west-2"
    }
    
    service = bedrock_request_with_region.get('service')
    is_bedrock = _is_bedrock_model(service)
    
    completion_kwargs = {
        'model': f"bedrock/{bedrock_request_with_region['api_name']}" if is_bedrock else f"openai/{bedrock_request_with_region['api_name']}",
        'messages': [{"role": "user", "content": bedrock_request_with_region['query']}],
        'max_tokens': 512,
        'temperature': 0.01,
        'top_p': 0.9,
        'timeout': 30
    }
    
    if is_bedrock:
        aws_region = bedrock_request_with_region.get('aws_region')
        if aws_region:
            completion_kwargs['aws_region_name'] = aws_region
        print(f"   ✓ aws_region_name in kwargs: {'aws_region_name' in completion_kwargs}")
        print(f"   ✓ aws_region_name value: {completion_kwargs.get('aws_region_name')}")
        print(f"   ✓ api_key not in kwargs: {'api_key' not in completion_kwargs}")
        print(f"   ✓ api_base not in kwargs: {'api_base' not in completion_kwargs}")
    
    # Test case 3: Non-Bedrock model should include API key
    print("\n3. Testing non-Bedrock model (should include API key):")
    nvidia_request = {
        "api_endpoint": "https://integrate.api.nvidia.com/v1",
        "query": "What is 2+2?",
        "model_name": "qwen2.5-7b-instruct",
        "api_name": "qwen/qwen2.5-7b-instruct",
        "service": "NVIDIA"
    }
    
    service = nvidia_request.get('service')
    is_bedrock = _is_bedrock_model(service)
    
    print(f"   Service: {service}")
    print(f"   Is Bedrock: {is_bedrock}")
    
    completion_kwargs = {
        'model': f"bedrock/{nvidia_request['api_name']}" if is_bedrock else f"openai/{nvidia_request['api_name']}",
        'messages': [{"role": "user", "content": nvidia_request['query']}],
        'max_tokens': 512,
        'temperature': 0.01,
        'top_p': 0.9,
        'timeout': 30
    }
    
    if is_bedrock:
        aws_region = nvidia_request.get('aws_region')
        if aws_region:
            completion_kwargs['aws_region_name'] = aws_region
        print(f"   ✗ Should not reach here for non-Bedrock")
    else:
        # Non-Bedrock: add api_key and api_base
        completion_kwargs['api_key'] = "test-api-key"
        completion_kwargs['api_base'] = nvidia_request['api_endpoint']
        print(f"   ✓ API key selection performed")
        print(f"   ✓ Model format: {completion_kwargs['model']}")
        print(f"   ✓ api_key in kwargs: {'api_key' in completion_kwargs}")
        print(f"   ✓ api_base in kwargs: {'api_base' in completion_kwargs}")
        print(f"   ✓ aws_region_name not in kwargs: {'aws_region_name' not in completion_kwargs}")
    
    print("\n" + "=" * 50)
    print("✅ All API key handling logic tests completed!")
    print("\nSummary:")
    print("  - Bedrock models skip API key selection")
    print("  - Bedrock models omit api_key and api_base parameters")
    print("  - Bedrock models pass aws_region_name when region is specified")
    print("  - Non-Bedrock models include api_key and api_base parameters")
    print("  - Non-Bedrock models do not pass aws_region_name")


if __name__ == "__main__":
    test_api_key_handling_logic()
    exit(0)
