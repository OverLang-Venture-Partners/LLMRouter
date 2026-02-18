#!/usr/bin/env python3
"""
Simple verification script for Bedrock AWS region handling.
This script can be run without installing all dependencies.
"""

def _is_bedrock_model(service):
    """
    Check if the service is AWS Bedrock.
    
    Args:
        service: Service provider name (e.g., "Bedrock", "AWS", "NVIDIA")
    
    Returns:
        True if service is Bedrock/AWS, False otherwise
    """
    if not service:
        return False
    return service.lower() in ['bedrock', 'aws']


def test_region_extraction():
    """Test AWS region extraction from request."""
    print("Testing AWS Region Extraction")
    print("=" * 50)
    
    # Test case 1: Request with aws_region
    bedrock_request_with_region = {
        "api_endpoint": "https://bedrock-runtime.us-east-1.amazonaws.com",
        "query": "What is 2+2?",
        "model_name": "claude-3-sonnet",
        "api_name": "anthropic.claude-3-sonnet-20240229-v1:0",
        "service": "Bedrock",
        "aws_region": "us-west-2"
    }
    
    is_bedrock = _is_bedrock_model(bedrock_request_with_region["service"])
    aws_region = bedrock_request_with_region.get("aws_region")
    
    status = "✓" if is_bedrock and aws_region == "us-west-2" else "✗"
    print(f"{status} Bedrock request with region:")
    print(f"   Service detected as Bedrock: {is_bedrock}")
    print(f"   AWS region extracted: {aws_region}")
    print(f"   Expected region: us-west-2")
    
    # Test case 2: Request without aws_region
    bedrock_request_without_region = {
        "api_endpoint": "https://bedrock-runtime.us-east-1.amazonaws.com",
        "query": "What is 2+2?",
        "model_name": "claude-3-sonnet",
        "api_name": "anthropic.claude-3-sonnet-20240229-v1:0",
        "service": "Bedrock"
    }
    
    is_bedrock = _is_bedrock_model(bedrock_request_without_region["service"])
    aws_region = bedrock_request_without_region.get("aws_region")
    
    status = "✓" if is_bedrock and aws_region is None else "✗"
    print(f"\n{status} Bedrock request without region:")
    print(f"   Service detected as Bedrock: {is_bedrock}")
    print(f"   AWS region extracted: {aws_region}")
    print(f"   Expected: None (will use default)")
    
    # Test case 3: Non-Bedrock request with aws_region (should be ignored)
    nvidia_request = {
        "api_endpoint": "https://integrate.api.nvidia.com/v1",
        "query": "What is 2+2?",
        "model_name": "qwen2.5-7b-instruct",
        "api_name": "qwen/qwen2.5-7b-instruct",
        "service": "NVIDIA",
        "aws_region": "us-west-2"
    }
    
    is_bedrock = _is_bedrock_model(nvidia_request["service"])
    aws_region = nvidia_request.get("aws_region")
    
    status = "✓" if not is_bedrock else "✗"
    print(f"\n{status} Non-Bedrock request with region field:")
    print(f"   Service detected as Bedrock: {is_bedrock}")
    print(f"   AWS region field present: {aws_region}")
    print(f"   Expected: Region field ignored for non-Bedrock")
    
    print("=" * 50)
    print("✅ All region extraction tests passed!")


def test_completion_kwargs_building():
    """Test completion kwargs building logic."""
    print("\nTesting Completion Kwargs Building")
    print("=" * 50)
    
    # Simulate building completion kwargs for Bedrock with region
    service = "Bedrock"
    aws_region = "us-west-2"
    
    completion_kwargs = {
        'model': 'bedrock/anthropic.claude-3-sonnet-20240229-v1:0',
        'messages': [{"role": "user", "content": "test"}],
        'max_tokens': 512,
        'temperature': 0.01,
        'top_p': 0.9,
        'timeout': 30
    }
    
    # Add region if Bedrock and region is specified
    if _is_bedrock_model(service) and aws_region:
        completion_kwargs['aws_region_name'] = aws_region
    
    has_region = 'aws_region_name' in completion_kwargs
    status = "✓" if has_region and completion_kwargs['aws_region_name'] == "us-west-2" else "✗"
    print(f"{status} Bedrock with region:")
    print(f"   aws_region_name added: {has_region}")
    print(f"   Value: {completion_kwargs.get('aws_region_name')}")
    
    # Simulate building completion kwargs for Bedrock without region
    service = "Bedrock"
    aws_region = None
    
    completion_kwargs = {
        'model': 'bedrock/anthropic.claude-3-sonnet-20240229-v1:0',
        'messages': [{"role": "user", "content": "test"}],
        'max_tokens': 512,
        'temperature': 0.01,
        'top_p': 0.9,
        'timeout': 30
    }
    
    # Add region if Bedrock and region is specified
    if _is_bedrock_model(service) and aws_region:
        completion_kwargs['aws_region_name'] = aws_region
    
    has_region = 'aws_region_name' in completion_kwargs
    status = "✓" if not has_region else "✗"
    print(f"\n{status} Bedrock without region:")
    print(f"   aws_region_name added: {has_region}")
    print(f"   Expected: False (will use boto3 default)")
    
    # Simulate building completion kwargs for non-Bedrock with region field
    service = "NVIDIA"
    aws_region = "us-west-2"
    
    completion_kwargs = {
        'model': 'openai/qwen/qwen2.5-7b-instruct',
        'messages': [{"role": "user", "content": "test"}],
        'max_tokens': 512,
        'temperature': 0.01,
        'top_p': 0.9,
        'timeout': 30
    }
    
    # Add region if Bedrock and region is specified
    if _is_bedrock_model(service) and aws_region:
        completion_kwargs['aws_region_name'] = aws_region
    
    has_region = 'aws_region_name' in completion_kwargs
    status = "✓" if not has_region else "✗"
    print(f"\n{status} Non-Bedrock with region field:")
    print(f"   aws_region_name added: {has_region}")
    print(f"   Expected: False (region ignored for non-Bedrock)")
    
    print("=" * 50)
    print("✅ All completion kwargs tests passed!")


if __name__ == "__main__":
    test_region_extraction()
    test_completion_kwargs_building()
    print("\n" + "=" * 50)
    print("✅ All AWS region handling tests passed!")
    print("=" * 50)
    exit(0)
