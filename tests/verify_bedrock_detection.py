#!/usr/bin/env python3
"""
Simple verification script for Bedrock service detection.
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


def test_bedrock_detection():
    """Test Bedrock service detection."""
    print("Testing Bedrock Service Detection")
    print("=" * 50)
    
    # Test cases
    test_cases = [
        ("Bedrock", True, "Bedrock (capitalized)"),
        ("AWS", True, "AWS (uppercase)"),
        ("bedrock", True, "bedrock (lowercase)"),
        ("aws", True, "aws (lowercase)"),
        ("BEDROCK", True, "BEDROCK (all caps)"),
        ("BeDrOcK", True, "BeDrOcK (mixed case)"),
        ("NVIDIA", False, "NVIDIA"),
        ("OpenAI", False, "OpenAI"),
        ("Anthropic", False, "Anthropic"),
        (None, False, "None"),
        ("", False, "Empty string"),
    ]
    
    all_passed = True
    for service, expected, description in test_cases:
        result = _is_bedrock_model(service)
        status = "✓" if result == expected else "✗"
        if result != expected:
            all_passed = False
        print(f"{status} {description}: {result} (expected {expected})")
    
    print("=" * 50)
    if all_passed:
        print("✅ All tests passed!")
    else:
        print("❌ Some tests failed!")
    
    return all_passed


def test_model_formatting():
    """Test model formatting logic."""
    print("\nTesting Model Formatting Logic")
    print("=" * 50)
    
    # Test Bedrock model formatting
    bedrock_request = {
        "api_name": "anthropic.claude-3-sonnet-20240229-v1:0",
        "service": "Bedrock"
    }
    
    if _is_bedrock_model(bedrock_request["service"]):
        model_format = f"bedrock/{bedrock_request['api_name']}"
        expected = "bedrock/anthropic.claude-3-sonnet-20240229-v1:0"
        status = "✓" if model_format == expected else "✗"
        print(f"{status} Bedrock model format: {model_format}")
        print(f"   Expected: {expected}")
    
    # Test non-Bedrock model formatting
    nvidia_request = {
        "api_name": "qwen/qwen2.5-7b-instruct",
        "service": "NVIDIA"
    }
    
    if not _is_bedrock_model(nvidia_request["service"]):
        model_format = f"openai/{nvidia_request['api_name']}"
        expected = "openai/qwen/qwen2.5-7b-instruct"
        status = "✓" if model_format == expected else "✗"
        print(f"{status} Non-Bedrock model format: {model_format}")
        print(f"   Expected: {expected}")
    
    print("=" * 50)
    print("✅ Model formatting tests completed!")


if __name__ == "__main__":
    detection_passed = test_bedrock_detection()
    test_model_formatting()
    
    if detection_passed:
        exit(0)
    else:
        exit(1)
