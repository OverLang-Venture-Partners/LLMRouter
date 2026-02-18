"""
Verification script for Bedrock inference parameter passthrough.

This script verifies that temperature, top_p, and max_tokens are correctly
passed to LiteLLM for Bedrock models.

Requirements: 5.5
"""

import os
from unittest.mock import Mock, patch
from llmrouter.utils.api_calling import call_api


def test_bedrock_inference_parameters_passthrough():
    """
    Verify that inference parameters (temperature, top_p, max_tokens) are
    correctly passed to LiteLLM for Bedrock models.
    """
    print("Testing Bedrock inference parameter passthrough...")
    
    # Test data
    bedrock_request = {
        "api_endpoint": "https://bedrock-runtime.us-east-1.amazonaws.com",
        "query": "What is 2+2?",
        "model_name": "claude-3-sonnet",
        "api_name": "anthropic.claude-3-sonnet-20240229-v1:0",
        "service": "Bedrock",
        "aws_region": "us-east-1"
    }
    
    # Custom inference parameters
    custom_max_tokens = 1024
    custom_temperature = 0.7
    custom_top_p = 0.95
    custom_timeout = 60
    
    # Mock LiteLLM completion response
    mock_response = Mock()
    mock_response.choices = [Mock()]
    mock_response.choices[0].message.content = "The answer is 4."
    mock_response.usage = Mock()
    mock_response.usage.__dict__ = {
        "total_tokens": 20,
        "prompt_tokens": 10,
        "completion_tokens": 10
    }
    
    # Mock the completion function and capture the call
    with patch('llmrouter.utils.api_calling.completion', return_value=mock_response) as mock_completion:
        # Set dummy API_KEYS (not used for Bedrock but required by call_api)
        os.environ['API_KEYS'] = '{"Bedrock": "dummy"}'
        
        try:
            # Call the API with custom parameters
            result = call_api(
                request=bedrock_request,
                max_tokens=custom_max_tokens,
                temperature=custom_temperature,
                top_p=custom_top_p,
                timeout=custom_timeout
            )
            
            # Verify the completion function was called
            assert mock_completion.called, "LiteLLM completion was not called"
            
            # Get the actual call arguments
            call_args = mock_completion.call_args
            actual_kwargs = call_args.kwargs if call_args.kwargs else call_args[1] if len(call_args) > 1 else {}
            
            # Verify model format
            assert 'model' in actual_kwargs, "model parameter not passed to LiteLLM"
            assert actual_kwargs['model'] == f"bedrock/{bedrock_request['api_name']}", \
                f"Expected model format 'bedrock/{bedrock_request['api_name']}', got '{actual_kwargs['model']}'"
            
            # Verify inference parameters are passed
            assert 'max_tokens' in actual_kwargs, "max_tokens parameter not passed to LiteLLM"
            assert actual_kwargs['max_tokens'] == custom_max_tokens, \
                f"Expected max_tokens={custom_max_tokens}, got {actual_kwargs['max_tokens']}"
            
            assert 'temperature' in actual_kwargs, "temperature parameter not passed to LiteLLM"
            assert actual_kwargs['temperature'] == custom_temperature, \
                f"Expected temperature={custom_temperature}, got {actual_kwargs['temperature']}"
            
            assert 'top_p' in actual_kwargs, "top_p parameter not passed to LiteLLM"
            assert actual_kwargs['top_p'] == custom_top_p, \
                f"Expected top_p={custom_top_p}, got {actual_kwargs['top_p']}"
            
            assert 'timeout' in actual_kwargs, "timeout parameter not passed to LiteLLM"
            assert actual_kwargs['timeout'] == custom_timeout, \
                f"Expected timeout={custom_timeout}, got {actual_kwargs['timeout']}"
            
            # Verify AWS region is passed
            assert 'aws_region_name' in actual_kwargs, "aws_region_name parameter not passed to LiteLLM"
            assert actual_kwargs['aws_region_name'] == bedrock_request['aws_region'], \
                f"Expected aws_region_name={bedrock_request['aws_region']}, got {actual_kwargs['aws_region_name']}"
            
            # Verify API key and api_base are NOT passed for Bedrock
            assert 'api_key' not in actual_kwargs, "api_key should not be passed for Bedrock models"
            assert 'api_base' not in actual_kwargs, "api_base should not be passed for Bedrock models"
            
            # Verify messages are formatted correctly
            assert 'messages' in actual_kwargs, "messages parameter not passed to LiteLLM"
            messages = actual_kwargs['messages']
            assert len(messages) == 1, f"Expected 1 message, got {len(messages)}"
            assert messages[0]['role'] == 'user', f"Expected role 'user', got '{messages[0]['role']}'"
            assert messages[0]['content'] == bedrock_request['query'], \
                f"Expected content '{bedrock_request['query']}', got '{messages[0]['content']}'"
            
            # Verify result contains expected fields
            assert 'response' in result, "response field missing from result"
            assert result['response'] == "The answer is 4.", f"Unexpected response: {result['response']}"
            assert 'token_num' in result, "token_num field missing from result"
            assert result['token_num'] == 20, f"Expected token_num=20, got {result['token_num']}"
            
            print("✓ All inference parameters correctly passed to LiteLLM")
            print(f"  - Model format: {actual_kwargs['model']}")
            print(f"  - max_tokens: {actual_kwargs['max_tokens']}")
            print(f"  - temperature: {actual_kwargs['temperature']}")
            print(f"  - top_p: {actual_kwargs['top_p']}")
            print(f"  - timeout: {actual_kwargs['timeout']}")
            print(f"  - aws_region_name: {actual_kwargs['aws_region_name']}")
            print(f"  - API key NOT passed (correct for Bedrock)")
            print(f"  - api_base NOT passed (correct for Bedrock)")
            
        finally:
            # Clean up environment
            if 'API_KEYS' in os.environ:
                del os.environ['API_KEYS']


def test_bedrock_with_system_prompt():
    """
    Verify that system prompt is correctly included in messages for Bedrock models.
    """
    print("\nTesting Bedrock with system prompt...")
    
    # Test data with system prompt
    bedrock_request = {
        "api_endpoint": "https://bedrock-runtime.us-east-1.amazonaws.com",
        "query": "What is 2+2?",
        "model_name": "claude-3-sonnet",
        "api_name": "anthropic.claude-3-sonnet-20240229-v1:0",
        "service": "Bedrock",
        "system_prompt": "You are a helpful math tutor."
    }
    
    # Mock LiteLLM completion response
    mock_response = Mock()
    mock_response.choices = [Mock()]
    mock_response.choices[0].message.content = "The answer is 4."
    mock_response.usage = Mock()
    mock_response.usage.__dict__ = {
        "total_tokens": 25,
        "prompt_tokens": 15,
        "completion_tokens": 10
    }
    
    # Mock the completion function
    with patch('llmrouter.utils.api_calling.completion', return_value=mock_response) as mock_completion:
        # Set dummy API_KEYS
        os.environ['API_KEYS'] = '{"Bedrock": "dummy"}'
        
        try:
            # Call the API
            result = call_api(request=bedrock_request)
            
            # Get the actual call arguments
            call_args = mock_completion.call_args
            actual_kwargs = call_args.kwargs if call_args.kwargs else call_args[1] if len(call_args) > 1 else {}
            
            # Verify messages include system prompt
            assert 'messages' in actual_kwargs, "messages parameter not passed to LiteLLM"
            messages = actual_kwargs['messages']
            assert len(messages) == 2, f"Expected 2 messages (system + user), got {len(messages)}"
            
            # Verify system message
            assert messages[0]['role'] == 'system', f"Expected first message role 'system', got '{messages[0]['role']}'"
            assert messages[0]['content'] == bedrock_request['system_prompt'], \
                f"Expected system prompt '{bedrock_request['system_prompt']}', got '{messages[0]['content']}'"
            
            # Verify user message
            assert messages[1]['role'] == 'user', f"Expected second message role 'user', got '{messages[1]['role']}'"
            assert messages[1]['content'] == bedrock_request['query'], \
                f"Expected query '{bedrock_request['query']}', got '{messages[1]['content']}'"
            
            print("✓ System prompt correctly included in messages")
            print(f"  - System message: {messages[0]}")
            print(f"  - User message: {messages[1]}")
            
        finally:
            # Clean up environment
            if 'API_KEYS' in os.environ:
                del os.environ['API_KEYS']


def test_non_bedrock_parameters():
    """
    Verify that non-Bedrock models still receive inference parameters correctly.
    """
    print("\nTesting non-Bedrock model parameter passthrough...")
    
    # Test data for non-Bedrock model
    nvidia_request = {
        "api_endpoint": "https://integrate.api.nvidia.com/v1",
        "query": "What is 2+2?",
        "model_name": "qwen2.5-7b-instruct",
        "api_name": "qwen/qwen2.5-7b-instruct",
        "service": "NVIDIA"
    }
    
    # Custom inference parameters
    custom_max_tokens = 2048
    custom_temperature = 0.5
    custom_top_p = 0.9
    
    # Mock LiteLLM completion response
    mock_response = Mock()
    mock_response.choices = [Mock()]
    mock_response.choices[0].message.content = "4"
    mock_response.usage = Mock()
    mock_response.usage.__dict__ = {
        "total_tokens": 15,
        "prompt_tokens": 10,
        "completion_tokens": 5
    }
    
    # Mock the completion function
    with patch('llmrouter.utils.api_calling.completion', return_value=mock_response) as mock_completion:
        # Set API_KEYS
        os.environ['API_KEYS'] = '{"NVIDIA": "test-key"}'
        
        try:
            # Call the API with custom parameters
            result = call_api(
                request=nvidia_request,
                max_tokens=custom_max_tokens,
                temperature=custom_temperature,
                top_p=custom_top_p
            )
            
            # Get the actual call arguments
            call_args = mock_completion.call_args
            actual_kwargs = call_args.kwargs if call_args.kwargs else call_args[1] if len(call_args) > 1 else {}
            
            # Verify inference parameters are passed
            assert actual_kwargs['max_tokens'] == custom_max_tokens, \
                f"Expected max_tokens={custom_max_tokens}, got {actual_kwargs['max_tokens']}"
            assert actual_kwargs['temperature'] == custom_temperature, \
                f"Expected temperature={custom_temperature}, got {actual_kwargs['temperature']}"
            assert actual_kwargs['top_p'] == custom_top_p, \
                f"Expected top_p={custom_top_p}, got {actual_kwargs['top_p']}"
            
            # Verify API key and api_base ARE passed for non-Bedrock
            assert 'api_key' in actual_kwargs, "api_key should be passed for non-Bedrock models"
            assert 'api_base' in actual_kwargs, "api_base should be passed for non-Bedrock models"
            
            # Verify aws_region_name is NOT passed for non-Bedrock
            assert 'aws_region_name' not in actual_kwargs, "aws_region_name should not be passed for non-Bedrock models"
            
            print("✓ Non-Bedrock model parameters correctly passed")
            print(f"  - max_tokens: {actual_kwargs['max_tokens']}")
            print(f"  - temperature: {actual_kwargs['temperature']}")
            print(f"  - top_p: {actual_kwargs['top_p']}")
            print(f"  - API key passed (correct for non-Bedrock)")
            print(f"  - api_base passed (correct for non-Bedrock)")
            
        finally:
            # Clean up environment
            if 'API_KEYS' in os.environ:
                del os.environ['API_KEYS']


if __name__ == "__main__":
    print("=" * 80)
    print("Bedrock Inference Parameter Passthrough Verification")
    print("=" * 80)
    
    try:
        test_bedrock_inference_parameters_passthrough()
        test_bedrock_with_system_prompt()
        test_non_bedrock_parameters()
        
        print("\n" + "=" * 80)
        print("✓ All verification tests passed!")
        print("=" * 80)
        print("\nSummary:")
        print("- Inference parameters (temperature, top_p, max_tokens) are correctly passed to LiteLLM")
        print("- System prompts are correctly included in messages array")
        print("- AWS region is correctly passed for Bedrock models")
        print("- API key/base are correctly omitted for Bedrock models")
        print("- Non-Bedrock models continue to work correctly")
        
    except AssertionError as e:
        print(f"\n✗ Verification failed: {e}")
        exit(1)
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
