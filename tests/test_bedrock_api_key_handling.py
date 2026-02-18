"""
Tests for Bedrock API key handling in API calling module.
Verifies that Bedrock models skip API key selection and use different parameters.
"""

import pytest
from unittest.mock import patch, MagicMock
from llmrouter.utils.api_calling import call_api


class TestBedrockAPIKeyHandling:
    """Test that Bedrock models skip API key selection and use AWS credentials."""
    
    @patch('llmrouter.utils.api_calling.completion')
    def test_bedrock_omits_api_key_and_base(self, mock_completion):
        """Test that Bedrock models don't pass api_key and api_base to LiteLLM."""
        # Setup mock response
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Test response"
        mock_response.usage = MagicMock()
        mock_response.usage.__dict__ = {
            "total_tokens": 10,
            "prompt_tokens": 5,
            "completion_tokens": 5
        }
        mock_completion.return_value = mock_response
        
        # Bedrock request
        bedrock_request = {
            "api_endpoint": "https://bedrock-runtime.us-east-1.amazonaws.com",
            "query": "What is 2+2?",
            "model_name": "claude-3-sonnet",
            "api_name": "anthropic.claude-3-sonnet-20240229-v1:0",
            "service": "Bedrock"
        }
        
        # Call API
        result = call_api(bedrock_request, api_keys_env="test-key")
        
        # Verify completion was called
        assert mock_completion.called
        
        # Get the kwargs passed to completion
        call_kwargs = mock_completion.call_args[1]
        
        # Verify Bedrock-specific behavior
        assert call_kwargs['model'] == "bedrock/anthropic.claude-3-sonnet-20240229-v1:0"
        assert 'api_key' not in call_kwargs, "api_key should not be passed for Bedrock"
        assert 'api_base' not in call_kwargs, "api_base should not be passed for Bedrock"
        
        # Verify common parameters are still present
        assert 'messages' in call_kwargs
        assert 'max_tokens' in call_kwargs
        assert 'temperature' in call_kwargs
        assert 'top_p' in call_kwargs
        assert 'timeout' in call_kwargs
    
    @patch('llmrouter.utils.api_calling.completion')
    def test_bedrock_with_region_passes_aws_region_name(self, mock_completion):
        """Test that Bedrock models with aws_region pass aws_region_name to LiteLLM."""
        # Setup mock response
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Test response"
        mock_response.usage = MagicMock()
        mock_response.usage.__dict__ = {
            "total_tokens": 10,
            "prompt_tokens": 5,
            "completion_tokens": 5
        }
        mock_completion.return_value = mock_response
        
        # Bedrock request with region
        bedrock_request = {
            "api_endpoint": "https://bedrock-runtime.us-west-2.amazonaws.com",
            "query": "What is 2+2?",
            "model_name": "claude-3-sonnet",
            "api_name": "anthropic.claude-3-sonnet-20240229-v1:0",
            "service": "Bedrock",
            "aws_region": "us-west-2"
        }
        
        # Call API
        result = call_api(bedrock_request, api_keys_env="test-key")
        
        # Verify completion was called
        assert mock_completion.called
        
        # Get the kwargs passed to completion
        call_kwargs = mock_completion.call_args[1]
        
        # Verify aws_region_name is passed
        assert call_kwargs['aws_region_name'] == "us-west-2"
        assert 'api_key' not in call_kwargs
        assert 'api_base' not in call_kwargs
    
    @patch('llmrouter.utils.api_calling.completion')
    def test_non_bedrock_includes_api_key_and_base(self, mock_completion):
        """Test that non-Bedrock models still pass api_key and api_base to LiteLLM."""
        # Setup mock response
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Test response"
        mock_response.usage = MagicMock()
        mock_response.usage.__dict__ = {
            "total_tokens": 10,
            "prompt_tokens": 5,
            "completion_tokens": 5
        }
        mock_completion.return_value = mock_response
        
        # Non-Bedrock request
        nvidia_request = {
            "api_endpoint": "https://integrate.api.nvidia.com/v1",
            "query": "What is 2+2?",
            "model_name": "qwen2.5-7b-instruct",
            "api_name": "qwen/qwen2.5-7b-instruct",
            "service": "NVIDIA"
        }
        
        # Call API
        result = call_api(nvidia_request, api_keys_env="test-api-key")
        
        # Verify completion was called
        assert mock_completion.called
        
        # Get the kwargs passed to completion
        call_kwargs = mock_completion.call_args[1]
        
        # Verify non-Bedrock behavior
        assert call_kwargs['model'] == "openai/qwen/qwen2.5-7b-instruct"
        assert 'api_key' in call_kwargs, "api_key should be passed for non-Bedrock"
        assert 'api_base' in call_kwargs, "api_base should be passed for non-Bedrock"
        assert call_kwargs['api_key'] == "test-api-key"
        assert call_kwargs['api_base'] == "https://integrate.api.nvidia.com/v1"
        
        # Verify aws_region_name is not passed for non-Bedrock
        assert 'aws_region_name' not in call_kwargs
    
    @patch('llmrouter.utils.api_calling.completion')
    def test_bedrock_without_region_omits_aws_region_name(self, mock_completion):
        """Test that Bedrock models without aws_region don't pass aws_region_name."""
        # Setup mock response
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Test response"
        mock_response.usage = MagicMock()
        mock_response.usage.__dict__ = {
            "total_tokens": 10,
            "prompt_tokens": 5,
            "completion_tokens": 5
        }
        mock_completion.return_value = mock_response
        
        # Bedrock request without region
        bedrock_request = {
            "api_endpoint": "https://bedrock-runtime.us-east-1.amazonaws.com",
            "query": "What is 2+2?",
            "model_name": "claude-3-sonnet",
            "api_name": "anthropic.claude-3-sonnet-20240229-v1:0",
            "service": "Bedrock"
        }
        
        # Call API
        result = call_api(bedrock_request, api_keys_env="test-key")
        
        # Verify completion was called
        assert mock_completion.called
        
        # Get the kwargs passed to completion
        call_kwargs = mock_completion.call_args[1]
        
        # Verify aws_region_name is not passed when not specified
        assert 'aws_region_name' not in call_kwargs
        assert 'api_key' not in call_kwargs
        assert 'api_base' not in call_kwargs


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
