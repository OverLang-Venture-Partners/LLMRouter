"""
Tests for Bedrock response parsing in API calling module.

Task 8.1: Verify response parsing works for Bedrock
- Test that response text is extracted correctly
- Ensure original request fields are preserved
- Requirements: 3.3
"""

import pytest
from unittest.mock import Mock, patch
from llmrouter.utils.api_calling import call_api


class TestBedrockResponseParsing:
    """Test Bedrock response parsing functionality."""
    
    @patch('llmrouter.utils.api_calling.completion')
    def test_bedrock_response_text_extraction(self, mock_completion):
        """Test that response text is extracted correctly from Bedrock API response."""
        # Mock LiteLLM response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message = Mock()
        mock_response.choices[0].message.content = "The answer is 4."
        mock_response.usage = Mock()
        mock_response.usage.__dict__ = {
            "total_tokens": 20,
            "prompt_tokens": 10,
            "completion_tokens": 10
        }
        mock_completion.return_value = mock_response
        
        # Test request
        request = {
            "api_endpoint": "https://bedrock-runtime.us-east-1.amazonaws.com",
            "query": "What is 2+2?",
            "model_name": "claude-3-sonnet",
            "api_name": "anthropic.claude-3-sonnet-20240229-v1:0",
            "service": "Bedrock"
        }
        
        # Call API (no API_KEYS needed for Bedrock)
        with patch.dict('os.environ', {'API_KEYS': '{"NVIDIA": "dummy-key"}'}):
            result = call_api(request)
        
        # Verify response text is extracted
        assert result['response'] == "The answer is 4."
        assert isinstance(result['response'], str)
        assert len(result['response']) > 0
    
    @patch('llmrouter.utils.api_calling.completion')
    def test_bedrock_response_preserves_request_fields(self, mock_completion):
        """Test that original request fields are preserved in response."""
        # Mock LiteLLM response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message = Mock()
        mock_response.choices[0].message.content = "Response text"
        mock_response.usage = Mock()
        mock_response.usage.__dict__ = {
            "total_tokens": 15,
            "prompt_tokens": 8,
            "completion_tokens": 7
        }
        mock_completion.return_value = mock_response
        
        # Test request with all fields
        request = {
            "api_endpoint": "https://bedrock-runtime.us-east-1.amazonaws.com",
            "query": "Test query",
            "model_name": "claude-3-sonnet",
            "api_name": "anthropic.claude-3-sonnet-20240229-v1:0",
            "service": "Bedrock",
            "aws_region": "us-west-2",
            "system_prompt": "You are a helpful assistant."
        }
        
        # Call API
        with patch.dict('os.environ', {'API_KEYS': '{"NVIDIA": "dummy-key"}'}):
            result = call_api(request)
        
        # Verify all original request fields are preserved
        assert result['api_endpoint'] == request['api_endpoint']
        assert result['query'] == request['query']
        assert result['model_name'] == request['model_name']
        assert result['api_name'] == request['api_name']
        assert result['service'] == request['service']
        assert result['aws_region'] == request['aws_region']
        assert result['system_prompt'] == request['system_prompt']
    
    @patch('llmrouter.utils.api_calling.completion')
    def test_bedrock_response_adds_output_fields(self, mock_completion):
        """Test that response includes all required output fields."""
        # Mock LiteLLM response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message = Mock()
        mock_response.choices[0].message.content = "Test response"
        mock_response.usage = Mock()
        mock_response.usage.__dict__ = {
            "total_tokens": 25,
            "prompt_tokens": 15,
            "completion_tokens": 10
        }
        mock_completion.return_value = mock_response
        
        # Test request
        request = {
            "api_endpoint": "https://bedrock-runtime.us-east-1.amazonaws.com",
            "query": "Test query",
            "model_name": "claude-3-sonnet",
            "api_name": "anthropic.claude-3-sonnet-20240229-v1:0",
            "service": "Bedrock"
        }
        
        # Call API
        with patch.dict('os.environ', {'API_KEYS': '{"NVIDIA": "dummy-key"}'}):
            result = call_api(request)
        
        # Verify all output fields are present
        assert 'response' in result
        assert 'token_num' in result
        assert 'prompt_tokens' in result
        assert 'completion_tokens' in result
        assert 'response_time' in result
        
        # Verify output field values
        assert result['response'] == "Test response"
        assert result['token_num'] == 25
        assert result['prompt_tokens'] == 15
        assert result['completion_tokens'] == 10
        assert result['response_time'] > 0
    
    @patch('llmrouter.utils.api_calling.completion')
    def test_bedrock_response_with_empty_text(self, mock_completion):
        """Test response parsing when Bedrock returns empty text."""
        # Mock LiteLLM response with empty content
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message = Mock()
        mock_response.choices[0].message.content = ""
        mock_response.usage = Mock()
        mock_response.usage.__dict__ = {
            "total_tokens": 5,
            "prompt_tokens": 5,
            "completion_tokens": 0
        }
        mock_completion.return_value = mock_response
        
        # Test request
        request = {
            "api_endpoint": "https://bedrock-runtime.us-east-1.amazonaws.com",
            "query": "Test query",
            "model_name": "claude-3-sonnet",
            "api_name": "anthropic.claude-3-sonnet-20240229-v1:0",
            "service": "Bedrock"
        }
        
        # Call API
        with patch.dict('os.environ', {'API_KEYS': '{"NVIDIA": "dummy-key"}'}):
            result = call_api(request)
        
        # Verify empty response is handled correctly
        assert result['response'] == ""
        assert result['completion_tokens'] == 0
    
    @patch('llmrouter.utils.api_calling.completion')
    def test_bedrock_response_with_long_text(self, mock_completion):
        """Test response parsing with long response text."""
        # Mock LiteLLM response with long content
        long_text = "This is a very long response. " * 100  # ~3000 characters
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message = Mock()
        mock_response.choices[0].message.content = long_text
        mock_response.usage = Mock()
        mock_response.usage.__dict__ = {
            "total_tokens": 1000,
            "prompt_tokens": 200,
            "completion_tokens": 800
        }
        mock_completion.return_value = mock_response
        
        # Test request
        request = {
            "api_endpoint": "https://bedrock-runtime.us-east-1.amazonaws.com",
            "query": "Write a long essay",
            "model_name": "claude-3-sonnet",
            "api_name": "anthropic.claude-3-sonnet-20240229-v1:0",
            "service": "Bedrock"
        }
        
        # Call API
        with patch.dict('os.environ', {'API_KEYS': '{"NVIDIA": "dummy-key"}'}):
            result = call_api(request)
        
        # Verify long response is preserved completely
        assert result['response'] == long_text
        assert len(result['response']) == len(long_text)
    
    @patch('llmrouter.utils.api_calling.completion')
    def test_bedrock_response_with_special_characters(self, mock_completion):
        """Test response parsing with special characters."""
        # Mock LiteLLM response with special characters
        special_text = "Response with special chars: \n\t\"quotes\" 'apostrophes' & symbols: @#$%^&*()"
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message = Mock()
        mock_response.choices[0].message.content = special_text
        mock_response.usage = Mock()
        mock_response.usage.__dict__ = {
            "total_tokens": 30,
            "prompt_tokens": 10,
            "completion_tokens": 20
        }
        mock_completion.return_value = mock_response
        
        # Test request
        request = {
            "api_endpoint": "https://bedrock-runtime.us-east-1.amazonaws.com",
            "query": "Test query",
            "model_name": "claude-3-sonnet",
            "api_name": "anthropic.claude-3-sonnet-20240229-v1:0",
            "service": "Bedrock"
        }
        
        # Call API
        with patch.dict('os.environ', {'API_KEYS': '{"NVIDIA": "dummy-key"}'}):
            result = call_api(request)
        
        # Verify special characters are preserved
        assert result['response'] == special_text
    
    @patch('llmrouter.utils.api_calling.completion')
    def test_bedrock_batch_response_parsing(self, mock_completion):
        """Test response parsing for batch Bedrock requests."""
        # Mock LiteLLM responses for batch
        def mock_completion_side_effect(**kwargs):
            mock_response = Mock()
            mock_response.choices = [Mock()]
            mock_response.choices[0].message = Mock()
            # Different responses for different calls
            if "first" in kwargs['messages'][-1]['content']:
                mock_response.choices[0].message.content = "First response"
            else:
                mock_response.choices[0].message.content = "Second response"
            mock_response.usage = Mock()
            mock_response.usage.__dict__ = {
                "total_tokens": 20,
                "prompt_tokens": 10,
                "completion_tokens": 10
            }
            return mock_response
        
        mock_completion.side_effect = mock_completion_side_effect
        
        # Test batch requests
        requests = [
            {
                "api_endpoint": "https://bedrock-runtime.us-east-1.amazonaws.com",
                "query": "first query",
                "model_name": "claude-3-sonnet",
                "api_name": "anthropic.claude-3-sonnet-20240229-v1:0",
                "service": "Bedrock"
            },
            {
                "api_endpoint": "https://bedrock-runtime.us-east-1.amazonaws.com",
                "query": "second query",
                "model_name": "claude-3-haiku",
                "api_name": "anthropic.claude-3-haiku-20240307-v1:0",
                "service": "Bedrock"
            }
        ]
        
        # Call API with batch
        with patch.dict('os.environ', {'API_KEYS': '{"NVIDIA": "dummy-key"}'}):
            results = call_api(requests)
        
        # Verify batch results
        assert len(results) == 2
        
        # Verify first result
        assert results[0]['response'] == "First response"
        assert results[0]['query'] == "first query"
        assert results[0]['model_name'] == "claude-3-sonnet"
        
        # Verify second result
        assert results[1]['response'] == "Second response"
        assert results[1]['query'] == "second query"
        assert results[1]['model_name'] == "claude-3-haiku"


class TestBedrockResponseParsingEdgeCases:
    """Test edge cases in Bedrock response parsing."""
    
    @patch('llmrouter.utils.api_calling.completion')
    def test_bedrock_response_without_usage_metadata(self, mock_completion):
        """Test response parsing when usage metadata is missing."""
        # Mock LiteLLM response without usage
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message = Mock()
        mock_response.choices[0].message.content = "Response without usage"
        mock_response.usage = None  # No usage metadata
        mock_completion.return_value = mock_response
        
        # Test request
        request = {
            "api_endpoint": "https://bedrock-runtime.us-east-1.amazonaws.com",
            "query": "Test query",
            "model_name": "claude-3-sonnet",
            "api_name": "anthropic.claude-3-sonnet-20240229-v1:0",
            "service": "Bedrock"
        }
        
        # Call API
        with patch.dict('os.environ', {'API_KEYS': '{"NVIDIA": "dummy-key"}'}):
            result = call_api(request)
        
        # Verify fallback token counting is used
        assert result['response'] == "Response without usage"
        assert result['token_num'] >= 0
        assert result['prompt_tokens'] >= 0
        assert result['completion_tokens'] >= 0
    
    @patch('llmrouter.utils.api_calling.completion')
    def test_bedrock_response_preserves_optional_fields(self, mock_completion):
        """Test that optional request fields are preserved in response."""
        # Mock LiteLLM response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message = Mock()
        mock_response.choices[0].message.content = "Response"
        mock_response.usage = Mock()
        mock_response.usage.__dict__ = {
            "total_tokens": 10,
            "prompt_tokens": 5,
            "completion_tokens": 5
        }
        mock_completion.return_value = mock_response
        
        # Test request with optional fields
        request = {
            "api_endpoint": "https://bedrock-runtime.us-east-1.amazonaws.com",
            "query": "Test query",
            "model_name": "claude-3-sonnet",
            "api_name": "anthropic.claude-3-sonnet-20240229-v1:0",
            "service": "Bedrock",
            "custom_field": "custom_value",  # Custom field
            "another_field": 123  # Another custom field
        }
        
        # Call API
        with patch.dict('os.environ', {'API_KEYS': '{"NVIDIA": "dummy-key"}'}):
            result = call_api(request)
        
        # Verify custom fields are preserved
        assert result['custom_field'] == "custom_value"
        assert result['another_field'] == 123


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
