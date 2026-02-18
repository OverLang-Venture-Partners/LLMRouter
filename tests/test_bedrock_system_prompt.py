"""
Tests for system prompt handling with Bedrock models.

This test verifies that system prompts are correctly passed to LiteLLM
for Bedrock models (Requirement 5.3).
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from llmrouter.utils.api_calling import call_api


class TestBedrockSystemPrompt:
    """Test system prompt handling for Bedrock models."""

    @patch('llmrouter.utils.api_calling.completion')
    def test_system_prompt_included_in_messages(self, mock_completion):
        """
        Test that system prompts are included in the messages array for Bedrock.
        
        Validates: Requirements 5.3
        """
        # Setup mock response
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="Test response"))]
        mock_response.usage = Mock(
            total_tokens=100,
            prompt_tokens=50,
            completion_tokens=50
        )
        mock_completion.return_value = mock_response

        # Create request with system prompt
        request = {
            "api_endpoint": "https://bedrock.amazonaws.com",
            "query": "What is 2+2?",
            "model_name": "claude-3-sonnet",
            "api_name": "anthropic.claude-3-sonnet-20240229-v1:0",
            "service": "Bedrock",
            "system_prompt": "You are a helpful math tutor.",
            "aws_region": "us-east-1"
        }

        # Call API
        result = call_api(request, api_keys_env='{"Bedrock": "dummy"}')

        # Verify completion was called
        assert mock_completion.called
        call_kwargs = mock_completion.call_args[1]

        # Verify messages array structure
        messages = call_kwargs['messages']
        assert len(messages) == 2, "Should have system and user messages"
        
        # Verify system message
        assert messages[0]['role'] == 'system'
        assert messages[0]['content'] == "You are a helpful math tutor."
        
        # Verify user message
        assert messages[1]['role'] == 'user'
        assert messages[1]['content'] == "What is 2+2?"

        # Verify model format for Bedrock
        assert call_kwargs['model'] == "bedrock/anthropic.claude-3-sonnet-20240229-v1:0"

        # Verify result
        assert result['response'] == "Test response"
        assert 'error' not in result

    @patch('llmrouter.utils.api_calling.completion')
    def test_no_system_prompt_only_user_message(self, mock_completion):
        """
        Test that requests without system prompts only include user message.
        
        Validates: Requirements 5.3
        """
        # Setup mock response
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="Test response"))]
        mock_response.usage = Mock(
            total_tokens=100,
            prompt_tokens=50,
            completion_tokens=50
        )
        mock_completion.return_value = mock_response

        # Create request WITHOUT system prompt
        request = {
            "api_endpoint": "https://bedrock.amazonaws.com",
            "query": "What is 2+2?",
            "model_name": "claude-3-sonnet",
            "api_name": "anthropic.claude-3-sonnet-20240229-v1:0",
            "service": "Bedrock",
            "aws_region": "us-east-1"
        }

        # Call API
        result = call_api(request, api_keys_env='{"Bedrock": "dummy"}')

        # Verify completion was called
        assert mock_completion.called
        call_kwargs = mock_completion.call_args[1]

        # Verify messages array structure
        messages = call_kwargs['messages']
        assert len(messages) == 1, "Should only have user message"
        
        # Verify user message
        assert messages[0]['role'] == 'user'
        assert messages[0]['content'] == "What is 2+2?"

        # Verify result
        assert result['response'] == "Test response"
        assert 'error' not in result

    @patch('llmrouter.utils.api_calling.completion')
    def test_empty_system_prompt_not_included(self, mock_completion):
        """
        Test that empty system prompts are not included in messages array.
        
        Validates: Requirements 5.3
        """
        # Setup mock response
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="Test response"))]
        mock_response.usage = Mock(
            total_tokens=100,
            prompt_tokens=50,
            completion_tokens=50
        )
        mock_completion.return_value = mock_response

        # Create request with empty system prompt
        request = {
            "api_endpoint": "https://bedrock.amazonaws.com",
            "query": "What is 2+2?",
            "model_name": "claude-3-sonnet",
            "api_name": "anthropic.claude-3-sonnet-20240229-v1:0",
            "service": "Bedrock",
            "system_prompt": "",  # Empty string
            "aws_region": "us-east-1"
        }

        # Call API
        result = call_api(request, api_keys_env='{"Bedrock": "dummy"}')

        # Verify completion was called
        assert mock_completion.called
        call_kwargs = mock_completion.call_args[1]

        # Verify messages array structure - empty string is falsy, so should not be included
        messages = call_kwargs['messages']
        assert len(messages) == 1, "Empty system prompt should not be included"
        
        # Verify only user message
        assert messages[0]['role'] == 'user'
        assert messages[0]['content'] == "What is 2+2?"

    @patch('llmrouter.utils.api_calling.completion')
    def test_system_prompt_with_non_bedrock_model(self, mock_completion):
        """
        Test that system prompts work for non-Bedrock models too (backward compatibility).
        
        Validates: Requirements 5.3, 8.5
        """
        # Setup mock response
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="Test response"))]
        mock_response.usage = Mock(
            total_tokens=100,
            prompt_tokens=50,
            completion_tokens=50
        )
        mock_completion.return_value = mock_response

        # Create request for non-Bedrock model with system prompt
        request = {
            "api_endpoint": "https://api.openai.com/v1",
            "query": "What is 2+2?",
            "model_name": "gpt-4",
            "api_name": "gpt-4",
            "service": "OpenAI",
            "system_prompt": "You are a helpful assistant."
        }

        # Call API
        result = call_api(request, api_keys_env='{"OpenAI": "test-key"}')

        # Verify completion was called
        assert mock_completion.called
        call_kwargs = mock_completion.call_args[1]

        # Verify messages array structure
        messages = call_kwargs['messages']
        assert len(messages) == 2, "Should have system and user messages"
        
        # Verify system message
        assert messages[0]['role'] == 'system'
        assert messages[0]['content'] == "You are a helpful assistant."
        
        # Verify user message
        assert messages[1]['role'] == 'user'
        assert messages[1]['content'] == "What is 2+2?"

        # Verify model format for non-Bedrock
        assert call_kwargs['model'] == "openai/gpt-4"

        # Verify result
        assert result['response'] == "Test response"
        assert 'error' not in result
