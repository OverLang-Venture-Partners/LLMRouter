"""
Test OpenClaw Bedrock synchronous API calls.

This test verifies that the _call_bedrock_sync method correctly:
- Converts OpenClaw message format to LLMRouter request format
- Calls the Bedrock API via llmrouter.utils.api_calling.call_api()
- Converts responses to OpenAI-compatible format
- Handles errors appropriately

**Validates: Requirements 1.1, 1.2, 1.4, 1.5, 3.1, 3.2, 3.3, 3.4, 3.5**
**Properties: Message Format Preservation, Response Format Compatibility, AWS Region Configuration, Error Message Clarity**
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from hypothesis import given, strategies as st, settings
from openclaw_router.config import LLMConfig
from openclaw_router.server import LLMBackend
from openclaw_router.config import OpenClawConfig
from fastapi import HTTPException


class TestBedrockSyncCalls:
    """Unit tests for synchronous Bedrock API calls."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Create a minimal config
        self.config = OpenClawConfig()
        self.config.llms = {}
        
        # Create a Bedrock LLM config
        self.bedrock_config = LLMConfig(
            name="claude",
            provider="bedrock",
            model_id="anthropic.claude-3-sonnet-20240229-v1:0",
            base_url="https://bedrock.us-east-1.amazonaws.com",
            aws_region="us-east-1"
        )
        
        self.backend = LLMBackend(self.config)
    
    @pytest.mark.asyncio
    async def test_successful_bedrock_call_returns_openai_format(self):
        """Test that successful Bedrock API call returns OpenAI-compatible format."""
        messages = [
            {"role": "user", "content": "Hello, how are you?"}
        ]
        
        # Mock call_api response
        mock_response = {
            "response": "I'm doing well, thank you!",
            "token_num": 20,
            "prompt_tokens": 5,
            "completion_tokens": 15,
            "response_time": 1.5
        }
        
        with patch('llmrouter.utils.api_calling.call_api', return_value=[mock_response]):
            result = await self.backend._call_bedrock_sync(
                self.bedrock_config,
                messages,
                max_tokens=4096,
                temperature=0.7
            )
        
        # Verify OpenAI format
        assert "id" in result
        assert result["object"] == "chat.completion"
        assert result["model"] == self.bedrock_config.model_id
        assert len(result["choices"]) == 1
        assert result["choices"][0]["message"]["role"] == "assistant"
        assert result["choices"][0]["message"]["content"] == "I'm doing well, thank you!"
        assert result["choices"][0]["finish_reason"] == "stop"
    
    @pytest.mark.asyncio
    async def test_token_usage_fields_included(self):
        """Test that token usage fields are included in response."""
        messages = [
            {"role": "user", "content": "Test message"}
        ]
        
        mock_response = {
            "response": "Test response",
            "token_num": 100,
            "prompt_tokens": 30,
            "completion_tokens": 70,
            "response_time": 1.0
        }
        
        with patch('llmrouter.utils.api_calling.call_api', return_value=[mock_response]):
            result = await self.backend._call_bedrock_sync(
                self.bedrock_config,
                messages,
                max_tokens=4096,
                temperature=0.7
            )
        
        # Verify token usage
        assert "usage" in result
        assert result["usage"]["prompt_tokens"] == 30
        assert result["usage"]["completion_tokens"] == 70
        assert result["usage"]["total_tokens"] == 100
    
    @pytest.mark.asyncio
    async def test_system_prompt_extracted_correctly(self):
        """Test that system prompt is extracted and passed correctly."""
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello"}
        ]
        
        mock_response = {
            "response": "Hi there!",
            "token_num": 10,
            "prompt_tokens": 5,
            "completion_tokens": 5,
            "response_time": 1.0
        }
        
        with patch('llmrouter.utils.api_calling.call_api', return_value=[mock_response]) as mock_call:
            await self.backend._call_bedrock_sync(
                self.bedrock_config,
                messages,
                max_tokens=4096,
                temperature=0.7
            )
            
            # Verify call_api was called with system_prompt
            call_args = mock_call.call_args
            request = call_args[0][0][0]  # First positional arg, first request in list
            assert request["system_prompt"] == "You are a helpful assistant."
    
    @pytest.mark.asyncio
    async def test_query_extracted_from_last_user_message(self):
        """Test that query is extracted from last user message."""
        messages = [
            {"role": "user", "content": "First message"},
            {"role": "assistant", "content": "Response"},
            {"role": "user", "content": "Second message"}
        ]
        
        mock_response = {
            "response": "Reply to second message",
            "token_num": 10,
            "prompt_tokens": 5,
            "completion_tokens": 5,
            "response_time": 1.0
        }
        
        with patch('llmrouter.utils.api_calling.call_api', return_value=[mock_response]) as mock_call:
            await self.backend._call_bedrock_sync(
                self.bedrock_config,
                messages,
                max_tokens=4096,
                temperature=0.7
            )
            
            # Verify call_api was called with last user message
            call_args = mock_call.call_args
            request = call_args[0][0][0]
            assert request["query"] == "Second message"
    
    @pytest.mark.asyncio
    async def test_multi_turn_conversation_uses_last_user_message(self):
        """Test that multi-turn conversations use last user message as query."""
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "4"},
            {"role": "user", "content": "What about 3+3?"}
        ]
        
        mock_response = {
            "response": "6",
            "token_num": 5,
            "prompt_tokens": 2,
            "completion_tokens": 3,
            "response_time": 1.0
        }
        
        with patch('llmrouter.utils.api_calling.call_api', return_value=[mock_response]) as mock_call:
            await self.backend._call_bedrock_sync(
                self.bedrock_config,
                messages,
                max_tokens=4096,
                temperature=0.7
            )
            
            # Verify last user message is used
            call_args = mock_call.call_args
            request = call_args[0][0][0]
            assert request["query"] == "What about 3+3?"
    
    @pytest.mark.asyncio
    async def test_aws_region_passed_when_specified(self):
        """Test that aws_region is passed to call_api when specified."""
        messages = [{"role": "user", "content": "Test"}]
        
        mock_response = {
            "response": "Response",
            "token_num": 5,
            "prompt_tokens": 2,
            "completion_tokens": 3,
            "response_time": 1.0
        }
        
        with patch('llmrouter.utils.api_calling.call_api', return_value=[mock_response]) as mock_call:
            await self.backend._call_bedrock_sync(
                self.bedrock_config,
                messages,
                max_tokens=4096,
                temperature=0.7
            )
            
            # Verify aws_region is passed
            call_args = mock_call.call_args
            request = call_args[0][0][0]
            assert request["aws_region"] == "us-east-1"
    
    @pytest.mark.asyncio
    async def test_aws_region_defaults_to_none(self):
        """Test that aws_region defaults to None when not specified."""
        config_no_region = LLMConfig(
            name="claude",
            provider="bedrock",
            model_id="anthropic.claude-3-sonnet-20240229-v1:0",
            base_url="https://bedrock.us-east-1.amazonaws.com"
        )
        
        messages = [{"role": "user", "content": "Test"}]
        
        mock_response = {
            "response": "Response",
            "token_num": 5,
            "prompt_tokens": 2,
            "completion_tokens": 3,
            "response_time": 1.0
        }
        
        with patch('llmrouter.utils.api_calling.call_api', return_value=[mock_response]) as mock_call:
            await self.backend._call_bedrock_sync(
                config_no_region,
                messages,
                max_tokens=4096,
                temperature=0.7
            )
            
            # Verify aws_region is None
            call_args = mock_call.call_args
            request = call_args[0][0][0]
            assert request["aws_region"] is None
    
    @pytest.mark.asyncio
    async def test_missing_boto3_error_message(self):
        """Test that missing boto3 returns installation instructions."""
        messages = [{"role": "user", "content": "Test"}]
        
        with patch('llmrouter.utils.api_calling.call_api', side_effect=ImportError("No module named 'boto3'")):
            with pytest.raises(HTTPException) as exc_info:
                await self.backend._call_bedrock_sync(
                    self.bedrock_config,
                    messages,
                    max_tokens=4096,
                    temperature=0.7
                )
            
            assert exc_info.value.status_code == 500
            assert "boto3" in exc_info.value.detail.lower()
            assert "pip install" in exc_info.value.detail.lower()
    
    @pytest.mark.asyncio
    async def test_error_in_response_raises_exception(self):
        """Test that error in call_api response raises HTTPException."""
        messages = [{"role": "user", "content": "Test"}]
        
        mock_response = {
            "response": "Error occurred",
            "token_num": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "response_time": 1.0,
            "error": "AWS credentials not found"
        }
        
        with patch('llmrouter.utils.api_calling.call_api', return_value=[mock_response]):
            with pytest.raises(HTTPException) as exc_info:
                await self.backend._call_bedrock_sync(
                    self.bedrock_config,
                    messages,
                    max_tokens=4096,
                    temperature=0.7
                )
            
            assert exc_info.value.status_code == 500
            assert "AWS credentials not found" in exc_info.value.detail
    
    @pytest.mark.asyncio
    async def test_general_exception_raises_http_exception(self):
        """Test that general exceptions are converted to HTTPException."""
        messages = [{"role": "user", "content": "Test"}]
        
        with patch('llmrouter.utils.api_calling.call_api', side_effect=Exception("Unexpected error")):
            with pytest.raises(HTTPException) as exc_info:
                await self.backend._call_bedrock_sync(
                    self.bedrock_config,
                    messages,
                    max_tokens=4096,
                    temperature=0.7
                )
            
            assert exc_info.value.status_code == 500
            assert "Unexpected error" in exc_info.value.detail


class TestBedrockSyncCallsProperties:
    """Property-based tests for Bedrock synchronous calls."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = OpenClawConfig()
        self.config.llms = {}
        
        self.bedrock_config = LLMConfig(
            name="claude",
            provider="bedrock",
            model_id="anthropic.claude-3-sonnet-20240229-v1:0",
            base_url="https://bedrock.us-east-1.amazonaws.com",
            aws_region="us-east-1"
        )
        
        self.backend = LLMBackend(self.config)
    
    # Feature: openclaw-bedrock-support, Property: Message Format Preservation
    @given(st.text(min_size=1, max_size=500))
    @settings(max_examples=50)
    @pytest.mark.asyncio
    async def test_property_system_messages_preserved(self, system_content):
        """
        Property: System messages are preserved through format conversion.
        
        **Validates: Requirements 1.1, 2.1**
        """
        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": "Test query"}
        ]
        
        mock_response = {
            "response": "Response",
            "token_num": 5,
            "prompt_tokens": 2,
            "completion_tokens": 3,
            "response_time": 1.0
        }
        
        with patch('llmrouter.utils.api_calling.call_api', return_value=[mock_response]) as mock_call:
            await self.backend._call_bedrock_sync(
                self.bedrock_config,
                messages,
                max_tokens=4096,
                temperature=0.7
            )
            
            # Verify system prompt is preserved
            call_args = mock_call.call_args
            request = call_args[0][0][0]
            assert request["system_prompt"] == system_content
    
    # Feature: openclaw-bedrock-support, Property: Message Format Preservation
    @given(st.text(min_size=1, max_size=500))
    @settings(max_examples=50)
    @pytest.mark.asyncio
    async def test_property_user_messages_preserved(self, user_content):
        """
        Property: User messages are preserved through format conversion.
        
        **Validates: Requirements 1.1, 2.1**
        """
        messages = [
            {"role": "user", "content": user_content}
        ]
        
        mock_response = {
            "response": "Response",
            "token_num": 5,
            "prompt_tokens": 2,
            "completion_tokens": 3,
            "response_time": 1.0
        }
        
        with patch('llmrouter.utils.api_calling.call_api', return_value=[mock_response]) as mock_call:
            await self.backend._call_bedrock_sync(
                self.bedrock_config,
                messages,
                max_tokens=4096,
                temperature=0.7
            )
            
            # Verify user message is preserved
            call_args = mock_call.call_args
            request = call_args[0][0][0]
            assert request["query"] == user_content
    
    # Feature: openclaw-bedrock-support, Property: Message Format Preservation
    @given(
        st.lists(
            st.text(min_size=1, max_size=100),
            min_size=1,
            max_size=5
        )
    )
    @settings(max_examples=50)
    @pytest.mark.asyncio
    async def test_property_multi_turn_extracts_last_user_message(self, user_messages):
        """
        Property: Multi-turn conversations extract the last user message as query.
        
        **Validates: Requirements 1.1**
        """
        # Build multi-turn conversation
        messages = []
        for i, content in enumerate(user_messages):
            messages.append({"role": "user", "content": content})
            if i < len(user_messages) - 1:  # Don't add assistant response after last user message
                messages.append({"role": "assistant", "content": f"Response {i}"})
        
        mock_response = {
            "response": "Final response",
            "token_num": 5,
            "prompt_tokens": 2,
            "completion_tokens": 3,
            "response_time": 1.0
        }
        
        with patch('llmrouter.utils.api_calling.call_api', return_value=[mock_response]) as mock_call:
            await self.backend._call_bedrock_sync(
                self.bedrock_config,
                messages,
                max_tokens=4096,
                temperature=0.7
            )
            
            # Verify last user message is used
            call_args = mock_call.call_args
            request = call_args[0][0][0]
            assert request["query"] == user_messages[-1]
    
    # Feature: openclaw-bedrock-support, Property: Response Format Compatibility
    @given(
        response_text=st.text(min_size=1, max_size=1000),
        prompt_tokens=st.integers(min_value=1, max_value=10000),
        completion_tokens=st.integers(min_value=1, max_value=10000)
    )
    @settings(max_examples=50)
    @pytest.mark.asyncio
    async def test_property_response_format_compatibility(self, response_text, prompt_tokens, completion_tokens):
        """
        Property: All Bedrock responses are converted to OpenAI-compatible format.
        
        **Validates: Requirements 1.1, 2.2**
        """
        messages = [{"role": "user", "content": "Test"}]
        
        mock_response = {
            "response": response_text,
            "token_num": prompt_tokens + completion_tokens,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "response_time": 1.0
        }
        
        with patch('llmrouter.utils.api_calling.call_api', return_value=[mock_response]):
            result = await self.backend._call_bedrock_sync(
                self.bedrock_config,
                messages,
                max_tokens=4096,
                temperature=0.7
            )
            
            # Verify OpenAI format
            assert "id" in result
            assert "object" in result
            assert result["object"] == "chat.completion"
            assert "model" in result
            assert "choices" in result
            assert len(result["choices"]) > 0
            assert "message" in result["choices"][0]
            assert "role" in result["choices"][0]["message"]
            assert "content" in result["choices"][0]["message"]
            assert result["choices"][0]["message"]["content"] == response_text
            assert "usage" in result
            assert result["usage"]["prompt_tokens"] == prompt_tokens
            assert result["usage"]["completion_tokens"] == completion_tokens
            assert result["usage"]["total_tokens"] == prompt_tokens + completion_tokens


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
