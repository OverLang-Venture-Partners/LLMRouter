"""
Test OpenClaw Bedrock streaming API calls.

This test verifies that the _call_bedrock_streaming method correctly:
- Converts OpenClaw message format to LiteLLM format
- Calls the Bedrock API via LiteLLM with streaming enabled
- Converts streaming chunks to OpenAI SSE format
- Sends [DONE] marker at the end
- Handles errors appropriately

**Validates: Requirements 2.1, 2.2, 2.3, 3.1**
**Properties: Response Format Compatibility, Error Message Clarity**
"""

import pytest
import json
from unittest.mock import Mock, patch, MagicMock
from hypothesis import given, strategies as st, settings
from openclaw_router.config import LLMConfig
from openclaw_router.server import LLMBackend
from openclaw_router.config import OpenClawConfig


class MockStreamChunk:
    """Mock LiteLLM streaming chunk."""
    
    def __init__(self, role=None, content=None, finish_reason=None):
        self.choices = [MockChoice(role, content, finish_reason)]


class MockChoice:
    """Mock choice in streaming chunk."""
    
    def __init__(self, role=None, content=None, finish_reason=None):
        self.delta = MockDelta(role, content)
        self.finish_reason = finish_reason


class MockDelta:
    """Mock delta in streaming chunk."""
    
    def __init__(self, role=None, content=None):
        self.role = role
        self.content = content


class TestBedrockStreamingCalls:
    """Unit tests for streaming Bedrock API calls."""
    
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
    async def test_streaming_response_format_matches_openai_sse(self):
        """Test that streaming response format matches OpenAI SSE."""
        messages = [
            {"role": "user", "content": "Hello"}
        ]
        
        # Mock streaming chunks
        mock_chunks = [
            MockStreamChunk(role="assistant", content=None),
            MockStreamChunk(content="Hello"),
            MockStreamChunk(content=" there"),
            MockStreamChunk(content="!", finish_reason="stop")
        ]
        
        with patch('litellm.completion', return_value=iter(mock_chunks)):
            chunks = []
            async for chunk in self.backend._call_bedrock_streaming(
                self.bedrock_config,
                messages,
                max_tokens=4096,
                temperature=0.7
            ):
                chunks.append(chunk)
        
        # Verify all chunks are in SSE format
        for chunk in chunks[:-1]:  # Exclude [DONE] marker
            assert chunk.startswith("data: ")
            assert chunk.endswith("\n\n")
            
            # Parse JSON
            json_str = chunk[6:-2]  # Remove "data: " and "\n\n"
            data = json.loads(json_str)
            
            # Verify OpenAI format
            assert "id" in data
            assert "object" in data
            assert data["object"] == "chat.completion.chunk"
            assert "choices" in data
            assert len(data["choices"]) == 1
            assert "delta" in data["choices"][0]
        
        # Verify [DONE] marker
        assert chunks[-1] == "data: [DONE]\n\n"
    
    @pytest.mark.asyncio
    async def test_chunks_contain_delta_with_role_and_content(self):
        """Test that chunks contain delta with role and content."""
        messages = [
            {"role": "user", "content": "Test"}
        ]
        
        mock_chunks = [
            MockStreamChunk(role="assistant", content=None),
            MockStreamChunk(content="Response text")
        ]
        
        with patch('litellm.completion', return_value=iter(mock_chunks)):
            chunks = []
            async for chunk in self.backend._call_bedrock_streaming(
                self.bedrock_config,
                messages,
                max_tokens=4096,
                temperature=0.7
            ):
                if not chunk.startswith("data: [DONE]"):
                    chunks.append(chunk)
        
        # Parse first chunk (should have role)
        first_data = json.loads(chunks[0][6:-2])
        assert "role" in first_data["choices"][0]["delta"]
        assert first_data["choices"][0]["delta"]["role"] == "assistant"
        
        # Parse second chunk (should have content)
        second_data = json.loads(chunks[1][6:-2])
        assert "content" in second_data["choices"][0]["delta"]
        assert second_data["choices"][0]["delta"]["content"] == "Response text"
    
    @pytest.mark.asyncio
    async def test_done_marker_sent_at_end(self):
        """Test that [DONE] marker is sent at end."""
        messages = [
            {"role": "user", "content": "Test"}
        ]
        
        mock_chunks = [
            MockStreamChunk(content="Response")
        ]
        
        with patch('litellm.completion', return_value=iter(mock_chunks)):
            chunks = []
            async for chunk in self.backend._call_bedrock_streaming(
                self.bedrock_config,
                messages,
                max_tokens=4096,
                temperature=0.7
            ):
                chunks.append(chunk)
        
        # Verify last chunk is [DONE]
        assert chunks[-1] == "data: [DONE]\n\n"
    
    @pytest.mark.asyncio
    async def test_error_messages_in_sse_format(self):
        """Test that error messages are in SSE format."""
        messages = [
            {"role": "user", "content": "Test"}
        ]
        
        with patch('litellm.completion', side_effect=Exception("Test error")):
            chunks = []
            async for chunk in self.backend._call_bedrock_streaming(
                self.bedrock_config,
                messages,
                max_tokens=4096,
                temperature=0.7
            ):
                chunks.append(chunk)
        
        # Should have one error chunk
        assert len(chunks) == 1
        assert chunks[0].startswith("data: ")
        
        # Parse error
        error_data = json.loads(chunks[0][6:-2])
        assert "error" in error_data
        assert "Test error" in error_data["error"]
    
    @pytest.mark.asyncio
    async def test_aws_region_passed_to_litellm(self):
        """Test that aws_region is passed to LiteLLM when specified."""
        messages = [{"role": "user", "content": "Test"}]
        
        mock_chunks = [MockStreamChunk(content="Response")]
        
        with patch('litellm.completion', return_value=iter(mock_chunks)) as mock_completion:
            async for _ in self.backend._call_bedrock_streaming(
                self.bedrock_config,
                messages,
                max_tokens=4096,
                temperature=0.7
            ):
                pass
            
            # Verify aws_region_name is passed
            call_kwargs = mock_completion.call_args[1]
            assert "aws_region_name" in call_kwargs
            assert call_kwargs["aws_region_name"] == "us-east-1"
    
    @pytest.mark.asyncio
    async def test_missing_litellm_error(self):
        """Test that missing LiteLLM returns error in SSE format."""
        messages = [{"role": "user", "content": "Test"}]
        
        # Mock the import to fail
        with patch.dict('sys.modules', {'litellm': None}):
            chunks = []
            async for chunk in self.backend._call_bedrock_streaming(
                self.bedrock_config,
                messages,
                max_tokens=4096,
                temperature=0.7
            ):
                chunks.append(chunk)
            
            # Should have one error chunk
            assert len(chunks) == 1
            assert "error" in chunks[0]
            assert "LiteLLM not installed" in chunks[0]
    
    @pytest.mark.asyncio
    async def test_missing_boto3_error(self):
        """Test that missing boto3 returns error in SSE format."""
        messages = [{"role": "user", "content": "Test"}]
        
        with patch('litellm.completion', side_effect=ImportError("No module named 'boto3'")):
            chunks = []
            async for chunk in self.backend._call_bedrock_streaming(
                self.bedrock_config,
                messages,
                max_tokens=4096,
                temperature=0.7
            ):
                chunks.append(chunk)
            
            # Should have one error chunk
            assert len(chunks) == 1
            error_data = json.loads(chunks[0][6:-2])
            assert "error" in error_data
            assert "boto3" in error_data["error"].lower()
            assert "pip install" in error_data["error"].lower()
    
    @pytest.mark.asyncio
    async def test_finish_reason_included_in_final_chunk(self):
        """Test that finish_reason is included in final chunk."""
        messages = [{"role": "user", "content": "Test"}]
        
        mock_chunks = [
            MockStreamChunk(content="Response", finish_reason="stop")
        ]
        
        with patch('litellm.completion', return_value=iter(mock_chunks)):
            chunks = []
            async for chunk in self.backend._call_bedrock_streaming(
                self.bedrock_config,
                messages,
                max_tokens=4096,
                temperature=0.7
            ):
                if not chunk.startswith("data: [DONE]"):
                    chunks.append(chunk)
        
        # Parse chunk
        data = json.loads(chunks[0][6:-2])
        assert data["choices"][0]["finish_reason"] == "stop"


class TestBedrockStreamingProperties:
    """Property-based tests for Bedrock streaming calls."""
    
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
    
    # Feature: openclaw-bedrock-support, Property: Response Format Compatibility
    @given(
        st.lists(
            st.text(min_size=1, max_size=100),
            min_size=1,
            max_size=10
        )
    )
    @settings(max_examples=50)
    @pytest.mark.asyncio
    async def test_property_all_chunks_conform_to_openai_sse(self, content_chunks):
        """
        Property: All streaming chunks conform to OpenAI SSE format.
        
        **Validates: Requirements 2.1, 2.2**
        """
        messages = [{"role": "user", "content": "Test"}]
        
        # Create mock chunks
        mock_chunks = [MockStreamChunk(content=content) for content in content_chunks]
        
        with patch('litellm.completion', return_value=iter(mock_chunks)):
            chunks = []
            async for chunk in self.backend._call_bedrock_streaming(
                self.bedrock_config,
                messages,
                max_tokens=4096,
                temperature=0.7
            ):
                chunks.append(chunk)
        
        # Verify all chunks (except [DONE]) are valid SSE
        for chunk in chunks[:-1]:
            assert chunk.startswith("data: ")
            assert chunk.endswith("\n\n")
            
            # Parse and verify JSON structure
            json_str = chunk[6:-2]
            data = json.loads(json_str)
            
            assert "id" in data
            assert "object" in data
            assert data["object"] == "chat.completion.chunk"
            assert "choices" in data
            assert isinstance(data["choices"], list)
            assert len(data["choices"]) > 0
            assert "delta" in data["choices"][0]
            assert "index" in data["choices"][0]
    
    # Feature: openclaw-bedrock-support, Property: Response Format Compatibility
    @given(st.lists(st.text(min_size=1, max_size=50), min_size=1, max_size=5))
    @settings(max_examples=50)
    @pytest.mark.asyncio
    async def test_property_streaming_ends_with_done_marker(self, content_chunks):
        """
        Property: All streaming responses end with [DONE] marker.
        
        **Validates: Requirements 2.2**
        """
        messages = [{"role": "user", "content": "Test"}]
        
        mock_chunks = [MockStreamChunk(content=content) for content in content_chunks]
        
        with patch('litellm.completion', return_value=iter(mock_chunks)):
            chunks = []
            async for chunk in self.backend._call_bedrock_streaming(
                self.bedrock_config,
                messages,
                max_tokens=4096,
                temperature=0.7
            ):
                chunks.append(chunk)
        
        # Verify last chunk is [DONE]
        assert len(chunks) > 0
        assert chunks[-1] == "data: [DONE]\n\n"
    
    # Feature: openclaw-bedrock-support, Property: Response Format Compatibility
    @given(st.lists(st.text(min_size=1, max_size=100), min_size=1, max_size=10))
    @settings(max_examples=50)
    @pytest.mark.asyncio
    async def test_property_chunk_content_preserved(self, content_chunks):
        """
        Property: Chunk content is preserved from Bedrock response.
        
        **Validates: Requirements 2.1**
        """
        messages = [{"role": "user", "content": "Test"}]
        
        mock_chunks = [MockStreamChunk(content=content) for content in content_chunks]
        
        with patch('litellm.completion', return_value=iter(mock_chunks)):
            chunks = []
            async for chunk in self.backend._call_bedrock_streaming(
                self.bedrock_config,
                messages,
                max_tokens=4096,
                temperature=0.7
            ):
                if not chunk.startswith("data: [DONE]"):
                    chunks.append(chunk)
        
        # Extract content from chunks
        extracted_content = []
        for chunk in chunks:
            data = json.loads(chunk[6:-2])
            if "content" in data["choices"][0]["delta"]:
                extracted_content.append(data["choices"][0]["delta"]["content"])
        
        # Verify all content is preserved
        assert extracted_content == content_chunks


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
