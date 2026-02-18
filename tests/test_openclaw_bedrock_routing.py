"""
Test OpenClaw Bedrock routing logic.

This test verifies that the LLMBackend.call() method correctly routes requests
to Bedrock methods for Bedrock providers and to HTTP methods for other providers.

**Validates: Requirements 1.1, 1.2, 1.3**
**Properties: Provider Routing Correctness, Non-Bedrock Provider Preservation**
"""

import pytest
from unittest.mock import Mock, patch, AsyncMock
from hypothesis import given, strategies as st, settings
from openclaw_router.config import LLMConfig, OpenClawConfig
from openclaw_router.server import LLMBackend


class TestBedrockRouting:
    """Integration tests for Bedrock routing."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = OpenClawConfig()
        self.config.llms = {}
        
        # Create Bedrock config
        self.bedrock_config = LLMConfig(
            name="claude",
            provider="bedrock",
            model_id="anthropic.claude-3-sonnet-20240229-v1:0",
            base_url="https://bedrock.us-east-1.amazonaws.com",
            aws_region="us-east-1"
        )
        
        # Create non-Bedrock config
        self.nvidia_config = LLMConfig(
            name="llama",
            provider="nvidia",
            model_id="meta/llama-3.1-8b-instruct",
            base_url="https://integrate.api.nvidia.com/v1",
            api_key="test-key"
        )
        
        self.config.llms["claude"] = self.bedrock_config
        self.config.llms["llama"] = self.nvidia_config
        
        self.backend = LLMBackend(self.config)
    
    @pytest.mark.asyncio
    async def test_bedrock_requests_routed_to_bedrock_sync_method(self):
        """Test that Bedrock requests are routed to _call_bedrock_sync."""
        messages = [{"role": "user", "content": "Test"}]
        
        with patch.object(self.backend, '_call_bedrock_sync', new_callable=AsyncMock) as mock_bedrock:
            mock_bedrock.return_value = {"response": "test"}
            
            await self.backend.call("claude", messages, stream=False)
            
            # Verify _call_bedrock_sync was called
            mock_bedrock.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_bedrock_streaming_requests_routed_to_bedrock_streaming_method(self):
        """Test that streaming Bedrock requests are routed to _call_bedrock_streaming."""
        messages = [{"role": "user", "content": "Test"}]
        
        async def mock_stream():
            yield "data: test\n\n"
        
        with patch.object(self.backend, '_call_bedrock_streaming', return_value=mock_stream()) as mock:
            result = await self.backend.call("claude", messages, stream=True)
            
            # Verify it returns a generator
            assert hasattr(result, '__aiter__')
            mock.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_non_bedrock_requests_routed_to_http_sync_method(self):
        """Test that non-Bedrock requests are routed to _call_sync."""
        messages = [{"role": "user", "content": "Test"}]
        
        with patch.object(self.backend, '_call_sync', new_callable=AsyncMock) as mock_http:
            mock_http.return_value = {"response": "test"}
            
            await self.backend.call("llama", messages, stream=False)
            
            # Verify _call_sync was called
            mock_http.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_non_bedrock_streaming_requests_routed_to_http_streaming_method(self):
        """Test that streaming non-Bedrock requests are routed to _call_streaming."""
        messages = [{"role": "user", "content": "Test"}]
        
        async def mock_stream():
            yield "data: test\n\n"
        
        with patch.object(self.backend, '_call_streaming', return_value=mock_stream()) as mock:
            result = await self.backend.call("llama", messages, stream=True)
            
            # Verify it returns a generator
            assert hasattr(result, '__aiter__')
            mock.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_mixed_provider_scenario(self):
        """Test that both Bedrock and non-Bedrock models work in same backend."""
        messages = [{"role": "user", "content": "Test"}]
        
        with patch.object(self.backend, '_call_bedrock_sync', new_callable=AsyncMock) as mock_bedrock, \
             patch.object(self.backend, '_call_sync', new_callable=AsyncMock) as mock_http:
            
            mock_bedrock.return_value = {"response": "bedrock"}
            mock_http.return_value = {"response": "http"}
            
            # Call Bedrock model
            await self.backend.call("claude", messages, stream=False)
            mock_bedrock.assert_called_once()
            
            # Call non-Bedrock model
            await self.backend.call("llama", messages, stream=False)
            mock_http.assert_called_once()


class TestBedrockRoutingProperties:
    """Property-based tests for Bedrock routing."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = OpenClawConfig()
        self.config.llms = {}
        self.backend = LLMBackend(self.config)
    
    # Feature: openclaw-bedrock-support, Property: Provider Routing Correctness
    @given(
        provider=st.sampled_from(['bedrock', 'aws', 'Bedrock', 'AWS', 'BEDROCK', 'AwS'])
    )
    @settings(max_examples=20)
    @pytest.mark.asyncio
    async def test_property_bedrock_providers_always_route_to_bedrock_methods(self, provider):
        """
        Property: Bedrock providers always route to Bedrock methods.
        
        **Validates: Requirements 1.2**
        """
        config = LLMConfig(
            name="test",
            provider=provider,
            model_id="test-model",
            base_url="https://bedrock.us-east-1.amazonaws.com",
            aws_region="us-east-1"
        )
        
        self.config.llms["test"] = config
        messages = [{"role": "user", "content": "Test"}]
        
        with patch.object(self.backend, '_call_bedrock_sync', new_callable=AsyncMock) as mock_bedrock:
            mock_bedrock.return_value = {"response": "test"}
            
            await self.backend.call("test", messages, stream=False)
            
            # Verify Bedrock method was called
            mock_bedrock.assert_called_once()
    
    # Feature: openclaw-bedrock-support, Property: Non-Bedrock Provider Preservation
    @given(
        provider=st.sampled_from(['openai', 'nvidia', 'anthropic', 'custom'])
    )
    @settings(max_examples=20)
    @pytest.mark.asyncio
    async def test_property_non_bedrock_providers_always_route_to_http_methods(self, provider):
        """
        Property: Non-Bedrock providers always route to HTTP methods.
        
        **Validates: Requirements 1.3**
        """
        config = LLMConfig(
            name="test",
            provider=provider,
            model_id="test-model",
            base_url="https://api.example.com/v1",
            api_key="test-key"
        )
        
        self.config.llms["test"] = config
        messages = [{"role": "user", "content": "Test"}]
        
        with patch.object(self.backend, '_call_sync', new_callable=AsyncMock) as mock_http:
            mock_http.return_value = {"response": "test"}
            
            await self.backend.call("test", messages, stream=False)
            
            # Verify HTTP method was called
            mock_http.assert_called_once()
    
    # Feature: openclaw-bedrock-support, Property: Provider Routing Correctness
    @given(
        is_bedrock=st.booleans(),
        stream=st.booleans()
    )
    @settings(max_examples=20)
    @pytest.mark.asyncio
    async def test_property_routing_deterministic_based_on_provider(self, is_bedrock, stream):
        """
        Property: Routing is deterministic based solely on provider field.
        
        **Validates: Requirements 1.2, 1.3**
        """
        provider = "bedrock" if is_bedrock else "openai"
        config = LLMConfig(
            name="test",
            provider=provider,
            model_id="test-model",
            base_url="https://api.example.com/v1",
            api_key="test-key" if not is_bedrock else None,
            aws_region="us-east-1" if is_bedrock else None
        )
        
        self.config.llms["test"] = config
        messages = [{"role": "user", "content": "Test"}]
        
        if is_bedrock:
            if stream:
                async def mock_stream():
                    yield "data: test\n\n"
                with patch.object(self.backend, '_call_bedrock_streaming', return_value=mock_stream()):
                    result = await self.backend.call("test", messages, stream=stream)
                    assert hasattr(result, '__aiter__')
            else:
                with patch.object(self.backend, '_call_bedrock_sync', new_callable=AsyncMock) as mock:
                    mock.return_value = {"response": "test"}
                    await self.backend.call("test", messages, stream=stream)
                    mock.assert_called_once()
        else:
            if stream:
                async def mock_stream():
                    yield "data: test\n\n"
                with patch.object(self.backend, '_call_streaming', return_value=mock_stream()):
                    result = await self.backend.call("test", messages, stream=stream)
                    assert hasattr(result, '__aiter__')
            else:
                with patch.object(self.backend, '_call_sync', new_callable=AsyncMock) as mock:
                    mock.return_value = {"response": "test"}
                    await self.backend.call("test", messages, stream=stream)
                    mock.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
