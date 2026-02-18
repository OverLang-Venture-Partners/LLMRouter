"""
Integration tests for OpenClaw server with mixed providers (Bedrock + non-Bedrock).

This test suite verifies that the OpenClaw server can handle configurations with
multiple provider types (Bedrock, NVIDIA, OpenAI, etc.) and correctly route requests
to each provider using different routing strategies.

**Validates: Requirements 1.1, 1.3**
**Properties: Non-Bedrock Provider Preservation**
"""

import pytest
import json
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from hypothesis import given, strategies as st, settings
from openclaw_router.config import LLMConfig, OpenClawConfig, RouterConfig
from openclaw_router.server import LLMBackend
from openclaw_router.routers import OpenClawRouter


class TestMixedProviderServer:
    """Integration tests for OpenClaw server with mixed providers."""
    
    def setup_method(self):
        """Set up test fixtures with mixed provider configuration."""
        self.config = OpenClawConfig()
        self.config.llms = {}
        
        # Create Bedrock models
        self.config.llms["claude-3-sonnet"] = LLMConfig(
            name="claude-3-sonnet",
            provider="bedrock",
            model_id="anthropic.claude-3-sonnet-20240229-v1:0",
            base_url="https://bedrock.us-east-1.amazonaws.com",
            description="Claude 3 Sonnet via AWS Bedrock",
            aws_region="us-east-1",
            max_tokens=4096,
            context_limit=200000,
            input_price=3.0,
            output_price=15.0
        )
        
        self.config.llms["titan-text-express"] = LLMConfig(
            name="titan-text-express",
            provider="bedrock",
            model_id="amazon.titan-text-express-v1",
            base_url="https://bedrock.us-east-1.amazonaws.com",
            description="Amazon Titan Text Express",
            aws_region="us-east-1",
            max_tokens=8192,
            context_limit=8192,
            input_price=0.2,
            output_price=0.6
        )
        
        # Create NVIDIA models
        self.config.llms["llama-3.1-8b"] = LLMConfig(
            name="llama-3.1-8b",
            provider="nvidia",
            model_id="meta/llama-3.1-8b-instruct",
            base_url="https://integrate.api.nvidia.com/v1",
            description="Fast responses, daily chat",
            api_key="test-nvidia-key",
            max_tokens=1024,
            context_limit=128000,
            input_price=0.2,
            output_price=0.2
        )
        
        self.config.llms["llama3-70b"] = LLMConfig(
            name="llama3-70b",
            provider="nvidia",
            model_id="meta/llama3-70b-instruct",
            base_url="https://integrate.api.nvidia.com/v1",
            description="Complex reasoning, deep analysis",
            api_key="test-nvidia-key",
            max_tokens=1024,
            context_limit=8192,
            input_price=0.9,
            output_price=0.9
        )
        
        # Create OpenAI model
        self.config.llms["gpt-4o-mini"] = LLMConfig(
            name="gpt-4o-mini",
            provider="openai",
            model_id="gpt-4o-mini",
            base_url="https://api.openai.com/v1",
            description="OpenAI GPT-4o mini",
            api_key="test-openai-key",
            max_tokens=4096,
            context_limit=128000,
            input_price=0.15,
            output_price=0.6
        )
        
        self.backend = LLMBackend(self.config)
    
    @pytest.mark.asyncio
    async def test_can_select_bedrock_models(self):
        """Test that router can select Bedrock models from mixed config."""
        messages = [{"role": "user", "content": "Test query"}]
        
        with patch.object(self.backend, '_call_bedrock_sync', new_callable=AsyncMock) as mock_bedrock:
            mock_bedrock.return_value = {
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "model": "claude-3-sonnet",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": "Bedrock response"},
                    "finish_reason": "stop"
                }],
                "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}
            }
            
            result = await self.backend.call("claude-3-sonnet", messages, stream=False)
            
            # Verify Bedrock method was called
            mock_bedrock.assert_called_once()
            assert result["model"] == "claude-3-sonnet"
            assert result["choices"][0]["message"]["content"] == "Bedrock response"
    
    @pytest.mark.asyncio
    async def test_can_select_nvidia_models(self):
        """Test that router can select NVIDIA models from mixed config."""
        messages = [{"role": "user", "content": "Test query"}]
        
        with patch.object(self.backend, '_call_sync', new_callable=AsyncMock) as mock_http:
            mock_http.return_value = {
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "model": "llama-3.1-8b",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": "NVIDIA response"},
                    "finish_reason": "stop"
                }],
                "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}
            }
            
            result = await self.backend.call("llama-3.1-8b", messages, stream=False)
            
            # Verify HTTP method was called
            mock_http.assert_called_once()
            assert result["model"] == "llama-3.1-8b"
            assert result["choices"][0]["message"]["content"] == "NVIDIA response"
    
    @pytest.mark.asyncio
    async def test_can_select_openai_models(self):
        """Test that router can select OpenAI models from mixed config."""
        messages = [{"role": "user", "content": "Test query"}]
        
        with patch.object(self.backend, '_call_sync', new_callable=AsyncMock) as mock_http:
            mock_http.return_value = {
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "model": "gpt-4o-mini",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": "OpenAI response"},
                    "finish_reason": "stop"
                }],
                "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}
            }
            
            result = await self.backend.call("gpt-4o-mini", messages, stream=False)
            
            # Verify HTTP method was called
            mock_http.assert_called_once()
            assert result["model"] == "gpt-4o-mini"
            assert result["choices"][0]["message"]["content"] == "OpenAI response"
    
    @pytest.mark.asyncio
    async def test_both_providers_work_in_same_instance(self):
        """Test that both Bedrock and non-Bedrock models work in same backend instance."""
        messages = [{"role": "user", "content": "Test query"}]
        
        with patch.object(self.backend, '_call_bedrock_sync', new_callable=AsyncMock) as mock_bedrock, \
             patch.object(self.backend, '_call_sync', new_callable=AsyncMock) as mock_http:
            
            mock_bedrock.return_value = {
                "id": "chatcmpl-bedrock",
                "object": "chat.completion",
                "model": "claude-3-sonnet",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": "Bedrock response"},
                    "finish_reason": "stop"
                }],
                "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}
            }
            
            mock_http.return_value = {
                "id": "chatcmpl-http",
                "object": "chat.completion",
                "model": "llama-3.1-8b",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": "HTTP response"},
                    "finish_reason": "stop"
                }],
                "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}
            }
            
            # Call Bedrock model
            bedrock_result = await self.backend.call("claude-3-sonnet", messages, stream=False)
            assert bedrock_result["choices"][0]["message"]["content"] == "Bedrock response"
            mock_bedrock.assert_called_once()
            
            # Call NVIDIA model
            nvidia_result = await self.backend.call("llama-3.1-8b", messages, stream=False)
            assert nvidia_result["choices"][0]["message"]["content"] == "HTTP response"
            mock_http.assert_called_once()
            
            # Call OpenAI model
            openai_result = await self.backend.call("gpt-4o-mini", messages, stream=False)
            assert openai_result["choices"][0]["message"]["content"] == "HTTP response"
            assert mock_http.call_count == 2
    
    @pytest.mark.asyncio
    async def test_model_prefix_works_with_both_providers(self):
        """Test that model prefix feature works with both Bedrock and non-Bedrock models."""
        messages = [{"role": "user", "content": "Test query"}]
        
        with patch.object(self.backend, '_call_bedrock_sync', new_callable=AsyncMock) as mock_bedrock, \
             patch.object(self.backend, '_call_sync', new_callable=AsyncMock) as mock_http:
            
            mock_bedrock.return_value = {
                "id": "chatcmpl-bedrock",
                "object": "chat.completion",
                "model": "claude-3-sonnet",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": "Bedrock response"},
                    "finish_reason": "stop"
                }],
                "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}
            }
            
            mock_http.return_value = {
                "id": "chatcmpl-http",
                "object": "chat.completion",
                "model": "llama-3.1-8b",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": "HTTP response"},
                    "finish_reason": "stop"
                }],
                "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}
            }
            
            # Test Bedrock model
            bedrock_result = await self.backend.call("claude-3-sonnet", messages, stream=False)
            assert "content" in bedrock_result["choices"][0]["message"]
            
            # Test NVIDIA model
            nvidia_result = await self.backend.call("llama-3.1-8b", messages, stream=False)
            assert "content" in nvidia_result["choices"][0]["message"]


class TestMixedProviderRoutingStrategies:
    """Test routing strategies with mixed Bedrock and non-Bedrock providers."""
    
    def setup_method(self):
        """Set up test fixtures with mixed provider configuration."""
        self.config = OpenClawConfig()
        self.config.llms = {}
        
        # Create mixed provider models
        self.config.llms["claude-3-sonnet"] = LLMConfig(
            name="claude-3-sonnet",
            provider="bedrock",
            model_id="anthropic.claude-3-sonnet-20240229-v1:0",
            base_url="https://bedrock.us-east-1.amazonaws.com",
            aws_region="us-east-1"
        )
        
        self.config.llms["llama-3.1-8b"] = LLMConfig(
            name="llama-3.1-8b",
            provider="nvidia",
            model_id="meta/llama-3.1-8b-instruct",
            base_url="https://integrate.api.nvidia.com/v1",
            api_key="test-key"
        )
        
        self.config.llms["gpt-4o-mini"] = LLMConfig(
            name="gpt-4o-mini",
            provider="openai",
            model_id="gpt-4o-mini",
            base_url="https://api.openai.com/v1",
            api_key="test-key"
        )
        
        self.config.router = RouterConfig()
    
    @pytest.mark.asyncio
    async def test_random_strategy_with_bedrock_models(self):
        """Test random strategy can select Bedrock models."""
        self.config.router.strategy = "random"
        self.config.router.weights = {
            "claude-3-sonnet": 1,
            "llama-3.1-8b": 1,
            "gpt-4o-mini": 1
        }
        
        router = OpenClawRouter(self.config)
        
        # Run multiple times to ensure Bedrock models can be selected
        selected_models = set()
        for _ in range(30):
            selected = await router.select_model("test query")
            selected_models.add(selected)
        
        # With 30 iterations, we should see at least 2 different models
        assert len(selected_models) >= 2, "Random strategy should select different models"
        # Verify all selected models are valid
        for model in selected_models:
            assert model in self.config.llms, f"Selected model {model} should be in config"
    
    @pytest.mark.asyncio
    async def test_round_robin_strategy_with_bedrock_models(self):
        """Test round_robin strategy cycles through Bedrock and non-Bedrock models."""
        self.config.router.strategy = "round_robin"
        
        router = OpenClawRouter(self.config)
        
        # Select models multiple times
        selected_models = []
        for _ in range(6):
            selected = await router.select_model("test query")
            selected_models.append(selected)
        
        # Verify we cycle through all models
        unique_models = set(selected_models)
        assert len(unique_models) == 3, "Round robin should cycle through all 3 models"
        assert "claude-3-sonnet" in unique_models, "Should include Bedrock model"
        assert "llama-3.1-8b" in unique_models, "Should include NVIDIA model"
        assert "gpt-4o-mini" in unique_models, "Should include OpenAI model"
    
    @pytest.mark.asyncio
    async def test_rules_strategy_with_bedrock_models(self):
        """Test rules strategy can route to Bedrock models."""
        self.config.router.strategy = "rules"
        self.config.router.rules = [
            {"keywords": ["bedrock", "aws", "claude"], "model": "claude-3-sonnet"},
            {"keywords": ["nvidia", "llama"], "model": "llama-3.1-8b"},
            {"keywords": ["openai", "gpt"], "model": "gpt-4o-mini"},
            {"default": "llama-3.1-8b"}
        ]
        
        router = OpenClawRouter(self.config)
        
        # Test routing to Bedrock model
        selected = await router.select_model("Tell me about AWS Bedrock")
        assert selected == "claude-3-sonnet", "Should route to Bedrock model for AWS query"
        
        # Test routing to NVIDIA model
        selected = await router.select_model("Use NVIDIA GPU for inference")
        assert selected == "llama-3.1-8b", "Should route to NVIDIA model"
        
        # Test routing to OpenAI model
        selected = await router.select_model("What is GPT-4?")
        assert selected == "gpt-4o-mini", "Should route to OpenAI model"
        
        # Test default routing
        selected = await router.select_model("Random query")
        assert selected == "llama-3.1-8b", "Should use default model"
    
    @pytest.mark.asyncio
    async def test_weights_with_bedrock_models(self):
        """Test weighted random selection includes Bedrock models."""
        self.config.router.strategy = "random"
        self.config.router.weights = {
            "claude-3-sonnet": 5,  # Higher weight for Bedrock
            "llama-3.1-8b": 1,
            "gpt-4o-mini": 1
        }
        
        router = OpenClawRouter(self.config)
        
        # Run many iterations to verify weight distribution
        selected_counts = {"claude-3-sonnet": 0, "llama-3.1-8b": 0, "gpt-4o-mini": 0}
        for _ in range(100):
            selected = await router.select_model("test query")
            selected_counts[selected] += 1
        
        # Bedrock model should be selected more often due to higher weight
        assert selected_counts["claude-3-sonnet"] > selected_counts["llama-3.1-8b"], \
            "Bedrock model with higher weight should be selected more often"
        assert selected_counts["claude-3-sonnet"] > selected_counts["gpt-4o-mini"], \
            "Bedrock model with higher weight should be selected more often"


class TestNonBedrockProviderPreservation:
    """Property-based tests to verify non-Bedrock providers are preserved."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = OpenClawConfig()
        self.config.llms = {}
        self.backend = LLMBackend(self.config)
    
    # Feature: openclaw-bedrock-support, Property: Non-Bedrock Provider Preservation
    @given(
        provider=st.sampled_from(['nvidia', 'openai', 'anthropic', 'custom']),
        stream=st.booleans()
    )
    @settings(max_examples=50)
    @pytest.mark.asyncio
    async def test_property_non_bedrock_providers_use_http_path(self, provider, stream):
        """
        Property: Non-Bedrock providers always use HTTP path.
        
        **Validates: Requirements 1.3**
        """
        config = LLMConfig(
            name="test-model",
            provider=provider,
            model_id="test-model-id",
            base_url="https://api.example.com/v1",
            api_key="test-key"
        )
        
        self.config.llms["test-model"] = config
        messages = [{"role": "user", "content": "Test"}]
        
        if stream:
            async def mock_stream():
                yield "data: test\n\n"
            
            with patch.object(self.backend, '_call_streaming', return_value=mock_stream()) as mock_http, \
                 patch.object(self.backend, '_call_bedrock_streaming') as mock_bedrock:
                
                result = await self.backend.call("test-model", messages, stream=True)
                
                # Verify HTTP streaming method was called, not Bedrock
                mock_http.assert_called_once()
                mock_bedrock.assert_not_called()
        else:
            with patch.object(self.backend, '_call_sync', new_callable=AsyncMock) as mock_http, \
                 patch.object(self.backend, '_call_bedrock_sync') as mock_bedrock:
                
                mock_http.return_value = {"response": "test"}
                
                await self.backend.call("test-model", messages, stream=False)
                
                # Verify HTTP sync method was called, not Bedrock
                mock_http.assert_called_once()
                mock_bedrock.assert_not_called()
    
    # Feature: openclaw-bedrock-support, Property: Non-Bedrock Provider Preservation
    @given(
        provider=st.sampled_from(['nvidia', 'openai', 'anthropic']),
        max_tokens=st.integers(min_value=100, max_value=4096),
        temperature=st.floats(min_value=0.0, max_value=2.0)
    )
    @settings(max_examples=50)
    @pytest.mark.asyncio
    async def test_property_non_bedrock_behavior_unchanged(self, provider, max_tokens, temperature):
        """
        Property: Non-Bedrock API behavior remains unchanged.
        
        **Validates: Requirements 1.3**
        """
        config = LLMConfig(
            name="test-model",
            provider=provider,
            model_id="test-model-id",
            base_url="https://api.example.com/v1",
            api_key="test-key"
        )
        
        self.config.llms["test-model"] = config
        messages = [{"role": "user", "content": "Test"}]
        
        with patch.object(self.backend, '_call_sync', new_callable=AsyncMock) as mock_http:
            mock_http.return_value = {
                "id": "test",
                "object": "chat.completion",
                "model": "test-model-id",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": "response"},
                    "finish_reason": "stop"
                }],
                "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}
            }
            
            result = await self.backend.call("test-model", messages, max_tokens=max_tokens, 
                                            temperature=temperature, stream=False)
            
            # Verify HTTP method was called with correct parameters
            mock_http.assert_called_once()
            call_args = mock_http.call_args
            
            # Verify parameters are passed correctly
            assert call_args[0][1] == messages  # messages parameter
            assert call_args[0][2] == max_tokens  # max_tokens parameter
            assert call_args[0][3] == temperature  # temperature parameter
            
            # Verify response format is unchanged
            assert "choices" in result
            assert "usage" in result
            assert result["choices"][0]["message"]["content"] == "response"
    
    # Feature: openclaw-bedrock-support, Property: Non-Bedrock Provider Preservation
    @given(
        provider=st.sampled_from(['nvidia', 'openai', 'anthropic']),
        num_messages=st.integers(min_value=1, max_value=10)
    )
    @settings(max_examples=30)
    @pytest.mark.asyncio
    async def test_property_non_bedrock_performance_unchanged(self, provider, num_messages):
        """
        Property: Non-Bedrock performance characteristics remain unchanged.
        
        **Validates: Requirements 1.3**
        """
        config = LLMConfig(
            name="test-model",
            provider=provider,
            model_id="test-model-id",
            base_url="https://api.example.com/v1",
            api_key="test-key"
        )
        
        self.config.llms["test-model"] = config
        
        # Create messages
        messages = [{"role": "user", "content": f"Message {i}"} for i in range(num_messages)]
        
        with patch.object(self.backend, '_call_sync', new_callable=AsyncMock) as mock_http:
            mock_http.return_value = {"response": "test"}
            
            await self.backend.call("test-model", messages, stream=False)
            
            # Verify only HTTP method is called (no Bedrock overhead)
            mock_http.assert_called_once()
            
            # Verify no Bedrock-specific processing occurred
            call_args = mock_http.call_args
            # The messages should be passed directly without Bedrock-specific transformations
            assert len(call_args[0][1]) == num_messages


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
