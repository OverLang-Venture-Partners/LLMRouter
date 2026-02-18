"""
Test OpenClaw Bedrock provider detection.

This test verifies that the _is_bedrock_provider helper function correctly
identifies Bedrock/AWS providers and distinguishes them from other providers.

**Validates: Requirements 1.2, 1.3**
**Property: Provider Routing Correctness**
"""

import pytest
from hypothesis import given, strategies as st, settings
from openclaw_router.config import LLMConfig
from openclaw_router.server import _is_bedrock_provider


class TestBedrockProviderDetection:
    """Test _is_bedrock_provider helper function with unit tests."""
    
    def test_bedrock_provider_lowercase(self):
        """Test that 'bedrock' provider is detected."""
        config = LLMConfig(
            name="test-model",
            provider="bedrock",
            model_id="anthropic.claude-v2",
            base_url="https://bedrock.us-east-1.amazonaws.com"
        )
        assert _is_bedrock_provider(config) is True
    
    def test_bedrock_provider_uppercase(self):
        """Test that 'BEDROCK' provider is detected (case-insensitive)."""
        config = LLMConfig(
            name="test-model",
            provider="BEDROCK",
            model_id="anthropic.claude-v2",
            base_url="https://bedrock.us-east-1.amazonaws.com"
        )
        assert _is_bedrock_provider(config) is True
    
    def test_bedrock_provider_mixed_case(self):
        """Test that 'Bedrock' provider is detected (case-insensitive)."""
        config = LLMConfig(
            name="test-model",
            provider="Bedrock",
            model_id="anthropic.claude-v2",
            base_url="https://bedrock.us-east-1.amazonaws.com"
        )
        assert _is_bedrock_provider(config) is True
    
    def test_aws_provider_lowercase(self):
        """Test that 'aws' provider is detected."""
        config = LLMConfig(
            name="test-model",
            provider="aws",
            model_id="anthropic.claude-v2",
            base_url="https://bedrock.us-east-1.amazonaws.com"
        )
        assert _is_bedrock_provider(config) is True
    
    def test_aws_provider_uppercase(self):
        """Test that 'AWS' provider is detected (case-insensitive)."""
        config = LLMConfig(
            name="test-model",
            provider="AWS",
            model_id="anthropic.claude-v2",
            base_url="https://bedrock.us-east-1.amazonaws.com"
        )
        assert _is_bedrock_provider(config) is True
    
    def test_aws_provider_mixed_case(self):
        """Test that 'Aws' provider is detected (case-insensitive)."""
        config = LLMConfig(
            name="test-model",
            provider="Aws",
            model_id="anthropic.claude-v2",
            base_url="https://bedrock.us-east-1.amazonaws.com"
        )
        assert _is_bedrock_provider(config) is True
    
    def test_openai_provider_not_bedrock(self):
        """Test that 'openai' provider is NOT detected as Bedrock."""
        config = LLMConfig(
            name="gpt4",
            provider="openai",
            model_id="gpt-4",
            base_url="https://api.openai.com/v1"
        )
        assert _is_bedrock_provider(config) is False
    
    def test_nvidia_provider_not_bedrock(self):
        """Test that 'nvidia' provider is NOT detected as Bedrock."""
        config = LLMConfig(
            name="llama",
            provider="nvidia",
            model_id="meta/llama-3.1-70b-instruct",
            base_url="https://integrate.api.nvidia.com/v1"
        )
        assert _is_bedrock_provider(config) is False
    
    def test_anthropic_provider_not_bedrock(self):
        """Test that 'anthropic' provider is NOT detected as Bedrock."""
        config = LLMConfig(
            name="claude",
            provider="anthropic",
            model_id="claude-3-opus-20240229",
            base_url="https://api.anthropic.com/v1"
        )
        assert _is_bedrock_provider(config) is False
    
    def test_none_provider_not_bedrock(self):
        """Test that None provider is NOT detected as Bedrock."""
        config = LLMConfig(
            name="test-model",
            provider=None,
            model_id="test-model",
            base_url="https://api.example.com/v1"
        )
        assert _is_bedrock_provider(config) is False
    
    def test_empty_string_provider_not_bedrock(self):
        """Test that empty string provider is NOT detected as Bedrock."""
        config = LLMConfig(
            name="test-model",
            provider="",
            model_id="test-model",
            base_url="https://api.example.com/v1"
        )
        assert _is_bedrock_provider(config) is False


class TestBedrockProviderDetectionProperties:
    """Property-based tests for provider detection."""
    
    # Feature: openclaw-bedrock-support, Property: Provider Routing Correctness
    @given(st.sampled_from(['bedrock', 'BEDROCK', 'Bedrock', 'BeDrOcK', 'aws', 'AWS', 'Aws', 'AwS']))
    @settings(max_examples=100)
    def test_property_bedrock_providers_always_detected(self, provider):
        """
        Property: Any case variation of 'bedrock' or 'aws' should be detected as Bedrock.
        
        **Validates: Requirements 1.2, 1.3**
        """
        config = LLMConfig(
            name="test-model",
            provider=provider,
            model_id="anthropic.claude-v2",
            base_url="https://bedrock.us-east-1.amazonaws.com"
        )
        assert _is_bedrock_provider(config) is True, \
            f"Provider '{provider}' should be detected as Bedrock"
    
    # Feature: openclaw-bedrock-support, Property: Provider Routing Correctness
    @given(st.sampled_from(['openai', 'nvidia', 'anthropic', 'google', 'cohere', 'mistral', 'huggingface']))
    @settings(max_examples=100)
    def test_property_non_bedrock_providers_not_detected(self, provider):
        """
        Property: Non-Bedrock providers should NOT be detected as Bedrock.
        
        **Validates: Requirements 1.3**
        """
        config = LLMConfig(
            name="test-model",
            provider=provider,
            model_id="test-model-id",
            base_url="https://api.example.com/v1"
        )
        assert _is_bedrock_provider(config) is False, \
            f"Provider '{provider}' should NOT be detected as Bedrock"
    
    # Feature: openclaw-bedrock-support, Property: Provider Routing Correctness
    @given(st.text(min_size=1, max_size=20).filter(
        lambda x: x.lower() not in ['bedrock', 'aws'] and x.strip() != ''
    ))
    @settings(max_examples=100)
    def test_property_random_non_bedrock_providers_not_detected(self, provider):
        """
        Property: Random provider strings (except bedrock/aws) should NOT be detected as Bedrock.
        
        **Validates: Requirements 1.3**
        """
        config = LLMConfig(
            name="test-model",
            provider=provider,
            model_id="test-model-id",
            base_url="https://api.example.com/v1"
        )
        assert _is_bedrock_provider(config) is False, \
            f"Random provider '{provider}' should NOT be detected as Bedrock"
    
    # Feature: openclaw-bedrock-support, Property: Provider Routing Correctness
    @given(st.sampled_from([None, '']))
    @settings(max_examples=100)
    def test_property_none_or_empty_provider_not_detected(self, provider):
        """
        Property: None or empty provider should NOT be detected as Bedrock.
        
        **Validates: Requirements 1.3**
        """
        config = LLMConfig(
            name="test-model",
            provider=provider,
            model_id="test-model-id",
            base_url="https://api.example.com/v1"
        )
        assert _is_bedrock_provider(config) is False, \
            f"Provider '{provider}' (None/empty) should NOT be detected as Bedrock"
    
    # Feature: openclaw-bedrock-support, Property: Provider Routing Correctness
    @given(
        provider=st.sampled_from(['bedrock', 'aws']),
        case_transform=st.sampled_from([str.lower, str.upper, str.title])
    )
    @settings(max_examples=100)
    def test_property_case_insensitivity(self, provider, case_transform):
        """
        Property: Provider detection should be case-insensitive for bedrock/aws.
        
        **Validates: Requirements 1.2**
        """
        transformed_provider = case_transform(provider)
        config = LLMConfig(
            name="test-model",
            provider=transformed_provider,
            model_id="anthropic.claude-v2",
            base_url="https://bedrock.us-east-1.amazonaws.com"
        )
        assert _is_bedrock_provider(config) is True, \
            f"Provider '{transformed_provider}' should be detected as Bedrock (case-insensitive)"
    
    # Feature: openclaw-bedrock-support, Property: Provider Routing Correctness
    @given(
        provider=st.sampled_from(['bedrock', 'aws', 'BEDROCK', 'AWS']),
        name=st.text(min_size=1, max_size=50),
        model_id=st.text(min_size=1, max_size=100),
        base_url=st.text(min_size=1, max_size=200)
    )
    @settings(max_examples=100)
    def test_property_detection_independent_of_other_fields(self, provider, name, model_id, base_url):
        """
        Property: Provider detection should depend only on provider field, not other fields.
        
        **Validates: Requirements 1.2**
        """
        config = LLMConfig(
            name=name,
            provider=provider,
            model_id=model_id,
            base_url=base_url
        )
        # Should always be True for bedrock/aws regardless of other fields
        assert _is_bedrock_provider(config) is True, \
            f"Provider '{provider}' should be detected regardless of other fields"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
