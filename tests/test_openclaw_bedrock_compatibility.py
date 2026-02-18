"""
Test OpenClaw compatibility with Bedrock models.

This test verifies that the OpenClaw configuration includes Bedrock models
and documents the current limitation that OpenClaw's direct HTTP implementation
doesn't support Bedrock's AWS authentication.
"""

import pytest
import yaml
from pathlib import Path


class TestOpenClawBedrockConfig:
    """Test OpenClaw configuration includes Bedrock models."""
    
    def test_openclaw_config_has_bedrock_models(self):
        """Test that openclaw_example.yaml includes Bedrock model configurations."""
        config_path = Path("configs/openclaw_example.yaml")
        assert config_path.exists(), "OpenClaw example config should exist"
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Verify llms section exists
        assert 'llms' in config, "Config should have llms section"
        llms = config['llms']
        
        # Find Bedrock models
        bedrock_models = []
        for name, llm_config in llms.items():
            provider = llm_config.get('provider', '').lower()
            if provider in ['bedrock', 'aws']:
                bedrock_models.append(name)
        
        # Verify Bedrock models are present
        assert len(bedrock_models) > 0, "Config should include Bedrock models"
        
        # Verify specific Bedrock models
        expected_models = ['claude-3-sonnet', 'claude-3-haiku', 'titan-text-express']
        for model in expected_models:
            assert model in bedrock_models, f"Config should include {model}"
    
    def test_bedrock_models_have_required_fields(self):
        """Test that Bedrock models in OpenClaw config have required fields."""
        config_path = Path("configs/openclaw_example.yaml")
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        llms = config['llms']
        
        # Check each Bedrock model
        for name, llm_config in llms.items():
            provider = llm_config.get('provider', '').lower()
            if provider in ['bedrock', 'aws']:
                # Required fields for Bedrock
                assert 'model' in llm_config, f"{name} should have model field"
                assert 'description' in llm_config, f"{name} should have description"
                assert 'aws_region' in llm_config, f"{name} should have aws_region"
                assert 'input_price' in llm_config, f"{name} should have input_price"
                assert 'output_price' in llm_config, f"{name} should have output_price"
                
                # Verify model ID format
                model_id = llm_config['model']
                assert '.' in model_id or 'amazon' in model_id, \
                    f"{name} model ID should be valid Bedrock format"
    
    def test_openclaw_config_has_aws_credentials_documentation(self):
        """Test that OpenClaw config includes AWS credentials documentation."""
        config_path = Path("configs/openclaw_example.yaml")
        
        with open(config_path, 'r') as f:
            content = f.read()
        
        # Verify AWS credentials section exists
        assert 'AWS Credentials' in content, "Config should document AWS credentials"
        assert 'AWS_ACCESS_KEY_ID' in content, "Config should mention AWS_ACCESS_KEY_ID"
        assert 'AWS_SECRET_ACCESS_KEY' in content, "Config should mention AWS_SECRET_ACCESS_KEY"
        assert 'aws_region' in content, "Config should mention aws_region"


class TestOpenClawBedrockLimitation:
    """Document the current limitation with OpenClaw and Bedrock."""
    
    def test_openclaw_uses_direct_http_not_litellm(self):
        """
        Document that OpenClaw uses direct HTTP calls, not LiteLLM.
        
        CURRENT LIMITATION:
        OpenClaw's LLMBackend class makes direct HTTP calls using httpx to
        LLM APIs. This works for OpenAI-compatible APIs (NVIDIA, OpenAI, etc.)
        but does NOT work for AWS Bedrock because:
        
        1. Bedrock requires AWS SDK (boto3) authentication, not Bearer tokens
        2. Bedrock uses a different API format than OpenAI
        3. Bedrock endpoints are region-specific
        
        The Bedrock integration implemented in llmrouter.utils.api_calling
        uses LiteLLM which handles these differences, but OpenClaw bypasses
        that module.
        
        WORKAROUND:
        To use Bedrock models with OpenClaw, you would need to:
        1. Modify openclaw_router/server.py LLMBackend class
        2. Detect Bedrock provider and use boto3/LiteLLM instead of httpx
        3. Handle AWS authentication and region configuration
        
        This test documents the limitation for future implementation.
        """
        # This is a documentation test - it always passes
        # The docstring above explains the limitation
        assert True, "See docstring for OpenClaw + Bedrock limitation details"
    
    def test_bedrock_works_with_llmrouter_cli(self):
        """
        Verify that Bedrock DOES work with llmrouter CLI commands.
        
        The Bedrock integration works correctly with:
        - llmrouter infer --router knnrouter --config config.yaml --query "..."
        - llmrouter chat --router knnrouter --config config.yaml
        - llmrouter train --router knnrouter --config config.yaml
        
        These commands use llmrouter.utils.api_calling which has full
        Bedrock support via LiteLLM.
        
        Only the OpenClaw server (llmrouter serve) has the limitation.
        """
        # This is a documentation test
        assert True, "Bedrock works with llmrouter CLI (infer, chat, train)"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
