"""
Test OpenClaw configuration support for AWS region field.

This test verifies that the LLMConfig dataclass properly supports the aws_region
field and that it is correctly loaded from YAML configuration files.

**Validates: Requirements 4.1, 4.2, 4.3**
"""

import pytest
import yaml
import tempfile
import os
from pathlib import Path
from openclaw_router.config import OpenClawConfig, LLMConfig


class TestLLMConfigAwsRegion:
    """Test LLMConfig dataclass aws_region field."""
    
    def test_llm_config_has_aws_region_field(self):
        """Test that LLMConfig has aws_region field with proper type."""
        # Create a minimal LLMConfig instance
        config = LLMConfig(
            name="test-model",
            provider="bedrock",
            model_id="anthropic.claude-v2",
            base_url="https://bedrock.us-east-1.amazonaws.com",
            aws_region="us-east-1"
        )
        
        # Verify aws_region field exists and has correct value
        assert hasattr(config, 'aws_region'), "LLMConfig should have aws_region field"
        assert config.aws_region == "us-east-1", "aws_region should be set correctly"
    
    def test_llm_config_aws_region_defaults_to_none(self):
        """Test that aws_region defaults to None when not specified."""
        # Create LLMConfig without aws_region
        config = LLMConfig(
            name="test-model",
            provider="openai",
            model_id="gpt-4",
            base_url="https://api.openai.com/v1"
        )
        
        # Verify aws_region defaults to None
        assert config.aws_region is None, "aws_region should default to None"
    
    def test_llm_config_aws_region_accepts_none(self):
        """Test that aws_region can be explicitly set to None."""
        config = LLMConfig(
            name="test-model",
            provider="bedrock",
            model_id="anthropic.claude-v2",
            base_url="https://bedrock.us-east-1.amazonaws.com",
            aws_region=None
        )
        
        assert config.aws_region is None, "aws_region should accept None value"


class TestConfigLoadingAwsRegion:
    """Test that aws_region is correctly loaded from YAML configuration."""
    
    def test_aws_region_loaded_from_yaml(self):
        """Test that aws_region is loaded from YAML config."""
        # Create a temporary YAML config with aws_region
        config_content = """
serve:
  host: "0.0.0.0"
  port: 8000

llms:
  claude-bedrock:
    provider: bedrock
    model: anthropic.claude-v2
    base_url: https://bedrock.us-east-1.amazonaws.com
    aws_region: us-east-1
    description: "Claude via Bedrock"
"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(config_content)
            temp_path = f.name
        
        try:
            # Load config
            config = OpenClawConfig.from_yaml(temp_path)
            
            # Verify aws_region is loaded
            assert 'claude-bedrock' in config.llms, "Model should be loaded"
            llm_config = config.llms['claude-bedrock']
            assert llm_config.aws_region == "us-east-1", "aws_region should be loaded from YAML"
        finally:
            os.unlink(temp_path)
    
    def test_aws_region_defaults_to_none_when_not_in_yaml(self):
        """Test that aws_region defaults to None when not specified in YAML."""
        # Create a temporary YAML config without aws_region
        config_content = """
serve:
  host: "0.0.0.0"
  port: 8000

llms:
  gpt4:
    provider: openai
    model: gpt-4
    base_url: https://api.openai.com/v1
    description: "GPT-4"
"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(config_content)
            temp_path = f.name
        
        try:
            # Load config
            config = OpenClawConfig.from_yaml(temp_path)
            
            # Verify aws_region defaults to None
            assert 'gpt4' in config.llms, "Model should be loaded"
            llm_config = config.llms['gpt4']
            assert llm_config.aws_region is None, "aws_region should default to None"
        finally:
            os.unlink(temp_path)
    
    def test_existing_configs_without_aws_region_still_load(self):
        """Test backward compatibility - existing configs without aws_region still work."""
        # Create a config that looks like an old config (no aws_region)
        config_content = """
serve:
  host: "0.0.0.0"
  port: 8000

llms:
  nvidia-llama:
    provider: nvidia
    model: meta/llama-3.1-70b-instruct
    base_url: https://integrate.api.nvidia.com/v1
    api_key_env: NVIDIA_API_KEY
    description: "Llama 3.1 70B via NVIDIA"
    max_tokens: 4096
    context_limit: 32768
  
  gpt4:
    provider: openai
    model: gpt-4
    base_url: https://api.openai.com/v1
    api_key_env: OPENAI_API_KEY
    description: "GPT-4"
"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(config_content)
            temp_path = f.name
        
        try:
            # Load config - should not raise any errors
            config = OpenClawConfig.from_yaml(temp_path)
            
            # Verify both models are loaded
            assert 'nvidia-llama' in config.llms, "NVIDIA model should be loaded"
            assert 'gpt4' in config.llms, "OpenAI model should be loaded"
            
            # Verify aws_region is None for both
            assert config.llms['nvidia-llama'].aws_region is None
            assert config.llms['gpt4'].aws_region is None
        finally:
            os.unlink(temp_path)
    
    def test_aws_region_value_preserved_in_loaded_config(self):
        """Test that aws_region value is preserved exactly as specified in YAML."""
        # Test various region values
        regions = ["us-east-1", "us-west-2", "eu-west-1", "ap-southeast-1"]
        
        for region in regions:
            config_content = f"""
serve:
  host: "0.0.0.0"
  port: 8000

llms:
  bedrock-model:
    provider: bedrock
    model: anthropic.claude-v2
    base_url: https://bedrock.{region}.amazonaws.com
    aws_region: {region}
"""
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
                f.write(config_content)
                temp_path = f.name
            
            try:
                config = OpenClawConfig.from_yaml(temp_path)
                llm_config = config.llms['bedrock-model']
                assert llm_config.aws_region == region, f"aws_region should be {region}"
            finally:
                os.unlink(temp_path)
    
    def test_multiple_models_with_different_regions(self):
        """Test that multiple Bedrock models can have different aws_region values."""
        config_content = """
serve:
  host: "0.0.0.0"
  port: 8000

llms:
  claude-us-east:
    provider: bedrock
    model: anthropic.claude-v2
    base_url: https://bedrock.us-east-1.amazonaws.com
    aws_region: us-east-1
  
  claude-eu-west:
    provider: bedrock
    model: anthropic.claude-v2
    base_url: https://bedrock.eu-west-1.amazonaws.com
    aws_region: eu-west-1
  
  titan-us-west:
    provider: bedrock
    model: amazon.titan-text-express-v1
    base_url: https://bedrock.us-west-2.amazonaws.com
    aws_region: us-west-2
"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(config_content)
            temp_path = f.name
        
        try:
            config = OpenClawConfig.from_yaml(temp_path)
            
            # Verify each model has correct region
            assert config.llms['claude-us-east'].aws_region == "us-east-1"
            assert config.llms['claude-eu-west'].aws_region == "eu-west-1"
            assert config.llms['titan-us-west'].aws_region == "us-west-2"
        finally:
            os.unlink(temp_path)


class TestConfigAwsRegionWithExistingExample:
    """Test aws_region with the actual openclaw_example.yaml if it exists."""
    
    def test_openclaw_example_config_loads_with_aws_region(self):
        """Test that openclaw_example.yaml loads correctly (with or without aws_region)."""
        config_path = Path("configs/openclaw_example.yaml")
        
        if not config_path.exists():
            pytest.skip("openclaw_example.yaml not found")
        
        # Load the config - should not raise any errors
        config = OpenClawConfig.from_yaml(str(config_path))
        
        # Verify config loaded successfully
        assert config is not None
        assert len(config.llms) > 0, "Config should have at least one LLM"
        
        # Check if any Bedrock models exist and verify aws_region field
        for name, llm_config in config.llms.items():
            # All models should have aws_region field (even if None)
            assert hasattr(llm_config, 'aws_region'), f"{name} should have aws_region field"
            
            # If it's a Bedrock model, aws_region might be set
            if llm_config.provider and llm_config.provider.lower() in ['bedrock', 'aws']:
                # aws_region can be None (uses default) or a string
                assert llm_config.aws_region is None or isinstance(llm_config.aws_region, str), \
                    f"{name} aws_region should be None or string"
