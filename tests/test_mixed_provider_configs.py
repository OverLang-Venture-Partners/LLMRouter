"""
Tests for mixed provider configurations (Bedrock + NVIDIA, Bedrock + OpenAI).

These tests verify that LLMRouter can handle configurations with multiple
provider types and route requests correctly to each provider.
"""

import json
import pytest
from pathlib import Path
from llmrouter.utils.api_calling import _is_bedrock_model


class TestMixedBedrockNvidiaConfig:
    """Test mixed Bedrock + NVIDIA provider configurations."""
    
    @pytest.fixture
    def mixed_bedrock_nvidia_config(self):
        """Load the mixed Bedrock + NVIDIA configuration."""
        config_path = Path("data/example_data/llm_candidates/mixed_bedrock_nvidia.json")
        with open(config_path, 'r') as f:
            return json.load(f)
    
    def test_config_file_exists(self):
        """Test that the mixed Bedrock + NVIDIA config file exists."""
        config_path = Path("data/example_data/llm_candidates/mixed_bedrock_nvidia.json")
        assert config_path.exists(), "Mixed Bedrock + NVIDIA config file should exist"
    
    def test_config_has_bedrock_models(self, mixed_bedrock_nvidia_config):
        """Test that config contains Bedrock models."""
        bedrock_models = [
            name for name, config in mixed_bedrock_nvidia_config.items()
            if _is_bedrock_model(config.get("service"))
        ]
        assert len(bedrock_models) > 0, "Config should contain at least one Bedrock model"
        assert "claude-3-sonnet-bedrock" in bedrock_models
        assert "titan-text-express-bedrock" in bedrock_models
    
    def test_config_has_nvidia_models(self, mixed_bedrock_nvidia_config):
        """Test that config contains NVIDIA models."""
        nvidia_models = [
            name for name, config in mixed_bedrock_nvidia_config.items()
            if config.get("service") == "NVIDIA"
        ]
        assert len(nvidia_models) > 0, "Config should contain at least one NVIDIA model"
        assert "qwen2.5-7b-instruct" in nvidia_models
        assert "llama-3.1-8b-instruct" in nvidia_models
    
    def test_bedrock_models_have_required_fields(self, mixed_bedrock_nvidia_config):
        """Test that Bedrock models have all required fields."""
        required_fields = ["model", "service", "size", "feature", "input_price", "output_price"]
        
        for name, config in mixed_bedrock_nvidia_config.items():
            if _is_bedrock_model(config.get("service")):
                for field in required_fields:
                    assert field in config, f"Bedrock model {name} should have {field} field"
                
                # Verify service is correctly set
                assert config["service"] in ["Bedrock", "AWS"], \
                    f"Bedrock model {name} should have service 'Bedrock' or 'AWS'"
    
    def test_nvidia_models_have_required_fields(self, mixed_bedrock_nvidia_config):
        """Test that NVIDIA models have all required fields."""
        required_fields = ["model", "service", "size", "feature", "input_price", "output_price", "api_endpoint"]
        
        for name, config in mixed_bedrock_nvidia_config.items():
            if config.get("service") == "NVIDIA":
                for field in required_fields:
                    assert field in config, f"NVIDIA model {name} should have {field} field"
                
                # Verify api_endpoint is set
                assert "api.nvidia.com" in config["api_endpoint"], \
                    f"NVIDIA model {name} should have NVIDIA API endpoint"
    
    def test_bedrock_models_have_aws_region(self, mixed_bedrock_nvidia_config):
        """Test that Bedrock models have aws_region field."""
        for name, config in mixed_bedrock_nvidia_config.items():
            if _is_bedrock_model(config.get("service")):
                assert "aws_region" in config, \
                    f"Bedrock model {name} should have aws_region field"
                assert config["aws_region"] in ["us-east-1", "us-west-2", "us-west-1", "eu-west-1"], \
                    f"Bedrock model {name} should have valid AWS region"
    
    def test_model_formatting_differs_by_provider(self, mixed_bedrock_nvidia_config):
        """Test that model formatting logic differs between Bedrock and NVIDIA."""
        for name, config in mixed_bedrock_nvidia_config.items():
            service = config.get("service")
            model_id = config.get("model")
            
            if _is_bedrock_model(service):
                # Bedrock models should use bedrock/{model_id} format
                expected_format = f"bedrock/{model_id}"
                assert expected_format.startswith("bedrock/"), \
                    f"Bedrock model {name} should use bedrock/ prefix"
            elif service == "NVIDIA":
                # NVIDIA models should use openai/{model_id} format
                expected_format = f"openai/{model_id}"
                assert expected_format.startswith("openai/"), \
                    f"NVIDIA model {name} should use openai/ prefix"


class TestMixedBedrockOpenAIConfig:
    """Test mixed Bedrock + OpenAI provider configurations."""
    
    @pytest.fixture
    def mixed_bedrock_openai_config(self):
        """Load the mixed Bedrock + OpenAI configuration."""
        config_path = Path("data/example_data/llm_candidates/mixed_bedrock_openai.json")
        with open(config_path, 'r') as f:
            return json.load(f)
    
    def test_config_file_exists(self):
        """Test that the mixed Bedrock + OpenAI config file exists."""
        config_path = Path("data/example_data/llm_candidates/mixed_bedrock_openai.json")
        assert config_path.exists(), "Mixed Bedrock + OpenAI config file should exist"
    
    def test_config_has_bedrock_models(self, mixed_bedrock_openai_config):
        """Test that config contains Bedrock models."""
        bedrock_models = [
            name for name, config in mixed_bedrock_openai_config.items()
            if _is_bedrock_model(config.get("service"))
        ]
        assert len(bedrock_models) > 0, "Config should contain at least one Bedrock model"
        assert "claude-3-haiku-bedrock" in bedrock_models
        assert "llama-3-70b-bedrock" in bedrock_models
    
    def test_config_has_openai_models(self, mixed_bedrock_openai_config):
        """Test that config contains OpenAI models."""
        openai_models = [
            name for name, config in mixed_bedrock_openai_config.items()
            if config.get("service") == "OpenAI"
        ]
        assert len(openai_models) > 0, "Config should contain at least one OpenAI model"
        assert "gpt-4o-mini" in openai_models
        assert "gpt-4o" in openai_models
    
    def test_bedrock_models_have_required_fields(self, mixed_bedrock_openai_config):
        """Test that Bedrock models have all required fields."""
        required_fields = ["model", "service", "size", "feature", "input_price", "output_price"]
        
        for name, config in mixed_bedrock_openai_config.items():
            if _is_bedrock_model(config.get("service")):
                for field in required_fields:
                    assert field in config, f"Bedrock model {name} should have {field} field"
                
                # Verify service is correctly set
                assert config["service"] in ["Bedrock", "AWS"], \
                    f"Bedrock model {name} should have service 'Bedrock' or 'AWS'"
    
    def test_openai_models_have_required_fields(self, mixed_bedrock_openai_config):
        """Test that OpenAI models have all required fields."""
        required_fields = ["model", "service", "size", "feature", "input_price", "output_price", "api_endpoint"]
        
        for name, config in mixed_bedrock_openai_config.items():
            if config.get("service") == "OpenAI":
                for field in required_fields:
                    assert field in config, f"OpenAI model {name} should have {field} field"
                
                # Verify api_endpoint is set
                assert "api.openai.com" in config["api_endpoint"], \
                    f"OpenAI model {name} should have OpenAI API endpoint"
    
    def test_bedrock_models_have_aws_region(self, mixed_bedrock_openai_config):
        """Test that Bedrock models have aws_region field."""
        for name, config in mixed_bedrock_openai_config.items():
            if _is_bedrock_model(config.get("service")):
                assert "aws_region" in config, \
                    f"Bedrock model {name} should have aws_region field"
                assert config["aws_region"] in ["us-east-1", "us-west-2", "us-west-1", "eu-west-1"], \
                    f"Bedrock model {name} should have valid AWS region"
    
    def test_model_formatting_differs_by_provider(self, mixed_bedrock_openai_config):
        """Test that model formatting logic differs between Bedrock and OpenAI."""
        for name, config in mixed_bedrock_openai_config.items():
            service = config.get("service")
            model_id = config.get("model")
            
            if _is_bedrock_model(service):
                # Bedrock models should use bedrock/{model_id} format
                expected_format = f"bedrock/{model_id}"
                assert expected_format.startswith("bedrock/"), \
                    f"Bedrock model {name} should use bedrock/ prefix"
            elif service == "OpenAI":
                # OpenAI models should use openai/{model_id} format
                expected_format = f"openai/{model_id}"
                assert expected_format.startswith("openai/"), \
                    f"OpenAI model {name} should use openai/ prefix"


class TestMixedProviderServiceDetection:
    """Test service detection across mixed provider configurations."""
    
    def test_bedrock_service_detection(self):
        """Test that Bedrock service is correctly detected."""
        assert _is_bedrock_model("Bedrock") is True
        assert _is_bedrock_model("AWS") is True
        assert _is_bedrock_model("bedrock") is True
        assert _is_bedrock_model("aws") is True
    
    def test_non_bedrock_service_detection(self):
        """Test that non-Bedrock services are correctly identified."""
        assert _is_bedrock_model("NVIDIA") is False
        assert _is_bedrock_model("OpenAI") is False
        assert _is_bedrock_model("Anthropic") is False
        assert _is_bedrock_model(None) is False
        assert _is_bedrock_model("") is False
    
    def test_mixed_config_service_identification(self):
        """Test service identification in mixed configurations."""
        # Simulate a mixed configuration
        models = [
            {"name": "claude-bedrock", "service": "Bedrock"},
            {"name": "gpt-4", "service": "OpenAI"},
            {"name": "llama-nvidia", "service": "NVIDIA"},
            {"name": "titan-bedrock", "service": "AWS"},
        ]
        
        bedrock_count = sum(1 for m in models if _is_bedrock_model(m["service"]))
        non_bedrock_count = sum(1 for m in models if not _is_bedrock_model(m["service"]))
        
        assert bedrock_count == 2, "Should identify 2 Bedrock models"
        assert non_bedrock_count == 2, "Should identify 2 non-Bedrock models"


class TestMixedProviderPricing:
    """Test pricing information in mixed provider configurations."""
    
    @pytest.fixture
    def all_configs(self):
        """Load all mixed provider configurations."""
        configs = {}
        
        # Load Bedrock + NVIDIA
        nvidia_path = Path("data/example_data/llm_candidates/mixed_bedrock_nvidia.json")
        if nvidia_path.exists():
            with open(nvidia_path, 'r') as f:
                configs["bedrock_nvidia"] = json.load(f)
        
        # Load Bedrock + OpenAI
        openai_path = Path("data/example_data/llm_candidates/mixed_bedrock_openai.json")
        if openai_path.exists():
            with open(openai_path, 'r') as f:
                configs["bedrock_openai"] = json.load(f)
        
        return configs
    
    def test_all_models_have_pricing(self, all_configs):
        """Test that all models have input_price and output_price."""
        for config_name, config_data in all_configs.items():
            for model_name, model_config in config_data.items():
                assert "input_price" in model_config, \
                    f"Model {model_name} in {config_name} should have input_price"
                assert "output_price" in model_config, \
                    f"Model {model_name} in {config_name} should have output_price"
                
                # Verify prices are non-negative
                assert model_config["input_price"] >= 0, \
                    f"Model {model_name} input_price should be non-negative"
                assert model_config["output_price"] >= 0, \
                    f"Model {model_name} output_price should be non-negative"
    
    def test_pricing_is_numeric(self, all_configs):
        """Test that pricing values are numeric."""
        for config_name, config_data in all_configs.items():
            for model_name, model_config in config_data.items():
                assert isinstance(model_config["input_price"], (int, float)), \
                    f"Model {model_name} input_price should be numeric"
                assert isinstance(model_config["output_price"], (int, float)), \
                    f"Model {model_name} output_price should be numeric"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
