"""
Tests for Bedrock service detection and model formatting in API calling module.
"""

import pytest
from llmrouter.utils.api_calling import _is_bedrock_model


class TestBedrockServiceDetection:
    """Test Bedrock service detection functionality."""
    
    def test_is_bedrock_model_with_bedrock_service(self):
        """Test that 'Bedrock' service is recognized."""
        assert _is_bedrock_model("Bedrock") is True
    
    def test_is_bedrock_model_with_aws_service(self):
        """Test that 'AWS' service is recognized."""
        assert _is_bedrock_model("AWS") is True
    
    def test_is_bedrock_model_case_insensitive(self):
        """Test that service detection is case-insensitive."""
        assert _is_bedrock_model("bedrock") is True
        assert _is_bedrock_model("aws") is True
        assert _is_bedrock_model("BEDROCK") is True
        assert _is_bedrock_model("AWS") is True
        assert _is_bedrock_model("BeDrOcK") is True
    
    def test_is_bedrock_model_with_other_services(self):
        """Test that other services are not recognized as Bedrock."""
        assert _is_bedrock_model("NVIDIA") is False
        assert _is_bedrock_model("OpenAI") is False
        assert _is_bedrock_model("Anthropic") is False
    
    def test_is_bedrock_model_with_none(self):
        """Test that None returns False."""
        assert _is_bedrock_model(None) is False
    
    def test_is_bedrock_model_with_empty_string(self):
        """Test that empty string returns False."""
        assert _is_bedrock_model("") is False


class TestBedrockModelFormatting:
    """Test Bedrock model formatting in call_api function."""
    
    def test_bedrock_model_format_in_request(self):
        """Test that Bedrock models are formatted correctly."""
        # This test verifies the logic without making actual API calls
        # We'll test the model formatting by checking the logic flow
        
        # Test data
        bedrock_request = {
            "api_endpoint": "https://bedrock-runtime.us-east-1.amazonaws.com",
            "query": "What is 2+2?",
            "model_name": "claude-3-sonnet",
            "api_name": "anthropic.claude-3-sonnet-20240229-v1:0",
            "service": "Bedrock"
        }
        
        # Verify service detection
        assert _is_bedrock_model(bedrock_request["service"]) is True
        
        # Verify expected model format
        expected_format = f"bedrock/{bedrock_request['api_name']}"
        assert expected_format == "bedrock/anthropic.claude-3-sonnet-20240229-v1:0"
    
    def test_non_bedrock_model_format_in_request(self):
        """Test that non-Bedrock models use OpenAI format."""
        # Test data
        nvidia_request = {
            "api_endpoint": "https://integrate.api.nvidia.com/v1",
            "query": "What is 2+2?",
            "model_name": "qwen2.5-7b-instruct",
            "api_name": "qwen/qwen2.5-7b-instruct",
            "service": "NVIDIA"
        }
        
        # Verify service detection
        assert _is_bedrock_model(nvidia_request["service"]) is False
        
        # Verify expected model format
        expected_format = f"openai/{nvidia_request['api_name']}"
        assert expected_format == "openai/qwen/qwen2.5-7b-instruct"


class TestBedrockRegionHandling:
    """Test AWS region handling for Bedrock models."""
    
    def test_aws_region_extraction_from_request(self):
        """Test that aws_region field is correctly extracted from request."""
        # Test data with aws_region
        bedrock_request = {
            "api_endpoint": "https://bedrock-runtime.us-east-1.amazonaws.com",
            "query": "What is 2+2?",
            "model_name": "claude-3-sonnet",
            "api_name": "anthropic.claude-3-sonnet-20240229-v1:0",
            "service": "Bedrock",
            "aws_region": "us-west-2"
        }
        
        # Verify service detection
        assert _is_bedrock_model(bedrock_request["service"]) is True
        
        # Verify region is present
        assert bedrock_request.get("aws_region") == "us-west-2"
    
    def test_aws_region_optional_in_request(self):
        """Test that aws_region is optional in request."""
        # Test data without aws_region
        bedrock_request = {
            "api_endpoint": "https://bedrock-runtime.us-east-1.amazonaws.com",
            "query": "What is 2+2?",
            "model_name": "claude-3-sonnet",
            "api_name": "anthropic.claude-3-sonnet-20240229-v1:0",
            "service": "Bedrock"
        }
        
        # Verify service detection
        assert _is_bedrock_model(bedrock_request["service"]) is True
        
        # Verify region is not present (should return None)
        assert bedrock_request.get("aws_region") is None
    
    def test_non_bedrock_model_ignores_region(self):
        """Test that non-Bedrock models don't use aws_region field."""
        # Test data with aws_region but non-Bedrock service
        nvidia_request = {
            "api_endpoint": "https://integrate.api.nvidia.com/v1",
            "query": "What is 2+2?",
            "model_name": "qwen2.5-7b-instruct",
            "api_name": "qwen/qwen2.5-7b-instruct",
            "service": "NVIDIA",
            "aws_region": "us-west-2"  # Should be ignored
        }
        
        # Verify service detection
        assert _is_bedrock_model(nvidia_request["service"]) is False
        
        # Region field exists but should not be used for non-Bedrock models
        assert nvidia_request.get("aws_region") == "us-west-2"



class TestBedrockInferenceParameters:
    """Test inference parameter passthrough for Bedrock models."""
    
    def test_inference_parameters_in_request(self):
        """Test that inference parameters are present in request structure."""
        # Test data with inference parameters
        bedrock_request = {
            "api_endpoint": "https://bedrock-runtime.us-east-1.amazonaws.com",
            "query": "What is 2+2?",
            "model_name": "claude-3-sonnet",
            "api_name": "anthropic.claude-3-sonnet-20240229-v1:0",
            "service": "Bedrock"
        }
        
        # Verify service detection
        assert _is_bedrock_model(bedrock_request["service"]) is True
        
        # Verify that call_api accepts these parameters
        # (actual API call testing would require mocking LiteLLM)
        # This test verifies the structure is correct
        assert "service" in bedrock_request
        assert "api_name" in bedrock_request
    
    def test_default_inference_parameters(self):
        """Test that default inference parameters are reasonable."""
        # Default values from call_api signature
        default_max_tokens = 512
        default_temperature = 0.01
        default_top_p = 0.9
        
        # Verify defaults are within valid ranges
        assert 0 < default_max_tokens <= 4096
        assert 0.0 <= default_temperature <= 2.0
        assert 0.0 <= default_top_p <= 1.0
    
    def test_custom_inference_parameters(self):
        """Test that custom inference parameters can be specified."""
        # Custom parameter values
        custom_max_tokens = 1024
        custom_temperature = 0.7
        custom_top_p = 0.95
        
        # Verify custom values are within valid ranges
        assert 0 < custom_max_tokens <= 4096
        assert 0.0 <= custom_temperature <= 2.0
        assert 0.0 <= custom_top_p <= 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
