"""
Unit tests for Bedrock-specific error handling in API calling module.
Tests Requirements: 7.1, 7.2, 7.3, 7.4, 7.5
"""

import sys
import os

# Add parent directory to path to import llmrouter
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from unittest.mock import Mock, patch, MagicMock
from llmrouter.utils.api_calling import call_api


class TestBedrockErrorHandling:
    """Test Bedrock-specific error handling."""
    
    def test_missing_boto3_dependency_error(self):
        """Test error handling for missing boto3 dependency (Requirement 7.1)."""
        # Create a Bedrock request
        request = {
            "api_endpoint": "https://bedrock-runtime.us-east-1.amazonaws.com",
            "query": "What is 2+2?",
            "model_name": "claude-3-sonnet",
            "api_name": "anthropic.claude-3-sonnet-20240229-v1:0",
            "service": "Bedrock"
        }
        
        # Mock completion to raise ImportError with boto3 message
        with patch('llmrouter.utils.api_calling.completion') as mock_completion:
            mock_completion.side_effect = ImportError("No module named 'boto3'")
            
            result = call_api(request, api_keys_env="test-key")
            
            # Verify error message contains installation instructions
            assert 'error' in result
            assert 'boto3' in result['error']
            assert 'pip install boto3' in result['error']
            assert result['token_num'] == 0
            assert result['response_time'] > 0
    
    def test_missing_aws_credentials_error(self):
        """Test error handling for missing AWS credentials (Requirements 1.3, 4.5, 7.3)."""
        request = {
            "api_endpoint": "https://bedrock-runtime.us-east-1.amazonaws.com",
            "query": "What is 2+2?",
            "model_name": "claude-3-sonnet",
            "api_name": "anthropic.claude-3-sonnet-20240229-v1:0",
            "service": "Bedrock"
        }
        
        # Mock completion to raise credential error
        with patch('llmrouter.utils.api_calling.completion') as mock_completion:
            mock_completion.side_effect = Exception("Unable to locate credentials")
            
            result = call_api(request, api_keys_env="test-key")
            
            # Verify error message contains credential setup instructions
            assert 'error' in result
            assert 'credentials' in result['error'].lower()
            assert 'AWS_ACCESS_KEY_ID' in result['error']
            assert 'AWS_SECRET_ACCESS_KEY' in result['error']
            assert '~/.aws/credentials' in result['error']
            assert 'IAM role' in result['error']
            assert result['token_num'] == 0
    
    def test_invalid_model_id_error(self):
        """Test error handling for invalid model IDs (Requirement 7.2)."""
        request = {
            "api_endpoint": "https://bedrock-runtime.us-east-1.amazonaws.com",
            "query": "What is 2+2?",
            "model_name": "invalid-model",
            "api_name": "invalid.model-id",
            "service": "Bedrock"
        }
        
        # Mock completion to raise model not found error
        with patch('llmrouter.utils.api_calling.completion') as mock_completion:
            mock_completion.side_effect = Exception("Model invalid.model-id not found")
            
            result = call_api(request, api_keys_env="test-key")
            
            # Verify error message contains common model IDs
            assert 'error' in result
            assert 'invalid.model-id' in result['error']
            assert 'anthropic.claude-3-sonnet' in result['error']
            assert 'amazon.titan-text-express' in result['error']
            assert 'meta.llama3-70b-instruct' in result['error']
            assert 'models-supported.html' in result['error']
            assert result['token_num'] == 0
    
    def test_region_mismatch_error(self):
        """Test error handling for region mismatches (Requirement 7.5)."""
        request = {
            "api_endpoint": "https://bedrock-runtime.us-east-1.amazonaws.com",
            "query": "What is 2+2?",
            "model_name": "claude-3-sonnet",
            "api_name": "anthropic.claude-3-sonnet-20240229-v1:0",
            "service": "Bedrock",
            "aws_region": "eu-west-1"
        }
        
        # Mock completion to raise region error
        with patch('llmrouter.utils.api_calling.completion') as mock_completion:
            mock_completion.side_effect = Exception("Model not available in region eu-west-1")
            
            result = call_api(request, api_keys_env="test-key")
            
            # Verify error message contains region troubleshooting
            assert 'error' in result
            assert 'eu-west-1' in result['error']
            assert 'different region' in result['error']
            assert 'models-regions.html' in result['error']
            assert 'Bedrock console' in result['error']
            assert result['token_num'] == 0
    
    def test_api_timeout_error(self):
        """Test error handling for API timeouts (Requirement 7.4)."""
        request = {
            "api_endpoint": "https://bedrock-runtime.us-east-1.amazonaws.com",
            "query": "What is 2+2?",
            "model_name": "claude-3-sonnet",
            "api_name": "anthropic.claude-3-sonnet-20240229-v1:0",
            "service": "Bedrock"
        }
        
        # Mock completion to raise timeout error
        with patch('llmrouter.utils.api_calling.completion') as mock_completion:
            mock_completion.side_effect = TimeoutError("Request timed out after 30 seconds")
            
            result = call_api(request, timeout=30, api_keys_env="test-key")
            
            # Verify error message contains timeout troubleshooting
            assert 'error' in result
            assert 'timed out' in result['error'].lower()
            assert '30 seconds' in result['error']
            assert 'Increase timeout' in result['error']
            assert 'network connectivity' in result['error']
            assert result['token_num'] == 0
    
    def test_generic_bedrock_error(self):
        """Test error handling for generic Bedrock errors."""
        request = {
            "api_endpoint": "https://bedrock-runtime.us-east-1.amazonaws.com",
            "query": "What is 2+2?",
            "model_name": "claude-3-sonnet",
            "api_name": "anthropic.claude-3-sonnet-20240229-v1:0",
            "service": "Bedrock"
        }
        
        # Mock completion to raise generic error
        with patch('llmrouter.utils.api_calling.completion') as mock_completion:
            mock_completion.side_effect = Exception("Some other Bedrock error")
            
            result = call_api(request, api_keys_env="test-key")
            
            # Verify error message is prefixed with "Bedrock API Error"
            assert 'error' in result
            assert 'Bedrock API Error' in result['error']
            assert 'Some other Bedrock error' in result['error']
            assert result['token_num'] == 0
    
    def test_non_bedrock_error_not_affected(self):
        """Test that non-Bedrock errors are not affected by Bedrock error handling."""
        request = {
            "api_endpoint": "https://integrate.api.nvidia.com/v1",
            "query": "What is 2+2?",
            "model_name": "qwen2.5-7b-instruct",
            "api_name": "qwen/qwen2.5-7b-instruct",
            "service": "NVIDIA"
        }
        
        # Mock completion to raise generic error
        with patch('llmrouter.utils.api_calling.completion') as mock_completion:
            mock_completion.side_effect = Exception("Some NVIDIA error")
            
            result = call_api(request, api_keys_env="test-key")
            
            # Verify error message is generic, not Bedrock-specific
            assert 'error' in result
            assert 'API Error' in result['error']
            assert 'Bedrock' not in result['error']
            assert 'Some NVIDIA error' in result['error']
            assert result['token_num'] == 0


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
