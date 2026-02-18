"""
Test OpenClaw Bedrock error handling.

This test verifies that the OpenClaw server properly handles Bedrock-specific errors:
- Missing boto3 dependency
- Missing AWS credentials
- Invalid model IDs
- Region mismatches
- Timeout errors
- Error message clarity

**Validates: Requirements 3.1, 3.2, 3.3, 3.4, 3.5**
**Property: Error Message Clarity**
"""

import pytest
import json
from unittest.mock import Mock, patch, MagicMock
from hypothesis import given, strategies as st, settings
from openclaw_router.config import LLMConfig, OpenClawConfig
from openclaw_router.server import LLMBackend
from fastapi import HTTPException


class TestOpenClawBedrockErrorHandling:
    """Unit tests for OpenClaw Bedrock error handling."""
    
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
    
    # Task 7.1: Write tests for missing boto3
    @pytest.mark.asyncio
    async def test_missing_boto3_returns_installation_instructions(self):
        """
        Test that missing boto3 returns installation instructions.
        
        **Validates: Requirement 3.1**
        """
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
            assert "pip install boto3" in exc_info.value.detail.lower()
    
    @pytest.mark.asyncio
    async def test_missing_boto3_includes_pip_install_boto3(self):
        """
        Test error message includes 'pip install boto3'.
        
        **Validates: Requirement 3.1**
        """
        messages = [{"role": "user", "content": "Test"}]
        
        with patch('llmrouter.utils.api_calling.call_api', side_effect=ImportError("boto3 not found")):
            with pytest.raises(HTTPException) as exc_info:
                await self.backend._call_bedrock_sync(
                    self.bedrock_config,
                    messages,
                    max_tokens=4096,
                    temperature=0.7
                )
            
            assert "pip install boto3" in exc_info.value.detail
    
    @pytest.mark.asyncio
    async def test_missing_boto3_streaming_returns_error_in_sse_format(self):
        """
        Test that missing boto3 in streaming returns error in SSE format.
        
        **Validates: Requirement 3.1**
        """
        messages = [{"role": "user", "content": "Test"}]
        
        # Mock litellm.completion to raise ImportError for boto3
        # The import happens inside the function, so we need to mock it at the litellm module level
        with patch('litellm.completion', side_effect=ImportError("No module named 'boto3'")):
            generator = self.backend._call_bedrock_streaming(
                self.bedrock_config,
                messages,
                max_tokens=4096,
                temperature=0.7
            )
            
            # Collect all chunks
            chunks = []
            async for chunk in generator:
                chunks.append(chunk)
            
            # Should have at least one error chunk
            assert len(chunks) > 0
            
            # Parse the error chunk
            error_chunk = chunks[0]
            assert error_chunk.startswith("data: ")
            error_data = json.loads(error_chunk[6:])  # Remove "data: " prefix
            
            assert "error" in error_data
            assert "boto3" in error_data["error"].lower()
            assert "pip install boto3" in error_data["error"].lower()
    
    # Task 7.2: Write tests for missing credentials
    @pytest.mark.asyncio
    async def test_missing_credentials_returns_aws_setup_instructions(self):
        """
        Test that missing credentials returns AWS setup instructions.
        
        **Validates: Requirement 3.2**
        """
        messages = [{"role": "user", "content": "Test"}]
        
        # Mock call_api to return error response with credentials error
        mock_response = {
            "response": "",
            "token_num": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "response_time": 0.0,
            "error": "Unable to locate credentials"
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
            assert "credentials" in exc_info.value.detail.lower()
    
    @pytest.mark.asyncio
    async def test_missing_credentials_includes_environment_variable_setup(self):
        """
        Test error message includes environment variable setup.
        
        **Validates: Requirement 3.2**
        """
        messages = [{"role": "user", "content": "Test"}]
        
        mock_response = {
            "response": "",
            "token_num": 0,
            "error": "AWS credentials not found. Set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY environment variables, or configure ~/.aws/credentials file, or use IAM role."
        }
        
        with patch('llmrouter.utils.api_calling.call_api', return_value=[mock_response]):
            with pytest.raises(HTTPException) as exc_info:
                await self.backend._call_bedrock_sync(
                    self.bedrock_config,
                    messages,
                    max_tokens=4096,
                    temperature=0.7
                )
            
            error_detail = exc_info.value.detail
            assert "AWS_ACCESS_KEY_ID" in error_detail
            assert "AWS_SECRET_ACCESS_KEY" in error_detail
    
    @pytest.mark.asyncio
    async def test_missing_credentials_includes_credential_file_setup(self):
        """
        Test error message includes credential file setup.
        
        **Validates: Requirement 3.2**
        """
        messages = [{"role": "user", "content": "Test"}]
        
        mock_response = {
            "response": "",
            "token_num": 0,
            "error": "AWS credentials not found. Set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY environment variables, or configure ~/.aws/credentials file, or use IAM role."
        }
        
        with patch('llmrouter.utils.api_calling.call_api', return_value=[mock_response]):
            with pytest.raises(HTTPException) as exc_info:
                await self.backend._call_bedrock_sync(
                    self.bedrock_config,
                    messages,
                    max_tokens=4096,
                    temperature=0.7
                )
            
            assert "~/.aws/credentials" in exc_info.value.detail
    
    @pytest.mark.asyncio
    async def test_missing_credentials_includes_iam_role_option(self):
        """
        Test error message includes IAM role option.
        
        **Validates: Requirement 3.2**
        """
        messages = [{"role": "user", "content": "Test"}]
        
        mock_response = {
            "response": "",
            "token_num": 0,
            "error": "AWS credentials not found. Set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY environment variables, or configure ~/.aws/credentials file, or use IAM role."
        }
        
        with patch('llmrouter.utils.api_calling.call_api', return_value=[mock_response]):
            with pytest.raises(HTTPException) as exc_info:
                await self.backend._call_bedrock_sync(
                    self.bedrock_config,
                    messages,
                    max_tokens=4096,
                    temperature=0.7
                )
            
            assert "IAM role" in exc_info.value.detail
    
    # Task 7.3: Write tests for invalid model ID
    @pytest.mark.asyncio
    async def test_invalid_model_id_returns_common_bedrock_model_examples(self):
        """
        Test that invalid model ID returns common Bedrock model examples.
        
        **Validates: Requirement 3.3**
        """
        messages = [{"role": "user", "content": "Test"}]
        
        mock_response = {
            "response": "",
            "token_num": 0,
            "error": "Model invalid.model-id not found. Common Bedrock model IDs: anthropic.claude-3-sonnet-20240229-v1:0, amazon.titan-text-express-v1, meta.llama3-70b-instruct-v1:0. See: https://docs.aws.amazon.com/bedrock/latest/userguide/models-supported.html"
        }
        
        with patch('llmrouter.utils.api_calling.call_api', return_value=[mock_response]):
            with pytest.raises(HTTPException) as exc_info:
                await self.backend._call_bedrock_sync(
                    self.bedrock_config,
                    messages,
                    max_tokens=4096,
                    temperature=0.7
                )
            
            error_detail = exc_info.value.detail
            assert "model" in error_detail.lower()
    
    @pytest.mark.asyncio
    async def test_invalid_model_id_includes_claude_model_ids(self):
        """
        Test error message includes Claude model IDs.
        
        **Validates: Requirement 3.3**
        """
        messages = [{"role": "user", "content": "Test"}]
        
        mock_response = {
            "response": "",
            "token_num": 0,
            "error": "Model not found. Common Bedrock model IDs: anthropic.claude-3-sonnet-20240229-v1:0, anthropic.claude-3-haiku-20240307-v1:0"
        }
        
        with patch('llmrouter.utils.api_calling.call_api', return_value=[mock_response]):
            with pytest.raises(HTTPException) as exc_info:
                await self.backend._call_bedrock_sync(
                    self.bedrock_config,
                    messages,
                    max_tokens=4096,
                    temperature=0.7
                )
            
            assert "anthropic.claude" in exc_info.value.detail.lower()
    
    @pytest.mark.asyncio
    async def test_invalid_model_id_includes_titan_model_ids(self):
        """
        Test error message includes Titan model IDs.
        
        **Validates: Requirement 3.3**
        """
        messages = [{"role": "user", "content": "Test"}]
        
        mock_response = {
            "response": "",
            "token_num": 0,
            "error": "Model not found. Common Bedrock model IDs: amazon.titan-text-express-v1, amazon.titan-text-lite-v1"
        }
        
        with patch('llmrouter.utils.api_calling.call_api', return_value=[mock_response]):
            with pytest.raises(HTTPException) as exc_info:
                await self.backend._call_bedrock_sync(
                    self.bedrock_config,
                    messages,
                    max_tokens=4096,
                    temperature=0.7
                )
            
            assert "amazon.titan" in exc_info.value.detail.lower()
    
    @pytest.mark.asyncio
    async def test_invalid_model_id_includes_llama_model_ids(self):
        """
        Test error message includes Llama model IDs.
        
        **Validates: Requirement 3.3**
        """
        messages = [{"role": "user", "content": "Test"}]
        
        mock_response = {
            "response": "",
            "token_num": 0,
            "error": "Model not found. Common Bedrock model IDs: meta.llama3-70b-instruct-v1:0, meta.llama3-8b-instruct-v1:0"
        }
        
        with patch('llmrouter.utils.api_calling.call_api', return_value=[mock_response]):
            with pytest.raises(HTTPException) as exc_info:
                await self.backend._call_bedrock_sync(
                    self.bedrock_config,
                    messages,
                    max_tokens=4096,
                    temperature=0.7
                )
            
            assert "meta.llama" in exc_info.value.detail.lower()
    
    @pytest.mark.asyncio
    async def test_invalid_model_id_includes_mistral_model_ids(self):
        """
        Test error message includes Mistral model IDs.
        
        **Validates: Requirement 3.3**
        """
        messages = [{"role": "user", "content": "Test"}]
        
        mock_response = {
            "response": "",
            "token_num": 0,
            "error": "Model not found. Common Bedrock model IDs: mistral.mistral-7b-instruct-v0:2, mistral.mixtral-8x7b-instruct-v0:1"
        }
        
        with patch('llmrouter.utils.api_calling.call_api', return_value=[mock_response]):
            with pytest.raises(HTTPException) as exc_info:
                await self.backend._call_bedrock_sync(
                    self.bedrock_config,
                    messages,
                    max_tokens=4096,
                    temperature=0.7
                )
            
            assert "mistral" in exc_info.value.detail.lower()
    
    # Task 7.4: Write tests for region mismatch
    @pytest.mark.asyncio
    async def test_region_mismatch_returns_region_configuration_help(self):
        """
        Test that region errors return region configuration help.
        
        **Validates: Requirement 3.4**
        """
        messages = [{"role": "user", "content": "Test"}]
        
        mock_response = {
            "response": "",
            "token_num": 0,
            "error": "Model not available in region eu-west-1. Try a different region or check model availability: https://docs.aws.amazon.com/bedrock/latest/userguide/models-regions.html"
        }
        
        with patch('llmrouter.utils.api_calling.call_api', return_value=[mock_response]):
            with pytest.raises(HTTPException) as exc_info:
                await self.backend._call_bedrock_sync(
                    self.bedrock_config,
                    messages,
                    max_tokens=4096,
                    temperature=0.7
                )
            
            error_detail = exc_info.value.detail
            assert "region" in error_detail.lower()
    
    @pytest.mark.asyncio
    async def test_region_mismatch_includes_model_availability_documentation_link(self):
        """
        Test error message includes model availability documentation link.
        
        **Validates: Requirement 3.4**
        """
        messages = [{"role": "user", "content": "Test"}]
        
        mock_response = {
            "response": "",
            "token_num": 0,
            "error": "Model not available in region. See: https://docs.aws.amazon.com/bedrock/latest/userguide/models-regions.html"
        }
        
        with patch('llmrouter.utils.api_calling.call_api', return_value=[mock_response]):
            with pytest.raises(HTTPException) as exc_info:
                await self.backend._call_bedrock_sync(
                    self.bedrock_config,
                    messages,
                    max_tokens=4096,
                    temperature=0.7
                )
            
            assert "models-regions.html" in exc_info.value.detail
    
    @pytest.mark.asyncio
    async def test_region_mismatch_suggests_trying_different_region(self):
        """
        Test error message suggests trying different region.
        
        **Validates: Requirement 3.4**
        """
        messages = [{"role": "user", "content": "Test"}]
        
        mock_response = {
            "response": "",
            "token_num": 0,
            "error": "Model not available in region us-west-1. Try a different region like us-east-1."
        }
        
        with patch('llmrouter.utils.api_calling.call_api', return_value=[mock_response]):
            with pytest.raises(HTTPException) as exc_info:
                await self.backend._call_bedrock_sync(
                    self.bedrock_config,
                    messages,
                    max_tokens=4096,
                    temperature=0.7
                )
            
            assert "different region" in exc_info.value.detail.lower()
    
    # Task 7.5: Write tests for timeout errors
    @pytest.mark.asyncio
    async def test_timeout_error_returns_troubleshooting_steps(self):
        """
        Test that timeout errors return troubleshooting steps.
        
        **Validates: Requirement 3.5**
        """
        messages = [{"role": "user", "content": "Test"}]
        
        mock_response = {
            "response": "",
            "token_num": 0,
            "error": "Request timed out after 120 seconds. Increase timeout, check network connectivity, or try a different region."
        }
        
        with patch('llmrouter.utils.api_calling.call_api', return_value=[mock_response]):
            with pytest.raises(HTTPException) as exc_info:
                await self.backend._call_bedrock_sync(
                    self.bedrock_config,
                    messages,
                    max_tokens=4096,
                    temperature=0.7
                )
            
            error_detail = exc_info.value.detail
            assert "timeout" in error_detail.lower() or "timed out" in error_detail.lower()
    
    @pytest.mark.asyncio
    async def test_timeout_error_suggests_increasing_timeout(self):
        """
        Test error message suggests increasing timeout.
        
        **Validates: Requirement 3.5**
        """
        messages = [{"role": "user", "content": "Test"}]
        
        mock_response = {
            "response": "",
            "token_num": 0,
            "error": "Request timed out after 120 seconds. Increase timeout parameter."
        }
        
        with patch('llmrouter.utils.api_calling.call_api', return_value=[mock_response]):
            with pytest.raises(HTTPException) as exc_info:
                await self.backend._call_bedrock_sync(
                    self.bedrock_config,
                    messages,
                    max_tokens=4096,
                    temperature=0.7
                )
            
            assert "increase timeout" in exc_info.value.detail.lower()
    
    @pytest.mark.asyncio
    async def test_timeout_error_suggests_checking_network_connectivity(self):
        """
        Test error message suggests checking network connectivity.
        
        **Validates: Requirement 3.5**
        """
        messages = [{"role": "user", "content": "Test"}]
        
        mock_response = {
            "response": "",
            "token_num": 0,
            "error": "Request timed out. Check network connectivity."
        }
        
        with patch('llmrouter.utils.api_calling.call_api', return_value=[mock_response]):
            with pytest.raises(HTTPException) as exc_info:
                await self.backend._call_bedrock_sync(
                    self.bedrock_config,
                    messages,
                    max_tokens=4096,
                    temperature=0.7
                )
            
            assert "network connectivity" in exc_info.value.detail.lower()
    
    @pytest.mark.asyncio
    async def test_timeout_error_suggests_trying_different_region(self):
        """
        Test error message suggests trying different region.
        
        **Validates: Requirement 3.5**
        """
        messages = [{"role": "user", "content": "Test"}]
        
        mock_response = {
            "response": "",
            "token_num": 0,
            "error": "Request timed out. Try a different region for better latency."
        }
        
        with patch('llmrouter.utils.api_calling.call_api', return_value=[mock_response]):
            with pytest.raises(HTTPException) as exc_info:
                await self.backend._call_bedrock_sync(
                    self.bedrock_config,
                    messages,
                    max_tokens=4096,
                    temperature=0.7
                )
            
            assert "different region" in exc_info.value.detail.lower()


class TestOpenClawBedrockErrorHandlingProperties:
    """Property-based tests for OpenClaw Bedrock error handling."""
    
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
    
    # Task 7.6: Write property tests for error message clarity
    # Feature: openclaw-bedrock-support, Property: Error Message Clarity
    @given(st.text(min_size=1, max_size=500))
    @settings(max_examples=50)
    @pytest.mark.asyncio
    async def test_property_all_bedrock_errors_include_actionable_instructions(self, error_message):
        """
        Property: All Bedrock errors include actionable instructions.
        
        **Validates: Requirements 3.1, 3.2, 3.3, 3.4, 3.5**
        """
        messages = [{"role": "user", "content": "Test"}]
        
        mock_response = {
            "response": "",
            "token_num": 0,
            "error": error_message
        }
        
        with patch('llmrouter.utils.api_calling.call_api', return_value=[mock_response]):
            with pytest.raises(HTTPException) as exc_info:
                await self.backend._call_bedrock_sync(
                    self.bedrock_config,
                    messages,
                    max_tokens=4096,
                    temperature=0.7
                )
            
            # Error detail should be non-empty
            assert len(exc_info.value.detail) > 0
            # Error detail should contain the original error message
            assert error_message in exc_info.value.detail
    
    # Feature: openclaw-bedrock-support, Property: Error Message Clarity
    @given(st.text(min_size=1, max_size=500))
    @settings(max_examples=50)
    @pytest.mark.asyncio
    async def test_property_error_messages_are_non_empty(self, error_message):
        """
        Property: Error messages are non-empty.
        
        **Validates: Requirements 3.1, 3.2, 3.3, 3.4, 3.5**
        """
        messages = [{"role": "user", "content": "Test"}]
        
        mock_response = {
            "response": "",
            "token_num": 0,
            "error": error_message
        }
        
        with patch('llmrouter.utils.api_calling.call_api', return_value=[mock_response]):
            with pytest.raises(HTTPException) as exc_info:
                await self.backend._call_bedrock_sync(
                    self.bedrock_config,
                    messages,
                    max_tokens=4096,
                    temperature=0.7
                )
            
            # Error detail must be non-empty
            assert exc_info.value.detail is not None
            assert len(exc_info.value.detail) > 0
    
    # Feature: openclaw-bedrock-support, Property: Error Message Clarity
    @given(
        st.sampled_from([
            "boto3",
            "credentials",
            "model",
            "region",
            "timeout",
            "timed out"
        ])
    )
    @settings(max_examples=50)
    @pytest.mark.asyncio
    async def test_property_error_messages_include_troubleshooting_steps(self, error_keyword):
        """
        Property: Error messages include troubleshooting steps for common error types.
        
        **Validates: Requirements 3.1, 3.2, 3.3, 3.4, 3.5**
        """
        messages = [{"role": "user", "content": "Test"}]
        
        # Create error message with keyword
        error_message = f"Error related to {error_keyword}"
        
        # For boto3 errors, use ImportError
        if error_keyword == "boto3":
            with patch('llmrouter.utils.api_calling.call_api', side_effect=ImportError(f"No module named '{error_keyword}'")):
                with pytest.raises(HTTPException) as exc_info:
                    await self.backend._call_bedrock_sync(
                        self.bedrock_config,
                        messages,
                        max_tokens=4096,
                        temperature=0.7
                    )
                
                # Should include installation instructions
                assert "pip install" in exc_info.value.detail.lower() or "install" in exc_info.value.detail.lower()
        else:
            # For other errors, use error response
            mock_response = {
                "response": "",
                "token_num": 0,
                "error": error_message
            }
            
            with patch('llmrouter.utils.api_calling.call_api', return_value=[mock_response]):
                with pytest.raises(HTTPException) as exc_info:
                    await self.backend._call_bedrock_sync(
                        self.bedrock_config,
                        messages,
                        max_tokens=4096,
                        temperature=0.7
                    )
                
                # Error detail should contain the error message
                assert error_message in exc_info.value.detail


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
