"""
Tests for Bedrock token usage tracking and response time measurement.

Task 8.2: Verify token usage tracking for Bedrock
- Test that token counts are extracted from response metadata
- Ensure response_time is recorded
- Requirements: 3.5, 3.6
"""

import pytest
import time
from unittest.mock import Mock, patch
from llmrouter.utils.api_calling import call_api


class TestBedrockTokenTracking:
    """Test token usage tracking for Bedrock models."""
    
    @patch('llmrouter.utils.api_calling.completion')
    def test_bedrock_extracts_token_counts_from_metadata(self, mock_completion):
        """Test that token counts are extracted from Bedrock response metadata.
        
        Validates: Requirements 3.5
        """
        # Mock LiteLLM response with usage metadata
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message = Mock()
        mock_response.choices[0].message.content = "Test response"
        mock_response.usage = Mock()
        mock_response.usage.__dict__ = {
            "total_tokens": 150,
            "prompt_tokens": 50,
            "completion_tokens": 100
        }
        mock_completion.return_value = mock_response
        
        # Test request
        request = {
            "api_endpoint": "https://bedrock-runtime.us-east-1.amazonaws.com",
            "query": "What is the capital of France?",
            "model_name": "claude-3-sonnet",
            "api_name": "anthropic.claude-3-sonnet-20240229-v1:0",
            "service": "Bedrock"
        }
        
        # Call API
        with patch.dict('os.environ', {'API_KEYS': '{"NVIDIA": "dummy-key"}'}):
            result = call_api(request)
        
        # Verify token counts are extracted correctly
        assert result['token_num'] == 150, "Total token count should match metadata"
        assert result['prompt_tokens'] == 50, "Prompt token count should match metadata"
        assert result['completion_tokens'] == 100, "Completion token count should match metadata"
        
        # Verify token counts are integers
        assert isinstance(result['token_num'], int)
        assert isinstance(result['prompt_tokens'], int)
        assert isinstance(result['completion_tokens'], int)
        
        # Verify token counts are non-negative
        assert result['token_num'] >= 0
        assert result['prompt_tokens'] >= 0
        assert result['completion_tokens'] >= 0
        
        # Verify total equals sum of prompt and completion
        assert result['token_num'] == result['prompt_tokens'] + result['completion_tokens']
    
    @patch('llmrouter.utils.api_calling.completion')
    def test_bedrock_records_response_time(self, mock_completion):
        """Test that response_time is recorded for Bedrock API calls.
        
        Validates: Requirements 3.6
        """
        # Mock LiteLLM response with simulated delay
        def mock_completion_with_delay(**kwargs):
            time.sleep(0.1)  # Simulate 100ms API call
            mock_response = Mock()
            mock_response.choices = [Mock()]
            mock_response.choices[0].message = Mock()
            mock_response.choices[0].message.content = "Response after delay"
            mock_response.usage = Mock()
            mock_response.usage.__dict__ = {
                "total_tokens": 50,
                "prompt_tokens": 20,
                "completion_tokens": 30
            }
            return mock_response
        
        mock_completion.side_effect = mock_completion_with_delay
        
        # Test request
        request = {
            "api_endpoint": "https://bedrock-runtime.us-east-1.amazonaws.com",
            "query": "Test query",
            "model_name": "claude-3-sonnet",
            "api_name": "anthropic.claude-3-sonnet-20240229-v1:0",
            "service": "Bedrock"
        }
        
        # Call API and measure time
        start = time.time()
        with patch.dict('os.environ', {'API_KEYS': '{"NVIDIA": "dummy-key"}'}):
            result = call_api(request)
        end = time.time()
        actual_duration = end - start
        
        # Verify response_time is recorded
        assert 'response_time' in result, "response_time field should be present"
        assert isinstance(result['response_time'], float), "response_time should be a float"
        assert result['response_time'] > 0, "response_time should be positive"
        
        # Verify response_time is reasonable (should be close to actual duration)
        assert result['response_time'] >= 0.1, "response_time should include the simulated delay"
        assert result['response_time'] <= actual_duration + 0.1, "response_time should not exceed actual duration significantly"
    
    @patch('llmrouter.utils.api_calling.completion')
    def test_bedrock_token_tracking_with_different_models(self, mock_completion):
        """Test token tracking works for different Bedrock model families.
        
        Validates: Requirements 3.5
        """
        # Test data for different models
        test_cases = [
            {
                "model_name": "claude-3-sonnet",
                "api_name": "anthropic.claude-3-sonnet-20240229-v1:0",
                "tokens": {"total": 200, "prompt": 80, "completion": 120}
            },
            {
                "model_name": "claude-3-haiku",
                "api_name": "anthropic.claude-3-haiku-20240307-v1:0",
                "tokens": {"total": 100, "prompt": 40, "completion": 60}
            },
            {
                "model_name": "titan-text-express",
                "api_name": "amazon.titan-text-express-v1",
                "tokens": {"total": 75, "prompt": 25, "completion": 50}
            }
        ]
        
        for test_case in test_cases:
            # Mock response for this model
            mock_response = Mock()
            mock_response.choices = [Mock()]
            mock_response.choices[0].message = Mock()
            mock_response.choices[0].message.content = f"Response from {test_case['model_name']}"
            mock_response.usage = Mock()
            mock_response.usage.__dict__ = {
                "total_tokens": test_case['tokens']['total'],
                "prompt_tokens": test_case['tokens']['prompt'],
                "completion_tokens": test_case['tokens']['completion']
            }
            mock_completion.return_value = mock_response
            
            # Test request
            request = {
                "api_endpoint": "https://bedrock-runtime.us-east-1.amazonaws.com",
                "query": "Test query",
                "model_name": test_case['model_name'],
                "api_name": test_case['api_name'],
                "service": "Bedrock"
            }
            
            # Call API
            with patch.dict('os.environ', {'API_KEYS': '{"NVIDIA": "dummy-key"}'}):
                result = call_api(request)
            
            # Verify token counts for this model
            assert result['token_num'] == test_case['tokens']['total'], \
                f"Token count mismatch for {test_case['model_name']}"
            assert result['prompt_tokens'] == test_case['tokens']['prompt'], \
                f"Prompt token count mismatch for {test_case['model_name']}"
            assert result['completion_tokens'] == test_case['tokens']['completion'], \
                f"Completion token count mismatch for {test_case['model_name']}"
    
    @patch('llmrouter.utils.api_calling.completion')
    def test_bedrock_token_tracking_with_zero_completion_tokens(self, mock_completion):
        """Test token tracking when model returns no completion tokens.
        
        Validates: Requirements 3.5
        """
        # Mock response with zero completion tokens
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message = Mock()
        mock_response.choices[0].message.content = ""
        mock_response.usage = Mock()
        mock_response.usage.__dict__ = {
            "total_tokens": 30,
            "prompt_tokens": 30,
            "completion_tokens": 0
        }
        mock_completion.return_value = mock_response
        
        # Test request
        request = {
            "api_endpoint": "https://bedrock-runtime.us-east-1.amazonaws.com",
            "query": "Test query",
            "model_name": "claude-3-sonnet",
            "api_name": "anthropic.claude-3-sonnet-20240229-v1:0",
            "service": "Bedrock"
        }
        
        # Call API
        with patch.dict('os.environ', {'API_KEYS': '{"NVIDIA": "dummy-key"}'}):
            result = call_api(request)
        
        # Verify zero completion tokens are handled correctly
        assert result['token_num'] == 30
        assert result['prompt_tokens'] == 30
        assert result['completion_tokens'] == 0
        assert result['response'] == ""
    
    @patch('llmrouter.utils.api_calling.completion')
    def test_bedrock_token_tracking_with_large_token_counts(self, mock_completion):
        """Test token tracking with large token counts.
        
        Validates: Requirements 3.5
        """
        # Mock response with large token counts
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message = Mock()
        mock_response.choices[0].message.content = "Long response " * 1000
        mock_response.usage = Mock()
        mock_response.usage.__dict__ = {
            "total_tokens": 10000,
            "prompt_tokens": 2000,
            "completion_tokens": 8000
        }
        mock_completion.return_value = mock_response
        
        # Test request
        request = {
            "api_endpoint": "https://bedrock-runtime.us-east-1.amazonaws.com",
            "query": "Write a very long essay",
            "model_name": "claude-3-sonnet",
            "api_name": "anthropic.claude-3-sonnet-20240229-v1:0",
            "service": "Bedrock"
        }
        
        # Call API
        with patch.dict('os.environ', {'API_KEYS': '{"NVIDIA": "dummy-key"}'}):
            result = call_api(request)
        
        # Verify large token counts are handled correctly
        assert result['token_num'] == 10000
        assert result['prompt_tokens'] == 2000
        assert result['completion_tokens'] == 8000
    
    @patch('llmrouter.utils.api_calling.completion')
    def test_bedrock_batch_token_tracking(self, mock_completion):
        """Test token tracking for batch Bedrock requests.
        
        Validates: Requirements 3.5, 3.6
        """
        # Mock responses for batch with different token counts
        call_count = [0]
        
        def mock_completion_side_effect(**kwargs):
            call_count[0] += 1
            mock_response = Mock()
            mock_response.choices = [Mock()]
            mock_response.choices[0].message = Mock()
            mock_response.choices[0].message.content = f"Response {call_count[0]}"
            mock_response.usage = Mock()
            # Different token counts for each call
            mock_response.usage.__dict__ = {
                "total_tokens": 100 * call_count[0],
                "prompt_tokens": 30 * call_count[0],
                "completion_tokens": 70 * call_count[0]
            }
            return mock_response
        
        mock_completion.side_effect = mock_completion_side_effect
        
        # Test batch requests
        requests = [
            {
                "api_endpoint": "https://bedrock-runtime.us-east-1.amazonaws.com",
                "query": "First query",
                "model_name": "claude-3-sonnet",
                "api_name": "anthropic.claude-3-sonnet-20240229-v1:0",
                "service": "Bedrock"
            },
            {
                "api_endpoint": "https://bedrock-runtime.us-east-1.amazonaws.com",
                "query": "Second query",
                "model_name": "claude-3-haiku",
                "api_name": "anthropic.claude-3-haiku-20240307-v1:0",
                "service": "Bedrock"
            },
            {
                "api_endpoint": "https://bedrock-runtime.us-east-1.amazonaws.com",
                "query": "Third query",
                "model_name": "titan-text-express",
                "api_name": "amazon.titan-text-express-v1",
                "service": "Bedrock"
            }
        ]
        
        # Call API with batch
        with patch.dict('os.environ', {'API_KEYS': '{"NVIDIA": "dummy-key"}'}):
            results = call_api(requests)
        
        # Verify token tracking for each result
        assert len(results) == 3
        
        for i, result in enumerate(results, 1):
            assert result['token_num'] == 100 * i, f"Token count mismatch for request {i}"
            assert result['prompt_tokens'] == 30 * i, f"Prompt token count mismatch for request {i}"
            assert result['completion_tokens'] == 70 * i, f"Completion token count mismatch for request {i}"
            assert result['response_time'] > 0, f"Response time not recorded for request {i}"


class TestBedrockTokenTrackingFallback:
    """Test fallback token counting when metadata is missing."""
    
    @patch('llmrouter.utils.api_calling.completion')
    @patch('llmrouter.utils.api_calling._count_tokens')
    def test_bedrock_fallback_token_counting_when_metadata_missing(self, mock_count_tokens, mock_completion):
        """Test fallback token counting when usage metadata is not available.
        
        Validates: Requirements 3.5
        """
        # Mock _count_tokens to return predictable values
        def count_tokens_side_effect(text):
            if text is None:
                return 0
            return len(text.split())  # Simple word count
        
        mock_count_tokens.side_effect = count_tokens_side_effect
        
        # Mock response without usage metadata
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message = Mock()
        mock_response.choices[0].message.content = "This is a test response"
        mock_response.usage = None  # No usage metadata
        mock_completion.return_value = mock_response
        
        # Test request
        request = {
            "api_endpoint": "https://bedrock-runtime.us-east-1.amazonaws.com",
            "query": "What is the weather today",
            "model_name": "claude-3-sonnet",
            "api_name": "anthropic.claude-3-sonnet-20240229-v1:0",
            "service": "Bedrock"
        }
        
        # Call API
        with patch.dict('os.environ', {'API_KEYS': '{"NVIDIA": "dummy-key"}'}):
            result = call_api(request)
        
        # Verify fallback token counting is used
        assert result['token_num'] > 0, "Token count should be estimated"
        assert result['prompt_tokens'] > 0, "Prompt tokens should be estimated"
        assert result['completion_tokens'] > 0, "Completion tokens should be estimated"
        
        # Verify _count_tokens was called for fallback
        assert mock_count_tokens.call_count >= 2, "Fallback token counting should be used"
    
    @patch('llmrouter.utils.api_calling.completion')
    def test_bedrock_token_tracking_with_partial_metadata(self, mock_completion):
        """Test token tracking when usage metadata is incomplete.
        
        Validates: Requirements 3.5
        """
        # Mock response with partial usage metadata
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message = Mock()
        mock_response.choices[0].message.content = "Response text"
        mock_response.usage = Mock()
        mock_response.usage.__dict__ = {
            "total_tokens": 50
            # Missing prompt_tokens and completion_tokens
        }
        mock_completion.return_value = mock_response
        
        # Test request
        request = {
            "api_endpoint": "https://bedrock-runtime.us-east-1.amazonaws.com",
            "query": "Test query",
            "model_name": "claude-3-sonnet",
            "api_name": "anthropic.claude-3-sonnet-20240229-v1:0",
            "service": "Bedrock"
        }
        
        # Call API
        with patch.dict('os.environ', {'API_KEYS': '{"NVIDIA": "dummy-key"}'}):
            result = call_api(request)
        
        # Verify token counts are present (using available metadata or fallback)
        assert 'token_num' in result
        assert 'prompt_tokens' in result
        assert 'completion_tokens' in result
        assert result['token_num'] >= 0
        assert result['prompt_tokens'] >= 0
        assert result['completion_tokens'] >= 0


class TestBedrockResponseTimeAccuracy:
    """Test response time measurement accuracy."""
    
    @patch('llmrouter.utils.api_calling.completion')
    def test_bedrock_response_time_includes_api_call_duration(self, mock_completion):
        """Test that response_time accurately measures API call duration.
        
        Validates: Requirements 3.6
        """
        # Mock completion with known delay
        def mock_completion_with_delay(**kwargs):
            time.sleep(0.2)  # 200ms delay
            mock_response = Mock()
            mock_response.choices = [Mock()]
            mock_response.choices[0].message = Mock()
            mock_response.choices[0].message.content = "Response"
            mock_response.usage = Mock()
            mock_response.usage.__dict__ = {
                "total_tokens": 50,
                "prompt_tokens": 20,
                "completion_tokens": 30
            }
            return mock_response
        
        mock_completion.side_effect = mock_completion_with_delay
        
        # Test request
        request = {
            "api_endpoint": "https://bedrock-runtime.us-east-1.amazonaws.com",
            "query": "Test query",
            "model_name": "claude-3-sonnet",
            "api_name": "anthropic.claude-3-sonnet-20240229-v1:0",
            "service": "Bedrock"
        }
        
        # Call API
        with patch.dict('os.environ', {'API_KEYS': '{"NVIDIA": "dummy-key"}'}):
            result = call_api(request)
        
        # Verify response_time is at least the delay duration
        assert result['response_time'] >= 0.2, "response_time should include the API call delay"
        # Allow some overhead but not too much
        assert result['response_time'] < 0.5, "response_time should not be excessively long"
    
    @patch('llmrouter.utils.api_calling.completion')
    def test_bedrock_response_time_for_fast_responses(self, mock_completion):
        """Test response_time measurement for fast API responses.
        
        Validates: Requirements 3.6
        """
        # Mock fast response (no artificial delay)
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message = Mock()
        mock_response.choices[0].message.content = "Quick response"
        mock_response.usage = Mock()
        mock_response.usage.__dict__ = {
            "total_tokens": 20,
            "prompt_tokens": 10,
            "completion_tokens": 10
        }
        mock_completion.return_value = mock_response
        
        # Test request
        request = {
            "api_endpoint": "https://bedrock-runtime.us-east-1.amazonaws.com",
            "query": "Quick query",
            "model_name": "claude-3-haiku",
            "api_name": "anthropic.claude-3-haiku-20240307-v1:0",
            "service": "Bedrock"
        }
        
        # Call API
        with patch.dict('os.environ', {'API_KEYS': '{"NVIDIA": "dummy-key"}'}):
            result = call_api(request)
        
        # Verify response_time is recorded even for fast responses
        assert result['response_time'] > 0, "response_time should be positive even for fast responses"
        assert result['response_time'] < 1.0, "response_time should be reasonable for fast responses"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
