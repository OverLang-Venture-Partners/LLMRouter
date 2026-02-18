"""
Test OpenClaw WebSocket endpoint with Bedrock models.

This test verifies that the WebSocket endpoint correctly:
- Connects and communicates with Bedrock models
- Streams responses from Bedrock models
- Handles errors appropriately
- Works with model prefix feature

**Validates: Requirements 2.3, 2.4**
"""

import pytest
import json
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from fastapi.testclient import TestClient
from openclaw_router.config import LLMConfig, OpenClawConfig
from openclaw_router.server import create_app


class MockStreamChunk:
    """Mock LiteLLM streaming chunk."""
    
    def __init__(self, role=None, content=None, finish_reason=None):
        self.choices = [MockChoice(role, content, finish_reason)]


class MockChoice:
    """Mock choice in streaming chunk."""
    
    def __init__(self, role=None, content=None, finish_reason=None):
        self.delta = MockDelta(role, content)
        self.finish_reason = finish_reason


class MockDelta:
    """Mock delta in streaming chunk."""
    
    def __init__(self, role=None, content=None):
        self.role = role
        self.content = content


class TestWebSocketBedrockConnection:
    """Test WebSocket connection with Bedrock models."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Create a minimal config with Bedrock model
        self.config = OpenClawConfig()
        self.config.llms = {
            "claude": LLMConfig(
                name="claude",
                provider="bedrock",
                model_id="anthropic.claude-3-sonnet-20240229-v1:0",
                base_url="https://bedrock.us-east-1.amazonaws.com",
                aws_region="us-east-1"
            )
        }
        self.config.routing_strategy = "random"
        self.config.show_model_prefix = False
        
        # Create app
        self.app = create_app(config=self.config)
        self.client = TestClient(self.app)
    
    def test_websocket_connection_with_bedrock_model(self):
        """Test that WebSocket can connect with Bedrock model."""
        # Mock streaming chunks
        mock_chunks = [
            MockStreamChunk(role="assistant", content=None),
            MockStreamChunk(content="Hello from Bedrock"),
            MockStreamChunk(content="!", finish_reason="stop")
        ]
        
        with patch('litellm.completion', return_value=iter(mock_chunks)):
            with self.client.websocket_connect("/v1/chat/ws") as websocket:
                # Send request
                request = {
                    "model": "claude",
                    "messages": [
                        {"role": "user", "content": "Hello"}
                    ],
                    "max_tokens": 100,
                    "temperature": 0.7
                }
                websocket.send_json(request)
                
                # Receive responses
                responses = []
                while True:
                    data = websocket.receive_text()
                    responses.append(data)
                    if "[DONE]" in data:
                        break
                
                # Verify we got responses
                assert len(responses) > 0
                assert any("[DONE]" in r for r in responses)
    
    def test_websocket_streaming_with_bedrock_model(self):
        """Test that WebSocket streaming works with Bedrock model."""
        mock_chunks = [
            MockStreamChunk(role="assistant", content=None),
            MockStreamChunk(content="Streaming"),
            MockStreamChunk(content=" response"),
            MockStreamChunk(content=" from"),
            MockStreamChunk(content=" Bedrock"),
            MockStreamChunk(content="!", finish_reason="stop")
        ]
        
        with patch('litellm.completion', return_value=iter(mock_chunks)):
            with self.client.websocket_connect("/v1/chat/ws") as websocket:
                request = {
                    "model": "claude",
                    "messages": [
                        {"role": "user", "content": "Test streaming"}
                    ],
                    "max_tokens": 100,
                    "temperature": 0.7
                }
                websocket.send_json(request)
                
                # Collect all chunks
                chunks = []
                content_parts = []
                while True:
                    data = websocket.receive_text()
                    chunks.append(data)
                    
                    if "[DONE]" in data:
                        break
                    
                    # Parse SSE format
                    if data.startswith("data: ") and not data.startswith("data: [DONE]"):
                        try:
                            json_str = data[6:].strip()
                            chunk_data = json.loads(json_str)
                            if "choices" in chunk_data and chunk_data["choices"]:
                                delta = chunk_data["choices"][0].get("delta", {})
                                if "content" in delta:
                                    content_parts.append(delta["content"])
                        except:
                            pass
                
                # Verify streaming format
                assert len(chunks) > 1  # Multiple chunks
                assert chunks[-1] == "data: [DONE]\n\n"  # Ends with [DONE]
                
                # Verify content was streamed
                full_content = "".join(content_parts)
                assert "Streaming" in full_content
                assert "Bedrock" in full_content
    
    def test_websocket_error_handling_with_bedrock_model(self):
        """Test that WebSocket handles Bedrock errors appropriately."""
        with patch('litellm.completion', side_effect=Exception("Bedrock API error")):
            with self.client.websocket_connect("/v1/chat/ws") as websocket:
                request = {
                    "model": "claude",
                    "messages": [
                        {"role": "user", "content": "Test error"}
                    ],
                    "max_tokens": 100,
                    "temperature": 0.7
                }
                websocket.send_json(request)
                
                # Should receive error message in SSE format
                data = websocket.receive_text()
                
                # Parse SSE format
                if data.startswith("data: "):
                    json_str = data[6:].strip()
                    error_data = json.loads(json_str)
                    assert "error" in error_data
                    assert "Bedrock API error" in str(error_data["error"])
                else:
                    # Fallback: check raw text
                    assert "error" in data.lower()
                    assert "bedrock" in data.lower()


class TestWebSocketBedrockModelPrefix:
    """Test WebSocket model prefix feature with Bedrock models."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Create config with model prefix enabled
        self.config = OpenClawConfig()
        self.config.llms = {
            "claude": LLMConfig(
                name="claude",
                provider="bedrock",
                model_id="anthropic.claude-3-sonnet-20240229-v1:0",
                base_url="https://bedrock.us-east-1.amazonaws.com",
                aws_region="us-east-1"
            )
        }
        self.config.routing_strategy = "random"
        self.config.show_model_prefix = True  # Enable prefix
        
        self.app = create_app(config=self.config)
        self.client = TestClient(self.app)
    
    def test_model_prefix_with_bedrock_websocket_responses(self):
        """Test that model prefix works with Bedrock WebSocket responses."""
        mock_chunks = [
            MockStreamChunk(role="assistant", content=None),
            MockStreamChunk(content="Response"),
            MockStreamChunk(content=" with"),
            MockStreamChunk(content=" prefix"),
            MockStreamChunk(content="!", finish_reason="stop")
        ]
        
        with patch('litellm.completion', return_value=iter(mock_chunks)):
            with self.client.websocket_connect("/v1/chat/ws") as websocket:
                request = {
                    "model": "claude",
                    "messages": [
                        {"role": "user", "content": "Test prefix"}
                    ],
                    "max_tokens": 100,
                    "temperature": 0.7
                }
                websocket.send_json(request)
                
                # Collect all chunks
                chunks = []
                first_content = None
                while True:
                    data = websocket.receive_text()
                    chunks.append(data)
                    
                    if "[DONE]" in data:
                        break
                    
                    # Parse first content chunk
                    if first_content is None and data.startswith("data: "):
                        try:
                            json_str = data[6:].strip()
                            chunk_data = json.loads(json_str)
                            if "choices" in chunk_data and chunk_data["choices"]:
                                delta = chunk_data["choices"][0].get("delta", {})
                                if "content" in delta and delta["content"]:
                                    first_content = delta["content"]
                        except:
                            pass
                
                # Verify prefix is present in first content
                # The prefix should be [claude] or [model_name]
                if first_content:
                    assert "[claude]" in first_content or "claude" in first_content.lower()


class TestWebSocketBedrockAutoRouting:
    """Test WebSocket with auto routing to Bedrock models."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Create config with multiple models including Bedrock
        self.config = OpenClawConfig()
        self.config.llms = {
            "claude": LLMConfig(
                name="claude",
                provider="bedrock",
                model_id="anthropic.claude-3-sonnet-20240229-v1:0",
                base_url="https://bedrock.us-east-1.amazonaws.com",
                aws_region="us-east-1"
            ),
            "nvidia": LLMConfig(
                name="nvidia",
                provider="nvidia",
                model_id="meta/llama-3.1-8b-instruct",
                base_url="https://integrate.api.nvidia.com/v1",
                api_key="test-key"
            )
        }
        self.config.routing_strategy = "random"
        self.config.show_model_prefix = False
        
        self.app = create_app(config=self.config)
        self.client = TestClient(self.app)
    
    def test_websocket_auto_routing_selects_bedrock(self):
        """Test that WebSocket auto routing can select Bedrock model."""
        mock_chunks = [
            MockStreamChunk(role="assistant", content=None),
            MockStreamChunk(content="Auto routed to Bedrock"),
            MockStreamChunk(content="!", finish_reason="stop")
        ]
        
        # Mock router to select Bedrock model
        with patch('litellm.completion', return_value=iter(mock_chunks)):
            with patch('openclaw_router.routers.OpenClawRouter.select_model', 
                      new_callable=AsyncMock) as mock_select:
                mock_select.return_value = "claude"
                
                with self.client.websocket_connect("/v1/chat/ws") as websocket:
                    request = {
                        "model": "auto",  # Auto routing
                        "messages": [
                            {"role": "user", "content": "Route me to Bedrock"}
                        ],
                        "max_tokens": 100,
                        "temperature": 0.7
                    }
                    websocket.send_json(request)
                    
                    # Receive responses
                    responses = []
                    while True:
                        data = websocket.receive_text()
                        responses.append(data)
                        if "[DONE]" in data:
                            break
                    
                    # Verify we got responses
                    assert len(responses) > 0
                    assert any("[DONE]" in r for r in responses)


class TestWebSocketBedrockMissingDependencies:
    """Test WebSocket error handling for missing Bedrock dependencies."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = OpenClawConfig()
        self.config.llms = {
            "claude": LLMConfig(
                name="claude",
                provider="bedrock",
                model_id="anthropic.claude-3-sonnet-20240229-v1:0",
                base_url="https://bedrock.us-east-1.amazonaws.com",
                aws_region="us-east-1"
            )
        }
        self.config.routing_strategy = "random"
        self.config.show_model_prefix = False
        
        self.app = create_app(config=self.config)
        self.client = TestClient(self.app)
    
    def test_websocket_missing_boto3_error(self):
        """Test that WebSocket returns error for missing boto3."""
        with patch('litellm.completion', side_effect=ImportError("No module named 'boto3'")):
            with self.client.websocket_connect("/v1/chat/ws") as websocket:
                request = {
                    "model": "claude",
                    "messages": [
                        {"role": "user", "content": "Test"}
                    ],
                    "max_tokens": 100,
                    "temperature": 0.7
                }
                websocket.send_json(request)
                
                # Should receive error message in SSE format
                data = websocket.receive_text()
                
                # Parse SSE format
                if data.startswith("data: "):
                    json_str = data[6:].strip()
                    error_data = json.loads(json_str)
                    error_msg = str(error_data.get("error", ""))
                else:
                    error_msg = data
                
                # Verify error mentions boto3 and installation
                assert "boto3" in error_msg.lower()
                assert "pip install" in error_msg.lower() or "install" in error_msg.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
