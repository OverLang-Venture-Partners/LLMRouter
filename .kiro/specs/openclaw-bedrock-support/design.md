# OpenClaw Bedrock Support - Design

## Architecture Overview

The design integrates Bedrock support into OpenClaw's `LLMBackend` class by detecting Bedrock providers and routing those requests to the existing `llmrouter.utils.api_calling.call_api()` function, while maintaining the current direct HTTP approach for OpenAI-compatible providers.

```
┌─────────────────────────────────────────────────────────────┐
│                    OpenClaw Server                          │
│                                                             │
│  ┌──────────────────────────────────────────────────────┐  │
│  │              LLMBackend.call()                       │  │
│  │                                                      │  │
│  │  ┌────────────────────────────────────────────┐    │  │
│  │  │  _is_bedrock_provider(llm_config)          │    │  │
│  │  │  Check if provider is "bedrock" or "aws"   │    │  │
│  │  └────────────────────────────────────────────┘    │  │
│  │                      │                              │  │
│  │         ┌────────────┴────────────┐                │  │
│  │         │                         │                │  │
│  │    YES  │                         │  NO            │  │
│  │         ▼                         ▼                │  │
│  │  ┌──────────────┐        ┌──────────────┐         │  │
│  │  │ _call_bedrock│        │ _call_sync   │         │  │
│  │  │              │        │ (existing)   │         │  │
│  │  │ Uses:        │        │              │         │  │
│  │  │ call_api()   │        │ Uses: httpx  │         │  │
│  │  └──────────────┘        └──────────────┘         │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                      │                         │
                      │                         │
                      ▼                         ▼
        ┌──────────────────────┐    ┌──────────────────────┐
        │ llmrouter.utils.     │    │ Direct HTTP to       │
        │ api_calling.call_api │    │ OpenAI-compatible    │
        │                      │    │ endpoints            │
        │ (boto3 + LiteLLM)    │    │ (httpx)              │
        └──────────────────────┘    └──────────────────────┘
                      │
                      ▼
        ┌──────────────────────┐
        │ AWS Bedrock API      │
        │ (via boto3)          │
        └──────────────────────┘
```

## Component Design

### 1. Configuration Changes

#### 1.1 Update `LLMConfig` dataclass

**File:** `openclaw_router/config.py`

Add `aws_region` field to `LLMConfig`:

```python
@dataclass
class LLMConfig:
    """Single LLM configuration"""
    name: str
    provider: str
    model_id: str
    base_url: str
    api_key: Optional[str] = None
    api_key_env: Optional[str] = None
    description: str = ""
    input_price: float = 0.0
    output_price: float = 0.0
    max_tokens: int = 4096
    context_limit: int = 32768
    aws_region: Optional[str] = None  # NEW: AWS region for Bedrock models
```

#### 1.2 Update config loading

**File:** `openclaw_router/config.py`

Modify `OpenClawConfig.from_yaml()` to load `aws_region`:

```python
# In the LLM configurations section
for name, llm_config in llms_data.items():
    config.llms[name] = LLMConfig(
        name=name,
        provider=llm_config.get("provider", "openai"),
        model_id=llm_config.get("model", name),
        base_url=llm_config.get("base_url", "https://api.openai.com/v1"),
        api_key=llm_config.get("api_key"),
        api_key_env=llm_config.get("api_key_env"),
        description=llm_config.get("description", ""),
        input_price=llm_config.get("input_price", 0.0),
        output_price=llm_config.get("output_price", 0.0),
        max_tokens=llm_config.get("max_tokens", 4096),
        context_limit=llm_config.get("context_limit", 32768),
        aws_region=llm_config.get("aws_region"),  # NEW
    )
```

### 2. Backend Changes

#### 2.1 Add Bedrock detection helper

**File:** `openclaw_router/server.py`

Add helper function to detect Bedrock providers:

```python
def _is_bedrock_provider(llm_config: LLMConfig) -> bool:
    """Check if LLM config uses Bedrock provider"""
    if not llm_config.provider:
        return False
    return llm_config.provider.lower() in ['bedrock', 'aws']
```

#### 2.2 Modify `LLMBackend.call()` method

**File:** `openclaw_router/server.py`

Update the `call()` method to route Bedrock requests:

```python
async def call(self, llm_name: str, messages: List[Dict], max_tokens: int = 4096,
               temperature: Optional[float] = None, stream: bool = False):
    """Call LLM API"""
    if llm_name not in self.config.llms:
        raise HTTPException(status_code=404, detail=f"LLM '{llm_name}' not found")

    llm_config = self.config.llms[llm_name]
    
    # Route based on provider type
    if _is_bedrock_provider(llm_config):
        # Use Bedrock path
        if stream:
            return self._call_bedrock_streaming(llm_config, messages, max_tokens, temperature)
        else:
            return await self._call_bedrock_sync(llm_config, messages, max_tokens, temperature)
    else:
        # Use existing HTTP path for OpenAI-compatible providers
        api_key = self.config.get_api_key(llm_config.provider, llm_config)
        if stream:
            return self._call_streaming(llm_config, messages, max_tokens, temperature, api_key)
        else:
            return await self._call_sync(llm_config, messages, max_tokens, temperature, api_key)
```

#### 2.3 Add synchronous Bedrock call method

**File:** `openclaw_router/server.py`

```python
async def _call_bedrock_sync(self, llm: LLMConfig, messages: List[Dict], max_tokens: int,
                             temperature: Optional[float]) -> Dict:
    """Synchronous Bedrock API call using llmrouter.utils.api_calling"""
    from llmrouter.utils.api_calling import call_api
    
    # Normalize messages
    normalized = normalize_messages(messages, llm.model_id)
    adjusted_max = adjust_max_tokens(normalized, llm.model_id, max_tokens)
    
    # Extract query (last user message) and system prompt
    query = ""
    system_prompt = ""
    
    for msg in normalized:
        if msg["role"] == "system":
            system_prompt = msg["content"]
        elif msg["role"] == "user":
            query = msg["content"]  # Use last user message
    
    # Build request for call_api
    request = {
        "query": query,
        "model_name": llm.name,
        "api_name": llm.model_id,
        "api_endpoint": llm.base_url,  # Not used for Bedrock, but required field
        "service": "Bedrock",
        "system_prompt": system_prompt if system_prompt else None,
        "aws_region": llm.aws_region,
    }
    
    # Call API
    try:
        result = call_api(
            [request],
            max_tokens=adjusted_max,
            temperature=temperature if temperature is not None else 0.7,
            top_p=1.0,
            timeout=120.0
        )[0]
        
        # Check for errors
        if 'error' in result:
            raise HTTPException(status_code=500, detail=result['error'])
        
        # Convert to OpenAI format
        return {
            "id": f"chatcmpl-{llm.name}",
            "object": "chat.completion",
            "model": llm.model_id,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": result['response']
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": result.get('prompt_tokens', 0),
                "completion_tokens": result.get('completion_tokens', 0),
                "total_tokens": result.get('token_num', 0)
            }
        }
    except ImportError as e:
        # Handle missing boto3
        if "boto3" in str(e).lower():
            raise HTTPException(
                status_code=500,
                detail="AWS Bedrock support requires boto3. Install with: pip install boto3"
            )
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

#### 2.4 Add streaming Bedrock call method

**File:** `openclaw_router/server.py`

```python
async def _call_bedrock_streaming(self, llm: LLMConfig, messages: List[Dict], max_tokens: int,
                                  temperature: Optional[float]) -> AsyncGenerator:
    """Streaming Bedrock API call using llmrouter.utils.api_calling with LiteLLM streaming"""
    import asyncio
    from llmrouter.utils.api_calling import _is_bedrock_model
    
    try:
        # Import LiteLLM for streaming
        from litellm import completion
    except ImportError:
        yield f'data: {json.dumps({"error": "LiteLLM not installed"})}\n\n'
        return
    
    # Normalize messages
    normalized = normalize_messages(messages, llm.model_id)
    adjusted_max = adjust_max_tokens(normalized, llm.model_id, max_tokens)
    
    # Build messages for LiteLLM (include system prompt in messages)
    litellm_messages = []
    for msg in normalized:
        litellm_messages.append({
            "role": msg["role"],
            "content": msg["content"]
        })
    
    # Build completion kwargs
    completion_kwargs = {
        'model': f"bedrock/{llm.model_id}",
        'messages': litellm_messages,
        'max_tokens': adjusted_max,
        'temperature': temperature if temperature is not None else 0.7,
        'top_p': 1.0,
        'timeout': 120.0,
        'stream': True
    }
    
    # Add AWS region if specified
    if llm.aws_region:
        completion_kwargs['aws_region_name'] = llm.aws_region
    
    try:
        # Call LiteLLM completion with streaming
        response = completion(**completion_kwargs)
        
        # Stream chunks in OpenAI format
        for chunk in response:
            if hasattr(chunk, 'choices') and chunk.choices:
                choice = chunk.choices[0]
                delta = {}
                
                if hasattr(choice, 'delta'):
                    if hasattr(choice.delta, 'role') and choice.delta.role:
                        delta['role'] = choice.delta.role
                    if hasattr(choice.delta, 'content') and choice.delta.content:
                        delta['content'] = choice.delta.content
                
                finish_reason = getattr(choice, 'finish_reason', None)
                
                chunk_data = {
                    "id": f"chatcmpl-{llm.name}",
                    "object": "chat.completion.chunk",
                    "choices": [{
                        "index": 0,
                        "delta": delta,
                        "finish_reason": finish_reason
                    }]
                }
                
                yield f"data: {json.dumps(chunk_data)}\n\n"
        
        # Send [DONE] marker
        yield "data: [DONE]\n\n"
        
    except ImportError as e:
        if "boto3" in str(e).lower():
            error_msg = "AWS Bedrock support requires boto3. Install with: pip install boto3"
        else:
            error_msg = str(e)
        yield f'data: {json.dumps({"error": error_msg})}\n\n'
    except Exception as e:
        yield f'data: {json.dumps({"error": str(e)})}\n\n'
```

## Data Flow

### Synchronous Request Flow

1. Client sends POST to `/v1/chat/completions` with Bedrock model
2. `chat_completions()` endpoint extracts messages and selects model
3. `LLMBackend.call()` checks if provider is Bedrock
4. `_call_bedrock_sync()` converts OpenClaw format to LLMRouter format:
   - Extract query from last user message
   - Extract system prompt from system message
   - Build request dict with required fields
5. Call `llmrouter.utils.api_calling.call_api()` with request
6. `call_api()` handles Bedrock authentication and API call via LiteLLM
7. `_call_bedrock_sync()` converts response to OpenAI format
8. Response returned to client

### Streaming Request Flow

1. Client sends POST to `/v1/chat/completions` with `stream: true`
2. `chat_completions()` endpoint calls `LLMBackend.call()` with `stream=True`
3. `_call_bedrock_streaming()` builds LiteLLM completion kwargs
4. Call LiteLLM `completion()` with `stream=True`
5. Iterate over streaming chunks
6. Convert each chunk to OpenAI SSE format
7. Yield formatted chunks to client
8. Send `[DONE]` marker when complete

## Error Handling

### Error Categories

1. **Missing boto3**: Catch `ImportError`, check for "boto3", return installation instructions
2. **Missing credentials**: Catch exceptions with "credentials" keywords, return AWS setup instructions
3. **Invalid model ID**: Catch exceptions with "model" keywords, return common Bedrock model IDs
4. **Region mismatch**: Catch exceptions with "region" keywords, return region configuration help
5. **Timeout**: Catch `TimeoutError` or timeout-related exceptions, return troubleshooting steps
6. **General errors**: Catch all other exceptions, return error message

### Error Response Format

All errors return HTTP 500 with detailed error message in response body:

```json
{
  "error": "Detailed error message with troubleshooting steps"
}
```

For streaming, errors are sent as SSE events:

```
data: {"error": "Detailed error message"}

```

## Testing Strategy

### Unit Tests

1. **Configuration tests**:
   - Test `aws_region` field is loaded from YAML
   - Test `aws_region` defaults to None
   - Test Bedrock provider detection

2. **Provider detection tests**:
   - Test `_is_bedrock_provider()` with various provider values
   - Test case-insensitive matching

3. **Request conversion tests**:
   - Test message format conversion
   - Test system prompt extraction
   - Test query extraction from multi-turn conversations

4. **Response conversion tests**:
   - Test LLMRouter response to OpenAI format
   - Test token usage preservation
   - Test error response handling

### Integration Tests

1. **Synchronous Bedrock calls**:
   - Test successful Bedrock API call
   - Test response format matches OpenAI
   - Test token counts are included

2. **Streaming Bedrock calls**:
   - Test streaming response format
   - Test chunk format matches OpenAI SSE
   - Test [DONE] marker is sent

3. **Error handling**:
   - Test missing boto3 error message
   - Test missing credentials error message
   - Test invalid model ID error message
   - Test region mismatch error message

4. **Mixed provider tests**:
   - Test OpenClaw server with both Bedrock and NVIDIA models
   - Test router can select between Bedrock and non-Bedrock models
   - Test model prefix works with Bedrock models

## Correctness Properties

### Property 1: Provider Routing Correctness
**Validates: Requirements 1.2, 1.3**

For any LLM configuration:
- If `provider` is "bedrock" or "aws" (case-insensitive), requests MUST be routed to `_call_bedrock_sync()` or `_call_bedrock_streaming()`
- If `provider` is any other value, requests MUST be routed to `_call_sync()` or `_call_streaming()`
- The routing decision MUST be deterministic based solely on the provider field

### Property 2: Message Format Preservation
**Validates: Requirements 1.1, 2.1**

For any valid OpenClaw message list:
- System messages MUST be extracted and passed as `system_prompt`
- User messages MUST be extracted and passed as `query`
- The semantic content of messages MUST be preserved through format conversion
- Multi-turn conversations MUST use the last user message as the query

### Property 3: Response Format Compatibility
**Validates: Requirements 1.1, 2.2**

For any successful Bedrock API response:
- The response MUST conform to OpenAI chat completion format
- Token usage fields (`prompt_tokens`, `completion_tokens`, `total_tokens`) MUST be present
- The response content MUST match the Bedrock API response
- Streaming chunks MUST conform to OpenAI SSE format

### Property 4: AWS Region Configuration
**Validates: Requirements 1.5, 4.2**

For any Bedrock model configuration:
- If `aws_region` is specified, it MUST be passed to the Bedrock API call
- If `aws_region` is not specified, the AWS default region MUST be used
- The region value MUST be preserved through the request pipeline

### Property 5: Error Message Clarity
**Validates: Requirements 3.1, 3.2, 3.3, 3.4, 3.5**

For any Bedrock API error:
- Missing boto3 errors MUST include installation instructions
- Credential errors MUST include AWS configuration instructions
- Invalid model errors MUST include examples of valid Bedrock model IDs
- Region errors MUST include region configuration guidance
- All error messages MUST be actionable and include troubleshooting steps

### Property 6: Non-Bedrock Provider Preservation
**Validates: Requirements 1.3**

For any non-Bedrock provider (NVIDIA, OpenAI, Anthropic):
- The existing HTTP-based call path MUST be used
- No Bedrock-specific code MUST be executed
- API behavior MUST remain unchanged from before the implementation
- Performance characteristics MUST remain unchanged

## Implementation Notes

### Reusing Existing Code

The implementation heavily reuses existing code:
- `llmrouter.utils.api_calling.call_api()` for Bedrock API calls
- `normalize_messages()` for message format normalization
- `adjust_max_tokens()` for token limit handling
- `clean_response()` for response formatting (with modifications for Bedrock)

### Async/Sync Considerations

- `call_api()` is synchronous, so `_call_bedrock_sync()` wraps it in async
- LiteLLM's `completion()` is synchronous, so streaming uses sync iteration
- The async generator pattern is maintained for consistency with existing streaming code

### Performance Considerations

- Bedrock calls may have higher latency than local/NVIDIA endpoints
- Streaming reduces perceived latency for long responses
- AWS region selection can significantly impact latency

### Security Considerations

- AWS credentials are handled by boto3 (environment variables, credential files, IAM roles)
- No credentials are logged or exposed in error messages
- API keys for non-Bedrock providers continue to use existing secure handling
