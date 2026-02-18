# Bedrock Tool Blocks Issue - Root Cause Analysis

## Problem
OpenClaw agent sends 50+ message conversation history containing `tool_use` and `tool_result` blocks to the router. When these messages reach Bedrock via LiteLLM, Bedrock rejects them with:

```
The toolConfig field must be defined when using toolUse and toolResult content blocks.
```

## Root Cause
1. OpenClaw's conversation history includes tool execution blocks from previous interactions
2. Bedrock's Converse API requires `toolConfig` parameter when ANY message contains tool blocks
3. We don't have the original tool definitions to pass along
4. LiteLLM's `modify_params=True` tries to MERGE tool blocks, not remove them

## Why Our Fixes Didn't Work

### Attempt 1: sanitize_messages_for_bedrock()
- **Issue**: Function runs AFTER normalize_messages()
- normalize_messages() converts list content to strings, but doesn't remove the list structure
- Tool blocks remain in the message structure that LiteLLM sees

### Attempt 2: Sanitize BEFORE normalize
- **Issue**: Code changes not taking effect (service caching or not restarting properly)
- Debug logs not appearing in output

### Attempt 3: force_text_only() + tools=None
- **Issue**: LiteLLM still sees tool blocks before our sanitization runs
- LiteLLM warning: "Potential consecutive user/tool blocks" proves tool blocks are present

## The Real Problem
**LiteLLM is receiving the ORIGINAL unsanitized messages**, not our cleaned version. This suggests:

1. Python is caching the module/function
2. The service isn't fully restarting
3. There's a code path we're not seeing that bypasses our sanitization
4. LiteLLM has internal message caching/transformation

## Solution Options

### Option A: Use Bedrock SDK Directly (RECOMMENDED)
Bypass LiteLLM entirely for Bedrock calls:
- Use boto3 bedrock-runtime client directly
- Full control over message sanitization
- No LiteLLM interference
- More code but more reliable

### Option B: Fix LiteLLM Message Passing
- Ensure sanitization happens at the EARLIEST point (in chat_completions endpoint)
- Pass pre-sanitized messages through the entire call chain
- Add extensive logging to verify sanitization is working

### Option C: Use LiteLLM's drop_params with Custom Preprocessing
- Create a LiteLLM custom callback to preprocess messages
- Strip tool blocks before LiteLLM sees them
- May require LiteLLM version upgrade

## Recommended Implementation (Option A)

```python
async def _call_bedrock_streaming_direct(self, llm: LLMConfig, messages: List[Dict], 
                                         max_tokens: int, temperature: Optional[float]) -> AsyncGenerator:
    """Direct Bedrock streaming using boto3 - bypasses LiteLLM"""
    import boto3
    
    # Force text-only extraction
    text_only = force_text_only(messages)
    normalized = normalize_messages(text_only, llm.model_id)
    
    # Create Bedrock client
    bedrock = boto3.client('bedrock-runtime', region_name=llm.aws_region)
    
    # Build Bedrock Converse API request
    request = {
        'modelId': llm.model_id,
        'messages': normalized,
        'inferenceConfig': {
            'maxTokens': max_tokens,
        }
    }
    
    if temperature is not None:
        request['inferenceConfig']['temperature'] = temperature
    
    # Call Bedrock Converse Stream API
    response = bedrock.converse_stream(**request)
    
    # Stream chunks
    for event in response['stream']:
        if 'contentBlockDelta' in event:
            delta = event['contentBlockDelta']['delta']
            if 'text' in delta:
                chunk_data = {
                    "id": f"chatcmpl-{llm.name}",
                    "object": "chat.completion.chunk",
                    "choices": [{
                        "index": 0,
                        "delta": {"content": delta['text']},
                        "finish_reason": None
                    }]
                }
                yield f"data: {json.dumps(chunk_data)}\n\n"
        
        elif 'messageStop' in event:
            # Send final chunk with finish_reason
            chunk_data = {
                "id": f"chatcmpl-{llm.name}",
                "object": "chat.completion.chunk",
                "choices": [{
                    "index": 0,
                    "delta": {},
                    "finish_reason": "stop"
                }]
            }
            yield f"data: {json.dumps(chunk_data)}\n\n"
            yield "data: [DONE]\n\n"
```

## Next Steps
1. Verify service is actually restarting and loading new code
2. Add logging at the EARLIEST point (chat_completions endpoint) to see raw messages
3. If sanitization still fails, implement Option A (direct Bedrock SDK)
4. Test with simple 2-message conversation first, then with full 50+ message history

## Testing Commands
```bash
# Restart service
sudo systemctl restart openclaw.service

# Watch logs in real-time
sudo journalctl -u openclaw.service -f

# Test with curl
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "haiku",
    "messages": [{"role": "user", "content": "Hello"}],
    "stream": true
  }'
```
