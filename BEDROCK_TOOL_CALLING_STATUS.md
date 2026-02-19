# Bedrock Tool Calling Integration - Current Status

## Problem Summary
OpenClaw agent is displaying raw XML tool invocation syntax (`<antml><invoke>`) instead of executing tools and rendering them as UI cards.

## Root Causes Identified

### 1. Tools Array Not Being Passed (FIXED)
- **Issue**: Models weren't receiving `tools` array in the request
- **Result**: Bedrock fell back to text-mode tool calling (XML in response body)
- **Fix**: Added `tools` and `tool_choice` parameters to ChatRequest and passed through to LiteLLM
- **Status**: ✅ Implemented

### 2. Tool Blocks in Message History Causing Validation Errors (IN PROGRESS)
- **Issue**: Bedrock error: "The number of toolResult blocks at messages.72.content exceeds the number of toolUse blocks of previous turn"
- **Cause**: OpenClaw sends conversation history with tool_use/tool_result blocks, but they're not properly paired
- **Attempted Fixes**:
  - `validate_tool_blocks()` - tried to validate pairing (didn't work)
  - `strip_tool_blocks_paired()` - strips tool_use/tool_result as paired units (NOT EXECUTING)
- **Status**: ❌ Function not being called - no debug output in logs

### 3. Clean Streaming Chunk Not Preserving Tool Calls (FIXED)
- **Issue**: `clean_streaming_chunk` was stripping `tool_calls` from delta
- **Fix**: Added preservation of `tool_calls` in delta
- **Status**: ✅ Implemented

### 4. Finish Reason Logging (FIXED)
- **Issue**: Need to verify finish_reason is `tool_calls` not `stop`
- **Fix**: Added logging for finish_reason
- **Status**: ✅ Implemented

## Current Code State

### Changes Made to `openclaw_router/server.py`:

1. **ChatRequest Model** - Added tools parameters:
```python
class ChatRequest(BaseModel):
    ...
    tools: Optional[List[Dict]] = None
    tool_choice: Optional[Any] = None
```

2. **LLMBackend.call()** - Passes tools through:
```python
async def call(self, llm_name: str, messages: List[Dict], max_tokens: int = 4096,
               temperature: Optional[float] = None, stream: bool = False,
               tools: Optional[List[Dict]] = None, tool_choice: Optional[Any] = None):
```

3. **strip_tool_blocks_paired()** - Strips tool blocks as paired units (NOT EXECUTING)

4. **LiteLLM Configuration**:
```python
litellm.drop_params = True
litellm.modify_params = True  # Auto-fix consecutive user/tool blocks
```

5. **clean_streaming_chunk()** - Preserves tool_calls in delta

## Critical Issue: Function Not Executing

The `strip_tool_blocks_paired()` function has comprehensive debug logging but produces NO output in logs:
- Expected: `[Strip Paired] Starting with X messages`
- Actual: Nothing

This means either:
1. The function isn't being called
2. An exception is happening before it runs
3. The service hasn't restarted with the new code

## Next Steps

1. **Verify service restart** - Ensure the new code is actually running
2. **Check function calls** - Verify `strip_tool_blocks_paired()` is being invoked
3. **Alternative approach** - If stripping doesn't work, consider:
   - Truncating conversation history to avoid tool blocks
   - Using a different message sanitization strategy
   - Letting LiteLLM's `modify_params=True` handle everything

## Test Commands

Restart service:
```bash
sudo systemctl restart openclaw
```

Check logs:
```bash
sudo journalctl -u openclaw -f | grep -E "Strip Paired|DEBUG Bedrock|ERROR"
```

## Files Modified
- `openclaw_router/server.py` - Main changes
- `openclaw_router/routers.py` - Tag-based routing (working)
- `configs/openclaw_bedrock_corrected.yaml` - Router config (working)

## Working Features
✅ Intelligent routing with haiku
✅ Tag-based manual model selection
✅ Memory collection for training data
✅ Cost optimization routing
✅ Bedrock authentication via LiteLLM

## Broken Features
❌ Tool calling (shows XML instead of executing)
❌ Multi-step agent execution (stops after one step)
❌ Tool result rendering in UI
