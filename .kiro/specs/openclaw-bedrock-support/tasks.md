# OpenClaw Bedrock Support - Implementation Tasks

## Task 1: Update Configuration to Support AWS Region

### 1.1 Add aws_region field to LLMConfig
- [ ] Add `aws_region: Optional[str] = None` field to `LLMConfig` dataclass in `openclaw_router/config.py`
- [ ] Update the dataclass to include proper type hints

### 1.2 Update config loading to parse aws_region
- [ ] Modify `OpenClawConfig.from_yaml()` method to load `aws_region` from YAML
- [ ] Add `aws_region=llm_config.get("aws_region")` to `LLMConfig` instantiation
- [ ] Ensure backward compatibility (existing configs without aws_region still work)

### 1.3 Write tests for configuration changes
- [ ] Test that `aws_region` is loaded from YAML config
- [ ] Test that `aws_region` defaults to None when not specified
- [ ] Test that existing configs without `aws_region` still load correctly
- [ ] Test that `aws_region` value is preserved in loaded config

**Validates:** Requirements 4.1, 4.2, 4.3

---

## Task 2: Implement Bedrock Provider Detection

### 2.1 Add _is_bedrock_provider helper function
- [ ] Add `_is_bedrock_provider(llm_config: LLMConfig) -> bool` function to `openclaw_router/server.py`
- [ ] Implement case-insensitive check for "bedrock" and "aws" provider values
- [ ] Handle None/empty provider values gracefully

### 2.2 Write property tests for provider detection
- [ ] Write property test that verifies Bedrock providers are correctly identified
- [ ] Write property test that verifies non-Bedrock providers return False
- [ ] Write property test for case-insensitivity (Bedrock, BEDROCK, bedrock, AWS, aws)
- [ ] Write property test for None/empty provider values

**Validates:** Requirements 1.2, 1.3

**Property:** Provider Routing Correctness

---

## Task 3: Implement Synchronous Bedrock API Calls

### 3.1 Add _call_bedrock_sync method to LLMBackend
- [ ] Add `_call_bedrock_sync()` method to `LLMBackend` class in `openclaw_router/server.py`
- [ ] Implement message normalization using existing `normalize_messages()`
- [ ] Implement token adjustment using existing `adjust_max_tokens()`
- [ ] Extract query from last user message
- [ ] Extract system prompt from system message
- [ ] Build request dict with required fields for `call_api()`
- [ ] Call `llmrouter.utils.api_calling.call_api()` with request
- [ ] Convert response to OpenAI format
- [ ] Handle token usage fields (prompt_tokens, completion_tokens, total_tokens)

### 3.2 Add error handling for Bedrock calls
- [ ] Catch `ImportError` and check for "boto3", return installation instructions
- [ ] Catch credential-related exceptions, return AWS setup instructions
- [ ] Catch model-related exceptions, return common Bedrock model IDs
- [ ] Catch region-related exceptions, return region configuration help
- [ ] Catch timeout exceptions, return troubleshooting steps
- [ ] Catch general exceptions, return error message
- [ ] Ensure all errors return HTTP 500 with detailed message

### 3.3 Write unit tests for synchronous Bedrock calls
- [ ] Test successful Bedrock API call returns OpenAI-compatible format
- [ ] Test token usage fields are included in response
- [ ] Test system prompt is extracted and passed correctly
- [ ] Test query is extracted from last user message
- [ ] Test multi-turn conversations use last user message
- [ ] Test aws_region is passed to call_api when specified
- [ ] Test aws_region defaults to None when not specified

### 3.4 Write property tests for message format preservation
- [ ] Write property test that system messages are preserved
- [ ] Write property test that user messages are preserved
- [ ] Write property test that message content is preserved through conversion
- [ ] Write property test that multi-turn conversations extract correct query

**Validates:** Requirements 1.1, 1.2, 1.4, 1.5, 3.1, 3.2, 3.3, 3.4, 3.5

**Properties:** Message Format Preservation, Response Format Compatibility, AWS Region Configuration, Error Message Clarity

---

## Task 4: Implement Streaming Bedrock API Calls

### 4.1 Add _call_bedrock_streaming method to LLMBackend
- [ ] Add `_call_bedrock_streaming()` method to `LLMBackend` class in `openclaw_router/server.py`
- [ ] Import LiteLLM's `completion` function
- [ ] Implement message normalization using existing `normalize_messages()`
- [ ] Implement token adjustment using existing `adjust_max_tokens()`
- [ ] Build LiteLLM messages list (include system prompt in messages)
- [ ] Build completion kwargs with `stream=True`
- [ ] Add aws_region to kwargs if specified
- [ ] Call LiteLLM `completion()` with streaming enabled
- [ ] Iterate over streaming chunks
- [ ] Convert each chunk to OpenAI SSE format
- [ ] Yield formatted chunks
- [ ] Send `[DONE]` marker when complete

### 4.2 Add error handling for streaming calls
- [ ] Catch `ImportError` for missing boto3, yield error in SSE format
- [ ] Catch `ImportError` for missing LiteLLM, yield error in SSE format
- [ ] Catch general exceptions, yield error in SSE format
- [ ] Ensure error format matches OpenAI SSE format

### 4.3 Write unit tests for streaming Bedrock calls
- [ ] Test streaming response format matches OpenAI SSE
- [ ] Test chunks contain delta with role and content
- [ ] Test [DONE] marker is sent at end
- [ ] Test error messages are in SSE format
- [ ] Test aws_region is passed to LiteLLM when specified

### 4.4 Write property tests for streaming format
- [ ] Write property test that all chunks conform to OpenAI SSE format
- [ ] Write property test that streaming ends with [DONE] marker
- [ ] Write property test that chunk content is preserved from Bedrock response

**Validates:** Requirements 2.1, 2.2, 2.3, 3.1

**Properties:** Response Format Compatibility, Error Message Clarity

---

## Task 5: Update LLMBackend.call() to Route Bedrock Requests

### 5.1 Modify call() method to detect and route Bedrock
- [ ] Update `LLMBackend.call()` method in `openclaw_router/server.py`
- [ ] Add Bedrock provider check using `_is_bedrock_provider()`
- [ ] Route to `_call_bedrock_sync()` for non-streaming Bedrock requests
- [ ] Route to `_call_bedrock_streaming()` for streaming Bedrock requests
- [ ] Preserve existing routing for non-Bedrock providers
- [ ] Ensure API key retrieval only happens for non-Bedrock providers

### 5.2 Write integration tests for routing
- [ ] Test Bedrock requests are routed to Bedrock methods
- [ ] Test non-Bedrock requests are routed to existing methods
- [ ] Test streaming Bedrock requests use streaming method
- [ ] Test non-streaming Bedrock requests use sync method
- [ ] Test mixed provider scenarios (Bedrock + NVIDIA in same config)

### 5.3 Write property tests for routing correctness
- [ ] Write property test that Bedrock providers always route to Bedrock methods
- [ ] Write property test that non-Bedrock providers always route to HTTP methods
- [ ] Write property test that routing is deterministic based on provider field

**Validates:** Requirements 1.1, 1.2, 1.3

**Properties:** Provider Routing Correctness, Non-Bedrock Provider Preservation

---

## Task 6: Add WebSocket Support for Bedrock

### 6.1 Update WebSocket endpoint to support Bedrock
- [x] Review `chat_websocket()` function in `openclaw_router/server.py`
- [x] Verify that WebSocket endpoint uses `LLMBackend.call()` (should work automatically)
- [x] Test that Bedrock streaming works through WebSocket
- [x] Ensure model prefix feature works with Bedrock WebSocket responses

### 6.2 Write tests for WebSocket Bedrock support
- [x] Test WebSocket connection with Bedrock model
- [x] Test WebSocket streaming with Bedrock model
- [x] Test WebSocket error handling with Bedrock model
- [x] Test model prefix with Bedrock WebSocket responses

**Validates:** Requirements 2.3, 2.4

---

## Task 7: Write Comprehensive Error Handling Tests

### 7.1 Write tests for missing boto3
- [ ] Test that missing boto3 returns installation instructions
- [ ] Test error message includes "pip install boto3"
- [ ] Test error message includes "pip install llmrouter-lib[bedrock]"

### 7.2 Write tests for missing credentials
- [ ] Test that missing credentials returns AWS setup instructions
- [ ] Test error message includes environment variable setup
- [ ] Test error message includes credential file setup
- [ ] Test error message includes IAM role option

### 7.3 Write tests for invalid model ID
- [ ] Test that invalid model ID returns common Bedrock model examples
- [ ] Test error message includes Claude model IDs
- [ ] Test error message includes Titan model IDs
- [ ] Test error message includes Llama model IDs
- [ ] Test error message includes Mistral model IDs

### 7.4 Write tests for region mismatch
- [ ] Test that region errors return region configuration help
- [ ] Test error message includes model availability documentation link
- [ ] Test error message suggests trying different region

### 7.5 Write tests for timeout errors
- [ ] Test that timeout errors return troubleshooting steps
- [ ] Test error message suggests increasing timeout
- [ ] Test error message suggests checking network connectivity
- [ ] Test error message suggests trying different region

### 7.6 Write property tests for error message clarity
- [ ] Write property test that all Bedrock errors include actionable instructions
- [ ] Write property test that error messages are non-empty
- [ ] Write property test that error messages include troubleshooting steps

**Validates:** Requirements 3.1, 3.2, 3.3, 3.4, 3.5

**Property:** Error Message Clarity

---

## Task 8: Write Integration Tests for Mixed Providers

### 8.1 Test OpenClaw server with mixed providers
- [ ] Create test config with both Bedrock and NVIDIA models
- [ ] Test that router can select Bedrock models
- [ ] Test that router can select NVIDIA models
- [ ] Test that both providers work in same server instance
- [ ] Test that model prefix works with both providers

### 8.2 Test routing strategies with Bedrock
- [ ] Test random strategy with Bedrock models
- [ ] Test round_robin strategy with Bedrock models
- [ ] Test rules strategy with Bedrock models
- [ ] Test weights with Bedrock models

### 8.3 Write property tests for non-Bedrock preservation
- [ ] Write property test that non-Bedrock providers use HTTP path
- [ ] Write property test that non-Bedrock behavior is unchanged
- [ ] Write property test that non-Bedrock performance is unchanged

**Validates:** Requirements 1.1, 1.3

**Property:** Non-Bedrock Provider Preservation

---

## Task 9: Update Documentation

### 9.1 Update OPENCLAW_BEDROCK_STATUS.md
- [x] Update status from "Limited" to "Full Support"
- [x] Update "What Works" section to include OpenClaw server
- [x] Remove "What Doesn't Work" section
- [x] Update "Workaround Options" to "Usage Examples"
- [x] Add examples of using Bedrock with OpenClaw server

### 9.2 Update README.md
- [x] Add note about OpenClaw Bedrock support
- [x] Add example of starting OpenClaw server with Bedrock models
- [x] Update feature list to include Bedrock support in OpenClaw

### 9.3 Update openclaw_example.yaml comments
- [x] Update comments to reflect that Bedrock models work at runtime
- [x] Add examples of using Bedrock models in router strategies
- [x] Add notes about AWS credential configuration

**Validates:** All requirements (documentation)

---

## Task 10: Run Full Test Suite and Verify

### 10.1 Run all OpenClaw tests
- [ ] Run existing OpenClaw tests to ensure no regressions
- [ ] Verify all tests pass
- [ ] Fix any failing tests

### 10.2 Run all Bedrock tests
- [ ] Run all new Bedrock integration tests
- [ ] Verify all tests pass
- [ ] Fix any failing tests

### 10.3 Manual testing
- [ ] Start OpenClaw server with Bedrock models
- [ ] Test synchronous Bedrock API call via curl/httpie
- [ ] Test streaming Bedrock API call via curl/httpie
- [ ] Test WebSocket Bedrock call
- [ ] Test mixed provider routing (Bedrock + NVIDIA)
- [ ] Test error scenarios (missing credentials, invalid model, etc.)

**Validates:** All requirements (verification)

---

## Summary

Total tasks: 10 major tasks with 60+ subtasks

Estimated effort:
- Configuration changes: 2-3 hours
- Bedrock implementation: 6-8 hours
- Testing: 8-10 hours
- Documentation: 2-3 hours
- Total: 18-24 hours

Dependencies:
- Tasks 1-2 can be done in parallel
- Task 3 depends on Tasks 1-2
- Task 4 depends on Tasks 1-2
- Task 5 depends on Tasks 3-4
- Task 6 depends on Task 5
- Tasks 7-8 depend on Tasks 3-6
- Task 9 depends on all implementation tasks
- Task 10 depends on all tasks
