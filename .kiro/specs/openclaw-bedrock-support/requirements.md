# OpenClaw Bedrock Support - Requirements

## Overview

Enable the OpenClaw server (`llmrouter serve`) to call AWS Bedrock models at runtime by integrating with the existing Bedrock implementation in `llmrouter.utils.api_calling`.

## Background

Currently:
- ✅ LLMRouter CLI commands (`infer`, `chat`, `train`) fully support Bedrock via `llmrouter.utils.api_calling.call_api()`
- ✅ Bedrock models are properly configured in `configs/openclaw_example.yaml`
- ⚠️ OpenClaw server cannot call Bedrock models at runtime

The limitation exists because OpenClaw's `LLMBackend` class makes direct HTTP calls via `httpx` to OpenAI-compatible APIs, but Bedrock requires AWS SDK (boto3) authentication and uses a different API format.

## User Stories

### 1. As a developer, I want to use Bedrock models through the OpenClaw server
**Acceptance Criteria:**
- 1.1: OpenClaw server can successfully call Bedrock models configured in the YAML config
- 1.2: Bedrock calls use the existing `llmrouter.utils.api_calling.call_api()` implementation
- 1.3: Non-Bedrock providers (NVIDIA, OpenAI, Anthropic) continue to work via direct HTTP calls
- 1.4: AWS credentials are properly passed to Bedrock calls
- 1.5: AWS region configuration from model config is respected

### 2. As a developer, I want streaming responses from Bedrock models
**Acceptance Criteria:**
- 2.1: Bedrock models support streaming responses via `/v1/chat/completions` endpoint
- 2.2: Streaming format matches OpenAI's Server-Sent Events (SSE) format
- 2.3: WebSocket endpoint (`/v1/chat/ws`) works with Bedrock models
- 2.4: Model prefix feature works with Bedrock streaming responses

### 3. As a developer, I want clear error messages for Bedrock configuration issues
**Acceptance Criteria:**
- 3.1: Missing boto3 dependency shows installation instructions
- 3.2: Missing AWS credentials show configuration instructions
- 3.3: Invalid model IDs show common Bedrock model examples
- 3.4: Region mismatches show available regions and model availability docs
- 3.5: Timeout errors show troubleshooting steps

### 4. As a developer, I want the OpenClaw config to support Bedrock-specific fields
**Acceptance Criteria:**
- 4.1: `LLMConfig` class includes `aws_region` field
- 4.2: Existing configs with `aws_region` are properly loaded
- 4.3: `aws_region` defaults to None (uses AWS default region)
- 4.4: Config validation ensures Bedrock models have required fields

## Technical Requirements

### 1. Provider Detection
- Detect Bedrock provider from `LLMConfig.provider` field
- Support provider values: "bedrock", "aws", "Bedrock", "AWS" (case-insensitive)

### 2. Request Format Conversion
- Convert OpenClaw message format to LLMRouter request format
- Map OpenClaw parameters (temperature, max_tokens) to request fields
- Include AWS region from model config

### 3. Response Format Conversion
- Convert LLMRouter response format to OpenAI-compatible format
- Preserve token usage information
- Handle error responses appropriately

### 4. Streaming Support
- Convert LLMRouter streaming responses to OpenAI SSE format
- Handle streaming chunks properly
- Support both HTTP streaming and WebSocket streaming

## Non-Goals

- Creating a new Bedrock implementation (reuse existing `call_api()`)
- Modifying the existing Bedrock implementation in `llmrouter.utils.api_calling`
- Supporting Bedrock-specific features not available in OpenAI API (e.g., guardrails)
- Changing how non-Bedrock providers work

## Dependencies

- Existing: `llmrouter.utils.api_calling.call_api()` function
- Existing: `boto3` package (optional dependency, checked at runtime)
- Existing: AWS credentials configuration

## Success Metrics

- All existing OpenClaw tests continue to pass
- New tests verify Bedrock integration works correctly
- Documentation updated to reflect Bedrock support
- Example configs demonstrate Bedrock usage
