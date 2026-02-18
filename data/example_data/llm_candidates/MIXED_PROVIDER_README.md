# Mixed Provider Configurations

This directory contains example configurations that demonstrate how to use multiple LLM providers together in LLMRouter. These configurations allow you to route queries across different providers (Bedrock, NVIDIA, OpenAI) based on your routing logic.

## Available Mixed Configurations

### 1. Bedrock + NVIDIA (`mixed_bedrock_nvidia.json`)

Combines AWS Bedrock models with NVIDIA models:

**Bedrock Models:**
- `claude-3-sonnet-bedrock`: Claude 3 Sonnet via AWS Bedrock (us-east-1)
- `titan-text-express-bedrock`: Amazon Titan Text Express (us-east-1)

**NVIDIA Models:**
- `qwen2.5-7b-instruct`: Qwen 2.5 7B Instruct
- `llama-3.1-8b-instruct`: Llama 3.1 8B Instruct
- `mistral-7b-instruct-v0.3`: Mistral 7B Instruct

### 2. Bedrock + OpenAI (`mixed_bedrock_openai.json`)

Combines AWS Bedrock models with OpenAI models:

**Bedrock Models:**
- `claude-3-haiku-bedrock`: Claude 3 Haiku via AWS Bedrock (us-west-2)
- `llama-3-70b-bedrock`: Llama 3 70B Instruct via AWS Bedrock (us-west-2)

**OpenAI Models:**
- `gpt-4o-mini`: GPT-4o Mini
- `gpt-4o`: GPT-4o
- `gpt-3.5-turbo`: GPT-3.5 Turbo

## Configuration Structure

Each model in the configuration file has the following fields:

### Common Fields (All Providers)
- `model`: The model identifier (required)
- `service`: The provider name (required)
- `size`: Model parameter count (required)
- `feature`: Human-readable description (required)
- `input_price`: Cost per million input tokens in USD (required)
- `output_price`: Cost per million output tokens in USD (required)

### Provider-Specific Fields

**Bedrock Models:**
- `aws_region`: AWS region for the model (optional, defaults to AWS_DEFAULT_REGION)
- No `api_endpoint` field needed (determined by AWS region)

**NVIDIA/OpenAI Models:**
- `api_endpoint`: API endpoint URL (required)
- No `aws_region` field needed

## Usage

### 1. Using Mixed Configurations in LLMRouter

```python
from llmrouter.utils.api_calling import call_api

# Load your mixed configuration
llm_config_path = "data/example_data/llm_candidates/mixed_bedrock_nvidia.json"

# Make a request to a Bedrock model
bedrock_request = {
    "api_endpoint": "",  # Not used for Bedrock
    "query": "What is machine learning?",
    "model_name": "claude-3-sonnet-bedrock",
    "api_name": "anthropic.claude-3-sonnet-20240229-v1:0",
    "service": "Bedrock",
    "aws_region": "us-east-1"
}

# Make a request to an NVIDIA model
nvidia_request = {
    "api_endpoint": "https://integrate.api.nvidia.com/v1",
    "query": "What is machine learning?",
    "model_name": "qwen2.5-7b-instruct",
    "api_name": "qwen/qwen2.5-7b-instruct",
    "service": "NVIDIA"
}

# Both requests use the same call_api function
# bedrock_response = call_api(bedrock_request)
# nvidia_response = call_api(nvidia_request)
```

### 2. Authentication Setup

**For Bedrock Models:**
```bash
# Option 1: Environment variables
export AWS_ACCESS_KEY_ID="your-access-key"
export AWS_SECRET_ACCESS_KEY="your-secret-key"
export AWS_DEFAULT_REGION="us-east-1"

# Option 2: AWS credential file (~/.aws/credentials)
[default]
aws_access_key_id = your-access-key
aws_secret_access_key = your-secret-key

# Option 3: IAM role (automatic on AWS infrastructure)
```

**For NVIDIA/OpenAI Models:**
```bash
# Set API keys using service-specific format
export API_KEYS='{"NVIDIA": "your-nvidia-key", "OpenAI": "your-openai-key"}'

# Or use comma-separated format for single provider
export API_KEYS="your-api-key-1,your-api-key-2"
```

### 3. Router Configuration

When using mixed providers with routers, the routing logic works the same way:

```python
# The router will select the best model based on your routing logic
# It can choose from any provider in your configuration
selected_model = router.select_model(query)

# The API calling logic automatically handles the provider-specific details
response = call_api(selected_model)
```

## Model Formatting

LLMRouter automatically formats model identifiers based on the provider:

- **Bedrock models**: `bedrock/{model_id}`
  - Example: `bedrock/anthropic.claude-3-sonnet-20240229-v1:0`

- **NVIDIA models**: `openai/{model_id}`
  - Example: `openai/qwen/qwen2.5-7b-instruct`

- **OpenAI models**: `openai/{model_id}`
  - Example: `openai/gpt-4o-mini`

This formatting is handled automatically by the `call_api` function based on the `service` field.

## Testing

To verify your mixed provider configuration:

```bash
# Run the test suite
python3 -m pytest tests/test_mixed_provider_configs.py -v

# Run the verification script
PYTHONPATH=. python3 tests/verify_mixed_provider_configs.py
```

## Benefits of Mixed Provider Configurations

1. **Cost Optimization**: Route expensive queries to premium models (GPT-4o, Claude 3 Sonnet) and simple queries to cost-effective models (Titan, GPT-3.5)

2. **Performance Optimization**: Use fast models (Claude 3 Haiku, Titan) for latency-sensitive applications and powerful models (Llama 3 70B) for complex reasoning

3. **Redundancy**: If one provider has issues, automatically fall back to another provider

4. **Regional Optimization**: Use Bedrock models in specific AWS regions for lower latency

5. **Feature-Specific Routing**: Route queries to models with specific capabilities (e.g., code generation, multilingual support)

## Example Use Cases

### Use Case 1: Cost-Aware Routing
```python
# Simple queries → Titan Text Express ($0.20/$0.60 per million tokens)
# Complex queries → Claude 3 Sonnet ($3.00/$15.00 per million tokens)
```

### Use Case 2: Latency-Aware Routing
```python
# Real-time chat → Claude 3 Haiku (fastest)
# Batch processing → Llama 3 70B (most powerful)
```

### Use Case 3: Regional Routing
```python
# US East users → Bedrock models in us-east-1
# US West users → Bedrock models in us-west-2
# Global users → OpenAI/NVIDIA models
```

## Troubleshooting

### Bedrock Models Not Working
1. Verify AWS credentials are configured
2. Check that the model is available in your AWS region
3. Ensure you have access to the model in AWS Bedrock console

### NVIDIA/OpenAI Models Not Working
1. Verify API keys are set in API_KEYS environment variable
2. Check that the API endpoint is correct
3. Ensure you have sufficient API credits

### Mixed Provider Routing Issues
1. Verify all models in your configuration have the required fields
2. Check that the `service` field is correctly set for each model
3. Ensure your routing logic handles multiple providers correctly

## Additional Resources

- [AWS Bedrock Documentation](https://docs.aws.amazon.com/bedrock/)
- [NVIDIA API Documentation](https://docs.api.nvidia.com/)
- [OpenAI API Documentation](https://platform.openai.com/docs/)
- [LLMRouter Documentation](../../README.md)
