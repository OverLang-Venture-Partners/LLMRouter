# LLMRouter Bedrock Examples

This directory contains practical, runnable examples demonstrating how to use AWS Bedrock models with LLMRouter.

## Prerequisites

### 1. Install Dependencies

```bash
# Install LLMRouter with Bedrock support
pip install llmrouter-lib boto3

# Or install from source
pip install -e .
pip install boto3
```

### 2. Configure AWS Credentials

You need AWS credentials to use Bedrock models. Choose one of these methods:

#### Option 1: Environment Variables (Recommended for Development)

```bash
export AWS_ACCESS_KEY_ID="your-access-key-id"
export AWS_SECRET_ACCESS_KEY="your-secret-access-key"
export AWS_DEFAULT_REGION="us-east-1"
```

#### Option 2: AWS Credential File

Create or edit `~/.aws/credentials`:

```ini
[default]
aws_access_key_id = your-access-key-id
aws_secret_access_key = your-secret-access-key
```

Create or edit `~/.aws/config`:

```ini
[default]
region = us-east-1
```

#### Option 3: IAM Role (Automatic on AWS Infrastructure)

If running on EC2, ECS, Lambda, or other AWS services, IAM roles are automatically used. No configuration needed!

For detailed instructions, see [docs/AWS_BEDROCK_CREDENTIALS.md](../docs/AWS_BEDROCK_CREDENTIALS.md).

### 3. Enable Bedrock Model Access

Before using Bedrock models, you must request access in the AWS Console:

1. Go to [AWS Bedrock Console](https://console.aws.amazon.com/bedrock/)
2. Navigate to "Model access" in the left sidebar
3. Click "Manage model access"
4. Select the models you want to use (Claude, Titan, Llama, etc.)
5. Submit the access request
6. Wait for approval (usually instant for most models)

## Examples

### 1. Single Bedrock Query (`bedrock_single_query.py`)

Demonstrates making a single query to a Bedrock model.

**Features:**
- Single API call to Claude 3 Haiku
- System prompt usage
- Token usage tracking
- Cost estimation

**Usage:**
```bash
python examples/bedrock_single_query.py
```

**Expected Output:**
```
Making request to Bedrock model...
Model: claude-3-haiku-bedrock
Region: us-west-2
Query: Explain the concept of machine learning in simple terms.

✅ Response: Machine learning is a type of artificial intelligence...

📊 Token Usage:
   - Input tokens: 45
   - Output tokens: 128
   - Total tokens: 173
   - Response time: 1.23s
   - Estimated cost: $0.000171
```

### 2. Batch Bedrock Queries (`bedrock_batch_queries.py`)

Demonstrates making multiple queries to different Bedrock models in batch.

**Features:**
- Batch processing of 4 queries
- Multiple Bedrock models (Claude, Titan, Llama)
- Different AWS regions
- Summary statistics

**Usage:**
```bash
python examples/bedrock_batch_queries.py
```

**Expected Output:**
```
Making 4 batch requests to Bedrock models...

================================================================================
Request 1/4
================================================================================
Model: claude-3-haiku-bedrock
Region: us-west-2
Query: What is the capital of France?

✅ Response: The capital of France is Paris.

📊 Stats:
   - Tokens: 28 (in: 12, out: 16)
   - Time: 0.89s

[... more results ...]

================================================================================
SUMMARY
================================================================================
✅ Successful: 4/4
❌ Failed: 0/4
📊 Total tokens: 456
⏱️  Total time: 4.32s
📈 Average time per request: 1.08s
```

### 3. Mixed Provider Routing (`mixed_provider_routing.py`)

Demonstrates advanced routing strategies using multiple providers.

**Features:**
- Route by query complexity (simple → Titan, complex → Claude Sonnet)
- Route by geographic region (lower latency)
- Mix Bedrock and NVIDIA providers in one application
- Practical routing strategies

**Usage:**
```bash
# For Bedrock-only examples (no NVIDIA API key needed)
python examples/mixed_provider_routing.py

# For full mixed provider demo (requires NVIDIA API key)
export API_KEYS='{"NVIDIA": "your-nvidia-api-key"}'
python examples/mixed_provider_routing.py
```

**Expected Output:**
```
================================================================================
EXAMPLE 1: Routing by Query Complexity
================================================================================

Query: What is 2+2?
Complexity: simple → Model: titan-text-express-bedrock

Query: Explain the difference between machine learning and deep learning.
Complexity: medium → Model: claude-3-haiku-bedrock

Query: Design a distributed system architecture for a real-time analytics platform.
Complexity: complex → Model: claude-3-sonnet-bedrock

[... results ...]

================================================================================
KEY TAKEAWAYS
================================================================================
1. Route by complexity: Use cost-effective models for simple queries
2. Route by region: Use nearby AWS regions for lower latency
3. Mix providers: Combine Bedrock, NVIDIA, OpenAI in one application
4. Same API: All providers use the same call_api() interface
```

## Common Issues and Solutions

### Issue: "AWS credentials not found"

**Solution:** Configure AWS credentials using one of the methods above. Verify with:
```bash
aws sts get-caller-identity
```

### Issue: "Bedrock model not found or not accessible"

**Solution:** 
1. Check that you've requested access to the model in AWS Bedrock Console
2. Verify the model is available in your AWS region
3. Check the model ID is correct (see [AWS Bedrock Models](https://docs.aws.amazon.com/bedrock/latest/userguide/models-supported.html))

### Issue: "Model may not be available in region"

**Solution:**
1. Check model availability by region: [AWS Bedrock Regions](https://docs.aws.amazon.com/bedrock/latest/userguide/models-regions.html)
2. Try a different region (e.g., `us-east-1` or `us-west-2`)
3. Update the `aws_region` field in your request

### Issue: "boto3 not installed"

**Solution:**
```bash
pip install boto3
```

### Issue: NVIDIA examples fail with "API Error"

**Solution:** The mixed provider example requires NVIDIA API keys for the NVIDIA portion. Either:
1. Set up NVIDIA API keys: `export API_KEYS='{"NVIDIA": "your-key"}'`
2. Or just run the Bedrock-only portions (Examples 1 and 2)

## Model Pricing Reference

| Model | Input (per 1M tokens) | Output (per 1M tokens) |
|-------|----------------------|------------------------|
| Claude 3 Haiku | $0.25 | $1.25 |
| Claude 3 Sonnet | $3.00 | $15.00 |
| Titan Text Express | $0.20 | $0.60 |
| Llama 3 70B | $0.99 | $0.99 |
| Mistral 7B | $0.15 | $0.20 |

*Prices as of 2024. Check [AWS Bedrock Pricing](https://aws.amazon.com/bedrock/pricing/) for current rates.*

## Next Steps

1. **Explore Model Configurations:** Check `data/example_data/llm_candidates/` for more model configurations
2. **Read Documentation:** See `docs/AWS_BEDROCK_CREDENTIALS.md` for detailed credential setup
3. **Try Mixed Providers:** Combine Bedrock with NVIDIA or OpenAI models
4. **Build Your Router:** Use these examples as a foundation for your own routing logic

## Additional Resources

- [AWS Bedrock Documentation](https://docs.aws.amazon.com/bedrock/)
- [LLMRouter Documentation](../README.md)
- [Bedrock Model IDs](https://docs.aws.amazon.com/bedrock/latest/userguide/model-ids.html)
- [Bedrock Pricing](https://aws.amazon.com/bedrock/pricing/)
