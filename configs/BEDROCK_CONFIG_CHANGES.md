# OpenClaw Bedrock Configuration - Key Changes

## What Was Fixed

### 1. Added Required `aws_region` Field
**Before:**
```yaml
nova-micro:
  provider: bedrock
  model: "us.amazon.nova-micro-v1:0"
  base_url: https://bedrock-runtime.us-west-2.amazonaws.com
```

**After:**
```yaml
nova-micro:
  provider: bedrock
  model: "us.amazon.nova-micro-v1:0"
  aws_region: "us-west-2"  # Required for Bedrock
```

**Why:** OpenClaw's Bedrock integration requires `aws_region` to construct the proper endpoint. The `base_url` field is not used for Bedrock providers.

---

### 2. Fixed Cost Field Names
**Before:**
```yaml
cost:
  input: 0.035
  output: 0.14
```

**After:**
```yaml
input_price: 0.035   # Per 1M input tokens
output_price: 0.14   # Per 1M output tokens
```

**Why:** OpenClaw uses `input_price` and `output_price` as top-level fields, not nested under `cost`.

---

### 3. Removed Unnecessary Fields

**Removed:**
- `api_keys.bedrock: ""` - Not needed, uses IAM role
- `base_url` - Not used for Bedrock providers

**Why:** 
- Bedrock authentication uses AWS credentials via boto3 (IAM role on EC2)
- OpenClaw automatically constructs Bedrock endpoints from `aws_region`

---

### 4. Fixed Cache Pricing Fields (for Claude models)
**Before:**
```yaml
cost:
  cache_read: 0.08
  cache_write: 1.00
```

**After:**
```yaml
cache_read_price: 0.08
cache_write_price: 1.00
```

**Why:** Consistent with OpenClaw's pricing field naming convention.

---

## How to Use

### 1. Start OpenClaw Server
```bash
python -m openclaw_router.server --config configs/openclaw_bedrock_corrected.yaml
```

### 2. Test Routing
```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "What is quantum computing?"}]
  }'
```

The router will:
1. Check memory for similar "what is" queries
2. If no memory match, use rules → selects `nova-micro` (simple query)
3. Save decision to `~/.llmrouter/bedrock_router_memory.jsonl`
4. Future similar queries will use memory suggestion

---

## Authentication on EC2

Since your EC2 instance has an IAM role with Bedrock permissions:

1. **No environment variables needed** - boto3 automatically uses instance profile
2. **No API keys in config** - IAM role provides credentials
3. **Automatic credential rotation** - AWS handles this for you

### Verify IAM Role Access
```bash
# On your EC2 instance, test boto3 can access Bedrock
python3 -c "import boto3; client = boto3.client('bedrock-runtime', region_name='us-west-2'); print('✓ Bedrock access confirmed')"
```

---

## Memory Learning

The router learns from every query:

**First time:**
- Query: "explain machine learning" → Rules → Selects `sonnet`

**After 10 similar queries:**
- Query: "explain quantum physics" → Memory recognizes "explain" pattern → Uses `sonnet` (learned)

**Memory file location:** `~/.llmrouter/bedrock_router_memory.jsonl`

---

## Model Selection Strategy

The router uses cost-optimized selection:

| Complexity | Primary Model | Fallbacks | Use Case |
|------------|---------------|-----------|----------|
| Simple | nova-micro | haiku, nova-2-lite | "What is X?", greetings |
| Standard | nova-2-lite | haiku, nova-pro | General queries, code |
| Reasoning | nova-pro | sonnet, nova-2-lite | Analysis, architecture |
| Deep | opus | sonnet, nova-pro | Research, comprehensive analysis |

---

## Next Steps

1. Copy `configs/openclaw_bedrock_corrected.yaml` to your deployment
2. Update `aws_region` if not using `us-west-2`
3. Start the server and test with sample queries
4. Monitor `~/.llmrouter/bedrock_router_memory.jsonl` to see learning progress
5. Adjust complexity thresholds in `custom_routers/bedrock_smart_router/config.yaml` if needed
