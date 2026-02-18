# OpenClaw Bedrock Configuration

This directory contains configuration files for running OpenClaw with AWS Bedrock models and the custom Bedrock Smart Router.

## Files

1. **config_bedrock.yaml** - OpenClaw Router server configuration for Bedrock
2. **openclaw_agent_config.json** - OpenClaw agent configuration that registers all Bedrock models
3. **BEDROCK_SETUP.md** - This file

## Setup Instructions

### 1. Update OpenClaw Agent Configuration

Copy the model definitions from `openclaw_agent_config.json` to your OpenClaw config file:

```bash
# Find your OpenClaw config (usually one of these locations):
# ~/.openclaw/config.json
# ~/.config/openclaw/config.json
# Or check the OpenClaw UI settings

# Merge the "models" section from openclaw_agent_config.json into your config
```

The key section to add is:

```json
{
  "models": {
    "mode": "merge",
    "providers": {
      "llmrouter": {
        "baseUrl": "http://localhost:8000/v1",
        "apiKey": "dummy-key-not-used",
        "api": "openai-completions",
        "models": [
          {
            "id": "bedrock_smart_router",
            "name": "Bedrock Smart Router (Auto)",
            ...
          },
          {
            "id": "nova-micro",
            "name": "Nova Micro (Ultra-cheap)",
            ...
          },
          ...
        ]
      }
    }
  }
}
```

### 2. Restart OpenClaw Service

After updating the configuration:

```bash
sudo systemctl restart openclaw
sudo systemctl status openclaw
```

### 3. Verify Models in OpenClaw UI

1. Open OpenClaw chat interface
2. Click on the model selector dropdown
3. You should now see:
   - Bedrock Smart Router (Auto) - Intelligent routing
   - Nova Micro (Ultra-cheap) - Simple queries
   - Nova 2 Lite (Workhorse) - General purpose
   - Nova Pro (Reasoning) - Complex tasks
   - Claude Haiku (Fast) - Quick responses
   - Claude Sonnet (Complex) - Deep reasoning
   - Claude Opus (Deep Think) - Maximum capability

### 4. Test the Router

Try these queries to see the router in action:

**Simple query (should route to nova-micro):**
```
What is 2+2?
```

**Standard query (should route to nova-2-lite or haiku):**
```
Explain how HTTP works
```

**Reasoning query (should route to nova-pro or sonnet):**
```
Analyze the trade-offs between microservices and monolithic architecture
```

**Deep analysis (should route to opus):**
```
Provide a comprehensive analysis of distributed consensus algorithms
```

## Router Behavior

The Bedrock Smart Router uses:

1. **Rule-based complexity estimation** - Analyzes query patterns and length
2. **Memory-based learning** - Learns from past routing decisions
3. **Cost optimization** - Prefers cheaper models when appropriate

### Complexity Levels

- **Simple**: Short queries, greetings, basic facts → nova-micro
- **Standard**: General questions, explanations → nova-2-lite, haiku
- **Reasoning**: Analysis, comparisons, design → nova-pro, sonnet
- **Deep**: Comprehensive research, theoretical → opus, sonnet

### Memory System

The router learns over time:
- Tracks which models work best for similar queries
- Stores patterns in `~/.llmrouter/bedrock_router_memory.jsonl`
- Prioritizes memory suggestions over rules when available

## Troubleshooting

### Models not showing in UI

1. Check OpenClaw config file has the models section
2. Restart OpenClaw service
3. Clear browser cache and reload UI

### Router not working

1. Check service logs: `sudo journalctl -u openclaw -f`
2. Verify router is registered: `llmrouter list-routers | grep bedrock_smart_router`
3. Test with curl:
   ```bash
   curl -X POST http://localhost:8000/v1/chat/completions \
     -H "Content-Type: application/json" \
     -d '{
       "model": "bedrock_smart_router",
       "messages": [{"role": "user", "content": "test"}],
       "temperature": 0.7
     }'
   ```

### Bedrock API errors

1. Verify IAM role has Bedrock permissions
2. Check AWS region is correct (us-east-1)
3. Verify models are available in your region
4. Check service logs for detailed error messages

## Cost Optimization

The router automatically optimizes costs:

- **Nova Micro**: $0.035/M input, $0.14/M output (cheapest)
- **Nova 2 Lite**: $0.30/M input, $2.50/M output (balanced)
- **Nova Pro**: $0.80/M input, $3.20/M output (reasoning)
- **Haiku**: $0.80/M input, $4.00/M output (fast Claude)
- **Sonnet**: $3.00/M input, $15.00/M output (complex)
- **Opus**: $15.00/M input, $75.00/M output (maximum capability)

The router will use cheaper models whenever possible while maintaining quality.

## Advanced Configuration

### Disable Memory Learning

Edit `custom_routers/bedrock_smart_router/config.yaml`:

```yaml
memory:
  enabled: false
```

### Adjust Complexity Thresholds

Edit the router config:

```yaml
router:
  thresholds:
    simple_max_words: 10
    reasoning_min_words: 50
```

### Change Fallback Model

```yaml
router:
  fallback_model: nova-2-lite
```

## Support

For issues or questions:
1. Check service logs: `sudo journalctl -u openclaw -f`
2. Review router memory: `cat ~/.llmrouter/bedrock_router_memory.jsonl`
3. Test router directly: `llmrouter infer --router bedrock_smart_router --config custom_routers/bedrock_smart_router/config.yaml --query "test"`
