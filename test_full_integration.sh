#!/bin/bash
# Full integration test for OpenClaw + Bedrock

echo "============================================================"
echo "Full OpenClaw Bedrock Integration Test"
echo "============================================================"

# Test 1: Check if OpenClaw router server is running
echo ""
echo "1. Checking OpenClaw router server..."
if curl -s http://localhost:8000/health > /dev/null 2>&1; then
    echo "✅ Router server is running"
    curl -s http://localhost:8000/health | python3 -m json.tool
else
    echo "❌ Router server is NOT running on port 8000"
    echo "   Start with: sudo systemctl start openclaw"
    exit 1
fi

# Test 2: List available models
echo ""
echo "2. Listing available models..."
curl -s http://localhost:8000/v1/models | python3 -m json.tool

# Test 3: Test direct model call (nova-micro)
echo ""
echo "3. Testing direct model call (nova-micro)..."
RESPONSE=$(curl -s -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "nova-micro",
    "messages": [{"role": "user", "content": "What is 2+2?"}],
    "temperature": 0.7
  }')

echo "$RESPONSE" | python3 -m json.tool

# Check if content is present
CONTENT=$(echo "$RESPONSE" | python3 -c "import sys, json; data=json.load(sys.stdin); print(data.get('choices', [{}])[0].get('message', {}).get('content', 'NULL'))" 2>/dev/null)

if [ "$CONTENT" != "NULL" ] && [ -n "$CONTENT" ]; then
    echo "✅ Direct model call succeeded with content"
else
    echo "❌ Direct model call returned null content"
fi

# Test 4: Test router call (bedrock_smart_router)
echo ""
echo "4. Testing smart router call..."
RESPONSE=$(curl -s -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "bedrock_smart_router",
    "messages": [{"role": "user", "content": "Explain machine learning"}],
    "temperature": 0.7
  }')

echo "$RESPONSE" | python3 -m json.tool

CONTENT=$(echo "$RESPONSE" | python3 -c "import sys, json; data=json.load(sys.stdin); print(data.get('choices', [{}])[0].get('message', {}).get('content', 'NULL'))" 2>/dev/null)

if [ "$CONTENT" != "NULL" ] && [ -n "$CONTENT" ]; then
    echo "✅ Router call succeeded with content"
else
    echo "❌ Router call returned null content"
fi

# Test 5: Check OpenClaw agent config
echo ""
echo "5. Checking OpenClaw agent configuration..."
if [ -f ~/.openclaw/config.json ]; then
    echo "Found agent config at: ~/.openclaw/config.json"
    echo "Models configured:"
    cat ~/.openclaw/config.json | python3 -c "import sys, json; data=json.load(sys.stdin); models=data.get('models', {}).get('providers', {}).get('llmrouter', {}).get('models', []); print(f'  Count: {len(models)}'); [print(f'  - {m.get(\"id\")}: {m.get(\"name\")}') for m in models]" 2>/dev/null || echo "  Unable to parse config"
else
    echo "❌ No agent config found at ~/.openclaw/config.json"
    echo "   The OpenClaw agent needs to be configured with the router models"
fi

echo ""
echo "============================================================"
echo "Integration test complete"
echo "============================================================"
echo ""
echo "NEXT STEPS:"
echo "1. If router server tests passed but agent has no models:"
echo "   - Copy model definitions from openclaw_router/openclaw_agent_config.json"
echo "   - Merge into ~/.openclaw/config.json"
echo "   - Restart OpenClaw agent"
echo ""
echo "2. If router server tests failed:"
echo "   - Check service logs: sudo journalctl -u openclaw -f"
echo "   - Verify config: configs/openclaw_bedrock_corrected.yaml"
echo ""
