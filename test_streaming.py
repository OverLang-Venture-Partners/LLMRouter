#!/usr/bin/env python3
"""
Test streaming vs non-streaming responses
"""

import requests
import json
import sys

BASE_URL = "http://localhost:8000"

print("=" * 60)
print("Testing Streaming vs Non-Streaming")
print("=" * 60)

# Test 1: Non-streaming (what curl does)
print("\n1. Non-streaming request (stream=false)")
print("-" * 60)

response = requests.post(
    f"{BASE_URL}/v1/chat/completions",
    json={
        "model": "nova-micro",
        "messages": [{"role": "user", "content": "What is 2+2?"}],
        "temperature": 0.7,
        "stream": False
    },
    timeout=30
)

print(f"Status: {response.status_code}")
print(f"Content-Type: {response.headers.get('content-type')}")
print(f"Response:")
try:
    data = response.json()
    print(json.dumps(data, indent=2))
    content = data.get("choices", [{}])[0].get("message", {}).get("content")
    if content:
        print(f"\n✅ Non-streaming works: {content[:50]}...")
    else:
        print(f"\n❌ Non-streaming returned null content")
except Exception as e:
    print(f"❌ Failed to parse: {e}")
    print(response.text[:500])

# Test 2: Streaming (what OpenClaw agent does)
print("\n\n2. Streaming request (stream=true)")
print("-" * 60)

response = requests.post(
    f"{BASE_URL}/v1/chat/completions",
    json={
        "model": "nova-micro",
        "messages": [{"role": "user", "content": "What is 2+2?"}],
        "temperature": 0.7,
        "stream": True
    },
    stream=True,
    timeout=30
)

print(f"Status: {response.status_code}")
print(f"Content-Type: {response.headers.get('content-type')}")
print(f"Response chunks:")

chunks_received = 0
content_parts = []
has_done = False

try:
    for line in response.iter_lines():
        if line:
            line_str = line.decode('utf-8')
            print(f"  {line_str[:100]}")
            
            if line_str.startswith("data: "):
                chunks_received += 1
                data_str = line_str[6:]  # Remove "data: " prefix
                
                if data_str == "[DONE]":
                    has_done = True
                    print("  ✅ Received [DONE] marker")
                else:
                    try:
                        chunk_data = json.loads(data_str)
                        if chunk_data.get("choices"):
                            delta = chunk_data["choices"][0].get("delta", {})
                            if "content" in delta:
                                content_parts.append(delta["content"])
                    except json.JSONDecodeError as e:
                        print(f"  ❌ Failed to parse chunk: {e}")
    
    full_content = "".join(content_parts)
    
    print(f"\nSummary:")
    print(f"  Chunks received: {chunks_received}")
    print(f"  Has [DONE]: {has_done}")
    print(f"  Content length: {len(full_content)}")
    
    if full_content:
        print(f"\n✅ Streaming works: {full_content[:100]}...")
    else:
        print(f"\n❌ Streaming returned no content")
        
except Exception as e:
    print(f"❌ Streaming failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("DIAGNOSIS:")
print("=" * 60)

if chunks_received == 0:
    print("❌ No SSE chunks received - server not returning text/event-stream")
elif not has_done:
    print("⚠️  Chunks received but no [DONE] marker - client may hang")
elif not content_parts:
    print("❌ Chunks received but no content in deltas")
else:
    print("✅ Streaming appears to be working correctly")

print("\nIf OpenClaw agent shows blank:")
print("1. Check if agent is sending stream=true")
print("2. Check if server returns Content-Type: text/event-stream")
print("3. Check if chunks have correct format: data: {json}\\n\\n")
print("4. Check if [DONE] marker is sent")
