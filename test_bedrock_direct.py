#!/usr/bin/env python3
"""
Direct test of Bedrock API call to diagnose null response issue
"""

import sys
import json

# Test 1: Direct LiteLLM call
print("=" * 60)
print("TEST 1: Direct LiteLLM Bedrock Call")
print("=" * 60)

try:
    from litellm import completion
    
    messages = [
        {"role": "user", "content": "What is 2+2?"}
    ]
    
    print(f"Calling Bedrock with messages: {messages}")
    
    response = completion(
        model="bedrock/us.amazon.nova-micro-v1:0",
        messages=messages,
        max_tokens=100,
        temperature=0.7,
        aws_region_name="us-west-2"
    )
    
    print(f"\nResponse object type: {type(response)}")
    print(f"Response: {response}")
    
    if hasattr(response, 'choices') and response.choices:
        print(f"\nChoices: {response.choices}")
        print(f"First choice: {response.choices[0]}")
        print(f"Message: {response.choices[0].message}")
        print(f"Content: {response.choices[0].message.content}")
    
    if hasattr(response, 'usage'):
        print(f"\nUsage: {response.usage}")
    
    print("\n✅ LiteLLM call succeeded")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Install with: pip install litellm boto3")
    sys.exit(1)
except Exception as e:
    print(f"❌ Error: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 2: Check what OpenClaw server would do
print("\n" + "=" * 60)
print("TEST 2: Simulate OpenClaw Server Processing")
print("=" * 60)

try:
    # Simulate the conversion
    openai_format = {
        "id": response.id if hasattr(response, 'id') else "chatcmpl-test",
        "object": "chat.completion",
        "model": "us.amazon.nova-micro-v1:0",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": response.choices[0].message.content if response.choices else ""
            },
            "finish_reason": response.choices[0].finish_reason if response.choices else "stop"
        }],
        "usage": {
            "prompt_tokens": response.usage.prompt_tokens if hasattr(response, 'usage') else 0,
            "completion_tokens": response.usage.completion_tokens if hasattr(response, 'usage') else 0,
            "total_tokens": response.usage.total_tokens if hasattr(response, 'usage') else 0
        }
    }
    
    print(f"OpenAI format response:")
    print(json.dumps(openai_format, indent=2))
    
    content = openai_format["choices"][0]["message"]["content"]
    if content:
        print(f"\n✅ Content present: {content}")
    else:
        print(f"\n❌ Content is empty or null!")
        
except Exception as e:
    print(f"❌ Conversion error: {e}")
    import traceback
    traceback.print_exc()

# Test 3: Check environment
print("\n" + "=" * 60)
print("TEST 3: Environment Check")
print("=" * 60)

import os
print(f"AWS_REGION: {os.getenv('AWS_REGION', 'not set')}")
print(f"AWS_DEFAULT_REGION: {os.getenv('AWS_DEFAULT_REGION', 'not set')}")
print(f"AWS_ACCESS_KEY_ID: {'set' if os.getenv('AWS_ACCESS_KEY_ID') else 'not set (using IAM role)'}")

try:
    import boto3
    print(f"boto3 version: {boto3.__version__}")
    
    # Try to get credentials
    session = boto3.Session()
    credentials = session.get_credentials()
    if credentials:
        print(f"✅ AWS credentials available (via {credentials.method})")
    else:
        print(f"❌ No AWS credentials found")
        
except Exception as e:
    print(f"❌ boto3 check failed: {e}")

print("\n" + "=" * 60)
print("DIAGNOSIS COMPLETE")
print("=" * 60)
