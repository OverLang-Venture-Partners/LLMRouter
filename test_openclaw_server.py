#!/usr/bin/env python3
"""
Test OpenClaw server Bedrock integration
"""

import asyncio
import sys
import json

async def test_server():
    print("=" * 60)
    print("Testing OpenClaw Server Bedrock Integration")
    print("=" * 60)
    
    # Import server components
    try:
        from openclaw_router.config import OpenClawConfig
        from openclaw_router.server import LLMBackend
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure you're in the LLMRouter directory")
        sys.exit(1)
    
    # Load config
    config_path = "configs/openclaw_bedrock_corrected.yaml"
    print(f"\n1. Loading config from: {config_path}")
    
    try:
        config = OpenClawConfig.from_yaml(config_path)
        print(f"✅ Config loaded")
        print(f"   LLMs: {list(config.llms.keys())}")
        print(f"   Router strategy: {config.router.strategy}")
        print(f"   API keys: {list(config.api_keys.keys())}")
    except Exception as e:
        print(f"❌ Config load failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Test backend
    print(f"\n2. Testing LLMBackend")
    backend = LLMBackend(config)
    
    # Test with nova-micro
    llm_name = "nova-micro"
    messages = [
        {"role": "user", "content": "What is 2+2?"}
    ]
    
    print(f"\n3. Calling {llm_name} with messages:")
    print(f"   {messages}")
    
    try:
        result = await backend.call(
            llm_name=llm_name,
            messages=messages,
            max_tokens=100,
            temperature=0.7,
            stream=False
        )
        
        print(f"\n4. Response received:")
        print(json.dumps(result, indent=2))
        
        # Check content
        if result.get("choices") and result["choices"][0].get("message"):
            content = result["choices"][0]["message"].get("content")
            if content:
                print(f"\n✅ SUCCESS: Got content: {content}")
            else:
                print(f"\n❌ FAIL: Content is null or empty")
                print(f"   Full message: {result['choices'][0]['message']}")
        else:
            print(f"\n❌ FAIL: No choices or message in response")
            
    except Exception as e:
        print(f"\n❌ Call failed: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    print("\n" + "=" * 60)
    print("Test complete")
    print("=" * 60)

if __name__ == "__main__":
    asyncio.run(test_server())
