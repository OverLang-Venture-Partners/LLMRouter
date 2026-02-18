#!/usr/bin/env python3
"""
Test script to inspect what messages OpenClaw is actually sending
"""
import json
import sys

# Simulate receiving a request from OpenClaw
test_messages = [
    {"role": "user", "content": "Hello"},
    {"role": "assistant", "content": [
        {"type": "text", "text": "Hi there!"},
        {"type": "tool_use", "id": "tool_123", "name": "search", "input": {"query": "test"}}
    ]},
    {"role": "user", "content": [
        {"type": "tool_result", "tool_use_id": "tool_123", "content": "Result here"},
        {"type": "text", "text": "What does this mean?"}
    ]},
]

def force_text_only(msgs):
    """Aggressively extract only text content, skip tool blocks entirely"""
    result = []
    for msg in msgs:
        content = msg.get("content", "")
        if isinstance(content, list):
            # Extract only text blocks
            text_parts = [
                b.get("text", "") for b in content 
                if isinstance(b, dict) and b.get("type") == "text" and b.get("text")
            ]
            text = " ".join(text_parts).strip()
            if text:  # Only include if there's actual text content
                result.append({"role": msg["role"], "content": text})
            # else: skip the message entirely (was pure tool block)
        elif isinstance(content, str) and content.strip():
            result.append({"role": msg["role"], "content": content})
    return result

print("Original messages:")
print(json.dumps(test_messages, indent=2))

print("\n" + "="*60)
print("After force_text_only:")
cleaned = force_text_only(test_messages)
print(json.dumps(cleaned, indent=2))

print("\n" + "="*60)
print(f"Original: {len(test_messages)} messages")
print(f"Cleaned: {len(cleaned)} messages")
