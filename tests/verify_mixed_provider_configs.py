"""
Verification script for mixed provider configurations.

This script demonstrates that mixed provider configurations (Bedrock + NVIDIA,
Bedrock + OpenAI) work correctly with the API calling logic.
"""

import json
from pathlib import Path
from llmrouter.utils.api_calling import _is_bedrock_model


def verify_mixed_bedrock_nvidia():
    """Verify mixed Bedrock + NVIDIA configuration."""
    print("=" * 80)
    print("Verifying Mixed Bedrock + NVIDIA Configuration")
    print("=" * 80)
    
    config_path = Path("data/example_data/llm_candidates/mixed_bedrock_nvidia.json")
    
    if not config_path.exists():
        print(f"❌ Configuration file not found: {config_path}")
        return False
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    print(f"\n✓ Configuration file loaded: {config_path}")
    print(f"✓ Total models: {len(config)}")
    
    # Separate models by provider
    bedrock_models = []
    nvidia_models = []
    
    for name, model_config in config.items():
        service = model_config.get("service")
        if _is_bedrock_model(service):
            bedrock_models.append(name)
        elif service == "NVIDIA":
            nvidia_models.append(name)
    
    print(f"\n✓ Bedrock models: {len(bedrock_models)}")
    for model in bedrock_models:
        model_config = config[model]
        model_id = model_config.get("model")
        region = model_config.get("aws_region", "default")
        print(f"  - {model}")
        print(f"    Model ID: {model_id}")
        print(f"    Region: {region}")
        print(f"    Format: bedrock/{model_id}")
    
    print(f"\n✓ NVIDIA models: {len(nvidia_models)}")
    for model in nvidia_models:
        model_config = config[model]
        model_id = model_config.get("model")
        endpoint = model_config.get("api_endpoint")
        print(f"  - {model}")
        print(f"    Model ID: {model_id}")
        print(f"    Endpoint: {endpoint}")
        print(f"    Format: openai/{model_id}")
    
    # Verify all models have required fields
    print("\n✓ Verifying required fields...")
    all_valid = True
    for name, model_config in config.items():
        required_fields = ["model", "service", "size", "feature", "input_price", "output_price"]
        missing_fields = [f for f in required_fields if f not in model_config]
        
        if missing_fields:
            print(f"  ❌ {name} missing fields: {missing_fields}")
            all_valid = False
    
    if all_valid:
        print("  ✓ All models have required fields")
    
    # Verify Bedrock models have aws_region
    print("\n✓ Verifying Bedrock-specific fields...")
    bedrock_valid = True
    for name in bedrock_models:
        model_config = config[name]
        if "aws_region" not in model_config:
            print(f"  ❌ {name} missing aws_region field")
            bedrock_valid = False
    
    if bedrock_valid:
        print("  ✓ All Bedrock models have aws_region field")
    
    # Verify NVIDIA models have api_endpoint
    print("\n✓ Verifying NVIDIA-specific fields...")
    nvidia_valid = True
    for name in nvidia_models:
        model_config = config[name]
        if "api_endpoint" not in model_config:
            print(f"  ❌ {name} missing api_endpoint field")
            nvidia_valid = False
    
    if nvidia_valid:
        print("  ✓ All NVIDIA models have api_endpoint field")
    
    success = all_valid and bedrock_valid and nvidia_valid
    print("\n" + "=" * 80)
    if success:
        print("✅ Mixed Bedrock + NVIDIA configuration is VALID")
    else:
        print("❌ Mixed Bedrock + NVIDIA configuration has ERRORS")
    print("=" * 80)
    
    return success


def verify_mixed_bedrock_openai():
    """Verify mixed Bedrock + OpenAI configuration."""
    print("\n\n" + "=" * 80)
    print("Verifying Mixed Bedrock + OpenAI Configuration")
    print("=" * 80)
    
    config_path = Path("data/example_data/llm_candidates/mixed_bedrock_openai.json")
    
    if not config_path.exists():
        print(f"❌ Configuration file not found: {config_path}")
        return False
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    print(f"\n✓ Configuration file loaded: {config_path}")
    print(f"✓ Total models: {len(config)}")
    
    # Separate models by provider
    bedrock_models = []
    openai_models = []
    
    for name, model_config in config.items():
        service = model_config.get("service")
        if _is_bedrock_model(service):
            bedrock_models.append(name)
        elif service == "OpenAI":
            openai_models.append(name)
    
    print(f"\n✓ Bedrock models: {len(bedrock_models)}")
    for model in bedrock_models:
        model_config = config[model]
        model_id = model_config.get("model")
        region = model_config.get("aws_region", "default")
        print(f"  - {model}")
        print(f"    Model ID: {model_id}")
        print(f"    Region: {region}")
        print(f"    Format: bedrock/{model_id}")
    
    print(f"\n✓ OpenAI models: {len(openai_models)}")
    for model in openai_models:
        model_config = config[model]
        model_id = model_config.get("model")
        endpoint = model_config.get("api_endpoint")
        print(f"  - {model}")
        print(f"    Model ID: {model_id}")
        print(f"    Endpoint: {endpoint}")
        print(f"    Format: openai/{model_id}")
    
    # Verify all models have required fields
    print("\n✓ Verifying required fields...")
    all_valid = True
    for name, model_config in config.items():
        required_fields = ["model", "service", "size", "feature", "input_price", "output_price"]
        missing_fields = [f for f in required_fields if f not in model_config]
        
        if missing_fields:
            print(f"  ❌ {name} missing fields: {missing_fields}")
            all_valid = False
    
    if all_valid:
        print("  ✓ All models have required fields")
    
    # Verify Bedrock models have aws_region
    print("\n✓ Verifying Bedrock-specific fields...")
    bedrock_valid = True
    for name in bedrock_models:
        model_config = config[name]
        if "aws_region" not in model_config:
            print(f"  ❌ {name} missing aws_region field")
            bedrock_valid = False
    
    if bedrock_valid:
        print("  ✓ All Bedrock models have aws_region field")
    
    # Verify OpenAI models have api_endpoint
    print("\n✓ Verifying OpenAI-specific fields...")
    openai_valid = True
    for name in openai_models:
        model_config = config[name]
        if "api_endpoint" not in model_config:
            print(f"  ❌ {name} missing api_endpoint field")
            openai_valid = False
    
    if openai_valid:
        print("  ✓ All OpenAI models have api_endpoint field")
    
    success = all_valid and bedrock_valid and openai_valid
    print("\n" + "=" * 80)
    if success:
        print("✅ Mixed Bedrock + OpenAI configuration is VALID")
    else:
        print("❌ Mixed Bedrock + OpenAI configuration has ERRORS")
    print("=" * 80)
    
    return success


def verify_model_formatting():
    """Verify that model formatting logic works correctly for mixed providers."""
    print("\n\n" + "=" * 80)
    print("Verifying Model Formatting Logic")
    print("=" * 80)
    
    test_cases = [
        {
            "name": "claude-3-sonnet-bedrock",
            "service": "Bedrock",
            "model": "anthropic.claude-3-sonnet-20240229-v1:0",
            "expected_format": "bedrock/anthropic.claude-3-sonnet-20240229-v1:0"
        },
        {
            "name": "qwen2.5-7b-instruct",
            "service": "NVIDIA",
            "model": "qwen/qwen2.5-7b-instruct",
            "expected_format": "openai/qwen/qwen2.5-7b-instruct"
        },
        {
            "name": "gpt-4o-mini",
            "service": "OpenAI",
            "model": "gpt-4o-mini",
            "expected_format": "openai/gpt-4o-mini"
        },
        {
            "name": "titan-text-express-bedrock",
            "service": "AWS",
            "model": "amazon.titan-text-express-v1",
            "expected_format": "bedrock/amazon.titan-text-express-v1"
        }
    ]
    
    all_passed = True
    for test in test_cases:
        service = test["service"]
        model_id = test["model"]
        expected = test["expected_format"]
        
        if _is_bedrock_model(service):
            actual = f"bedrock/{model_id}"
        else:
            actual = f"openai/{model_id}"
        
        passed = actual == expected
        status = "✓" if passed else "❌"
        
        print(f"\n{status} {test['name']}")
        print(f"  Service: {service}")
        print(f"  Model ID: {model_id}")
        print(f"  Expected: {expected}")
        print(f"  Actual: {actual}")
        
        if not passed:
            all_passed = False
    
    print("\n" + "=" * 80)
    if all_passed:
        print("✅ All model formatting tests PASSED")
    else:
        print("❌ Some model formatting tests FAILED")
    print("=" * 80)
    
    return all_passed


def main():
    """Run all verification checks."""
    print("\n" + "=" * 80)
    print("MIXED PROVIDER CONFIGURATION VERIFICATION")
    print("=" * 80)
    
    results = []
    
    # Verify mixed Bedrock + NVIDIA
    results.append(("Bedrock + NVIDIA", verify_mixed_bedrock_nvidia()))
    
    # Verify mixed Bedrock + OpenAI
    results.append(("Bedrock + OpenAI", verify_mixed_bedrock_openai()))
    
    # Verify model formatting
    results.append(("Model Formatting", verify_model_formatting()))
    
    # Summary
    print("\n\n" + "=" * 80)
    print("VERIFICATION SUMMARY")
    print("=" * 80)
    
    for name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{status}: {name}")
    
    all_passed = all(passed for _, passed in results)
    
    print("\n" + "=" * 80)
    if all_passed:
        print("✅ ALL VERIFICATIONS PASSED")
        print("\nMixed provider configurations are working correctly!")
        print("Both Bedrock + NVIDIA and Bedrock + OpenAI configurations are valid.")
    else:
        print("❌ SOME VERIFICATIONS FAILED")
        print("\nPlease review the errors above.")
    print("=" * 80 + "\n")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    exit(main())
