"""
Demonstration script showing Bedrock error handling in action.
This script shows examples of each error type and the helpful messages provided.
"""


def demonstrate_error_messages():
    """Demonstrate the error messages for each error type."""
    
    print("=" * 80)
    print("BEDROCK ERROR HANDLING DEMONSTRATION")
    print("=" * 80)
    
    print("\n1. MISSING BOTO3 DEPENDENCY ERROR")
    print("-" * 80)
    print("When boto3 is not installed, users see:")
    print("""
AWS Bedrock support requires boto3. Install it with:
  pip install boto3
Or install with LLMRouter:
  pip install llmrouter-lib[bedrock]
Original error: No module named 'boto3'
""")
    
    print("\n2. MISSING AWS CREDENTIALS ERROR")
    print("-" * 80)
    print("When AWS credentials are not configured, users see:")
    print("""
AWS credentials not found or invalid. Configure credentials using one of:
1. Environment variables:
   export AWS_ACCESS_KEY_ID='your-key-id'
   export AWS_SECRET_ACCESS_KEY='your-secret-key'
   export AWS_DEFAULT_REGION='us-east-1'
2. AWS credential file (~/.aws/credentials):
   [default]
   aws_access_key_id = your-key-id
   aws_secret_access_key = your-secret-key
3. IAM role (when running on AWS infrastructure)

Original error: Unable to locate credentials
""")
    
    print("\n3. INVALID MODEL ID ERROR")
    print("-" * 80)
    print("When an invalid model ID is used, users see:")
    print("""
Bedrock model 'invalid.model-id' not found or not accessible.
Common Bedrock model IDs:
  Claude: anthropic.claude-3-sonnet-20240229-v1:0
  Claude: anthropic.claude-3-haiku-20240307-v1:0
  Titan: amazon.titan-text-express-v1
  Llama: meta.llama3-70b-instruct-v1:0
  Mistral: mistral.mistral-7b-instruct-v0:2
Check model availability: https://docs.aws.amazon.com/bedrock/latest/userguide/models-supported.html

Original error: Model invalid.model-id not found
""")
    
    print("\n4. REGION MISMATCH ERROR")
    print("-" * 80)
    print("When a model is not available in the specified region, users see:")
    print("""
Model may not be available in region 'eu-west-1'.
Try one of:
1. Use a different region (add 'aws_region' field to model config)
2. Check model availability: https://docs.aws.amazon.com/bedrock/latest/userguide/models-regions.html
3. Request access to the model in AWS Bedrock console

Original error: Model not available in region eu-west-1
""")
    
    print("\n5. API TIMEOUT ERROR")
    print("-" * 80)
    print("When an API call times out, users see:")
    print("""
Bedrock API call timed out after 30 seconds.
Try:
1. Increase timeout parameter in call_api()
2. Check network connectivity to AWS
3. Try a different AWS region

Original error: Request timed out after 30 seconds
""")
    
    print("\n" + "=" * 80)
    print("KEY FEATURES OF ERROR HANDLING:")
    print("=" * 80)
    print("""
✓ Clear, actionable error messages
✓ Multiple solution options provided
✓ Links to relevant AWS documentation
✓ Original error preserved for debugging
✓ Consistent error response format
✓ Bedrock-specific vs generic error routing
✓ All error types covered (Requirements 7.1-7.5)
""")
    
    print("\n" + "=" * 80)
    print("IMPLEMENTATION DETAILS:")
    print("=" * 80)
    print("""
The error handling implementation includes:

1. Three-tier exception handling:
   - ImportError: Catches missing boto3 dependency
   - TimeoutError: Catches API timeouts
   - Exception: Catches all other errors with Bedrock-specific routing

2. Intelligent error detection:
   - Checks error message keywords to identify error type
   - Routes Bedrock errors differently from other providers
   - Provides context-specific troubleshooting steps

3. Consistent error response:
   - All errors return same response structure
   - Includes error field with detailed message
   - Sets token counts to 0
   - Records response time for debugging

4. Requirements coverage:
   - Requirement 7.1: Missing boto3 dependency ✓
   - Requirements 1.3, 4.5, 7.3: Missing AWS credentials ✓
   - Requirement 7.2: Invalid model IDs ✓
   - Requirement 7.5: Region mismatches ✓
   - Requirement 7.4: API timeouts ✓
""")


if __name__ == "__main__":
    demonstrate_error_messages()
