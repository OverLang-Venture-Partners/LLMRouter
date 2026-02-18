# AWS Bedrock Credentials Setup Guide

This guide provides comprehensive instructions for setting up AWS credentials to use AWS Bedrock models with LLMRouter.

## Overview

AWS Bedrock uses standard AWS authentication mechanisms through the AWS SDK for Python (boto3). LLMRouter automatically uses boto3's credential resolution chain, which checks multiple sources in a specific order to find your AWS credentials.

## Prerequisites

Before setting up AWS credentials, ensure you have:

1. **AWS Account**: An active AWS account with access to AWS Bedrock
2. **Bedrock Model Access**: Enabled access to the Bedrock models you want to use in the AWS Bedrock console
3. **boto3 Installed**: The AWS SDK for Python (installed automatically with LLMRouter)

```bash
# Verify boto3 is installed
python -c "import boto3; print(boto3.__version__)"
```

## Credential Configuration Methods

AWS credentials can be configured using three methods, checked in the following order:

1. **Environment Variables** (recommended for development)
2. **AWS Credential Files** (recommended for production)
3. **IAM Roles** (recommended for AWS infrastructure)

### Method 1: Environment Variables

Environment variables are the simplest method for development and testing.

#### Required Variables

```bash
# AWS Access Key ID (required)
export AWS_ACCESS_KEY_ID="your-access-key-id"

# AWS Secret Access Key (required)
export AWS_SECRET_ACCESS_KEY="your-secret-access-key"

# AWS Region (required)
export AWS_DEFAULT_REGION="us-east-1"

# AWS Session Token (optional, only needed for temporary credentials)
export AWS_SESSION_TOKEN="your-session-token"
```

#### Getting Your Credentials

1. Log in to the [AWS Console](https://console.aws.amazon.com/)
2. Navigate to **IAM** → **Users** → Select your user
3. Go to **Security credentials** tab
4. Click **Create access key**
5. Save the Access Key ID and Secret Access Key securely

#### Setting Environment Variables

**Linux/macOS:**
```bash
# Temporary (current session only)
export AWS_ACCESS_KEY_ID="AKIAIOSFODNN7EXAMPLE"
export AWS_SECRET_ACCESS_KEY="wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"
export AWS_DEFAULT_REGION="us-east-1"

# Persistent (add to ~/.bashrc or ~/.zshrc)
echo 'export AWS_ACCESS_KEY_ID="AKIAIOSFODNN7EXAMPLE"' >> ~/.bashrc
echo 'export AWS_SECRET_ACCESS_KEY="wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"' >> ~/.bashrc
echo 'export AWS_DEFAULT_REGION="us-east-1"' >> ~/.bashrc
source ~/.bashrc
```

**Windows (PowerShell):**
```powershell
# Temporary (current session only)
$env:AWS_ACCESS_KEY_ID="AKIAIOSFODNN7EXAMPLE"
$env:AWS_SECRET_ACCESS_KEY="wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"
$env:AWS_DEFAULT_REGION="us-east-1"

# Persistent (system-wide)
[System.Environment]::SetEnvironmentVariable("AWS_ACCESS_KEY_ID", "AKIAIOSFODNN7EXAMPLE", "User")
[System.Environment]::SetEnvironmentVariable("AWS_SECRET_ACCESS_KEY", "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY", "User")
[System.Environment]::SetEnvironmentVariable("AWS_DEFAULT_REGION", "us-east-1", "User")
```

**Windows (Command Prompt):**
```cmd
# Temporary (current session only)
set AWS_ACCESS_KEY_ID=AKIAIOSFODNN7EXAMPLE
set AWS_SECRET_ACCESS_KEY=wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY
set AWS_DEFAULT_REGION=us-east-1

# Persistent (system-wide)
setx AWS_ACCESS_KEY_ID "AKIAIOSFODNN7EXAMPLE"
setx AWS_SECRET_ACCESS_KEY "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"
setx AWS_DEFAULT_REGION "us-east-1"
```

#### Verification

```bash
# Test AWS credentials
aws sts get-caller-identity

# Or using Python
python -c "import boto3; print(boto3.client('sts').get_caller_identity())"
```

### Method 2: AWS Credential Files

AWS credential files provide a more secure and organized way to manage credentials, especially when working with multiple AWS accounts or profiles.

#### File Locations

- **Linux/macOS**: `~/.aws/credentials` and `~/.aws/config`
- **Windows**: `%USERPROFILE%\.aws\credentials` and `%USERPROFILE%\.aws\config`

#### Setting Up Credential Files

**Step 1: Create the AWS directory**

```bash
# Linux/macOS
mkdir -p ~/.aws

# Windows (PowerShell)
New-Item -ItemType Directory -Force -Path "$env:USERPROFILE\.aws"
```

**Step 2: Create credentials file**

Create or edit `~/.aws/credentials` (Linux/macOS) or `%USERPROFILE%\.aws\credentials` (Windows):

```ini
[default]
aws_access_key_id = AKIAIOSFODNN7EXAMPLE
aws_secret_access_key = wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY

[production]
aws_access_key_id = AKIAI44QH8DHBEXAMPLE
aws_secret_access_key = je7MtGbClwBF/2Zp9Utk/h3yCo8nvbEXAMPLEKEY

[development]
aws_access_key_id = AKIAIOSFODNN7EXAMPLE
aws_secret_access_key = wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY
```

**Step 3: Create config file**

Create or edit `~/.aws/config` (Linux/macOS) or `%USERPROFILE%\.aws\config` (Windows):

```ini
[default]
region = us-east-1
output = json

[profile production]
region = us-west-2
output = json

[profile development]
region = us-east-1
output = json
```

#### Using Different Profiles

By default, LLMRouter uses the `[default]` profile. To use a different profile:

```bash
# Set the profile via environment variable
export AWS_PROFILE=production

# Or specify in your Python code
import boto3
session = boto3.Session(profile_name='production')
```

#### Using AWS CLI to Configure

The easiest way to set up credential files is using the AWS CLI:

```bash
# Install AWS CLI (if not already installed)
pip install awscli

# Configure default profile
aws configure
# Enter your Access Key ID, Secret Access Key, region, and output format

# Configure additional profiles
aws configure --profile production
aws configure --profile development
```

#### File Permissions

For security, ensure your credential files have restricted permissions:

```bash
# Linux/macOS
chmod 600 ~/.aws/credentials
chmod 600 ~/.aws/config

# Windows (PowerShell)
icacls "$env:USERPROFILE\.aws\credentials" /inheritance:r /grant:r "$env:USERNAME:R"
```

### Method 3: IAM Roles

IAM roles provide the most secure method for AWS infrastructure, as credentials are automatically managed and rotated by AWS.

#### When to Use IAM Roles

- Running on **EC2 instances**
- Running on **ECS containers**
- Running on **AWS Lambda**
- Running on **AWS Batch**
- Running on **Amazon SageMaker**

#### Setting Up IAM Roles

**Step 1: Create an IAM Role**

1. Go to [IAM Console](https://console.aws.amazon.com/iam/)
2. Click **Roles** → **Create role**
3. Select the trusted entity type:
   - **AWS service** for EC2, ECS, Lambda, etc.
   - Select the specific service (e.g., EC2)
4. Click **Next**

**Step 2: Attach Bedrock Permissions**

Attach a policy that grants Bedrock access. You can use:

- **AWS Managed Policy**: `AmazonBedrockFullAccess` (full access)
- **Custom Policy** (recommended for least privilege):

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "bedrock:InvokeModel",
        "bedrock:InvokeModelWithResponseStream"
      ],
      "Resource": [
        "arn:aws:bedrock:*::foundation-model/*"
      ]
    }
  ]
}
```

**Step 3: Attach Role to Resource**

- **EC2**: Attach the role when launching the instance or modify an existing instance
- **ECS**: Specify the role in the task definition
- **Lambda**: Specify the role when creating the function

**Step 4: Verify Role Assignment**

```bash
# On EC2 instance, check instance metadata
curl http://169.254.169.254/latest/meta-data/iam/security-credentials/

# Or using Python
import boto3
print(boto3.client('sts').get_caller_identity())
```

#### No Configuration Needed

When using IAM roles, LLMRouter automatically detects and uses the role credentials. No environment variables or credential files are needed.

## Regional Configuration

AWS Bedrock models are available in specific regions. You can configure regions at two levels:

### Global Default Region

Set a default region for all Bedrock models:

```bash
export AWS_DEFAULT_REGION="us-east-1"
```

Or in `~/.aws/config`:

```ini
[default]
region = us-east-1
```

### Per-Model Region

Override the default region for specific models in your LLM candidate JSON:

```json
{
  "claude-3-sonnet": {
    "model": "anthropic.claude-3-sonnet-20240229-v1:0",
    "service": "Bedrock",
    "aws_region": "us-east-1",
    ...
  },
  "titan-text-express": {
    "model": "amazon.titan-text-express-v1",
    "service": "Bedrock",
    "aws_region": "us-west-2",
    ...
  }
}
```

### Available Regions

Common regions with Bedrock support:
- `us-east-1` (US East - N. Virginia)
- `us-west-2` (US West - Oregon)
- `eu-west-1` (Europe - Ireland)
- `eu-central-1` (Europe - Frankfurt)
- `ap-southeast-1` (Asia Pacific - Singapore)
- `ap-northeast-1` (Asia Pacific - Tokyo)

Check the [AWS Bedrock documentation](https://docs.aws.amazon.com/bedrock/latest/userguide/models-regions.html) for the latest region availability.

## Security Best Practices

### 1. Use IAM Roles When Possible

IAM roles are the most secure option as credentials are:
- Automatically rotated by AWS
- Never stored in code or configuration files
- Scoped to specific resources

### 2. Use Least Privilege Permissions

Grant only the minimum permissions needed:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "bedrock:InvokeModel"
      ],
      "Resource": [
        "arn:aws:bedrock:us-east-1::foundation-model/anthropic.claude-3-sonnet-20240229-v1:0",
        "arn:aws:bedrock:us-east-1::foundation-model/amazon.titan-text-express-v1"
      ]
    }
  ]
}
```

### 3. Rotate Access Keys Regularly

If using access keys:
- Rotate keys every 90 days
- Delete unused keys
- Use AWS Secrets Manager for automated rotation

### 4. Never Commit Credentials to Version Control

Add to `.gitignore`:

```gitignore
# AWS credentials
.aws/
*.pem
*.key
*credentials*
*secret*
```

### 5. Use AWS Secrets Manager

For production applications, store credentials in AWS Secrets Manager:

```python
import boto3
import json

def get_credentials_from_secrets_manager(secret_name):
    client = boto3.client('secretsmanager')
    response = client.get_secret_value(SecretId=secret_name)
    return json.loads(response['SecretString'])
```

### 6. Enable MFA for IAM Users

Enable Multi-Factor Authentication for IAM users with access to Bedrock:

1. Go to IAM Console → Users → Select user
2. Security credentials tab → Assign MFA device
3. Follow the setup wizard

### 7. Monitor Credential Usage

Enable AWS CloudTrail to monitor API calls:

```bash
# View recent Bedrock API calls
aws cloudtrail lookup-events \
  --lookup-attributes AttributeKey=ResourceType,AttributeValue=AWS::Bedrock::Model \
  --max-results 10
```

## Troubleshooting

### Error: "AWS credentials not found"

**Cause**: No credentials configured in any of the three methods.

**Solution**:
1. Verify environment variables are set: `echo $AWS_ACCESS_KEY_ID`
2. Check credential files exist: `ls -la ~/.aws/`
3. Verify IAM role is attached (if on AWS infrastructure)

### Error: "The security token included in the request is invalid"

**Cause**: Credentials are invalid, expired, or incorrect.

**Solution**:
1. Verify credentials are correct
2. Check if temporary credentials have expired
3. Regenerate access keys in IAM console

### Error: "Access Denied" or "UnauthorizedOperation"

**Cause**: IAM user/role lacks Bedrock permissions.

**Solution**:
1. Verify IAM policy includes `bedrock:InvokeModel` permission
2. Check if Bedrock model access is enabled in AWS console
3. Verify resource ARNs in IAM policy match the models you're using

### Error: "Model not found" or "Model not available in region"

**Cause**: Model not available in the configured region.

**Solution**:
1. Check [model availability by region](https://docs.aws.amazon.com/bedrock/latest/userguide/models-regions.html)
2. Change `aws_region` in model configuration
3. Request model access in AWS Bedrock console for your region

### Error: "Rate limit exceeded"

**Cause**: Too many requests to Bedrock API.

**Solution**:
1. Implement exponential backoff retry logic
2. Request quota increase in AWS Service Quotas console
3. Distribute requests across multiple regions

### Debugging Tips

**Check current credentials:**
```bash
aws sts get-caller-identity
```

**Test Bedrock access:**
```bash
aws bedrock list-foundation-models --region us-east-1
```

**Enable boto3 debug logging:**
```python
import boto3
boto3.set_stream_logger('boto3.resources', level='DEBUG')
```

**Verify credential resolution order:**
```python
import boto3
session = boto3.Session()
credentials = session.get_credentials()
print(f"Access Key: {credentials.access_key}")
print(f"Method: {credentials.method}")
```

## Testing Your Setup

### Quick Test Script

Create a file `test_bedrock_credentials.py`:

```python
#!/usr/bin/env python3
import boto3
import sys

def test_credentials():
    """Test AWS credentials and Bedrock access."""
    
    print("Testing AWS credentials...")
    
    try:
        # Test basic AWS access
        sts = boto3.client('sts')
        identity = sts.get_caller_identity()
        print(f"✓ AWS credentials valid")
        print(f"  Account: {identity['Account']}")
        print(f"  User ARN: {identity['Arn']}")
        
        # Test Bedrock access
        bedrock = boto3.client('bedrock', region_name='us-east-1')
        models = bedrock.list_foundation_models()
        print(f"✓ Bedrock access granted")
        print(f"  Available models: {len(models['modelSummaries'])}")
        
        # List some available models
        print("\nSample available models:")
        for model in models['modelSummaries'][:5]:
            print(f"  - {model['modelId']}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

if __name__ == "__main__":
    success = test_credentials()
    sys.exit(0 if success else 1)
```

Run the test:

```bash
python test_bedrock_credentials.py
```

### Test with LLMRouter

```python
from llmrouter.utils.api_calling import call_api

# Test request
request = {
    "api_endpoint": "",  # Not used for Bedrock
    "query": "What is 2+2?",
    "model_name": "claude-3-haiku",
    "api_name": "anthropic.claude-3-haiku-20240307-v1:0",
    "service": "Bedrock",
    "aws_region": "us-east-1"
}

# Make API call
response = call_api(request)

# Check response
if "error" in response:
    print(f"Error: {response['error']}")
else:
    print(f"Success: {response['response']}")
    print(f"Tokens used: {response['token_num']}")
```

## Additional Resources

- [AWS Bedrock Documentation](https://docs.aws.amazon.com/bedrock/)
- [AWS Bedrock Model Access](https://docs.aws.amazon.com/bedrock/latest/userguide/model-access.html)
- [AWS Bedrock Regions](https://docs.aws.amazon.com/bedrock/latest/userguide/models-regions.html)
- [AWS IAM Best Practices](https://docs.aws.amazon.com/IAM/latest/UserGuide/best-practices.html)
- [Boto3 Credentials Documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/guide/credentials.html)
- [AWS CLI Configuration](https://docs.aws.amazon.com/cli/latest/userguide/cli-configure-files.html)

## Getting Help

If you encounter issues not covered in this guide:

1. Check the [LLMRouter GitHub Issues](https://github.com/ulab-uiuc/LLMRouter/issues)
2. Review [AWS Bedrock Troubleshooting](https://docs.aws.amazon.com/bedrock/latest/userguide/troubleshooting.html)
3. Join the [LLMRouter Slack Community](https://join.slack.com/t/llmrouteropen-ri04588/shared_invite/zt-3mkx82cut-A25v5yR52xVKi7_jm_YK_w)
4. Contact AWS Support for Bedrock-specific issues
