# Requirements Document

## Introduction

This document specifies the requirements for adding AWS Bedrock support to the LLMRouter project. AWS Bedrock is a fully managed service that provides access to foundation models from leading AI companies through a unified API. This feature will enable LLMRouter to route queries to Bedrock-hosted models (Claude, Titan, Llama, Mistral, etc.) alongside existing providers (NVIDIA, OpenAI, Anthropic), expanding the available model pool and providing users with more routing options.

## Glossary

- **AWS_Bedrock**: Amazon Web Services Bedrock service, a fully managed service providing access to foundation models via API
- **LLMRouter**: The intelligent routing system that dynamically selects the most suitable LLM for each query
- **LiteLLM**: The unified LLM API library used by LLMRouter for making API calls to various providers
- **API_Calling_Module**: The module in LLMRouter responsible for making API calls to LLM providers (llmrouter/utils/api_calling.py)
- **LLM_Candidate_JSON**: The JSON configuration file that defines available LLM models and their metadata (default_llm.json)
- **Router_Config**: The YAML configuration file that defines router settings and parameters
- **Service_Provider**: The backend service providing LLM access (e.g., NVIDIA, OpenAI, Anthropic, AWS Bedrock)
- **Boto3**: The AWS SDK for Python, used for authenticating and interacting with AWS services
- **AWS_Credentials**: Authentication credentials for AWS services (access key ID, secret access key, session token, region)
- **Foundation_Model**: Pre-trained large language models available through AWS Bedrock (e.g., Claude, Titan, Llama)
- **Model_ID**: The unique identifier for a Bedrock model (e.g., "anthropic.claude-3-sonnet-20240229-v1:0")
- **Streaming_Response**: Real-time token-by-token response generation from LLM APIs
- **Embeddings**: Vector representations of text used for semantic similarity and routing decisions

## Requirements

### Requirement 1: AWS Bedrock Provider Integration

**User Story:** As a developer, I want to configure AWS Bedrock as a service provider in LLMRouter, so that I can route queries to Bedrock-hosted models.

#### Acceptance Criteria

1. WHEN a user adds "Bedrock" or "AWS" as a service in the LLM_Candidate_JSON, THE LLMRouter SHALL recognize it as a valid Service_Provider
2. WHEN the API_Calling_Module processes a request with service "Bedrock", THE system SHALL use AWS Bedrock-specific authentication and API calling logic
3. WHEN AWS_Credentials are not configured, THE system SHALL return a descriptive error message indicating missing AWS credentials
4. THE system SHALL support AWS_Credentials configuration through environment variables (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_SESSION_TOKEN, AWS_DEFAULT_REGION)
5. THE system SHALL support AWS_Credentials configuration through AWS credential files (~/.aws/credentials, ~/.aws/config)
6. THE system SHALL support AWS_Credentials configuration through IAM roles when running on AWS infrastructure

### Requirement 2: Bedrock Model Configuration

**User Story:** As a developer, I want to configure Bedrock models in the LLM candidate JSON, so that routers can select from Bedrock-hosted models.

#### Acceptance Criteria

1. WHEN a user adds a Bedrock model to LLM_Candidate_JSON with service "Bedrock", THE system SHALL validate the Model_ID format
2. THE LLM_Candidate_JSON SHALL support all required fields for Bedrock models: model (Model_ID), service, size, feature, input_price, output_price
3. WHEN a Bedrock model entry includes an aws_region field, THE system SHALL use that region for API calls to that specific model
4. WHEN a Bedrock model entry does not include an aws_region field, THE system SHALL use the default AWS region from credentials configuration
5. THE system SHALL support Model_IDs for all major Bedrock Foundation_Models including Claude, Titan, Llama, Mistral, Cohere, and AI21 models

### Requirement 3: Bedrock API Integration

**User Story:** As a developer, I want LLMRouter to make API calls to AWS Bedrock, so that queries can be routed to Bedrock models.

#### Acceptance Criteria

1. WHEN the API_Calling_Module receives a request for a Bedrock model, THE system SHALL use Boto3 to invoke the Bedrock runtime API
2. WHEN making a Bedrock API call, THE system SHALL format the request payload according to the specific Foundation_Model's requirements
3. WHEN a Bedrock API call succeeds, THE system SHALL parse the response and extract the generated text
4. WHEN a Bedrock API call fails, THE system SHALL return an error message with details from the AWS error response
5. THE system SHALL track token usage for Bedrock API calls by parsing the response metadata
6. THE system SHALL measure and record response time for Bedrock API calls

### Requirement 4: Authentication and Credentials Management

**User Story:** As a developer, I want to authenticate with AWS Bedrock using standard AWS credential mechanisms, so that I can securely access Bedrock models.

#### Acceptance Criteria

1. WHEN AWS_Credentials are configured via environment variables, THE system SHALL use those credentials for Bedrock authentication
2. WHEN AWS_Credentials are configured via AWS credential files, THE system SHALL use the default profile unless a specific profile is specified
3. WHEN running on AWS infrastructure with IAM roles, THE system SHALL automatically use the instance/container role credentials
4. WHEN multiple AWS regions are needed, THE system SHALL support per-model region configuration
5. WHEN AWS_Credentials are invalid or expired, THE system SHALL return a descriptive error message indicating authentication failure

### Requirement 5: Bedrock-Specific Features Support

**User Story:** As a developer, I want to use Bedrock-specific features like streaming and embeddings, so that I can leverage the full capabilities of Bedrock models.

#### Acceptance Criteria

1. WHEN a request specifies streaming mode for a Bedrock model, THE system SHALL use the Bedrock streaming API and yield tokens as they are generated
2. WHEN using Bedrock embedding models, THE system SHALL support the embedding API endpoint and return vector representations
3. WHEN a Bedrock model supports system prompts, THE system SHALL include the system_prompt field in the API request
4. WHEN a Bedrock model has specific parameter requirements (e.g., anthropic_version for Claude), THE system SHALL include those parameters in the request
5. THE system SHALL support common inference parameters for Bedrock models: temperature, top_p, max_tokens, stop_sequences

### Requirement 6: Model Metadata and Pricing

**User Story:** As a developer, I want to configure pricing and metadata for Bedrock models, so that routers can make cost-aware routing decisions.

#### Acceptance Criteria

1. WHEN a Bedrock model is configured in LLM_Candidate_JSON, THE system SHALL store input_price and output_price per million tokens
2. WHEN calculating routing costs, THE system SHALL use Bedrock model pricing from the LLM_Candidate_JSON
3. THE system SHALL support the feature field for Bedrock models to describe model capabilities
4. THE system SHALL support the size field for Bedrock models to indicate parameter count
5. WHEN generating LLM embeddings for routing, THE system SHALL include Bedrock model metadata in the embedding generation process

### Requirement 7: Error Handling and Validation

**User Story:** As a developer, I want clear error messages when Bedrock integration fails, so that I can quickly diagnose and fix configuration issues.

#### Acceptance Criteria

1. WHEN Boto3 is not installed, THE system SHALL return an error message indicating the missing dependency and installation instructions
2. WHEN a Bedrock Model_ID is invalid or not accessible, THE system SHALL return an error message with the invalid Model_ID and available models
3. WHEN AWS_Credentials are missing or invalid, THE system SHALL return an error message with instructions for configuring credentials
4. WHEN a Bedrock API call times out, THE system SHALL return an error message indicating timeout and suggest increasing the timeout parameter
5. WHEN a Bedrock model is not available in the configured region, THE system SHALL return an error message indicating the region mismatch

### Requirement 8: Backward Compatibility

**User Story:** As a developer, I want Bedrock support to integrate seamlessly with existing LLMRouter functionality, so that existing configurations and workflows continue to work.

#### Acceptance Criteria

1. WHEN Bedrock models are added to LLM_Candidate_JSON, THE system SHALL continue to support existing NVIDIA, OpenAI, and Anthropic models
2. WHEN using the service-specific dict format for API_KEYS, THE system SHALL support a "Bedrock" key for AWS credentials alongside existing service keys
3. WHEN routers are trained with mixed provider models, THE system SHALL route to Bedrock models using the same routing logic as other providers
4. WHEN using the chat interface, THE system SHALL support Bedrock models in the model selection dropdown
5. THE system SHALL maintain the existing API_Calling_Module interface while adding Bedrock support

### Requirement 9: Configuration Examples and Documentation

**User Story:** As a developer, I want clear examples of how to configure Bedrock models, so that I can quickly set up and use Bedrock integration.

#### Acceptance Criteria

1. THE system SHALL provide example LLM_Candidate_JSON entries for common Bedrock models (Claude, Titan, Llama)
2. THE system SHALL provide example Router_Config entries showing Bedrock model configuration
3. THE system SHALL provide documentation on setting up AWS_Credentials for Bedrock access
4. THE system SHALL provide examples of mixed-provider configurations with Bedrock, NVIDIA, and OpenAI models
5. THE system SHALL provide examples of using Bedrock models in the data generation pipeline

### Requirement 10: Testing and Validation

**User Story:** As a developer, I want to test Bedrock integration, so that I can verify it works correctly before using it in production.

#### Acceptance Criteria

1. THE system SHALL provide a test script or command to validate Bedrock connectivity and authentication
2. WHEN running inference with --route-only flag, THE system SHALL validate Bedrock model configuration without making API calls
3. WHEN testing Bedrock models, THE system SHALL support the same testing patterns as existing providers (single query, batch, chat interface)
4. THE system SHALL log Bedrock API call details (model, region, tokens, response time) for debugging
5. THE system SHALL provide clear success/failure indicators when testing Bedrock model availability
