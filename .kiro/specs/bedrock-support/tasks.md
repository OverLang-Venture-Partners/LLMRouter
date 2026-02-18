# Implementation Plan: AWS Bedrock Support

## Overview

This implementation plan adds AWS Bedrock as a supported LLM provider in LLMRouter. The integration leverages LiteLLM's existing Bedrock support while adding custom handling for AWS-specific authentication, regional configuration, and model-specific request formatting. The implementation maintains backward compatibility with existing providers and follows the established patterns in the codebase.

## Tasks

- [x] 1. Add Bedrock service detection and model formatting
  - Modify `llmrouter/utils/api_calling.py` to detect Bedrock service
  - Add `_is_bedrock_model()` helper function
  - Update model formatting logic to use "bedrock/{model_id}" format for Bedrock models
  - _Requirements: 1.1, 1.2, 3.1_

- [ ]* 1.1 Write property test for Bedrock service recognition
  - **Property 1: Bedrock Service Recognition**
  - **Validates: Requirements 1.1, 2.1**

- [ ]* 1.2 Write property test for Bedrock request routing
  - **Property 2: Bedrock Request Routing**
  - **Validates: Requirements 1.2, 3.1**

- [x] 2. Implement AWS region handling
  - [x] 2.1 Extract aws_region from request dictionary
    - Add logic to read aws_region field from request
    - Pass region to LiteLLM via aws_region_name parameter
    - _Requirements: 2.3, 4.4_
  
  - [ ]* 2.2 Write property test for regional configuration
    - **Property 4: Regional Configuration**
    - **Validates: Requirements 2.3, 4.4**

- [x] 3. Update API key handling for Bedrock
  - Modify `call_api()` to skip API key selection for Bedrock models
  - Update completion call to omit api_key and api_base for Bedrock
  - Add conditional logic to use different parameters for Bedrock vs other providers
  - _Requirements: 1.2, 4.1_

- [x] 4. Implement error handling for Bedrock-specific errors
  - [x] 4.1 Add error handling for missing boto3 dependency
    - Catch ImportError and provide installation instructions
    - _Requirements: 7.1_
  
  - [x] 4.2 Add error handling for missing AWS credentials
    - Catch credential-related exceptions
    - Provide setup instructions for environment variables, credential files, and IAM roles
    - _Requirements: 1.3, 4.5, 7.3_
  
  - [x] 4.3 Add error handling for invalid model IDs
    - Catch model not found errors
    - Provide list of common Bedrock model IDs
    - _Requirements: 7.2_
  
  - [x] 4.4 Add error handling for region mismatches
    - Catch region-related errors
    - Suggest checking model availability in region
    - _Requirements: 7.5_
  
  - [x] 4.5 Add error handling for API timeouts
    - Catch timeout exceptions
    - Suggest increasing timeout parameter
    - _Requirements: 7.4_

- [ ]* 4.6 Write unit tests for error conditions
  - Test missing boto3 (mock import failure)
  - Test missing credentials (mock credential error)
  - Test invalid model ID (mock API error)
  - Test region mismatch (mock region error)
  - _Requirements: 1.3, 7.1, 7.2, 7.3, 7.5_

- [x] 5. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 6. Create example Bedrock model configurations
  - [x] 6.1 Add Bedrock models to example LLM candidate JSON
    - Create examples for Claude 3 Sonnet, Claude 3 Haiku
    - Create examples for Amazon Titan Text Express
    - Create examples for Llama 3 70B, Mistral 7B
    - Include all required fields: model, service, size, feature, input_price, output_price, aws_region
    - _Requirements: 2.2, 9.1_
  
  - [x] 6.2 Update OpenClaw example config with Bedrock models
    - Add Bedrock model entries to `configs/openclaw_example.yaml`
    - Include examples with different regions
    - Add comments explaining AWS credential configuration
    - _Requirements: 9.2, 9.4_

- [ ]* 6.3 Write property test for configuration schema
  - **Property 3: Configuration Schema Completeness**
  - **Validates: Requirements 2.2, 6.1, 6.3, 6.4**

- [ ] 7. Add system prompt and inference parameter support
  - [x] 7.1 Ensure system prompt is passed to LiteLLM
    - Verify existing system_prompt handling works for Bedrock
    - Test that system prompts are included in messages array
    - _Requirements: 5.3_
  
  - [x] 7.2 Ensure inference parameters are passed to LiteLLM
    - Verify temperature, top_p, max_tokens are passed correctly
    - Test parameter passthrough for Bedrock models
    - _Requirements: 5.5_

- [ ]* 7.3 Write property tests for parameter passthrough
  - **Property 7: System Prompt Passthrough**
  - **Property 8: Inference Parameters**
  - **Validates: Requirements 5.3, 5.5**

- [ ] 8. Implement response parsing and metadata extraction
  - [x] 8.1 Verify response parsing works for Bedrock
    - Test that response text is extracted correctly
    - Ensure original request fields are preserved
    - _Requirements: 3.3_
  
  - [x] 8.2 Verify token usage tracking for Bedrock
    - Test that token counts are extracted from response metadata
    - Ensure response_time is recorded
    - _Requirements: 3.5, 3.6_

- [ ]* 8.3 Write property tests for response handling
  - **Property 5: Response Parsing**
  - **Property 6: Response Metadata**
  - **Validates: Requirements 3.3, 3.5, 3.6**

- [ ] 9. Test backward compatibility
  - [x] 9.1 Create mixed provider test configurations
    - Create test configs with Bedrock + NVIDIA models
    - Create test configs with Bedrock + OpenAI models
    - Test that both provider types work correctly
    - _Requirements: 8.1, 8.5_
  
  - [ ]* 9.2 Write property test for backward compatibility
    - **Property 9: Backward Compatibility**
    - **Validates: Requirements 8.1, 8.5**
  
  - [ ]* 9.3 Run existing test suites with Bedrock models
    - Verify existing tests pass with Bedrock models added
    - Test data generation pipeline with Bedrock models
    - _Requirements: 8.3_

- [ ] 10. Create documentation and examples
  - [x] 10.1 Add AWS credentials setup documentation
    - Document environment variable configuration
    - Document AWS credential file configuration
    - Document IAM role configuration
    - _Requirements: 9.3_
  
  - [x] 10.2 Add Bedrock usage examples
    - Create example script for single Bedrock query
    - Create example script for batch Bedrock queries
    - Create example for mixed provider routing
    - _Requirements: 9.4, 9.5_
  
  - [x] 10.3 Update README with Bedrock support information
    - Add Bedrock to list of supported providers
    - Add installation instructions for boto3
    - Add quick start example for Bedrock
    - _Requirements: 9.1, 9.2_

- [x] 11. Final checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP
- Each task references specific requirements for traceability
- Checkpoints ensure incremental validation
- Property tests validate universal correctness properties (minimum 100 iterations each)
- Unit tests validate specific examples and edge cases
- The implementation leverages LiteLLM's existing Bedrock support, minimizing custom code
- AWS credentials are handled by boto3's default credential chain (no custom credential management needed)
