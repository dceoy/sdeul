"""Constants and configuration for the JSON schema extraction task.

This module contains system prompts, templates, and default model configurations
used throughout the sdeul package for structured data extraction from text
using various Language Learning Models.

Constants:
    SYSTEM_PROMPT: Core instruction prompt for LLMs to extract structured JSON
    USER_PROMPT_TEMPLATE: Template for formatting user input and schema
    DEFAULT_MODEL_NAMES: Default model names for each LLM provider
"""

SYSTEM_PROMPT = """\
# Role
You are a precise information extraction engine that converts unstructured text into structured JSON data.

# Task
Extract information from user-provided input text and return valid JSON that strictly conforms to the provided JSON Schema.

# Output Requirements
- Return ONLY valid JSON wrapped in a markdown code block with `json` annotation
- No explanations, comments, or additional text outside the code block
- Ensure JSON is syntactically correct (balanced braces, quoted keys, no trailing commas)

# Extraction Rules
1. **Schema Compliance**: Follow the JSON Schema exactly - match all required fields, data types, and constraints
2. **Data Types**: Use precise data types as specified (string, number, boolean, array, object)
3. **Value Preservation**: Maintain original formatting from input when possible (dates, units, capitalization)
4. **Missing Data**: Use `null` for required fields when values are genuinely absent from input
5. **Scope**: Extract only information that maps to schema properties - ignore irrelevant data
6. **Completeness**: Include all required schema fields in your output

# Process
1. Analyze the JSON Schema to identify required fields and data types
2. Scan input text for relevant information matching schema properties
3. Extract and format data according to schema specifications
4. Validate JSON structure before output

Begin extraction when provided with input text and JSON Schema.
"""  # noqa: E501
USER_PROMPT_TEMPLATE = """\
Input text:
```
{input_text}
```

Provided JSON schema:
```json
{schema}
```
"""

DEFAULT_MODEL_NAMES = {
    "openai": "gpt-4.1",
    "google": "gemini-2.5-pro",
    "anthropic": "claude-3-5-sonnet-20241022",
    "bedrock": "us.anthropic.claude-3-7-sonnet-20250219-v1:0",
    "cerebras": "llama3.3-70b",
    "groq": "meta-llama/llama-4-maverick-17b-128e-instruct",
}

# LLM Configuration Defaults
DEFAULT_TEMPERATURE = 0.0
DEFAULT_TOP_P = 0.95
DEFAULT_TOP_K = 64
DEFAULT_REPEAT_PENALTY = 1.1
DEFAULT_REPEAT_LAST_N = 64
DEFAULT_CONTEXT_WINDOW = 8192
DEFAULT_MAX_TOKENS = 8192
DEFAULT_SEED = -1

# LlamaCpp Specific Defaults
DEFAULT_N_BATCH = 8
DEFAULT_N_THREADS = -1
DEFAULT_N_GPU_LAYERS = -1
DEFAULT_F16_KV = True
DEFAULT_USE_MLOCK = False
DEFAULT_USE_MMAP = True
DEFAULT_TOKEN_WISE_STREAMING = False

# API Server Defaults
DEFAULT_API_HOST = "0.0.0.0"  # noqa: S104
DEFAULT_API_PORT = 8000
DEFAULT_API_RELOAD = True

# Timeout and Retry Defaults
DEFAULT_TIMEOUT = None
DEFAULT_MAX_RETRIES = 2
