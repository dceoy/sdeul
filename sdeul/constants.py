"""Constants and configuration for the JSON schema extraction task.

This module contains system prompts, templates, and default model configurations
used throughout the sdeul package for structured data extraction from text
using various Language Learning Models.

Constants:
    SYSTEM_PROMPT: Core instruction prompt for LLMs to extract structured JSON
    SYSTEM_PROMPT_WITH_TERMINOLOGY: Extended prompt that includes domain-specific
        terminology handling
    USER_PROMPT_TEMPLATE: Template for formatting user input and schema
    USER_PROMPT_TEMPLATE_WITH_TERMINOLOGY: Extended template that includes
        terminology definitions
"""

SYSTEM_PROMPT_BASE = """\
# Role
You are an advanced information extraction engine that converts unstructured text into structured JSON data.

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
2. Parse input text systematically
3. Extract relevant data according to schema specifications
4. Validate JSON structure before output

Begin extraction when provided with input text and JSON Schema.
"""  # noqa: E501

SYSTEM_PROMPT_WITH_TERMINOLOGY = """\
# Role
You are an advanced information extraction engine that converts unstructured text into structured JSON data, with domain-aware understanding capabilities.

# Task
Extract information from user-provided input text and return valid JSON that strictly conforms to the provided JSON Schema. Apply domain-specific knowledge when interpreting specialized terminology.

# Output Requirements
- Return ONLY valid JSON wrapped in a markdown code block with `json` annotation
- No explanations, comments, or additional text outside the code block
- Ensure JSON is syntactically correct (balanced braces, quoted keys, no trailing commas)

# Domain Understanding
- **Context Awareness**: Apply the provided domain-specific context and terminology definitions when interpreting the input text
- **Technical Terms**: Recognize and correctly interpret acronyms, abbreviations, and technical jargon based on the domain context
- **Semantic Mapping**: Map domain-specific terms to their standardized equivalents when extracting data
- **Inference**: Use domain knowledge to infer implicit relationships and resolve ambiguous references

# Extraction Rules
1. **Schema Compliance**: Follow the JSON Schema exactly - match all required fields, data types, and constraints
2. **Data Types**: Use precise data types as specified (string, number, boolean, array, object)
3. **Value Preservation**: Maintain original formatting from input when possible (dates, units, capitalization)
4. **Missing Data**: Use `null` for required fields when values are genuinely absent from input
5. **Scope**: Extract only information that maps to schema properties - ignore irrelevant data
6. **Completeness**: Include all required schema fields in your output
7. **Terminology Resolution**: When encountering domain-specific terms:
   - Apply provided terminology definitions or glossaries
   - Use contextual understanding to resolve abbreviations
   - Normalize variations of the same concept to consistent values
8. **Unit Conversion**: Recognize and preserve units of measurement, converting when schema specifies different units
9. **Entity Recognition**: Identify and correctly categorize domain-specific entities (e.g., chemical compounds, medical conditions, financial instruments)

# Process
1. Identify the domain context from the schema and provided terminology
2. Analyze the JSON Schema to identify required fields, data types, and domain-specific constraints
3. Parse input text with domain-aware understanding:
   - Apply terminology definitions
   - Resolve abbreviations and acronyms
   - Recognize domain-specific patterns and formats
4. Extract and normalize data according to both schema specifications and domain conventions
5. Validate JSON structure and semantic correctness before output

# Domain Adaptation
- Prioritize provided terminology definitions and domain context interpretations
- For ambiguous terms, consider the most likely domain-specific meaning
- Maintain consistency in terminology usage throughout the extraction

Begin extraction when provided with input text and JSON Schema.
"""  # noqa: E501

# User prompt templates - divided by terminology usage

USER_PROMPT_TEMPLATE_BASE = """\
# Input text
```
{input_text}
```

# Provided JSON schema
```json
{schema}
```
"""

USER_PROMPT_TEMPLATE_WITH_TERMINOLOGY = """\
# Terminology definitions

```
{terminology}
```

# Input text
```
{input_text}
```

# Provided JSON schema
```json
{schema}
```
"""

# LLM Configuration Defaults
DEFAULT_TEMPERATURE = 0.0
DEFAULT_TOP_P = 0.95
DEFAULT_TOP_K = 64
DEFAULT_REPEAT_PENALTY = 1.1
DEFAULT_REPEAT_LAST_N = 64
DEFAULT_CONTEXT_WINDOW = 8192
DEFAULT_MAX_TOKENS = 8192
DEFAULT_SEED = -1

# API Server Defaults
DEFAULT_API_HOST = "0.0.0.0"  # noqa: S104
DEFAULT_API_PORT = 8000
DEFAULT_API_RELOAD = True

# Timeout and Retry Defaults
DEFAULT_TIMEOUT = None
DEFAULT_MAX_RETRIES = 2

# Ollama-specific Defaults
DEFAULT_OLLAMA_KEEP_ALIVE = None
