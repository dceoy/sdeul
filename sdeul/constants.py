"""Constants for the JSON schema extraction task."""

EXTRACTION_TEMPLATE = """\
Instructions:
- You are a structured data extraction engine.
- Extract ONLY the relevant entities defined by the provided JSON schema from the input text.
- Generate the extracted entities in JSON format according to the schema.
- Include ONLY the fields specified in the schema.
- For required fields in the schema, set the value to null if the information cannot be found in the input text.
- For optional fields not found in the input text, omit them from the output.
- Output the complete JSON data in a markdown code block.
- Provide complete, unabridged code in all responses without omitting any parts.

Input text:
```
{input_text}
```

Provided JSON schema:
```json
{schema}
```
"""  # noqa: E501

DEFAULT_MODEL_NAMES = {
    "openai": "gpt-4.1",
    "google": "gemini-2.5-pro",
    "bedrock": "us.anthropic.claude-3-7-sonnet-20250219-v1:0",
    "groq": "meta-llama/llama-4-maverick-17b-128e-instruct",
}
