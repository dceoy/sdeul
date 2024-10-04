#!/usr/bin/env python

TEST_TEXT = "This is a test input text."
TEST_SCHEMA = {
    "type": "object",
    "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
}
TEST_LLM_OUTPUT = {"name": "John Doe", "age": 30}
TEST_LLM_OUTPUT_JSON = """{
    "name": "John Doe",
    "age": 30
}"""
TEST_LLM_OUTPUT_MD = f"""Here is the output of the LLM:

```json
{TEST_LLM_OUTPUT_JSON}
```
"""
