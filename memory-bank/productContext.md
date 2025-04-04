# Product Context

## Problem Solved

Extracting specific, structured information (like JSON objects) from unstructured text (like emails, reports, or free-form notes) is often a manual, time-consuming, and error-prone process. Traditional methods like regular expressions or rule-based systems can be brittle and difficult to maintain, especially with varied input formats.

## How It Works

`sdeul` leverages the natural language understanding capabilities of Large Language Models (LLMs) to address this challenge.

1.  **Input:** The user provides unstructured text and a target JSON schema.
2.  **Processing:** `sdeul` sends the text and schema to a configured LLM provider with instructions to extract data matching the schema.
3.  **Validation:** The LLM's output (expected to be JSON) is validated against the provided JSON schema to ensure correctness and structure.
4.  **Output:** If validation passes, the structured JSON data is returned to the user. If validation fails or the LLM encounters an error, appropriate feedback is provided.

## User Experience Goals

*   **Simplicity:** Easy to use via both the command line and as a Python library.
*   **Flexibility:** Support for various popular LLM providers.
*   **Reliability:** Consistent and accurate data extraction.
*   **Transparency:** Clear error messages and validation feedback.
*   **Configuration:** Straightforward configuration of LLM providers and API keys.
