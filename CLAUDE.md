# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SDEUL (Structural Data Extractor using LLMs) is a Python tool that extracts structured data from text using various Large Language Models (LLMs) and validates it against a JSON Schema.

## Development Commands

### Environment Setup

```sh
# Install dependencies
uv sync

# Activate virtual environment (if needed)
source .venv/bin/activate
```

### Testing

```sh
# Run pytest tests
uv run pytest

# Run specific pytest test
uv run pytest test/unit/test_extraction.py -v

# Run tests with coverage report
uv run pytest

# Run bats tests
bats test/cli/test_cli.bats
bats test/cli/test_openai.bats  # Requires OpenAI API key
bats test/cli/test_bedrock.bats  # Requires AWS credentials
bats test/cli/test_google.bats  # Requires Google API key
bats test/cli/test_groq.bats  # Requires Groq API key
bats test/cli/test_ollama.bats  # Requires Ollama running
bats test/cli/test_llamacpp.bats  # Requires LLM file
```

### Code Quality

```sh
# Run linting
uv run ruff check .

# Run linting with auto-fix
uv run ruff check --fix .

# Run type checking
uv run pyright .
```

### Building and Packaging

```sh
# Build the package
uv build

# Install locally
uv pip install -e .
```

## Architecture

### Core Components

1. **CLI Interface (`cli.py`)**: Defines the command-line interface using Typer with two main commands:
   - `extract`: Extracts structured data from text using LLMs
   - `validate`: Validates JSON files against a JSON Schema

2. **Extraction Module (`extraction.py`)**: Contains the main functionality for:
   - Reading input text and JSON Schema
   - Creating appropriate LLM instances
   - Generating structured data with the LLM
   - Validating the output against the schema

3. **LLM Module (`llm.py`)**: Handles:
   - Creating LLM instances based on provider (OpenAI, Google, AWS Bedrock, Groq, Ollama, LLamaCpp)
   - Parsing LLM outputs (extracting JSON from responses)

4. **Utility Functions (`utility.py`)**: Provides helper functions for:
   - File I/O operations
   - Logging configuration
   - Environment variables management

5. **Validation Module (`validation.py`)**: Validates JSON data against JSON Schema

### Data Flow

1. User provides a JSON Schema and input text
2. CLI parses arguments and calls extraction function
3. The extraction function:
   - Reads the schema and input text
   - Creates an appropriate LLM instance based on user parameters
   - Prompts the LLM using a system prompt and user template
   - Parses the LLM output to extract valid JSON
   - Validates the output against the schema
   - Writes or prints the resulting structured data

### Key Design Patterns

- **Factory Pattern**: In `llm.py` to create appropriate LLM instances
- **Decorator Pattern**: Used for timing function execution with `@log_execution_time`
- **Adapter Pattern**: Each LLM provider has a consistent interface regardless of underlying implementation
