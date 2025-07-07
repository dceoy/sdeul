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

## Web Search Instructions

For tasks requiring web search, always use `gemini` command instead of the built-in web search tool.

### Usage

```sh
# Basic search query
gemini --prompt "WebSearch: <query>"

# Example: Search for latest news
gemini --prompt "WebSearch: What are the latest developments in AI?"
```

### Policy

When users request information that requires web search:

1. Use `gemini --prompt` command via terminal
2. Parse and present the Gemini response appropriately

This ensures consistent and reliable web search results through the Gemini API.

## Development Methodology

This section combines essential guidance from Martin Fowler's refactoring, Kent Beck's tidying, and t_wada's TDD approaches.

### Core Philosophy

- **Small, safe, behavior-preserving changes** - Every change should be tiny, reversible, and testable
- **Separate concerns** - Never mix adding features with refactoring/tidying
- **Test-driven workflow** - Tests provide safety net and drive design
- **Economic justification** - Only refactor/tidy when it makes immediate work easier

### The Development Cycle

1. **Red** - Write a failing test first (TDD)
2. **Green** - Write minimum code to pass the test
3. **Refactor/Tidy** - Clean up without changing behavior
4. **Commit** - Separate commits for features vs refactoring

### Essential Practices

#### Before Coding

- Create TODO list for complex tasks
- Ensure test coverage exists
- Identify code smells (long functions, duplication, etc.)

#### While Coding

- **Test-First**: Write the test before the implementation
- **Small Steps**: Each change should be easily reversible
- **Run Tests Frequently**: After each small change
- **Two Hats**: Either add features OR refactor, never both

#### Refactoring Techniques

1. **Extract Function/Variable** - Improve readability
2. **Rename** - Use meaningful names
3. **Guard Clauses** - Replace nested conditionals
4. **Remove Dead Code** - Delete unused code
5. **Normalize Symmetries** - Make similar code consistent

#### TDD Strategies

1. **Fake It** - Start with hardcoded values
2. **Obvious Implementation** - When solution is clear
3. **Triangulation** - Multiple tests to find general solution

### When to Apply

- **Rule of Three**: Refactor on third duplication
- **Preparatory**: Before adding features to messy code
- **Comprehension**: As you understand code better
- **Opportunistic**: Small improvements during daily work

### Key Reminders

- One assertion per test
- Commit refactoring separately from features
- Delete redundant tests
- Focus on making code understandable to humans

Quote: "Make the change easy, then make the easy change." - Kent Beck
