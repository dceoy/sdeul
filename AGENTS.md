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
   - Creating LLM instances based on provider (OpenAI, Google, AWS Bedrock, Groq, Ollama)
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

## Code Design Principles

Follow Robert C. Martin's SOLID and Clean Code principles:

### SOLID Principles

1. **SRP (Single Responsibility)**: One reason to change per class; separate concerns (e.g., storage vs formatting vs calculation)
2. **OCP (Open/Closed)**: Open for extension, closed for modification; use polymorphism over if/else chains
3. **LSP (Liskov Substitution)**: Subtypes must be substitutable for base types without breaking expectations
4. **ISP (Interface Segregation)**: Many specific interfaces over one general; no forced unused dependencies
5. **DIP (Dependency Inversion)**: Depend on abstractions, not concretions; inject dependencies

### Clean Code Practices

- **Naming**: Intention-revealing, pronounceable, searchable names (`daysSinceLastUpdate` not `d`)
- **Functions**: Small, single-task, verb names, 0-3 args, extract complex logic
- **Classes**: Follow SRP, high cohesion, descriptive names
- **Error Handling**: Exceptions over error codes, no null returns, provide context, try-catch-finally first
- **Testing**: TDD, one assertion/test, FIRST principles (Fast, Independent, Repeatable, Self-validating, Timely), Arrange-Act-Assert pattern
- **Code Organization**: Variables near usage, instance vars at top, public then private functions, conceptual affinity
- **Comments**: Self-documenting code preferred, explain "why" not "what", delete commented code
- **Formatting**: Consistent, vertical separation, 88-char limit, team rules override preferences
- **General**: DRY, KISS, YAGNI, Boy Scout Rule, fail fast

## Development Methodology

Follow Martin Fowler's Refactoring, Kent Beck's Tidy Code, and t_wada's TDD principles:

### Core Philosophy

- **Small, safe changes**: Tiny, reversible, testable modifications
- **Separate concerns**: Never mix features with refactoring
- **Test-driven**: Tests provide safety and drive design
- **Economic**: Only refactor when it aids immediate work

### TDD Cycle

1. **Red** → Write failing test
2. **Green** → Minimum code to pass
3. **Refactor** → Clean without changing behavior
4. **Commit** → Separate commits for features vs refactoring

### Practices

- **Before**: Create TODOs, ensure coverage, identify code smells
- **During**: Test-first, small steps, frequent tests, two hats rule
- **Refactoring**: Extract function/variable, rename, guard clauses, remove dead code, normalize symmetries
- **TDD Strategies**: Fake it, obvious implementation, triangulation

### When to Apply

- Rule of Three (3rd duplication)
- Preparatory (before features)
- Comprehension (as understanding grows)
- Opportunistic (daily improvements)

### Key Rules

- One assertion per test
- Separate refactoring commits
- Delete redundant tests
- Human-readable code first

> "Make the change easy, then make the easy change." - Kent Beck
