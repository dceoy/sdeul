# Tech Context

## Core Technology

*   **Language:** Python (based on `.py` files and `pyproject.toml`, `poetry.lock`)

## Key Libraries & Frameworks (Inferred)

*   **Dependency Management:** Poetry (indicated by `pyproject.toml` and `poetry.lock`).
*   **CLI Framework:** Typer (verified in `sdeul/cli.py`).
*   **JSON Schema Validation:** Likely `jsonschema`. Needs verification by reading `sdeul/validation.py` and `pyproject.toml`.
*   **LLM Clients:** Langchain libraries (`langchain-aws`, `langchain-google-genai`, etc.) are used (verified in `sdeul/llm.py`). `boto3` is likely a dependency of `langchain-aws`.
*   **Testing:**
    *   `pytest` (indicated by `test/pytest/` directory and `conftest.py`).
    *   `bats` (Bash Automated Testing System) for CLI/integration testing (indicated by `test/bats/` directory).
*   **Code Formatting/Linting:** Potentially tools like `black`, `ruff`, `flake8`, or `mypy` (often configured in `pyproject.toml`).

## Development Setup

*   Requires Python installation.
*   Requires Poetry installation (`pip install poetry`).
*   Install dependencies: `poetry install`.
*   Running tests: Likely `poetry run pytest` and `poetry run bats test/bats`.
*   Running CLI: `poetry run sdeul extract ...` or `poetry run python -m sdeul.cli extract ...` (Needs verification from `pyproject.toml`).

## Technical Constraints

*   Requires network access to communicate with cloud-based LLM providers.
*   API keys/credentials must be securely managed. Supported methods include environment variables (e.g., `OPENAI_API_KEY`, `GOOGLE_API_KEY`, `GROQ_API_KEY`, `AWS_PROFILE`) and CLI arguments (e.g., `--openai-api-key`, `--aws-profile`). Standard AWS credential chain (environment variables, shared credential file, config file, IAM role) is likely used by `boto3`/`langchain-aws`.
*   Performance depends on the chosen LLM provider's response time.
*   Handling potentially large text inputs and JSON outputs efficiently.

## Dependencies

*   Managed via `pyproject.toml` and `poetry.lock`. Key dependencies need to be identified by reading these files.

*(Note: This context is inferred. Code review is needed for confirmation and detail.)*
