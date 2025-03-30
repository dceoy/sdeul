# SDEUL Development Guidelines

## Build/Test Commands
- Install: `pip install -e .` or `poetry install`
- Lint: `ruff check .`
- Type check: `pyright .`
- Run all tests: `pytest`
- Run single test: `pytest test/pytest/test_file.py::TestClass::test_function -v`
- Run bats tests: `bats test/bats/test_*.bats`

## Code Style Guidelines
- Use Google docstring style; all public functions/classes must have docs
- Type annotations required with strict typing (`typeCheckingMode = "strict"`)
- Line length: 88 characters max
- Imports: sorted using isort (handled by ruff)
- Error handling: Use explicit exception types; avoid bare excepts
- Naming: snake_case for variables/functions, PascalCase for classes
- Keep test coverage at 100% (`fail_under = 100` in coverage settings)
- Use pathlib instead of os.path where possible
- Prefer f-strings over string formatting or concatenation
- Follow ruff linting rules defined in pyproject.toml