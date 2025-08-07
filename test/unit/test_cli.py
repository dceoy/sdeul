"""Tests for the CLI module."""

# pyright: reportPrivateUsage=false

import pytest
from pytest_mock import MockerFixture
from typer import Exit
from typer.testing import CliRunner

from sdeul.cli import _version_callback, app


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


@pytest.mark.parametrize("value", [True, False])
def test__version_callback(value: bool, mocker: MockerFixture) -> None:
    dummy_version = "1.0.0"
    mocker.patch("sdeul.cli.__version__", dummy_version)
    mock_print = mocker.patch("sdeul.cli.print")
    if value:
        with pytest.raises(Exit):
            _version_callback(value)
        mock_print.assert_called_once_with(dummy_version)
    else:
        _version_callback(value)
        mock_print.assert_not_called()


@pytest.mark.parametrize(
    "cli_args",
    [["--help"], ["extract", "--help"], ["validate", "--help"]],
)
def test_main_with_help_option(cli_args: list[str], runner: CliRunner) -> None:
    result = runner.invoke(app, cli_args)
    assert result.exit_code == 0
    assert "Usage:" in result.stdout


@pytest.mark.parametrize(
    "cli_args",
    [[], ["invalid-command"], ["--invalid-option"]],
)
def test_main_invalid_arguments(cli_args: list[str], runner: CliRunner) -> None:
    result = runner.invoke(app, cli_args)
    assert result.exit_code != 0
    # Check if error appears in stdout or stderr (typer can output to either)
    output = result.stdout + result.stderr
    assert "Error" in output or "Usage:" in output


def test_main_with_version_option(runner: CliRunner, mocker: MockerFixture) -> None:
    dummy_version = "1.0.0"
    mocker.patch("sdeul.cli.__version__", dummy_version)
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert dummy_version in result.stdout


@pytest.mark.parametrize(
    "cli_args",
    [
        ["extract", "schema.json", "input.txt"],
        ["extract", "--debug", "schema.json", "input.txt"],
        ["extract", "--info", "schema.json", "input.txt"],
        ["extract", "--output-json-file=output.json", "schema.json", "input.txt"],
        ["extract", "--compact-json", "schema.json", "input.txt"],
        ["extract", "--skip-validation", "schema.json", "input.txt"],
        [
            "extract",
            "--openai-model=gpt-4o-mini",
            "--openai-api-base=https://example.com/api",
            "--openai-organization=example",
            "--openai-api-key=dummy",
            "schema.json",
            "input.txt",
        ],
        [
            "extract",
            "--google-model=gemini-1.5-flash",
            "--google-api-key=dummy",
            "schema.json",
            "input.txt",
        ],
        [
            "extract",
            "--groq-model=llama-3.1-70b-versatile",
            "--groq-api-key=dummy",
            "schema.json",
            "input.txt",
        ],
        [
            "extract",
            "--bedrock-model=anthropic.claude-3-5-sonnet-20240620-v1:0",
            "schema.json",
            "input.txt",
        ],
        ["extract", "--llamacpp-model-file=llm.gguf", "schema.json", "input.txt"],
        ["extract", "--ollama-model=llama3.1", "schema.json", "input.txt"],
        [
            "extract",
            "--ollama-model=llama3.1",
            "--ollama-base-url=http://localhost:11434",
            "schema.json",
            "input.txt",
        ],
        [
            "extract",
            "--llamacpp-model-file=llm.gguf",
            "--temperature=0.5",
            "--top-p=0.2",
            "--max-tokens=4000",
            "--n-ctx=512",
            "--seed=42",
            "--n-batch=4",
            "--n-gpu-layers=2",
            "schema.json",
            "input.txt",
        ],
    ],
)
def test_extract_command(
    cli_args: list[str],
    runner: CliRunner,
    mocker: MockerFixture,
) -> None:
    mock_configure_logging = mocker.patch("sdeul.cli.configure_logging")
    mock_extract_json_from_text_file_with_config = mocker.patch(
        "sdeul.cli.extract_json_from_text_file_with_config",
    )
    result = runner.invoke(app, cli_args)
    assert result.exit_code == 0
    mock_configure_logging.assert_called_once_with(
        debug=("--debug" in cli_args),
        info=("--info" in cli_args),
    )
    mock_extract_json_from_text_file_with_config.assert_called_once()


@pytest.mark.parametrize(
    "cli_args",
    [
        ["validate", "schema.json", "input_0.json"],
        ["validate", "schema.json", "input_0.json", "input_1.json", "input_2.json"],
        ["validate", "--debug", "schema.json", "input_0.json"],
        ["validate", "--info", "schema.json", "input_0.json"],
    ],
)
def test_validate_command(
    cli_args: list[str],
    runner: CliRunner,
    mocker: MockerFixture,
) -> None:
    mock_configure_logging = mocker.patch("sdeul.cli.configure_logging")
    mock_validate_json_files_using_json_schema = mocker.patch(
        "sdeul.cli.validate_json_files_using_json_schema",
    )
    result = runner.invoke(app, cli_args)
    assert result.exit_code == 0
    mock_configure_logging.assert_called_once_with(
        debug=("--debug" in cli_args),
        info=("--info" in cli_args),
    )
    mock_validate_json_files_using_json_schema.assert_called_once()


def test_serve_command(runner: CliRunner, mocker: MockerFixture) -> None:
    """Test the serve command."""
    mock_configure_logging = mocker.patch("sdeul.cli.configure_logging")
    mock_run_server = mocker.patch("sdeul.cli.run_server")
    result = runner.invoke(app, ["serve"])
    assert result.exit_code == 0
    mock_configure_logging.assert_called_once_with(debug=False, info=False)
    mock_run_server.assert_called_once_with(host="0.0.0.0", port=8000, reload=True)


def test_serve_command_with_options(runner: CliRunner, mocker: MockerFixture) -> None:
    """Test the serve command with custom options."""
    mock_configure_logging = mocker.patch("sdeul.cli.configure_logging")
    mock_run_server = mocker.patch("sdeul.cli.run_server")
    result = runner.invoke(
        app,
        ["serve", "--host", "localhost", "--port", "9000", "--no-reload", "--debug"],
    )
    assert result.exit_code == 0
    mock_configure_logging.assert_called_once_with(debug=True, info=False)
    mock_run_server.assert_called_once_with(host="localhost", port=9000, reload=False)
