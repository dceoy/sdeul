#!/usr/bin/env python
"""Structural Data Extractor using LLMs."""

import typer
from rich import print

from . import __version__
from .extraction import extract_json_from_text_file
from .utility import configure_logging
from .validation import validate_json_files_using_json_schema

app = typer.Typer()


def _version_callback(value: bool) -> None:
    if value:
        print(__version__)
        raise typer.Exit


@app.callback()
def main(
    version: bool = typer.Option(
        None,
        "--version",
        callback=_version_callback,
        is_eager=True,
        help="Show version information and exit.",
    ),
) -> None:
    """Structural Data Extractor using LLMs."""
    pass


@app.command()
def extract(
    json_schema_file_path: str = typer.Argument(..., help="JSON Schema file path."),
    text_file_path: str = typer.Argument(..., help="Input text file path."),
    output_json_file_path: str | None = typer.Option(
        None,
        "--output-json-file",
        help="Output JSON file path.",
    ),
    compact_json: bool = typer.Option(
        False,
        "--compact-json",
        help="Compact instead of pretty-printed output.",
    ),
    skip_validation: bool = typer.Option(
        False,
        "--skip-validation",
        help="Skip output validation using JSON Schema.",
    ),
    temperature: float = typer.Option(
        0,
        "--temperature",
        help="Specify the temperature for sampling.",
    ),
    top_p: float = typer.Option(
        0.1,
        "--top-p",
        help="Specify the top-p value for sampling.",
    ),
    max_tokens: int = typer.Option(
        8000,
        "--max-tokens",
        help="Specify the max tokens to generate.",
    ),
    n_ctx: int = typer.Option(
        1024,
        "--n-ctx",
        help="Specify the token context window.",
    ),
    seed: int = typer.Option(-1, "--seed", help="Specify the random seed."),
    n_batch: int = typer.Option(
        8,
        "--n-batch",
        help="Specify the number of batch tokens.",
    ),
    n_gpu_layers: int = typer.Option(
        -1,
        "--n-gpu-layers",
        help="Specify the number of GPU layers.",
    ),
    openai_model_name: str | None = typer.Option(
        None,
        "--openai-model",
        envvar="OPENAI_MODEL",
        help="Use the OpenAI model. (e.g., gpt-4o-mini)",
    ),
    google_model_name: str | None = typer.Option(
        None,
        "--google-model",
        envvar="GOOGLE_MODEL",
        help="Use the Google Generative AI model. (e.g., gemini-1.5-flash)",
    ),
    groq_model_name: str | None = typer.Option(
        None,
        "--groq-model",
        envvar="GROQ_MODEL",
        help="Use the Groq model. (e.g., llama-3.1-70b-versatile)",
    ),
    bedrock_model_id: str | None = typer.Option(
        None,
        "--bedrock-model",
        envvar="BEDROCK_MODEL",
        help=(
            "Use the Amazon Bedrock model."
            " (e.g., anthropic.claude-3-5-sonnet-20240620-v1:0)"
        ),
    ),
    llamacpp_model_file_path: str | None = typer.Option(
        None,
        "--model-file",
        help="Use the model GGUF file for llama.cpp.",
    ),
    openai_api_key: str | None = typer.Option(
        None,
        "--openai-api-key",
        envvar="OPENAI_API_KEY",
        help="Override the OpenAI API key.",
    ),
    openai_api_base: str | None = typer.Option(
        None,
        "--openai-api-base",
        envvar="OPENAI_API_BASE",
        help="Override the OpenAI API base URL.",
    ),
    openai_organization: str | None = typer.Option(
        None,
        "--openai-organization",
        envvar="OPENAI_ORGANIZATION",
        help="Override the OpenAI organization ID.",
    ),
    google_api_key: str | None = typer.Option(
        None,
        "--google-api-key",
        envvar="GOOGLE_API_KEY",
        help="Override the Google API key.",
    ),
    groq_api_key: str | None = typer.Option(
        None,
        "--groq-api-key",
        envvar="GROQ_API_KEY",
        help="Override the Groq API key.",
    ),
    debug: bool = typer.Option(False, "--debug", help="Execute with debug messages."),
    info: bool = typer.Option(False, "--info", help="Execute with info messages."),
) -> None:
    """Extract data as JSON."""
    configure_logging(debug=debug, info=info)
    extract_json_from_text_file(
        json_schema_file_path=json_schema_file_path,
        text_file_path=text_file_path,
        output_json_file_path=output_json_file_path,
        compact_json=compact_json,
        skip_validation=skip_validation,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        n_ctx=n_ctx,
        seed=seed,
        n_batch=n_batch,
        n_gpu_layers=n_gpu_layers,
        openai_model_name=openai_model_name,
        google_model_name=google_model_name,
        groq_model_name=groq_model_name,
        bedrock_model_id=bedrock_model_id,
        llamacpp_model_file_path=llamacpp_model_file_path,
        openai_api_key=openai_api_key,
        openai_api_base=openai_api_base,
        openai_organization=openai_organization,
        google_api_key=google_api_key,
        groq_api_key=groq_api_key,
    )


@app.command()
def validate(
    json_schema_file_path: str = typer.Argument(..., help="JSON Schema file path."),
    json_file_paths: list[str] = typer.Argument(..., help="JSON file paths."),
    debug: bool = typer.Option(False, "--debug", help="Set DEBUG log level."),
    info: bool = typer.Option(False, "--info", help="Set INFO log level."),
) -> None:
    """Validate JSON files using JSON Schema."""
    configure_logging(debug=debug, info=info)
    validate_json_files_using_json_schema(
        json_schema_file_path=json_schema_file_path,
        json_file_paths=json_file_paths,
    )
