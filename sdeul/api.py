"""REST API for Structural Data Extractor using LLMs.

This module provides a FastAPI-based REST API for sdeul, offering endpoints
for extracting structured JSON data from text and validating JSON files
against schemas.

Endpoints:
    POST /extract: Extract structured JSON data from text
    POST /validate: Validate JSON data against a JSON schema
"""

import logging
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException
from jsonschema import ValidationError as JsonSchemaValidationError
from jsonschema import validate
from pydantic import BaseModel, Field

from .config import LLMConfig, ModelConfig
from .constants import (
    DEFAULT_CONTEXT_WINDOW,
    DEFAULT_MAX_RETRIES,
    DEFAULT_MAX_TOKENS,
    DEFAULT_REPEAT_LAST_N,
    DEFAULT_REPEAT_PENALTY,
    DEFAULT_SEED,
    DEFAULT_TEMPERATURE,
    DEFAULT_TIMEOUT,
    DEFAULT_TOP_K,
    DEFAULT_TOP_P,
)
from .extraction import extract_structured_data_from_text
from .llm import create_llm_instance
from .utility import configure_logging

app = FastAPI(
    title="SDEUL API",
    description="Structural Data Extractor using LLMs REST API",
    version="0.2.0",
)

configure_logging(debug=False, info=True)
logger = logging.getLogger(__name__)


class ExtractRequest(BaseModel):
    """Request model for the extract endpoint."""

    text: str = Field(..., description="Input text to extract data from")
    json_schema: dict[str, Any] = Field(
        ...,
        description="JSON Schema defining the output structure",
    )
    skip_validation: bool = Field(
        default=False,
        description="Skip JSON schema validation",
    )
    terminology: str | None = Field(
        default=None,
        description=(
            "Domain-specific terminology or glossary to help interpret "
            "specialized terms"
        ),
    )
    temperature: float = Field(
        default=DEFAULT_TEMPERATURE,
        ge=0.0,
        le=2.0,
        description="Sampling temperature",
    )
    top_p: float = Field(
        default=DEFAULT_TOP_P,
        ge=0.0,
        le=1.0,
        description="Top-p sampling parameter",
    )
    top_k: int = Field(
        default=DEFAULT_TOP_K,
        ge=1,
        description="Top-k sampling parameter",
    )
    repeat_penalty: float = Field(
        default=DEFAULT_REPEAT_PENALTY,
        ge=1.0,
        description="Repeat penalty",
    )
    repeat_last_n: int = Field(
        default=DEFAULT_REPEAT_LAST_N,
        ge=0,
        description="Tokens to consider for repeat penalty",
    )
    n_ctx: int = Field(
        default=DEFAULT_CONTEXT_WINDOW,
        ge=1,
        description="Context window size",
    )
    max_tokens: int = Field(
        default=DEFAULT_MAX_TOKENS,
        ge=1,
        description="Maximum tokens to generate",
    )
    seed: int = Field(
        default=DEFAULT_SEED,
        description="Random seed (-1 for random)",
    )
    timeout: int | None = Field(
        default=DEFAULT_TIMEOUT,
        description="Request timeout in seconds",
    )
    max_retries: int = Field(
        default=DEFAULT_MAX_RETRIES,
        ge=0,
        description="Maximum number of retries",
    )

    # Model selection (exactly one should be provided)
    openai_model: str | None = Field(default=None, description="OpenAI model name")
    openai_api_key: str | None = Field(default=None, description="OpenAI API key")
    openai_api_base: str | None = Field(default=None, description="OpenAI API base URL")
    openai_organization: str | None = Field(
        default=None,
        description="OpenAI organization ID",
    )

    google_model: str | None = Field(default=None, description="Google model name")
    google_api_key: str | None = Field(default=None, description="Google API key")

    anthropic_model: str | None = Field(
        default=None,
        description="Anthropic model name",
    )
    anthropic_api_key: str | None = Field(default=None, description="Anthropic API key")
    anthropic_api_base: str | None = Field(
        default=None,
        description="Anthropic API base URL",
    )

    cerebras_model: str | None = Field(default=None, description="Cerebras model name")
    cerebras_api_key: str | None = Field(default=None, description="Cerebras API key")

    groq_model: str | None = Field(default=None, description="Groq model name")
    groq_api_key: str | None = Field(default=None, description="Groq API key")

    bedrock_model: str | None = Field(default=None, description="AWS Bedrock model ID")
    aws_credentials_profile: str | None = Field(
        default=None,
        description="AWS credentials profile",
    )
    aws_region: str | None = Field(default=None, description="AWS region")
    bedrock_endpoint_url: str | None = Field(
        default=None,
        description="Bedrock endpoint URL",
    )

    ollama_model: str | None = Field(default=None, description="Ollama model name")
    ollama_base_url: str | None = Field(default=None, description="Ollama base URL")
    ollama_keep_alive: str | int | None = Field(
        default=None,
        description=(
            "Duration to keep Ollama model loaded in memory "
            "(e.g., '5m', '10m', or -1 for indefinite)"
        ),
    )


class ExtractResponse(BaseModel):
    """Response model for the extract endpoint."""

    data: Any = Field(..., description="Extracted structured data")
    validated: bool = Field(
        ...,
        description="Whether the data was validated against the schema",
    )


class ValidateRequest(BaseModel):
    """Request model for the validate endpoint."""

    data: Any = Field(..., description="JSON data to validate")
    json_schema: dict[str, Any] = Field(
        ...,
        description="JSON Schema to validate against",
    )


class ValidateResponse(BaseModel):
    """Response model for the validate endpoint."""

    valid: bool = Field(..., description="Whether the data is valid")
    error: str | None = Field(
        default=None,
        description="Validation error message if invalid",
    )


@app.post("/extract")
async def extract_data(request: ExtractRequest) -> ExtractResponse:
    """Extract structured JSON data from text using LLMs.

    This endpoint takes input text and a JSON schema, then uses a Language
    Learning Model to extract structured data that conforms to the provided schema.

    Args:
        request (ExtractRequest): The extraction request containing text, schema,
            and model configuration.

    Returns:
        ExtractResponse: The extracted data and validation status.

    Raises:
        HTTPException: If extraction fails or no model is specified.
    """
    # Handle empty inputs gracefully
    if not request.text and not request.json_schema:
        return ExtractResponse(data={}, validated=True)

    try:
        # Create configuration objects from request
        llm_config = LLMConfig(
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            repeat_penalty=request.repeat_penalty,
            repeat_last_n=request.repeat_last_n,
            n_ctx=request.n_ctx,
            max_tokens=request.max_tokens,
            seed=request.seed,
            timeout=request.timeout,
            max_retries=request.max_retries,
        )

        model_config = ModelConfig(
            openai_model=request.openai_model,
            google_model=request.google_model,
            anthropic_model=request.anthropic_model,
            cerebras_model=request.cerebras_model,
            groq_model=request.groq_model,
            bedrock_model=request.bedrock_model,
            ollama_model=request.ollama_model,
            ollama_base_url=request.ollama_base_url,
            ollama_keep_alive=request.ollama_keep_alive,
            openai_api_base=request.openai_api_base,
            anthropic_api_base=request.anthropic_api_base,
            bedrock_endpoint_url=request.bedrock_endpoint_url,
            openai_api_key=request.openai_api_key,
            openai_organization=request.openai_organization,
            google_api_key=request.google_api_key,
            anthropic_api_key=request.anthropic_api_key,
            cerebras_api_key=request.cerebras_api_key,
            groq_api_key=request.groq_api_key,
            aws_credentials_profile=request.aws_credentials_profile,
            aws_region=request.aws_region,
        )

        # Determine model_name and provider from the individual model arguments
        model_name = (
            request.openai_model
            or request.google_model
            or request.anthropic_model
            or request.cerebras_model
            or request.groq_model
            or request.bedrock_model
            or request.ollama_model
        )

        # Determine provider based on which model is specified
        provider = None
        if request.openai_model:
            provider = "openai"
        elif request.google_model:
            provider = "google"
        elif request.anthropic_model:
            provider = "anthropic"
        elif request.cerebras_model:
            provider = "cerebras"
        elif request.groq_model:
            provider = "groq"
        elif request.bedrock_model:
            provider = "bedrock"
        elif request.ollama_model:
            provider = "ollama"

        llm = create_llm_instance(
            model_name=model_name,
            provider=provider,
            ollama_base_url=model_config.ollama_base_url,
            ollama_keep_alive=model_config.ollama_keep_alive,
            cerebras_api_key=model_config.cerebras_api_key,
            groq_api_key=model_config.groq_api_key,
            google_api_key=model_config.google_api_key,
            anthropic_api_key=model_config.anthropic_api_key,
            anthropic_api_base=model_config.anthropic_api_base,
            openai_api_key=model_config.openai_api_key,
            openai_api_base=model_config.openai_api_base,
            openai_organization=model_config.openai_organization,
            temperature=llm_config.temperature,
            top_p=llm_config.top_p,
            top_k=llm_config.top_k,
            repeat_penalty=llm_config.repeat_penalty,
            repeat_last_n=llm_config.repeat_last_n,
            n_ctx=llm_config.n_ctx,
            max_tokens=llm_config.max_tokens,
            seed=llm_config.seed,
            timeout=llm_config.timeout,
            max_retries=llm_config.max_retries,
            aws_credentials_profile_name=model_config.aws_credentials_profile,
            aws_region=model_config.aws_region,
            bedrock_endpoint_base_url=model_config.bedrock_endpoint_url,
        )
        extracted_data = extract_structured_data_from_text(
            input_text=request.text,
            schema=request.json_schema,
            llm=llm,
            skip_validation=request.skip_validation,
            terminology=request.terminology,
        )
        response = ExtractResponse(
            data=extracted_data,
            validated=not request.skip_validation,
        )
    except ValueError as e:
        logger.exception("Invalid request")
        raise HTTPException(status_code=400, detail=str(e)) from e
    except JsonSchemaValidationError as e:
        logger.exception("Validation error")
        raise HTTPException(
            status_code=422,
            detail=f"Validation error: {e.message}",
        ) from e
    except Exception as e:
        logger.exception("Extraction failed")
        raise HTTPException(status_code=500, detail=f"Extraction failed: {e!s}") from e
    else:
        logger.info("Data extracted successfully")
        return response


@app.post("/validate")
async def validate_data(request: ValidateRequest) -> ValidateResponse:
    """Validate JSON data against a JSON Schema.

    Args:
        request (ValidateRequest): The validation request containing data
            and schema.

    Returns:
        ValidateResponse: The validation result.

    Raises:
        HTTPException: If validation encounters an unexpected error.
    """
    try:
        validate(instance=request.data, schema=request.json_schema)
        return ValidateResponse(valid=True, error=None)
    except JsonSchemaValidationError as e:
        return ValidateResponse(valid=False, error=e.message)
    except Exception as e:
        logger.exception("Validation error")
        raise HTTPException(status_code=500, detail=f"Validation error: {e!s}") from e


@app.get("/health")
async def health_check() -> dict[str, str]:
    """Health check endpoint.

    Returns:
        dict[str, str]: Health status.
    """
    return {"status": "healthy"}


def run_server(
    host: str = "0.0.0.0",  # noqa: S104
    port: int = 8000,
    reload: bool = True,
) -> None:
    """Run the FastAPI server using uvicorn.

    Args:
        host (str): The host to run the server on.
        port (int): The port to run the server on.
        reload (bool): Whether to enable auto-reload on code changes.
    """
    uvicorn.run(
        "sdeul.api:app",
        host=host,
        port=port,
        reload=reload,
    )
