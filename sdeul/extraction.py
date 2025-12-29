"""Functions for extracting structured JSON data from unstructured text.

This module provides the core functionality for extracting JSON data from text
files using various Language Learning Models. It handles the complete workflow
from reading input files to generating validated JSON output.

Functions:
    extract_json_from_text_file: Main function for extracting JSON from text files
    extract_structured_data_from_text: Function for LLM-based extraction
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any, cast

from jsonschema import validate
from jsonschema.exceptions import ValidationError
from langchain_core.prompts import ChatPromptTemplate

from .config import (
    ExtractConfig,
    LLMConfig,
    ModelConfig,
    ProcessingConfig,
)
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
    SYSTEM_PROMPT_BASE,
    SYSTEM_PROMPT_WITH_TERMINOLOGY,
    USER_PROMPT_TEMPLATE_BASE,
    USER_PROMPT_TEMPLATE_WITH_TERMINOLOGY,
)
from .llm import JsonCodeOutputParser, create_llm_instance
from .utility import (
    log_execution_time,
    read_json_file,
    read_text_file,
    write_or_print_json_data,
)

if TYPE_CHECKING:
    from langchain_core.language_models.chat_models import BaseChatModel


@log_execution_time
def extract_json_from_text_file(
    text_file_path: str,
    json_schema_file_path: str,
    model_name: str | None = None,
    provider: str | None = None,
    ollama_base_url: str | None = None,
    cerebras_api_key: str | None = None,
    groq_api_key: str | None = None,
    google_api_key: str | None = None,
    anthropic_api_key: str | None = None,
    anthropic_api_base: str | None = None,
    openai_api_key: str | None = None,
    openai_api_base: str | None = None,
    openai_organization: str | None = None,
    output_json_file_path: str | None = None,
    compact_json: bool = False,
    skip_validation: bool = False,
    terminology_file_path: str | None = None,
    temperature: float = DEFAULT_TEMPERATURE,
    top_p: float = DEFAULT_TOP_P,
    top_k: int = DEFAULT_TOP_K,
    repeat_penalty: float = DEFAULT_REPEAT_PENALTY,
    repeat_last_n: int = DEFAULT_REPEAT_LAST_N,
    n_ctx: int = DEFAULT_CONTEXT_WINDOW,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    seed: int = DEFAULT_SEED,
    timeout: int | None = DEFAULT_TIMEOUT,
    max_retries: int = DEFAULT_MAX_RETRIES,
    aws_credentials_profile_name: str | None = None,
    aws_region: str | None = None,
    bedrock_endpoint_base_url: str | None = None,
) -> None:
    """Extract structured JSON data from a text file using an LLM.

    Reads a text file and JSON schema, then uses a Language Learning Model
    to extract structured data that conforms to the provided schema. The
    extracted data can be validated and output to a file or stdout.

    Args:
        text_file_path (str): Path to the input text file containing
            unstructured data.
        json_schema_file_path (str): Path to the JSON schema file defining
            output structure.
        model_name (str | None): Name or ID of the model to use.
        provider (str | None): LLM provider to use (openai, google, anthropic,
            cerebras, groq, bedrock, ollama). If not specified, will be
            inferred from API keys and environment.
        ollama_base_url (str | None): Custom Ollama API base URL.
        cerebras_api_key (str | None): Cerebras API key (overrides environment
            variable).
        groq_api_key (str | None): Groq API key (overrides environment
            variable).
        google_api_key (str | None): Google API key (overrides environment
            variable).
        anthropic_api_key (str | None): Anthropic API key (overrides environment
            variable).
        anthropic_api_base (str | None): Custom Anthropic API base URL.
        openai_api_key (str | None): OpenAI API key (overrides environment
            variable).
        openai_api_base (str | None): Custom OpenAI API base URL.
        openai_organization (str | None): OpenAI organization ID.
        output_json_file_path (str | None): Optional path to save extracted JSON.
            If None, prints to stdout.
        compact_json (bool): If True, outputs JSON in compact format without
            indentation.
        skip_validation (bool): If True, skips JSON schema validation of
            extracted data.
        terminology_file_path (str | None): Optional path to a file containing
            domain-specific terminology definitions or glossary to help the LLM
            interpret specialized terms.
        temperature (float): Sampling temperature for randomness (0.0-2.0).
        top_p (float): Top-p value for nucleus sampling (0.0-1.0).
        top_k (int): Top-k value for limiting token choices.
        repeat_penalty (float): Penalty for repeating tokens (1.0 = no penalty).
        repeat_last_n (int): Number of tokens to consider for repeat penalty.
        n_ctx (int): Token context window size.
        max_tokens (int): Maximum number of tokens to generate.
        seed (int): Random seed for reproducible output (-1 for random).
        timeout (int | None): API request timeout in seconds.
        max_retries (int): Maximum number of API request retries.
        aws_credentials_profile_name (str | None): AWS credentials profile name
            for Bedrock.
        aws_region (str | None): AWS region for Bedrock service.
        bedrock_endpoint_base_url (str | None): Custom Bedrock endpoint URL.
    """
    config = ExtractConfig(
        llm=LLMConfig(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repeat_penalty=repeat_penalty,
            repeat_last_n=repeat_last_n,
            n_ctx=n_ctx,
            max_tokens=max_tokens,
            seed=seed,
            timeout=timeout,
            max_retries=max_retries,
        ),
        model=ModelConfig(
            model_name=model_name,
            provider=provider,
            ollama_base_url=ollama_base_url,
            openai_api_key=openai_api_key,
            openai_api_base=openai_api_base,
            openai_organization=openai_organization,
            google_api_key=google_api_key,
            anthropic_api_key=anthropic_api_key,
            anthropic_api_base=anthropic_api_base,
            cerebras_api_key=cerebras_api_key,
            groq_api_key=groq_api_key,
            aws_credentials_profile=aws_credentials_profile_name,
            aws_region=aws_region,
            bedrock_endpoint_url=bedrock_endpoint_base_url,
        ),
        processing=ProcessingConfig(
            output_json_file=output_json_file_path,
            compact_json=compact_json,
            skip_validation=skip_validation,
            terminology_file=terminology_file_path,
        ),
    )
    extract_json_from_text_file_with_config(
        text_file_path=text_file_path,
        json_schema_file_path=json_schema_file_path,
        config=config,
        infer_provider=False,
    )


def _build_prompt_and_params(
    input_text: str,
    schema: dict[str, Any],
    terminology: str | None,
) -> tuple[ChatPromptTemplate, dict[str, str]]:
    if terminology:
        system_prompt = SYSTEM_PROMPT_WITH_TERMINOLOGY
        user_prompt_template = USER_PROMPT_TEMPLATE_WITH_TERMINOLOGY
        invoke_params = {
            "schema": json.dumps(obj=schema),
            "input_text": input_text,
            "terminology": terminology,
        }
    else:
        system_prompt = SYSTEM_PROMPT_BASE
        user_prompt_template = USER_PROMPT_TEMPLATE_BASE
        invoke_params = {
            "schema": json.dumps(obj=schema),
            "input_text": input_text,
        }

    prompt = ChatPromptTemplate(
        [
            ("system", system_prompt),
            ("user", user_prompt_template),
        ],
    )
    return prompt, invoke_params


def _resolve_model_selection(model: ModelConfig) -> tuple[str | None, str | None]:
    legacy_order = (
        ("openai", model.openai_model),
        ("google", model.google_model),
        ("anthropic", model.anthropic_model),
        ("cerebras", model.cerebras_model),
        ("groq", model.groq_model),
        ("bedrock", model.bedrock_model),
        ("ollama", model.ollama_model),
    )
    for provider, model_name in legacy_order:
        if model_name:
            return model_name, provider
    return None, None


def extract_structured_data_from_text(
    input_text: str,
    schema: dict[str, Any],
    llm: BaseChatModel,
    skip_validation: bool = False,
    terminology: str | None = None,
) -> Any:  # noqa: ANN401
    """Extract structured data from text using an LLM and JSON schema.

    This function uses a Language Learning Model to extract structured data
    from unstructured text according to a provided JSON schema. The extracted
    data is optionally validated against the schema.

    Args:
        input_text (str): The unstructured text to extract data from.
        schema (dict[str, Any]): JSON schema defining the structure of the
            expected output.
        llm (BaseChatModel): The Language Learning Model instance to use for extraction.
        skip_validation (bool): Whether to skip JSON schema validation of
            the output.
        terminology (str | None): Optional domain-specific terminology definitions
            or glossary to help the LLM interpret specialized terms.

    Returns:
        Any: The extracted structured data as a Python object.

    Raises:
        ValidationError: If validation is enabled and the extracted data
            doesn't conform to the provided schema.
    """
    logger = logging.getLogger(extract_structured_data_from_text.__name__)
    logger.info("Start extracting structured data from the input text.")

    prompt, invoke_params = _build_prompt_and_params(
        input_text=input_text,
        schema=schema,
        terminology=terminology,
    )
    llm_chain = cast("Any", prompt | llm | JsonCodeOutputParser())
    logger.info("LLM chain: %s", llm_chain)
    parsed_output_data: Any = llm_chain.invoke(invoke_params)
    logger.info("LLM output: %s", parsed_output_data)
    if skip_validation:
        logger.info("Skip validation using JSON Schema.")
    else:
        logger.info("Validate data using JSON Schema.")
        try:
            validate(instance=parsed_output_data, schema=schema)
        except ValidationError:
            logger.exception("Validation failed: %s", parsed_output_data)
            raise
        else:
            logger.info("Validation succeeded.")
    return parsed_output_data


@log_execution_time
def extract_json_from_text_file_with_config(
    text_file_path: str,
    json_schema_file_path: str,
    config: ExtractConfig,
    infer_provider: bool = True,
) -> None:
    """Extract structured JSON data from a text file using configuration objects.

    This is a simplified version of extract_json_from_text_file that uses
    configuration dataclasses instead of a large number of individual parameters.
    This follows Kent Beck's tidying principle of grouping related parameters.

    Args:
        text_file_path (str): Path to the input text file containing
            unstructured data.
        json_schema_file_path (str): Path to the JSON schema file defining
            output structure.
        config (ExtractConfig): Configuration object containing all LLM,
            model, and processing settings.
        infer_provider (bool): Whether to infer provider and model name from
            legacy model settings in the configuration.
    """
    schema = read_json_file(path=json_schema_file_path)
    input_text = read_text_file(path=text_file_path)

    # Read terminology file if provided
    terminology = None
    if config.processing.terminology_file:
        terminology = read_text_file(path=config.processing.terminology_file)

    # Create LLM instance using config
    if infer_provider:
        model_name, provider = _resolve_model_selection(config.model)
    else:
        model_name = config.model.model_name
        provider = config.model.provider

    llm = create_llm_instance(
        # Model selection
        model_name=model_name,
        provider=provider,
        ollama_base_url=config.model.ollama_base_url,
        ollama_keep_alive=config.model.ollama_keep_alive,
        cerebras_api_key=config.model.cerebras_api_key,
        groq_api_key=config.model.groq_api_key,
        google_api_key=config.model.google_api_key,
        anthropic_api_key=config.model.anthropic_api_key,
        anthropic_api_base=config.model.anthropic_api_base,
        openai_api_key=config.model.openai_api_key,
        openai_api_base=config.model.openai_api_base,
        openai_organization=config.model.openai_organization,
        # LLM parameters
        temperature=config.llm.temperature,
        top_p=config.llm.top_p,
        top_k=config.llm.top_k,
        max_tokens=config.llm.max_tokens,
        seed=config.llm.seed,
        timeout=config.llm.timeout,
        max_retries=config.llm.max_retries,
        repeat_penalty=config.llm.repeat_penalty,
        repeat_last_n=config.llm.repeat_last_n,
        n_ctx=config.llm.n_ctx,
        # AWS parameters
        aws_credentials_profile_name=config.model.aws_credentials_profile,
        aws_region=config.model.aws_region,
        bedrock_endpoint_base_url=config.model.bedrock_endpoint_url,
    )

    # Extract structured data
    parsed_output_data = extract_structured_data_from_text(
        input_text=input_text,
        schema=schema,
        llm=llm,
        skip_validation=config.processing.skip_validation,
        terminology=terminology,
    )

    # Write or print output
    write_or_print_json_data(
        data=parsed_output_data,
        output_json_file_path=config.processing.output_json_file,
        compact_json=config.processing.compact_json,
    )
