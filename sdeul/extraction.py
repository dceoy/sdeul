"""Functions for extracting structured JSON data from unstructured text.

This module provides the core functionality for extracting JSON data from text
files using various Language Learning Models. It handles the complete workflow
from reading input files to generating validated JSON output.

Functions:
    extract_json_from_text_file: Main function for extracting JSON from text files
    extract_structured_data_from_text: Function for LLM-based extraction
"""

import json
import logging
from multiprocessing import cpu_count
from typing import TYPE_CHECKING, Any

from jsonschema import validate
from jsonschema.exceptions import ValidationError
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate

from .config import ExtractConfig
from .constants import (
    DEFAULT_CONTEXT_WINDOW,
    DEFAULT_F16_KV,
    DEFAULT_MAX_RETRIES,
    DEFAULT_MAX_TOKENS,
    DEFAULT_N_BATCH,
    DEFAULT_N_GPU_LAYERS,
    DEFAULT_N_THREADS,
    DEFAULT_REPEAT_LAST_N,
    DEFAULT_REPEAT_PENALTY,
    DEFAULT_SEED,
    DEFAULT_TEMPERATURE,
    DEFAULT_TIMEOUT,
    DEFAULT_TOKEN_WISE_STREAMING,
    DEFAULT_TOP_K,
    DEFAULT_TOP_P,
    DEFAULT_USE_MLOCK,
    DEFAULT_USE_MMAP,
    SYSTEM_PROMPT,
    USER_PROMPT_TEMPLATE,
)
from .llm import JsonCodeOutputParser, create_llm_instance
from .utility import (
    log_execution_time,
    read_json_file,
    read_text_file,
    write_or_print_json_data,
)

if TYPE_CHECKING:
    from langchain.chains import LLMChain


@log_execution_time
def extract_json_from_text_file(
    text_file_path: str,
    json_schema_file_path: str,
    model_name: str | None = None,
    provider: str | None = None,
    ollama_base_url: str | None = None,
    llamacpp_model_file_path: str | None = None,
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
    n_batch: int = DEFAULT_N_BATCH,
    n_threads: int = DEFAULT_N_THREADS,
    n_gpu_layers: int = DEFAULT_N_GPU_LAYERS,
    f16_kv: bool = DEFAULT_F16_KV,
    use_mlock: bool = DEFAULT_USE_MLOCK,
    use_mmap: bool = DEFAULT_USE_MMAP,
    token_wise_streaming: bool = DEFAULT_TOKEN_WISE_STREAMING,
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
            cerebras, groq, bedrock, ollama, llamacpp). If not specified, will be
            inferred from API keys and environment.
        ollama_base_url (str | None): Custom Ollama API base URL.
        llamacpp_model_file_path (str | None): Path to local GGUF model file
            for llama.cpp.
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
        n_batch (int): Number of tokens to process in parallel (llama.cpp only).
        n_threads (int): Number of CPU threads to use (llama.cpp only).
        n_gpu_layers (int): Number of layers to offload to GPU (llama.cpp only).
        f16_kv (bool): Use half-precision for key/value cache (llama.cpp only).
        use_mlock (bool): Force system to keep model in RAM (llama.cpp only).
        use_mmap (bool): Keep the model loaded in RAM (llama.cpp only).
        token_wise_streaming (bool): Enable token-wise streaming output
            (llama.cpp only).
        timeout (int | None): API request timeout in seconds.
        max_retries (int): Maximum number of API request retries.
        aws_credentials_profile_name (str | None): AWS credentials profile name
            for Bedrock.
        aws_region (str | None): AWS region for Bedrock service.
        bedrock_endpoint_base_url (str | None): Custom Bedrock endpoint URL.
    """
    llm = create_llm_instance(
        model_name=model_name,
        provider=provider,
        ollama_base_url=ollama_base_url,
        llamacpp_model_file_path=llamacpp_model_file_path,
        cerebras_api_key=cerebras_api_key,
        groq_api_key=groq_api_key,
        google_api_key=google_api_key,
        anthropic_api_key=anthropic_api_key,
        anthropic_api_base=anthropic_api_base,
        openai_api_key=openai_api_key,
        openai_api_base=openai_api_base,
        openai_organization=openai_organization,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        repeat_penalty=repeat_penalty,
        repeat_last_n=repeat_last_n,
        n_ctx=n_ctx,
        max_tokens=max_tokens,
        seed=seed,
        n_batch=n_batch,
        n_threads=(n_threads if n_threads > 0 else cpu_count()),
        n_gpu_layers=n_gpu_layers,
        f16_kv=f16_kv,
        use_mlock=use_mlock,
        use_mmap=use_mmap,
        token_wise_streaming=token_wise_streaming,
        timeout=timeout,
        max_retries=max_retries,
        aws_credentials_profile_name=aws_credentials_profile_name,
        aws_region=aws_region,
        bedrock_endpoint_base_url=bedrock_endpoint_base_url,
    )
    schema = read_json_file(path=json_schema_file_path)
    input_text = read_text_file(path=text_file_path)

    # Read terminology file if provided
    terminology = None
    if terminology_file_path:
        terminology = read_text_file(path=terminology_file_path)

    parsed_output_data = extract_structured_data_from_text(
        input_text=input_text,
        schema=schema,
        llm=llm,
        skip_validation=skip_validation,
        terminology=terminology,
    )
    write_or_print_json_data(
        data=parsed_output_data,
        output_json_file_path=output_json_file_path,
        compact_json=compact_json,
    )


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

    # Format terminology section for the prompt
    terminology_section = ""
    if terminology:
        terminology_section = f"""
Domain-specific terminology and context:
```
{terminology}
```
"""

    prompt = ChatPromptTemplate([
        ("system", SYSTEM_PROMPT),
        ("user", USER_PROMPT_TEMPLATE),
    ])
    llm_chain: LLMChain = prompt | llm | JsonCodeOutputParser()  # pyright: ignore[reportUnknownVariableType]
    logger.info("LLM chain: %s", llm_chain)
    parsed_output_data = llm_chain.invoke({
        "schema": json.dumps(obj=schema),
        "input_text": input_text,
        "terminology": terminology_section.strip(),
    })
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
    """
    schema = read_json_file(path=json_schema_file_path)
    input_text = read_text_file(path=text_file_path)

    # Create LLM instance using config
    # Determine model name and provider from config
    model_name = (
        config.model.openai_model
        or config.model.google_model
        or config.model.anthropic_model
        or config.model.cerebras_model
        or config.model.groq_model
        or config.model.bedrock_model
        or config.model.ollama_model
    )

    # Determine provider based on which model is specified
    provider = None
    if config.model.openai_model:
        provider = "openai"
    elif config.model.google_model:
        provider = "google"
    elif config.model.anthropic_model:
        provider = "anthropic"
    elif config.model.cerebras_model:
        provider = "cerebras"
    elif config.model.groq_model:
        provider = "groq"
    elif config.model.bedrock_model:
        provider = "bedrock"
    elif config.model.ollama_model:
        provider = "ollama"
    elif config.model.llamacpp_model_file:
        provider = "llamacpp"

    llm = create_llm_instance(
        # Model selection
        model_name=model_name,
        provider=provider,
        ollama_base_url=config.model.ollama_base_url,
        llamacpp_model_file_path=config.model.llamacpp_model_file,
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
        # LlamaCpp parameters
        repeat_penalty=config.llamacpp.repeat_penalty,
        repeat_last_n=config.llamacpp.repeat_last_n,
        n_ctx=config.llamacpp.n_ctx,
        n_batch=config.llamacpp.n_batch,
        n_threads=config.llamacpp.n_threads,
        n_gpu_layers=config.llamacpp.n_gpu_layers,
        f16_kv=config.llamacpp.f16_kv,
        use_mlock=config.llamacpp.use_mlock,
        use_mmap=config.llamacpp.use_mmap,
        token_wise_streaming=config.llamacpp.token_wise_streaming,
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
    )

    # Write or print output
    write_or_print_json_data(
        data=parsed_output_data,
        output_json_file_path=config.processing.output_json_file,
        compact_json=config.processing.compact_json,
    )
