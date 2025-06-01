"""Functions for extracting structured JSON data from unstructured text.

This module provides the core functionality for extracting JSON data from text
files using various Language Learning Models. It handles the complete workflow
from reading input files to generating validated JSON output.

Functions:
    extract_json_from_text_file: Main function for extracting JSON from text files
    _extract_structured_data_from_text: Internal function for LLM-based extraction
"""

import json
import logging
from multiprocessing import cpu_count
from typing import TYPE_CHECKING, Any

from jsonschema import validate
from jsonschema.exceptions import ValidationError
from langchain_aws import ChatBedrockConverse
from langchain_community.llms import LlamaCpp
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

from .constants import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE
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
    ollama_model_name: str | None = None,
    ollama_base_url: str | None = None,
    llamacpp_model_file_path: str | None = None,
    groq_model_name: str | None = None,
    groq_api_key: str | None = None,
    bedrock_model_id: str | None = None,
    google_model_name: str | None = None,
    google_api_key: str | None = None,
    openai_model_name: str | None = None,
    openai_api_key: str | None = None,
    openai_api_base: str | None = None,
    openai_organization: str | None = None,
    output_json_file_path: str | None = None,
    compact_json: bool = False,
    skip_validation: bool = False,
    temperature: float = 0.0,
    top_p: float = 0.95,
    top_k: int = 64,
    repeat_penalty: float = 1.1,
    repeat_last_n: int = 64,
    n_ctx: int = 512,
    max_tokens: int = 8192,
    seed: int = -1,
    n_batch: int = 8,
    n_threads: int = -1,
    n_gpu_layers: int = -1,
    f16_kv: bool = True,
    use_mlock: bool = False,
    use_mmap: bool = True,
    token_wise_streaming: bool = False,
    timeout: int | None = None,
    max_retries: int = 2,
    aws_credentials_profile_name: str | None = None,
    aws_region: str | None = None,
    bedrock_endpoint_base_url: str | None = None,
) -> None:
    """Extract structured JSON data from a text file using an LLM.

    Reads a text file and JSON schema, then uses a Language Learning Model
    to extract structured data that conforms to the provided schema. The
    extracted data can be validated and output to a file or stdout.

    Args:
        text_file_path: Path to the input text file containing unstructured data.
        json_schema_file_path: Path to the JSON schema file defining output structure.
        ollama_model_name: Ollama model name.
        ollama_base_url: Custom Ollama API base URL.
        llamacpp_model_file_path: Path to local GGUF model file for llama.cpp.
        groq_model_name: Groq model name.
        groq_api_key: Groq API key (overrides environment variable).
        bedrock_model_id: Amazon Bedrock model ID.
        google_model_name: Google Generative AI model name.
        google_api_key: Google API key (overrides environment variable).
        openai_model_name: OpenAI model name.
        openai_api_key: OpenAI API key (overrides environment variable).
        openai_api_base: Custom OpenAI API base URL.
        openai_organization: OpenAI organization ID.
        output_json_file_path: Optional path to save extracted JSON. If None,
            prints to stdout.
        compact_json: If True, outputs JSON in compact format without indentation.
        skip_validation: If True, skips JSON schema validation of extracted data.
        temperature: Sampling temperature for randomness (0.0-2.0).
        top_p: Top-p value for nucleus sampling (0.0-1.0).
        top_k: Top-k value for limiting token choices.
        repeat_penalty: Penalty for repeating tokens (1.0 = no penalty).
        repeat_last_n: Number of tokens to consider for repeat penalty.
        n_ctx: Token context window size.
        max_tokens: Maximum number of tokens to generate.
        seed: Random seed for reproducible output (-1 for random).
        n_batch: Number of tokens to process in parallel (llama.cpp only).
        n_threads: Number of CPU threads to use (llama.cpp only).
        n_gpu_layers: Number of layers to offload to GPU (llama.cpp only).
        f16_kv: Use half-precision for key/value cache (llama.cpp only).
        use_mlock: Force system to keep model in RAM (llama.cpp only).
        use_mmap: Keep the model loaded in RAM (llama.cpp only).
        token_wise_streaming: Enable token-wise streaming output (llama.cpp only).
        timeout: API request timeout in seconds.
        max_retries: Maximum number of API request retries.
        aws_credentials_profile_name: AWS credentials profile name for Bedrock.
        aws_region: AWS region for Bedrock service.
        bedrock_endpoint_base_url: Custom Bedrock endpoint URL.
    """
    llm = create_llm_instance(
        ollama_model_name=ollama_model_name,
        ollama_base_url=ollama_base_url,
        llamacpp_model_file_path=llamacpp_model_file_path,
        groq_model_name=groq_model_name,
        groq_api_key=groq_api_key,
        bedrock_model_id=bedrock_model_id,
        google_model_name=google_model_name,
        google_api_key=google_api_key,
        openai_model_name=openai_model_name,
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
    parsed_output_data = _extract_structured_data_from_text(
        input_text=input_text,
        schema=schema,
        llm=llm,
        skip_validation=skip_validation,
    )
    write_or_print_json_data(
        data=parsed_output_data,
        output_json_file_path=output_json_file_path,
        compact_json=compact_json,
    )


def _extract_structured_data_from_text(
    input_text: str,
    schema: dict[str, Any],
    llm: ChatOllama
    | LlamaCpp
    | ChatGroq
    | ChatBedrockConverse
    | ChatGoogleGenerativeAI
    | ChatOpenAI,
    skip_validation: bool = False,
) -> Any:  # noqa: ANN401
    """Extract structured data from text using an LLM and JSON schema.

    This function uses a Language Learning Model to extract structured data
    from unstructured text according to a provided JSON schema. The extracted
    data is optionally validated against the schema.

    Args:
        input_text: The unstructured text to extract data from.
        schema: JSON schema defining the structure of the expected output.
        llm: The Language Learning Model instance to use for extraction.
        skip_validation: Whether to skip JSON schema validation of the output.

    Returns:
        Any: The extracted structured data as a Python object.

    Raises:
        ValidationError: If validation is enabled and the extracted data
            doesn't conform to the provided schema.
    """
    logger = logging.getLogger(_extract_structured_data_from_text.__name__)
    logger.info("Start extracting structured data from the input text.")
    prompt = ChatPromptTemplate([
        ("system", SYSTEM_PROMPT),
        ("user", USER_PROMPT_TEMPLATE),
    ])
    llm_chain: LLMChain = prompt | llm | JsonCodeOutputParser()  # pyright: ignore[reportUnknownVariableType]
    logger.info("LLM chain: %s", llm_chain)
    parsed_output_data = llm_chain.invoke({
        "schema": json.dumps(obj=schema),
        "input_text": input_text,
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
