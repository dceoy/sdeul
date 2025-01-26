#!/usr/bin/env python
"""Functions for extracting JSON from text."""

import json
import logging
from typing import TYPE_CHECKING, Any

from jsonschema import validate
from jsonschema.exceptions import ValidationError
from langchain.prompts import PromptTemplate
from langchain_aws import ChatBedrockConverse
from langchain_community.llms import LlamaCpp
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI

from .llm import JsonCodeOutputParser, create_llm_instance
from .utility import (
    log_execution_time,
    read_json_file,
    read_text_file,
    write_or_print_json_data,
)

if TYPE_CHECKING:
    from langchain.chains import LLMChain

_EXTRACTION_TEMPLATE = """\
Input text:
```
{input_text}
```

Provided JSON schema:
```json
{schema}
```

Instructions:
- Extract only the relevant entities defined by the provided JSON schema from the input text.
- Generate the extracted entities in JSON format according to the schema.
- If a property is not present in the schema, DO NOT include it in the output.
- Output the JSON data in a markdown code block.
- Provide complete, unabridged code in all responses without omitting any parts.
"""  # noqa: E501
_EXTRACTION_INPUT_VARIABLES = ["input_text"]


@log_execution_time
def extract_json_from_text_file(
    text_file_path: str,
    json_schema_file_path: str,
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
    temperature: float = 0.8,
    top_p: float = 0.95,
    max_tokens: int = 8192,
    n_ctx: int = 512,
    seed: int = -1,
    n_batch: int = 8,
    n_gpu_layers: int = -1,
    token_wise_streaming: bool = False,
    timeout: int | None = None,
    max_retries: int = 2,
    aws_credentials_profile_name: str | None = None,
    aws_region: str | None = None,
    bedrock_endpoint_base_url: str | None = None,
) -> None:
    """Extract JSON from input text.

    Args:
        text_file_path: Path to the input text file.
        json_schema_file_path: Path to the JSON schema file.
        llamacpp_model_file_path: Path to the LlamaCpp model file.
        groq_model_name: Name of the Groq model.
        groq_api_key: API key
        bedrock_model_id: Bedrock model ID.
        google_model_name: Name of the Google model.
        google_api_key: API key of the Google model.
        openai_model_name: Name of the OpenAI model.
        openai_api_key: API key of the OpenAI model.
        openai_api_base: Base URL of the OpenAI API.
        openai_organization: Organization of the OpenAI.
        output_json_file_path: Path to the output JSON file.
        compact_json: Flag to output the JSON in compact format.
        skip_validation: Flag to skip JSON validation.
        temperature: Temperature of the model.
        top_p: Top-p of the model.
        max_tokens: Maximum number of tokens.
        n_ctx: Context size.
        seed: Seed of the model.
        n_batch: Batch size.
        n_gpu_layers: Number of GPU layers.
        token_wise_streaming: Flag to enable token-wise streaming.
        timeout: Timeout of the model.
        max_retries: Maximum number of retries.
        aws_credentials_profile_name: Name of the AWS credentials profile.
        aws_region: AWS region.
        bedrock_endpoint_base_url: Base URL of the Amazon Bedrock endpoint.
    """
    llm = create_llm_instance(
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
        max_tokens=max_tokens,
        n_ctx=n_ctx,
        seed=seed,
        n_batch=n_batch,
        n_gpu_layers=n_gpu_layers,
        token_wise_streaming=token_wise_streaming,
        timeout=timeout,
        max_retries=max_retries,
        aws_credentials_profile_name=aws_credentials_profile_name,
        aws_region=aws_region,
        bedrock_endpoint_base_url=bedrock_endpoint_base_url,
    )
    schema = read_json_file(path=json_schema_file_path)
    input_text = read_text_file(path=text_file_path)
    parsed_output_data = _extruct_structured_data_from_text(
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


def _extruct_structured_data_from_text(
    input_text: str,
    schema: dict[str, Any],
    llm: LlamaCpp
    | ChatGroq
    | ChatBedrockConverse
    | ChatGoogleGenerativeAI
    | ChatOpenAI,
    skip_validation: bool = False,
) -> Any:
    logger = logging.getLogger(_extruct_structured_data_from_text.__name__)
    logger.info("Start extracting structured data from the input text.")
    prompt = PromptTemplate(
        template=_EXTRACTION_TEMPLATE,
        input_variables=_EXTRACTION_INPUT_VARIABLES,
        partial_variables={"schema": json.dumps(obj=schema)},
    )
    llm_chain: LLMChain = prompt | llm | JsonCodeOutputParser()
    logger.info("LLM chain: %s", llm_chain)
    parsed_output_data = llm_chain.invoke({"input_text": input_text})
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
