#!/usr/bin/env python

import json
import logging
import os
from json.decoder import JSONDecodeError
from typing import Any

from jsonschema import validate
from jsonschema.exceptions import ValidationError
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.schema import StrOutputParser
from langchain_aws import ChatBedrockConverse
from langchain_community.llms import LlamaCpp
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from rich import print

from .llm import create_llm_instance
from .utility import log_execution_time, read_json_file, read_text_file, write_file

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
"""  # noqa: E501
_EXTRACTION_INPUT_VARIABLES = ["input_text"]
_DEFAULT_MODEL_NAMES = {
    "openai": "gpt-4o-mini",
    "google": "gemini-1.5-flash",
    "groq": "llama-3.1-70b-versatile",
    "bedrock": "anthropic.claude-3-5-sonnet-20240620-v1:0",
}
_DEFAULT_MAX_TOKENS = {
    "gpt-4o": 128000,
    "gpt-4o-2024-05-13": 128000,
    "gpt-4o-mini": 128000,
    "gpt-4o-mini-2024-07-18": 128000,
    "gpt-4o-2024-08-06": 128000,
    "o1-mini": 128000,
    "o1-mini-2024-09-12": 128000,
    "o1-preview": 128000,
    "o1-preview-2024-09-12": 128000,
    "claude-3-5-sonnet@20240620": 100000,
    "gemini-1.5-pro": 1048576,
    "gemini-1.5-flash": 1048576,
    "gemma2": 8200,
    "gemma2-9b-it": 8192,
    "claude-3-5-sonnet": 100000,
    "claude-3-5-sonnet-20240620": 100000,
    "anthropic.claude-3-5-sonnet-20240620-v1:0": 100000,
    "mixtral-8x7b-32768": 32768,
    "llama-3.1-8b-instant": 131072,
    "llama-3.1-70b-versatile": 131072,
    "llama-3.1-405b-reasoning": 131072,
}


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
    """Extract JSON from input text."""
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
    output_data_as_json(
        data=parsed_output_data,
        output_json_file_path=output_json_file_path,
        compact_json=compact_json,
    )


def output_data_as_json(
    data: Any,
    output_json_file_path: str | None = None,
    compact_json: bool = False,
) -> None:
    output_json_string = json.dumps(obj=data, indent=(None if compact_json else 2))
    if output_json_file_path:
        write_file(path=output_json_file_path, data=output_json_string)
    else:
        print(output_json_string)


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
    llm_chain: LLMChain = prompt | llm | StrOutputParser()
    logger.info(f"LLM chain: {llm_chain}")
    output_string = llm_chain.invoke({"input_text": input_text})
    logger.info(f"LLM output: {output_string}")
    if not output_string:
        raise RuntimeError("LLM output is empty.")
    else:
        parsed_output_data = _parse_llm_output(string=str(output_string))
        if skip_validation:
            logger.info("Skip validation using JSON Schema.")
        else:
            logger.info("Validate data using JSON Schema.")
            try:
                validate(instance=parsed_output_data, schema=schema)
            except ValidationError as e:
                logger.error(f"Validation failed: {parsed_output_data}")
                raise e
            else:
                logger.info("Validation succeeded.")
        return parsed_output_data


def _parse_llm_output(string: str) -> Any:
    logger = logging.getLogger(_parse_llm_output.__name__)
    json_string = None
    markdown = True
    for r in string.splitlines(keepends=False):
        if json_string is None:
            if r in {"```json", "```"}:
                json_string = ""
            elif r in {"[", "{"}:
                markdown = False
                json_string = r + os.linesep
            else:
                pass
        elif (markdown and r != "```") or (not markdown and r):
            json_string += r + os.linesep
        else:
            break
    logger.debug(f"json_string: {json_string}")
    if not json_string:
        raise RuntimeError(f"JSON code block is not found: {string}")
    else:
        try:
            output_data = json.loads(json_string)
        except JSONDecodeError as e:
            logger.error(f"Failed to parse the LLM output: {string}")
            raise e
        else:
            logger.info(f"Parsed output: {output_data}")
            return output_data
