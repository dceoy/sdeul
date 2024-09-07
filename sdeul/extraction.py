#!/usr/bin/env python

import json
import logging
import os
from json.decoder import JSONDecodeError
from typing import Any, Dict

from jsonschema import validate
from jsonschema.exceptions import ValidationError
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.schema import StrOutputParser
from langchain_aws import ChatBedrockConverse
from langchain_community.llms import LlamaCpp
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI

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

_DEFAULT_MODEL_NAMES = {
    "openai": "gpt-4o-mini",
    "google": "gemini-1.5-pro",
    "groq": "llama-3.1-70b-versatile",
    "bedrock": "anthropic.claude-3-5-sonnet-20240620-v1:0",
}


@log_execution_time
def extract_json_from_text(
    text_file_path: str,
    json_schema_file_path: str,
    model_file_path: str | None = None,
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
    pretty_json: bool = False,
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
    logger = logging.getLogger(extract_json_from_text.__name__)
    if model_file_path:
        llm = _read_llm_file(
            path=model_file_path,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            n_ctx=n_ctx,
            seed=seed,
            n_batch=n_batch,
            n_gpu_layers=n_gpu_layers,
            token_wise_streaming=token_wise_streaming,
        )
    else:
        overrided_env_vars = {
            "GROQ_API_KEY": groq_api_key,
            "GOOGLE_API_KEY": google_api_key,
            "OPENAI_API_KEY": openai_api_key,
            "OPENAI_ORGANIZATION": openai_organization,
        }
        for k, v in overrided_env_vars.items():
            if v:
                logger.info(f"Override environment variable: {k}")
                os.environ[k] = v
        if groq_model_name:
            llm = ChatGroq(
                model=(groq_model_name or _DEFAULT_MODEL_NAMES["groq"]),
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout,
                max_retries=max_retries,
                stop_sequences=None,
            )
        elif bedrock_model_id:
            llm = ChatBedrockConverse(
                model=(bedrock_model_id or _DEFAULT_MODEL_NAMES["bedrock"]),
                temperature=temperature,
                max_tokens=max_tokens,
                region_name=aws_region,
                base_url=bedrock_endpoint_base_url,
                credentials_profile_name=aws_credentials_profile_name,
            )
        elif google_model_name:
            llm = ChatGoogleGenerativeAI(
                model=(google_model_name or _DEFAULT_MODEL_NAMES["google"]),
                temperature=temperature,
                top_p=top_p,
                max_output_tokens=max_tokens,
                timeout=timeout,
                max_retries=max_retries,
            )
        elif openai_model_name:
            llm = ChatOpenAI(
                model=(openai_model_name or _DEFAULT_MODEL_NAMES["openai"]),
                temperature=temperature,
                top_p=top_p,
                seed=seed,
                max_tokens=max_tokens,
                timeout=timeout,
                max_retries=max_retries,
            )
        else:
            raise RuntimeError("No model is specified.")
    schema = read_json_file(path=json_schema_file_path)
    input_text = read_text_file(path=text_file_path)
    llm_chain = _create_llm_chain(schema=schema, llm=llm)

    logger.info("Start extracting JSON data from the input text.")
    output_string = llm_chain.invoke({"input_text": input_text})
    logger.info(f"LLM output: {output_string}")
    if not output_string:
        raise RuntimeError("LLM output is empty.")
    else:
        parsed_output_data = _parse_llm_output(output_string=str(output_string))
        logger.info(f"Parsed output: {parsed_output_data}")
        output_json_string = json.dumps(
            obj=parsed_output_data, indent=(2 if pretty_json else None)
        )
        if skip_validation:
            logger.info("Skip validation using JSON Schema.")
        else:
            logger.info("Validate the parsed output using JSON Schema.")
            try:
                validate(instance=parsed_output_data, schema=schema)
            except ValidationError as e:
                logger.error(f"Validation failed: {output_json_string}")
                raise e
            else:
                logger.info("Validation succeeded.")
        if output_json_file_path:
            write_file(path=output_json_file_path, data=output_json_string)
        else:
            print(output_json_string)


def _parse_llm_output(output_string: str) -> Any:
    logger = logging.getLogger(_parse_llm_output.__name__)
    json_string = None
    markdown = True
    for r in output_string.splitlines(keepends=False):
        if json_string is None:
            if r in ("```json", "```"):
                json_string = ""
            elif r in ("[", "{"):
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
        raise RuntimeError(f"JSON code block is not found: {output_string}")
    else:
        try:
            output_data = json.loads(json_string)
        except JSONDecodeError as e:
            logger.error(f"Failed to parse the LLM output: {output_string}")
            raise e
        else:
            logger.debug(f"output_data: {output_data}")
            return output_data


def _create_llm_chain(schema: Dict[str, Any], llm: LlamaCpp) -> LLMChain:
    logger = logging.getLogger(_create_llm_chain.__name__)
    prompt = PromptTemplate(
        template=_EXTRACTION_TEMPLATE,
        input_variables=["input_text"],
        partial_variables={"schema": json.dumps(obj=schema)},
    )
    chain = prompt | llm | StrOutputParser()
    logger.info(f"LLM chain: {chain}")
    return chain


def _read_llm_file(
    path: str,
    temperature: float = 0.8,
    top_p: float = 0.95,
    max_tokens: int = 256,
    n_ctx: int = 512,
    seed: int = -1,
    n_batch: int = 8,
    n_gpu_layers: int = -1,
    token_wise_streaming: bool = False,
) -> LlamaCpp:
    logger = logging.getLogger(_read_llm_file.__name__)
    logger.info(f"Read a Llama 2 model file: {path}")
    llm = LlamaCpp(
        model_path=path,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        n_ctx=n_ctx,
        seed=seed,
        n_batch=n_batch,
        n_gpu_layers=n_gpu_layers,
        verbose=(token_wise_streaming or logging.getLogger().level <= logging.DEBUG),
        callback_manager=(
            CallbackManager([StreamingStdOutCallbackHandler()])
            if token_wise_streaming
            else None
        ),
    )
    logger.debug(f"llm: {llm}")
    return llm
