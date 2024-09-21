#!/usr/bin/env python

import json
import logging
import os
from json.decoder import JSONDecodeError

import pytest
from jsonschema import ValidationError
from pytest_mock import MockerFixture

from sdeul.extraction import (
    _EXTRACTION_INPUT_VARIABLES,
    _EXTRACTION_TEMPLATE,
    _create_llm_instance,
    _parse_llm_output,
    _read_llm_file,
    extract_json_from_text,
)

_TEST_TEXT = "This is a test input text."
_TEST_SCHEMA = {
    "type": "object",
    "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
}
_TEST_LLM_OUTPUT = """
```json
{
    "name": "John Doe",
    "age": 30
}
```
"""
_TEST_LLM_OUTPUT_JSON = {"name": "John Doe", "age": 30}


@pytest.mark.parametrize(
    "skip_validation, pretty_json, output_json_file_path, expected_indent",
    [
        (False, False, None, None),
        (True, False, None, None),
        (True, True, None, 2),
        (True, False, "output.json", None),
    ],
)
def test_extract_json_from_text(
    skip_validation: bool,
    pretty_json: bool,
    output_json_file_path: str,
    expected_indent: int,
    capsys: pytest.CaptureFixture[str],
    mocker: MockerFixture,
) -> None:
    text_file_path = "input.txt"
    json_schema_file_path = "schema.json"
    model_file_path = "model.gguf"
    temperature = 0.8
    top_p = 0.95
    max_tokens = 8192
    n_ctx = 512
    seed = -1
    n_batch = 8
    n_gpu_layers = -1
    token_wise_streaming = False
    timeout = None
    max_retries = 2
    schema = {"type": "object"}
    input_text = "This is a test input text."
    expected_json_ouput = json.dumps(obj=_TEST_LLM_OUTPUT_JSON, indent=expected_indent)
    mock_llm_chain = mocker.MagicMock()
    mock__create_llm_instance = mocker.patch(
        "sdeul.extraction._create_llm_instance", return_value=mock_llm_chain
    )
    mock_read_json_file = mocker.patch(
        "sdeul.extraction.read_json_file", return_value=schema
    )
    mock_read_text_file = mocker.patch(
        "sdeul.extraction.read_text_file", return_value=input_text
    )
    mock_promot_template = mocker.patch(
        "sdeul.extraction.PromptTemplate", return_value=mock_llm_chain
    )
    mock_str_output_parser = mocker.patch(
        "sdeul.extraction.StrOutputParser", return_value=mock_llm_chain
    )
    mock_llm_chain.__or__.return_value = mock_llm_chain
    mock_llm_chain.invoke.return_value = _TEST_LLM_OUTPUT
    mock__parse_llm_output = mocker.patch(
        "sdeul.extraction._parse_llm_output", return_value=_TEST_LLM_OUTPUT_JSON
    )
    mock_validate = mocker.patch("sdeul.extraction.validate")
    mock_write_file = mocker.patch("sdeul.extraction.write_file")

    extract_json_from_text(
        text_file_path=text_file_path,
        json_schema_file_path=json_schema_file_path,
        model_file_path=model_file_path,
        output_json_file_path=output_json_file_path,
        pretty_json=pretty_json,
        skip_validation=skip_validation,
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
    )
    mock__create_llm_instance.assert_called_once_with(
        model_file_path=model_file_path,
        groq_model_name=None,
        groq_api_key=None,
        bedrock_model_id=None,
        google_model_name=None,
        google_api_key=None,
        openai_model_name=None,
        openai_api_key=None,
        openai_api_base=None,
        openai_organization=None,
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
        aws_credentials_profile_name=None,
        aws_region=None,
        bedrock_endpoint_base_url=None,
    )
    mock_read_json_file.assert_called_once_with(path=json_schema_file_path)
    mock_read_text_file.assert_called_once_with(path=text_file_path)
    mock_promot_template.assert_called_once_with(
        template=_EXTRACTION_TEMPLATE,
        input_variables=_EXTRACTION_INPUT_VARIABLES,
        partial_variables={"schema": json.dumps(obj=schema)},
    )
    mock_str_output_parser.assert_called_once_with()
    mock_llm_chain.invoke.assert_called_once_with({"input_text": input_text})
    mock__parse_llm_output.assert_called_once_with(string=_TEST_LLM_OUTPUT)
    if skip_validation:
        mock_validate.assert_not_called()
    else:
        mock_validate.assert_called_once_with(
            instance=_TEST_LLM_OUTPUT_JSON, schema=schema
        )
    if output_json_file_path:
        mock_write_file.assert_called_once_with(
            path=output_json_file_path, data=expected_json_ouput
        )
    else:
        assert capsys.readouterr().out.strip() == expected_json_ouput


def test_extract_json_from_text_with_empty_output(mocker: MockerFixture) -> None:
    mock_llm_chain = mocker.MagicMock()
    mocker.patch("sdeul.extraction._create_llm_instance", return_value=mock_llm_chain)
    mocker.patch("sdeul.extraction.read_json_file")
    mocker.patch("sdeul.extraction.read_text_file")
    mocker.patch("sdeul.extraction.json.dumps")
    mocker.patch("sdeul.extraction.PromptTemplate", return_value=mock_llm_chain)
    mocker.patch("sdeul.extraction.StrOutputParser", return_value=mock_llm_chain)
    mock_llm_chain.__or__.return_value = mock_llm_chain
    mock_llm_chain.invoke.return_value = ""
    with pytest.raises(RuntimeError, match="LLM output is empty."):
        extract_json_from_text(
            text_file_path="input.txt",
            json_schema_file_path="schema.json",
            model_file_path="model.gguf",
            skip_validation=False,
        )


def test_extract_json_from_text_with_invalid_json_output(
    mocker: MockerFixture,
) -> None:
    mock_logger = mocker.MagicMock()
    mocker.patch("logging.getLogger", return_value=mock_logger)
    mocker.patch("sdeul.extraction._create_llm_instance")
    mocker.patch("sdeul.extraction.read_json_file")
    mocker.patch("sdeul.extraction.read_text_file")
    mocker.patch("sdeul.extraction.json.dumps")
    mocker.patch("sdeul.extraction.PromptTemplate")
    mocker.patch("sdeul.extraction.StrOutputParser")
    mocker.patch("sdeul.extraction.LLMChain")
    mocker.patch("sdeul.extraction._parse_llm_output")
    mocker.patch(
        "sdeul.extraction.validate",
        side_effect=ValidationError("Schema validation failed."),
    )
    with pytest.raises(ValidationError):
        extract_json_from_text(
            text_file_path="input.txt",
            json_schema_file_path="schema.json",
            model_file_path="model.gguf",
            skip_validation=False,
        )
    assert mock_logger.error.call_count > 0


def test__parse_llm_output() -> None:
    result = _parse_llm_output(_TEST_LLM_OUTPUT)
    assert result == _TEST_LLM_OUTPUT_JSON


def test__parse_llm_output_with_raw_json() -> None:
    result = _parse_llm_output(
        os.linesep.join(
            r for r in _TEST_LLM_OUTPUT.splitlines() if not r.startswith("```")
        )
    )
    assert result == _TEST_LLM_OUTPUT_JSON


def test__parse_llm_output_without_json() -> None:
    string = "Output without JSON"
    with pytest.raises(RuntimeError, match=f"JSON code block is not found: {string}"):
        _parse_llm_output(string)


def test__parse_llm_output_with_unloadable_json(mocker: MockerFixture) -> None:
    mock_logger = mocker.MagicMock()
    mocker.patch("logging.getLogger", return_value=mock_logger)
    string = os.linesep.join(["```json", '{"unloadable"}', "```"])

    with pytest.raises(JSONDecodeError):
        _parse_llm_output(string)
    mock_logger.error.assert_called_once()


def test__create_llm_instance_with_model_file(mocker: MockerFixture) -> None:
    model_file_path = "/path/to/model"
    temperature = 0.8
    top_p = 0.95
    max_tokens = 8192
    n_ctx = 512
    seed = -1
    n_batch = 8
    n_gpu_layers = -1
    token_wise_streaming = False
    mocker.patch("sdeul.extraction.override_env_vars")
    llm = mocker.MagicMock()
    mock_read_llm_file = mocker.patch(
        "sdeul.extraction._read_llm_file", return_value=llm
    )

    result = _create_llm_instance(
        model_file_path=model_file_path,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        n_ctx=n_ctx,
        seed=seed,
        n_batch=n_batch,
        n_gpu_layers=n_gpu_layers,
        token_wise_streaming=token_wise_streaming,
    )
    assert result == llm
    mock_read_llm_file.assert_called_once_with(
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


def test__create_llm_instance_with_groq(mocker: MockerFixture) -> None:
    groq_model_name = "dummy-groq-model"
    temperature = 0.8
    max_tokens = 8192
    timeout = None
    max_retries = 2
    stop_sequences = None
    mocker.patch("sdeul.extraction.override_env_vars")
    llm = mocker.MagicMock()
    mock_chat_groq = mocker.patch("sdeul.extraction.ChatGroq", return_value=llm)

    result = _create_llm_instance(
        groq_model_name=groq_model_name,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
        max_retries=max_retries,
    )
    assert result == llm
    mock_chat_groq.assert_called_once_with(
        model=groq_model_name,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
        max_retries=max_retries,
        stop_sequences=stop_sequences,
    )


def test__create_llm_instance_with_bedrock(mocker: MockerFixture) -> None:
    bedrock_model_id = "dummy-bedrock-model"
    temperature = 0.8
    max_tokens = 8192
    aws_credentials_profile_name = None
    aws_region = "us-east-1"
    bedrock_endpoint_base_url = "https://api.bedrock.com"
    mocker.patch("sdeul.extraction.override_env_vars")
    mocker.patch("sdeul.extraction.has_aws_credentials")
    llm = mocker.MagicMock()
    mock_chat_bedrock_converse = mocker.patch(
        "sdeul.extraction.ChatBedrockConverse", return_value=llm
    )

    result = _create_llm_instance(
        bedrock_model_id=bedrock_model_id,
        temperature=temperature,
        max_tokens=max_tokens,
        aws_credentials_profile_name=aws_credentials_profile_name,
        aws_region=aws_region,
        bedrock_endpoint_base_url=bedrock_endpoint_base_url,
    )
    assert result == llm
    mock_chat_bedrock_converse.assert_called_once_with(
        model=bedrock_model_id,
        temperature=temperature,
        max_tokens=max_tokens,
        region_name=aws_region,
        base_url=bedrock_endpoint_base_url,
        credentials_profile_name=aws_credentials_profile_name,
    )


def test__create_llm_instance_with_google(mocker: MockerFixture) -> None:
    google_model_name = "dummy-google-model"
    temperature = 0.8
    top_p = 0.95
    max_tokens = 8192
    timeout = None
    max_retries = 2
    mocker.patch("sdeul.extraction.override_env_vars")
    llm = mocker.MagicMock()
    mock_chat_google_generative_ai = mocker.patch(
        "sdeul.extraction.ChatGoogleGenerativeAI", return_value=llm
    )

    result = _create_llm_instance(
        google_model_name=google_model_name,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        timeout=timeout,
        max_retries=max_retries,
    )
    assert result == llm
    mock_chat_google_generative_ai.assert_called_once_with(
        model=google_model_name,
        temperature=temperature,
        top_p=top_p,
        max_output_tokens=max_tokens,
        timeout=timeout,
        max_retries=max_retries,
    )


def test__create_llm_instance_with_openai(mocker: MockerFixture) -> None:
    openai_model_name = "dummy-openai-model"
    openai_api_base = "https://api.openai.com"
    openai_organization = "dummy-organization"
    temperature = 0.8
    top_p = 0.95
    seed = -1
    max_tokens = 8192
    timeout = None
    max_retries = 2
    mocker.patch("sdeul.extraction.override_env_vars")
    llm = mocker.MagicMock()
    mock_chat_openai = mocker.patch("sdeul.extraction.ChatOpenAI", return_value=llm)

    result = _create_llm_instance(
        openai_model_name=openai_model_name,
        openai_api_base=openai_api_base,
        openai_organization=openai_organization,
        temperature=temperature,
        top_p=top_p,
        seed=seed,
        max_tokens=max_tokens,
        timeout=timeout,
        max_retries=max_retries,
    )
    assert result == llm
    mock_chat_openai.assert_called_once_with(
        model=openai_model_name,
        base_url=openai_api_base,
        organization=openai_organization,
        temperature=temperature,
        top_p=top_p,
        seed=seed,
        max_tokens=max_tokens,
        timeout=timeout,
        max_retries=max_retries,
    )


def test__create_llm_instance_no_model_specified(mocker: MockerFixture) -> None:
    mocker.patch("sdeul.extraction.override_env_vars")
    mocker.patch.dict(os.environ, {}, clear=True)
    mocker.patch("sdeul.extraction.has_aws_credentials", return_value=False)
    with pytest.raises(RuntimeError, match="The model cannot be determined."):
        _create_llm_instance()


@pytest.mark.parametrize(
    "token_wise_streaming, logging_level, expected_verbose",
    [
        (False, logging.INFO, False),
        (False, logging.DEBUG, True),
        (True, logging.INFO, True),
    ],
)
def test__read_llm_file(
    token_wise_streaming: bool,
    logging_level: int,
    expected_verbose: bool,
    mocker: MockerFixture,
) -> None:
    llm_file_path = "llm.gguf"
    temperature = 0.8
    top_p = 0.95
    max_tokens = 256
    n_ctx = 512
    seed = -1
    n_batch = 8
    n_gpu_layers = -1
    mock_logger = mocker.MagicMock()
    mocker.patch("logging.getLogger", return_value=mock_logger)
    mock_logger.level = logging_level
    expected_result = mocker.Mock()
    mock_llamacpp = mocker.patch(
        "sdeul.extraction.LlamaCpp", return_value=expected_result
    )
    mocker.patch("sdeul.extraction.StreamingStdOutCallbackHandler")
    mock_callback_manager = mocker.MagicMock()
    mocker.patch(
        "sdeul.extraction.CallbackManager",
        return_value=mock_callback_manager,
    )
    result = _read_llm_file(
        path=llm_file_path,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        n_ctx=n_ctx,
        seed=seed,
        n_batch=n_batch,
        n_gpu_layers=n_gpu_layers,
        token_wise_streaming=token_wise_streaming,
    )
    assert result == expected_result
    if token_wise_streaming:
        mock_llamacpp.assert_called_once_with(
            model_path=llm_file_path,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            n_ctx=n_ctx,
            seed=seed,
            n_batch=n_batch,
            n_gpu_layers=n_gpu_layers,
            verbose=expected_verbose,
            callback_manager=mock_callback_manager,
        )
    else:
        mock_llamacpp.assert_called_once_with(
            model_path=llm_file_path,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            n_ctx=n_ctx,
            seed=seed,
            n_batch=n_batch,
            n_gpu_layers=n_gpu_layers,
            verbose=expected_verbose,
            callback_manager=None,
        )
