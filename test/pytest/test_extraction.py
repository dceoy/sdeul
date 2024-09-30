#!/usr/bin/env python

import json
import os
from json.decoder import JSONDecodeError
from typing import Any

import pytest
from jsonschema import ValidationError
from pytest_mock import MockerFixture

from sdeul.extraction import (
    _EXTRACTION_INPUT_VARIABLES,
    _EXTRACTION_TEMPLATE,
    _extruct_structured_data_from_text,
    _parse_llm_output,
    extract_json_from_text_file,
    output_data_as_json,
)

_TEST_TEXT = "This is a test input text."
_TEST_SCHEMA = {
    "type": "object",
    "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
}
_TEST_LLM_OUTPUT = """
{
    "name": "John Doe",
    "age": 30
}
"""
_TEST_LLM_OUTPUT_MD = f"""
```json
{_TEST_LLM_OUTPUT}
```
"""
_TEST_LLM_OUTPUT_JSON = {"name": "John Doe", "age": 30}


def test_extract_json_from_text_file(mocker: MockerFixture) -> None:
    text_file_path = "input.txt"
    json_schema_file_path = "schema.json"
    llamacpp_model_file_path = "model.gguf"
    output_json_file_path = None
    compact_json = False
    skip_validation = False
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
    mock_llm_chain = mocker.MagicMock()
    mock_create_llm_instance = mocker.patch(
        "sdeul.extraction.create_llm_instance", return_value=mock_llm_chain
    )
    mock_read_json_file = mocker.patch(
        "sdeul.extraction.read_json_file", return_value=_TEST_SCHEMA
    )
    mock_read_text_file = mocker.patch(
        "sdeul.extraction.read_text_file", return_value=_TEST_TEXT
    )
    mock__extract_structured_data_from_text = mocker.patch(
        "sdeul.extraction._extruct_structured_data_from_text",
        return_value=_TEST_LLM_OUTPUT_JSON,
    )
    mock_output_data_as_json = mocker.patch("sdeul.extraction.output_data_as_json")

    extract_json_from_text_file(
        text_file_path=text_file_path,
        json_schema_file_path=json_schema_file_path,
        llamacpp_model_file_path=llamacpp_model_file_path,
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
        token_wise_streaming=token_wise_streaming,
        timeout=timeout,
        max_retries=max_retries,
    )
    mock_create_llm_instance.assert_called_once_with(
        llamacpp_model_file_path=llamacpp_model_file_path,
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
    mock__extract_structured_data_from_text.assert_called_once_with(
        input_text=_TEST_TEXT,
        schema=_TEST_SCHEMA,
        llm=mock_llm_chain,
        skip_validation=skip_validation,
    )
    mock_output_data_as_json.assert_called_once_with(
        data=_TEST_LLM_OUTPUT_JSON,
        output_json_file_path=output_json_file_path,
        compact_json=compact_json,
    )


@pytest.mark.parametrize(
    "compact_json, output_json_file_path, expected_indent",
    [
        (False, None, 2),
        (True, None, None),
        (False, "output.json", 2),
    ],
)
def test_output_data_as_json(
    compact_json: bool,
    output_json_file_path: str | None,
    expected_indent: int,
    capsys: pytest.CaptureFixture[str],
    mocker: MockerFixture,
) -> None:
    data = {"description": "dummy"}
    expected_json_ouput = json.dumps(obj=data, indent=expected_indent)
    mock_write_file = mocker.patch("sdeul.extraction.write_file")

    output_data_as_json(
        data=data,
        output_json_file_path=output_json_file_path,
        compact_json=compact_json,
    )
    if output_json_file_path:
        mock_write_file.assert_called_once_with(
            path=output_json_file_path, data=expected_json_ouput
        )
    else:
        assert capsys.readouterr().out.strip() == expected_json_ouput


@pytest.mark.parametrize("skip_validation", [(False), (True)])
def test__extruct_structured_data_from_text(
    skip_validation: bool,
    mocker: MockerFixture,
) -> None:
    mock_logger = mocker.MagicMock()
    mocker.patch("logging.getLogger", return_value=mock_logger)
    mock_llm_chain = mocker.MagicMock()
    mock_prompt_template = mocker.patch(
        "sdeul.extraction.PromptTemplate", return_value=mock_llm_chain
    )
    mocker.patch("sdeul.extraction.StrOutputParser", return_value=mock_llm_chain)
    mock_llm_chain.__or__.return_value = mock_llm_chain
    mock_llm_chain.invoke.return_value = _TEST_LLM_OUTPUT_MD
    mock__parse_llm_output = mocker.patch(
        "sdeul.extraction._parse_llm_output", return_value=_TEST_LLM_OUTPUT_JSON
    )
    mock_validate = mocker.patch("sdeul.extraction.validate")

    result = _extruct_structured_data_from_text(
        input_text=_TEST_TEXT,
        schema=_TEST_SCHEMA,
        llm=mock_llm_chain,
        skip_validation=skip_validation,
    )
    assert result == _TEST_LLM_OUTPUT_JSON
    mock_prompt_template.assert_called_once_with(
        template=_EXTRACTION_TEMPLATE,
        input_variables=_EXTRACTION_INPUT_VARIABLES,
        partial_variables={"schema": json.dumps(obj=_TEST_SCHEMA)},
    )
    mock_llm_chain.invoke.assert_called_once_with({"input_text": _TEST_TEXT})
    mock__parse_llm_output.assert_called_once_with(string=_TEST_LLM_OUTPUT_MD)
    if skip_validation:
        mock_validate.assert_not_called()
    else:
        mock_validate.assert_called_once_with(
            instance=_TEST_LLM_OUTPUT_JSON, schema=_TEST_SCHEMA
        )
    assert mock_logger.error.call_count == 0


def test__extruct_structured_data_from_text_with_empty_output(
    mocker: MockerFixture,
) -> None:
    mock_llm_chain = mocker.MagicMock()
    mocker.patch("sdeul.extraction.json.dumps")
    mocker.patch("sdeul.extraction.PromptTemplate", return_value=mock_llm_chain)
    mocker.patch("sdeul.extraction.StrOutputParser", return_value=mock_llm_chain)
    mock_llm_chain.__or__.return_value = mock_llm_chain
    mock_llm_chain.invoke.return_value = ""
    with pytest.raises(RuntimeError, match="LLM output is empty."):
        _extruct_structured_data_from_text(
            input_text=_TEST_TEXT,
            schema=_TEST_SCHEMA,
            llm=mock_llm_chain,
            skip_validation=False,
        )


def test__extruct_structured_data_from_text_with_invalid_json_output(
    mocker: MockerFixture,
) -> None:
    mock_logger = mocker.MagicMock()
    mocker.patch("logging.getLogger", return_value=mock_logger)
    mock_llm_chain = mocker.MagicMock()
    mocker.patch("sdeul.extraction.json.dumps")
    mocker.patch("sdeul.extraction.PromptTemplate", return_value=mock_llm_chain)
    mocker.patch("sdeul.extraction.StrOutputParser", return_value=mock_llm_chain)
    mock_llm_chain.__or__.return_value = mock_llm_chain
    mock_llm_chain.invoke.return_value = "Invalid JSON output"
    mocker.patch("sdeul.extraction._parse_llm_output")
    mocker.patch(
        "sdeul.extraction.validate",
        side_effect=ValidationError("Schema validation failed."),
    )
    with pytest.raises(ValidationError):
        _extruct_structured_data_from_text(
            input_text=_TEST_TEXT,
            schema=_TEST_SCHEMA,
            llm=mock_llm_chain,
            skip_validation=False,
        )
    assert mock_logger.error.call_count > 0


@pytest.mark.parametrize(
    "string, expected_result",
    [
        (_TEST_LLM_OUTPUT_MD, _TEST_LLM_OUTPUT_JSON),
        (_TEST_LLM_OUTPUT, _TEST_LLM_OUTPUT_JSON),
    ],
)
def test__parse_llm_output(string: str, expected_result: dict[str, Any]) -> None:
    result = _parse_llm_output(string=string)
    assert result == expected_result


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
