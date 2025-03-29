#!/usr/bin/env python
# pyright: reportMissingImports=false
# pyright: reportUnknownVariableType=false
# pyright: reportPrivateUsage=false
# pyright: reportUnknownArgumentType=false

import json

import pytest
from conftest import TEST_LLM_OUTPUT, TEST_SCHEMA, TEST_TEXT
from jsonschema import ValidationError
from pytest_mock import MockerFixture

from sdeul.extraction import (
    _EXTRACTION_INPUT_VARIABLES,
    _EXTRACTION_TEMPLATE,
    _extruct_structured_data_from_text,
    extract_json_from_text_file,
)


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
    mock_llm_instance = mocker.MagicMock()
    mock_create_llm_instance = mocker.patch(
        "sdeul.extraction.create_llm_instance",
        return_value=mock_llm_instance,
    )
    mock_read_json_file = mocker.patch(
        "sdeul.extraction.read_json_file",
        return_value=TEST_SCHEMA,
    )
    mock_read_text_file = mocker.patch(
        "sdeul.extraction.read_text_file",
        return_value=TEST_TEXT,
    )
    mock__extract_structured_data_from_text = mocker.patch(
        "sdeul.extraction._extruct_structured_data_from_text",
        return_value=TEST_LLM_OUTPUT,
    )
    mock_write_or_print_json_data = mocker.patch(
        "sdeul.extraction.write_or_print_json_data",
    )

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
        ollama_model_name=None,
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
        input_text=TEST_TEXT,
        schema=TEST_SCHEMA,
        llm=mock_llm_instance,
        skip_validation=skip_validation,
    )
    mock_write_or_print_json_data.assert_called_once_with(
        data=TEST_LLM_OUTPUT,
        output_json_file_path=output_json_file_path,
        compact_json=compact_json,
    )


@pytest.mark.parametrize("skip_validation", [(False), (True)])
def test__extruct_structured_data_from_text(
    skip_validation: bool,
    mocker: MockerFixture,
) -> None:
    mock_logger = mocker.MagicMock()
    mocker.patch("logging.getLogger", return_value=mock_logger)
    mock_llm_chain = mocker.MagicMock()
    mock_prompt_template = mocker.patch(
        "sdeul.extraction.PromptTemplate",
        return_value=mock_llm_chain,
    )
    mocker.patch("sdeul.extraction.JsonCodeOutputParser", return_value=mock_llm_chain)
    mock_llm_chain.__or__.return_value = mock_llm_chain
    mock_llm_chain.invoke.return_value = TEST_LLM_OUTPUT
    mock_validate = mocker.patch("sdeul.extraction.validate")

    result = _extruct_structured_data_from_text(
        input_text=TEST_TEXT,
        schema=TEST_SCHEMA,
        llm=mock_llm_chain,
        skip_validation=skip_validation,
    )
    assert result == TEST_LLM_OUTPUT
    mock_prompt_template.assert_called_once_with(
        template=_EXTRACTION_TEMPLATE,
        input_variables=_EXTRACTION_INPUT_VARIABLES,
        partial_variables={"schema": json.dumps(obj=TEST_SCHEMA)},
    )
    mock_llm_chain.invoke.assert_called_once_with({"input_text": TEST_TEXT})
    if skip_validation:
        mock_validate.assert_not_called()
    else:
        mock_validate.assert_called_once_with(
            instance=TEST_LLM_OUTPUT,
            schema=TEST_SCHEMA,
        )
    assert mock_logger.error.call_count == 0


def test__extruct_structured_data_from_text_with_invalid_json_output(
    mocker: MockerFixture,
) -> None:
    mock_logger = mocker.MagicMock()
    mocker.patch("logging.getLogger", return_value=mock_logger)
    mock_llm_chain = mocker.MagicMock()
    mocker.patch("sdeul.extraction.json.dumps")
    mocker.patch("sdeul.extraction.PromptTemplate", return_value=mock_llm_chain)
    mocker.patch("sdeul.extraction.JsonCodeOutputParser", return_value=mock_llm_chain)
    mock_llm_chain.__or__.return_value = mock_llm_chain
    mock_llm_chain.invoke.return_value = "Invalid JSON output"
    mocker.patch(
        "sdeul.extraction.validate",
        side_effect=ValidationError("Schema validation failed."),
    )
    with pytest.raises(ValidationError):
        _extruct_structured_data_from_text(
            input_text=TEST_TEXT,
            schema=TEST_SCHEMA,
            llm=mock_llm_chain,
            skip_validation=False,
        )
    assert mock_logger.exception.call_count > 0
