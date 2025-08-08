"""Tests for the extraction module."""

# pyright: reportMissingImports=false
# pyright: reportUnknownVariableType=false
# pyright: reportPrivateUsage=false
# pyright: reportUnknownArgumentType=false

import json

import pytest
from jsonschema import ValidationError
from pytest_mock import MockerFixture

from sdeul.config import ExtractConfig
from sdeul.extraction import (
    extract_json_from_text_file,
    extract_json_from_text_file_with_config,
    extract_structured_data_from_text,
)

from .conftest import TEST_LLM_OUTPUT, TEST_SCHEMA, TEST_TEXT


def test_extract_json_from_text_file(mocker: MockerFixture) -> None:
    text_file_path = "input.txt"
    json_schema_file_path = "schema.json"
    llamacpp_model_file_path = "model.gguf"
    output_json_file_path = None
    compact_json = False
    skip_validation = False
    temperature = 0.0
    top_p = 0.95
    top_k = 64
    repeat_penalty = 1.1
    repeat_last_n = 64
    n_ctx = 8192
    max_tokens = 8192
    seed = -1
    n_batch = 8
    n_threads = 2
    n_gpu_layers = -1
    f16_kv = True
    use_mlock = False
    use_mmap = True
    token_wise_streaming = False
    timeout = None
    max_retries = 2
    ollama_base_url = "http://localhost:11434"
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
    mock_extract_structured_data_from_text = mocker.patch(
        "sdeul.extraction.extract_structured_data_from_text",
        return_value=TEST_LLM_OUTPUT,
    )
    mock_write_or_print_json_data = mocker.patch(
        "sdeul.extraction.write_or_print_json_data",
    )

    extract_json_from_text_file(
        text_file_path=text_file_path,
        json_schema_file_path=json_schema_file_path,
        ollama_base_url=ollama_base_url,
        llamacpp_model_file_path=llamacpp_model_file_path,
        output_json_file_path=output_json_file_path,
        compact_json=compact_json,
        skip_validation=skip_validation,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        repeat_penalty=repeat_penalty,
        repeat_last_n=repeat_last_n,
        n_ctx=n_ctx,
        max_tokens=max_tokens,
        seed=seed,
        n_batch=n_batch,
        n_threads=n_threads,
        n_gpu_layers=n_gpu_layers,
        f16_kv=f16_kv,
        use_mlock=use_mlock,
        use_mmap=use_mmap,
        token_wise_streaming=token_wise_streaming,
        timeout=timeout,
        max_retries=max_retries,
    )
    mock_create_llm_instance.assert_called_once_with(
        model_name=None,
        provider=None,
        ollama_base_url=ollama_base_url,
        llamacpp_model_file_path=llamacpp_model_file_path,
        cerebras_api_key=None,
        groq_api_key=None,
        google_api_key=None,
        anthropic_api_key=None,
        anthropic_api_base=None,
        openai_api_key=None,
        openai_api_base=None,
        openai_organization=None,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        repeat_penalty=repeat_penalty,
        repeat_last_n=repeat_last_n,
        n_ctx=n_ctx,
        max_tokens=max_tokens,
        seed=seed,
        n_batch=n_batch,
        n_threads=n_threads,
        n_gpu_layers=n_gpu_layers,
        f16_kv=f16_kv,
        use_mlock=use_mlock,
        use_mmap=use_mmap,
        token_wise_streaming=token_wise_streaming,
        timeout=timeout,
        max_retries=max_retries,
        aws_credentials_profile_name=None,
        aws_region=None,
        bedrock_endpoint_base_url=None,
    )
    mock_read_json_file.assert_called_once_with(path=json_schema_file_path)
    mock_read_text_file.assert_called_once_with(path=text_file_path)
    mock_extract_structured_data_from_text.assert_called_once_with(
        input_text=TEST_TEXT,
        schema=TEST_SCHEMA,
        llm=mock_llm_instance,
        skip_validation=skip_validation,
        terminology=None,
    )
    mock_write_or_print_json_data.assert_called_once_with(
        data=TEST_LLM_OUTPUT,
        output_json_file_path=output_json_file_path,
        compact_json=compact_json,
    )


@pytest.mark.parametrize("skip_validation", [(False), (True)])
def test_extract_structured_data_from_text(
    skip_validation: bool,
    mocker: MockerFixture,
) -> None:
    mock_logger = mocker.MagicMock()
    mocker.patch("logging.getLogger", return_value=mock_logger)
    mock_llm_chain = mocker.MagicMock()
    mocker.patch("sdeul.extraction.ChatPromptTemplate", return_value=mock_llm_chain)
    mocker.patch("sdeul.extraction.JsonCodeOutputParser", return_value=mock_llm_chain)
    mock_llm_chain.__or__.return_value = mock_llm_chain
    mock_llm_chain.invoke.return_value = TEST_LLM_OUTPUT
    mock_validate = mocker.patch("sdeul.extraction.validate")

    result = extract_structured_data_from_text(
        input_text=TEST_TEXT,
        schema=TEST_SCHEMA,
        llm=mock_llm_chain,
        skip_validation=skip_validation,
    )
    assert result == TEST_LLM_OUTPUT
    mock_llm_chain.invoke.assert_called_once_with({
        "schema": json.dumps(obj=TEST_SCHEMA),
        "input_text": TEST_TEXT,
    })
    if skip_validation:
        mock_validate.assert_not_called()
    else:
        mock_validate.assert_called_once_with(
            instance=TEST_LLM_OUTPUT,
            schema=TEST_SCHEMA,
        )
    assert mock_logger.error.call_count == 0


def test_extract_structured_data_from_text_with_terminology(
    mocker: MockerFixture,
) -> None:
    mock_logger = mocker.MagicMock()
    mocker.patch("logging.getLogger", return_value=mock_logger)
    mock_llm_chain = mocker.MagicMock()
    mocker.patch("sdeul.extraction.ChatPromptTemplate", return_value=mock_llm_chain)
    mocker.patch("sdeul.extraction.JsonCodeOutputParser", return_value=mock_llm_chain)
    mock_llm_chain.__or__.return_value = mock_llm_chain
    mock_llm_chain.invoke.return_value = TEST_LLM_OUTPUT
    mock_validate = mocker.patch("sdeul.extraction.validate")
    terminology_text = (
        "API: Application Programming Interface\nHTTP: HyperText Transfer Protocol"
    )

    result = extract_structured_data_from_text(
        input_text=TEST_TEXT,
        schema=TEST_SCHEMA,
        llm=mock_llm_chain,
        skip_validation=False,
        terminology=terminology_text,
    )

    assert result == TEST_LLM_OUTPUT
    mock_llm_chain.invoke.assert_called_once_with({
        "schema": json.dumps(obj=TEST_SCHEMA),
        "input_text": TEST_TEXT,
        "terminology": terminology_text,
    })
    mock_validate.assert_called_once_with(
        instance=TEST_LLM_OUTPUT,
        schema=TEST_SCHEMA,
    )


def test_extract_structured_data_from_text_with_invalid_json_output(
    mocker: MockerFixture,
) -> None:
    mock_logger = mocker.MagicMock()
    mocker.patch("logging.getLogger", return_value=mock_logger)
    mock_llm_chain = mocker.MagicMock()
    mocker.patch("sdeul.extraction.json.dumps")
    mocker.patch("sdeul.extraction.ChatPromptTemplate", return_value=mock_llm_chain)
    mocker.patch("sdeul.extraction.JsonCodeOutputParser", return_value=mock_llm_chain)
    mock_llm_chain.__or__.return_value = mock_llm_chain
    mock_llm_chain.invoke.return_value = "Invalid JSON output"
    mocker.patch(
        "sdeul.extraction.validate",
        side_effect=ValidationError("Schema validation failed."),
    )
    with pytest.raises(ValidationError):
        extract_structured_data_from_text(
            input_text=TEST_TEXT,
            schema=TEST_SCHEMA,
            llm=mock_llm_chain,
            skip_validation=False,
        )
    assert mock_logger.exception.call_count > 0


def test_extract_json_from_text_file_with_terminology(mocker: MockerFixture) -> None:
    text_file_path = "input.txt"
    json_schema_file_path = "schema.json"
    terminology_file_path = "terminology.txt"
    terminology_content = (
        "API: Application Programming Interface\nHTTP: HyperText Transfer Protocol"
    )

    mock_llm_instance = mocker.MagicMock()
    mocker.patch(
        "sdeul.extraction.create_llm_instance",
        return_value=mock_llm_instance,
    )
    mock_read_json_file = mocker.patch(
        "sdeul.extraction.read_json_file",
        return_value=TEST_SCHEMA,
    )
    mock_read_text_file = mocker.patch(
        "sdeul.extraction.read_text_file",
        side_effect=[TEST_TEXT, terminology_content],
    )
    mock_extract_structured_data_from_text = mocker.patch(
        "sdeul.extraction.extract_structured_data_from_text",
        return_value=TEST_LLM_OUTPUT,
    )
    mock_write_or_print_json_data = mocker.patch(
        "sdeul.extraction.write_or_print_json_data",
    )

    extract_json_from_text_file(
        text_file_path=text_file_path,
        json_schema_file_path=json_schema_file_path,
        terminology_file_path=terminology_file_path,
    )

    mock_read_json_file.assert_called_once_with(path=json_schema_file_path)
    assert mock_read_text_file.call_count == 2  # noqa: PLR2004
    mock_read_text_file.assert_any_call(path=text_file_path)
    mock_read_text_file.assert_any_call(path=terminology_file_path)
    mock_extract_structured_data_from_text.assert_called_once_with(
        input_text=TEST_TEXT,
        schema=TEST_SCHEMA,
        llm=mock_llm_instance,
        skip_validation=False,
        terminology=terminology_content,
    )
    mock_write_or_print_json_data.assert_called_once_with(
        data=TEST_LLM_OUTPUT,
        output_json_file_path=None,
        compact_json=False,
    )


def test_extract_json_from_text_file_with_config(mocker: MockerFixture) -> None:
    """Test extract_json_from_text_file_with_config function."""
    text_file_path = "input.txt"
    json_schema_file_path = "schema.json"
    config = ExtractConfig()

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
    mock_extract_structured_data_from_text = mocker.patch(
        "sdeul.extraction.extract_structured_data_from_text",
        return_value=TEST_LLM_OUTPUT,
    )
    mock_write_or_print_json_data = mocker.patch(
        "sdeul.extraction.write_or_print_json_data",
    )

    extract_json_from_text_file_with_config(
        text_file_path=text_file_path,
        json_schema_file_path=json_schema_file_path,
        config=config,
    )

    mock_read_json_file.assert_called_once_with(path=json_schema_file_path)
    mock_read_text_file.assert_called_once_with(path=text_file_path)
    mock_create_llm_instance.assert_called_once()
    mock_extract_structured_data_from_text.assert_called_once_with(
        input_text=TEST_TEXT,
        schema=TEST_SCHEMA,
        llm=mock_llm_instance,
        skip_validation=config.processing.skip_validation,
        terminology=None,
    )
    mock_write_or_print_json_data.assert_called_once_with(
        data=TEST_LLM_OUTPUT,
        output_json_file_path=config.processing.output_json_file,
        compact_json=config.processing.compact_json,
    )
