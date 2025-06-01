#!/usr/bin/env python
# pyright: reportMissingImports=false
# pyright: reportUnknownVariableType=false
# pyright: reportPrivateUsage=false
# pyright: reportUnknownArgumentType=false

import ctypes
import io
import logging
import os

import pytest
from conftest import TEST_LLM_OUTPUT, TEST_LLM_OUTPUT_JSON, TEST_LLM_OUTPUT_MD
from langchain_core.exceptions import OutputParserException
from pytest_mock import MockerFixture

from sdeul.llm import (
    JsonCodeOutputParser,
    _llama_log_callback,
    _read_llm_file,
    create_llm_instance,
)


def test_jsoncodeoutputparser_parse(mocker: MockerFixture) -> None:
    mock_logger = mocker.Mock(spec=logging.Logger)
    mocker.patch("logging.getLogger", return_value=mock_logger)
    parser = JsonCodeOutputParser()
    mock__detect_json_code_block = mocker.patch.object(
        parser,
        "_detect_json_code_block",
        return_value=TEST_LLM_OUTPUT_JSON,
    )
    result = parser.parse(text=TEST_LLM_OUTPUT_MD)
    assert result == TEST_LLM_OUTPUT
    assert mock_logger.info.call_count > 0
    mock__detect_json_code_block.assert_called_once_with(text=TEST_LLM_OUTPUT_MD)


def test_jsoncodeoutputparser_parse_with_invalid_input(mocker: MockerFixture) -> None:
    mock_logger = mocker.Mock(spec=logging.Logger)
    mocker.patch("logging.getLogger", return_value=mock_logger)
    parser = JsonCodeOutputParser()
    invalid_json_code = "invalid"
    mocker.patch.object(
        parser,
        "_detect_json_code_block",
        return_value=invalid_json_code,
    )
    with pytest.raises(OutputParserException) as exc_info:
        parser.parse(text=TEST_LLM_OUTPUT_MD)
    assert str(exc_info.value).startswith(f"Invalid JSON code: {invalid_json_code}")


@pytest.mark.parametrize(
    ("text", "expected_result"),
    [
        (TEST_LLM_OUTPUT_MD, TEST_LLM_OUTPUT_JSON),
        (str.replace(TEST_LLM_OUTPUT_MD, "```json", "```"), TEST_LLM_OUTPUT_JSON),
        (
            str.replace(TEST_LLM_OUTPUT_MD, "```", f"```{os.linesep}```"),
            TEST_LLM_OUTPUT_JSON,
        ),
        (TEST_LLM_OUTPUT_JSON, TEST_LLM_OUTPUT_JSON),
    ],
)
def test_jsoncodeoutputparser__detect_json_code_block(
    text: str,
    expected_result: str,
) -> None:
    result = JsonCodeOutputParser()._detect_json_code_block(text=text)
    assert result == expected_result


def test_jsoncodeoutputparser__detect_json_code_block_with_invalid_input() -> None:
    text = "This is not a valid JSON code block"
    msg = f"JSON code block not detected in the text: {text}"
    with pytest.raises(OutputParserException) as exc_info:
        JsonCodeOutputParser()._detect_json_code_block(text)
    assert str(exc_info.value).startswith(msg)


def test_create_llm_instance_with_model_file(mocker: MockerFixture) -> None:
    llamacpp_model_file_path = "/path/to/model"
    temperature = 0.0
    top_p = 0.95
    top_k = 64
    repeat_penalty = 1.1
    repeat_last_n = 64
    n_ctx = 8192
    max_tokens = 8192
    seed = -1
    n_batch = 8
    n_gpu_layers = -1
    token_wise_streaming = False
    mocker.patch("sdeul.llm.override_env_vars")
    llm = mocker.MagicMock()
    mock_read_llm_file = mocker.patch("sdeul.llm._read_llm_file", return_value=llm)

    result = create_llm_instance(
        llamacpp_model_file_path=llamacpp_model_file_path,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        repeat_penalty=repeat_penalty,
        repeat_last_n=repeat_last_n,
        n_ctx=n_ctx,
        max_tokens=max_tokens,
        seed=seed,
        n_batch=n_batch,
        n_gpu_layers=n_gpu_layers,
        token_wise_streaming=token_wise_streaming,
    )
    assert result == llm
    mock_read_llm_file.assert_called_once_with(
        path=llamacpp_model_file_path,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        repeat_penalty=repeat_penalty,
        last_n_tokens_size=repeat_last_n,
        n_ctx=n_ctx,
        max_tokens=max_tokens,
        seed=seed,
        n_batch=n_batch,
        n_threads=None,
        n_gpu_layers=n_gpu_layers,
        f16_kv=True,
        use_mlock=True,
        use_mmap=True,
        token_wise_streaming=token_wise_streaming,
    )


def test_create_llm_instance_with_groq(mocker: MockerFixture) -> None:
    groq_model_name = "dummy-groq-model"
    temperature = 0.8
    max_tokens = 8192
    timeout = None
    max_retries = 2
    stop_sequences = None
    mocker.patch("sdeul.llm.override_env_vars")
    llm = mocker.MagicMock()
    mock_chat_groq = mocker.patch("sdeul.llm.ChatGroq", return_value=llm)

    result = create_llm_instance(
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


def test_create_llm_instance_with_bedrock(mocker: MockerFixture) -> None:
    bedrock_model_id = "dummy-bedrock-model"
    temperature = 0.8
    max_tokens = 8192
    aws_credentials_profile_name = None
    aws_region = "us-east-1"
    bedrock_endpoint_base_url = "https://api.bedrock.com"
    mocker.patch("sdeul.llm.override_env_vars")
    mocker.patch("sdeul.llm.has_aws_credentials")
    llm = mocker.MagicMock()
    mock_chat_bedrock_converse = mocker.patch(
        "sdeul.llm.ChatBedrockConverse",
        return_value=llm,
    )

    result = create_llm_instance(
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


def test_create_llm_instance_with_google(mocker: MockerFixture) -> None:
    google_model_name = "dummy-google-model"
    temperature = 0.0
    top_p = 0.95
    top_k = 64
    max_tokens = 8192
    timeout = None
    max_retries = 2
    mocker.patch("sdeul.llm.override_env_vars")
    llm = mocker.MagicMock()
    mock_chat_google_generative_ai = mocker.patch(
        "sdeul.llm.ChatGoogleGenerativeAI",
        return_value=llm,
    )

    result = create_llm_instance(
        google_model_name=google_model_name,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        max_tokens=max_tokens,
        timeout=timeout,
        max_retries=max_retries,
    )
    assert result == llm
    mock_chat_google_generative_ai.assert_called_once_with(
        model=google_model_name,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        max_tokens=max_tokens,
        timeout=timeout,
        max_retries=max_retries,
    )


def test_create_llm_instance_with_openai(mocker: MockerFixture) -> None:
    openai_model_name = "dummy-openai-model"
    openai_api_base = "https://api.openai.com"
    openai_organization = "dummy-organization"
    temperature = 0.8
    top_p = 0.95
    seed = -1
    max_tokens = 8192
    timeout = None
    max_retries = 2
    mocker.patch("sdeul.llm.override_env_vars")
    llm = mocker.MagicMock()
    mock_chat_openai = mocker.patch("sdeul.llm.ChatOpenAI", return_value=llm)

    result = create_llm_instance(
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
        max_completion_tokens=max_tokens,
        timeout=timeout,
        max_retries=max_retries,
    )


def test_create_llm_instance_no_model_specified(mocker: MockerFixture) -> None:
    mocker.patch("sdeul.llm.override_env_vars")
    mocker.patch.dict(os.environ, {}, clear=True)
    mocker.patch("sdeul.llm.has_aws_credentials", return_value=False)
    with pytest.raises(RuntimeError, match=r"The model cannot be determined."):
        create_llm_instance()


@pytest.mark.parametrize(
    ("token_wise_streaming", "logging_level", "expected_verbose"),
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
    top_k = 40
    n_ctx = 512
    repeat_penalty = 1.1
    last_n_tokens_size = 64
    max_tokens = 256
    seed = -1
    n_batch = 8
    n_gpu_layers = -1
    mock_logger = mocker.MagicMock()
    mocker.patch("logging.getLogger", return_value=mock_logger)
    mock_logger.level = logging_level
    mocker.patch("sdeul.llm.llama_log_set")
    expected_result = mocker.Mock()
    mock_llamacpp = mocker.patch("sdeul.llm.LlamaCpp", return_value=expected_result)
    mocker.patch("sdeul.llm.StreamingStdOutCallbackHandler")
    mock_callback_manager = mocker.MagicMock()
    mocker.patch(
        "sdeul.llm.CallbackManager",
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
            top_k=top_k,
            repeat_penalty=repeat_penalty,
            last_n_tokens_size=last_n_tokens_size,
            n_ctx=n_ctx,
            max_tokens=max_tokens,
            seed=seed,
            n_batch=n_batch,
            n_threads=None,
            n_gpu_layers=n_gpu_layers,
            f16_kv=True,
            use_mlock=False,
            use_mmap=True,
            verbose=expected_verbose,
            callback_manager=mock_callback_manager,
        )
    else:
        mock_llamacpp.assert_called_once_with(
            model_path=llm_file_path,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repeat_penalty=repeat_penalty,
            last_n_tokens_size=last_n_tokens_size,
            n_ctx=n_ctx,
            max_tokens=max_tokens,
            seed=seed,
            n_batch=n_batch,
            n_threads=None,
            n_gpu_layers=n_gpu_layers,
            f16_kv=True,
            use_mlock=False,
            use_mmap=True,
            verbose=expected_verbose,
            callback_manager=None,
        )


@pytest.mark.parametrize(
    ("text", "root_level", "expected_output"),
    [
        (b"info message", logging.DEBUG, "info message"),
        (b"debug message", logging.INFO, "debug message"),
        (b"warning message", logging.WARNING, ""),
    ],
)
def test__llama_log_callback(
    text: bytes,
    root_level: int,
    expected_output: str,
    mocker: MockerFixture,
) -> None:
    mocker.patch.object(logging.root, "level", root_level)
    mock_stderr = mocker.patch("sdeul.llm.sys.stderr", new_callable=io.StringIO)
    _llama_log_callback(0, text, ctypes.c_void_p(0))
    assert mock_stderr.getvalue() == expected_output


def test_create_llm_instance_with_ollama(mocker: MockerFixture) -> None:
    ollama_model_name = "dummy-ollama-model"
    ollama_base_url = "http://localhost:11434"
    temperature = 0.0
    top_p = 0.95
    top_k = 64
    repeat_penalty = 1.1
    repeat_last_n = 64
    n_ctx = 8192
    seed = -1
    mocker.patch("sdeul.llm.override_env_vars")
    llm = mocker.MagicMock()
    mock_chat_ollama = mocker.patch("sdeul.llm.ChatOllama", return_value=llm)

    result = create_llm_instance(
        ollama_model_name=ollama_model_name,
        ollama_base_url=ollama_base_url,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        repeat_penalty=repeat_penalty,
        repeat_last_n=repeat_last_n,
        n_ctx=n_ctx,
        seed=seed,
    )
    assert result == llm
    mock_chat_ollama.assert_called_once_with(
        model=ollama_model_name,
        base_url=ollama_base_url,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        repeat_penalty=repeat_penalty,
        repeat_last_n=repeat_last_n,
        num_ctx=n_ctx,
        seed=seed,
    )
