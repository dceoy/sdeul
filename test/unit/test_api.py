"""Tests for the API module."""

# pyright: reportMissingImports=false
# pyright: reportUnknownVariableType=false
# pyright: reportPrivateUsage=false
# pyright: reportUnknownArgumentType=false

from typing import Any

import pytest
from fastapi.testclient import TestClient
from jsonschema import ValidationError as JsonSchemaValidationError
from pydantic import ValidationError
from pytest_mock import MockerFixture

from sdeul.api import (
    ExtractRequest,
    ExtractResponse,
    ValidateRequest,
    ValidateResponse,
    app,
    run_server,
)

from .conftest import (
    TEST_LLM_OUTPUT,
    TEST_MAX_RETRIES,
    TEST_MAX_TOKENS,
    TEST_SCHEMA,
    TEST_TEMPERATURE,
    TEST_TEXT,
    TEST_TIMEOUT,
    TEST_TOP_P,
)

# HTTP status codes
_HTTP_200_OK = 200
_HTTP_400_BAD_REQUEST = 400
_HTTP_422_UNPROCESSABLE_ENTITY = 422
_HTTP_500_INTERNAL_SERVER_ERROR = 500

# Test parameter values


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


@pytest.fixture
def extract_request_data() -> dict[str, Any]:
    return {
        "text": TEST_TEXT,
        "json_schema": TEST_SCHEMA,
        "openai_model": "gpt-4o-mini",
        "openai_api_key": "test-api-key",
    }


@pytest.fixture
def validate_request_data() -> dict[str, Any]:
    return {
        "data": TEST_LLM_OUTPUT,
        "json_schema": TEST_SCHEMA,
    }


def test_health_endpoint(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == _HTTP_200_OK
    assert response.json() == {"status": "healthy"}


def test_extract_endpoint_success(
    client: TestClient,
    extract_request_data: dict[str, Any],
    mocker: MockerFixture,
) -> None:
    mock_llm = mocker.MagicMock()
    mock_create_llm_instance = mocker.patch(
        "sdeul.api.create_llm_instance",
        return_value=mock_llm,
    )
    mock_extract_structured_data = mocker.patch(
        "sdeul.api.extract_structured_data_from_text",
        return_value=TEST_LLM_OUTPUT,
    )

    response = client.post("/extract", json=extract_request_data)

    assert response.status_code == _HTTP_200_OK
    response_data = response.json()
    assert response_data["data"] == TEST_LLM_OUTPUT
    assert response_data["validated"] is True

    # Verify the mocks were called correctly
    mock_create_llm_instance.assert_called_once()
    mock_extract_structured_data.assert_called_once_with(
        input_text=TEST_TEXT,
        schema=TEST_SCHEMA,
        llm=mock_llm,
        skip_validation=False,
        terminology=None,
    )


def test_extract_endpoint_with_skip_validation(
    client: TestClient,
    extract_request_data: dict[str, Any],
    mocker: MockerFixture,
) -> None:
    extract_request_data["skip_validation"] = True

    mock_llm = mocker.MagicMock()
    mocker.patch("sdeul.api.create_llm_instance", return_value=mock_llm)
    mock_extract_structured_data = mocker.patch(
        "sdeul.api.extract_structured_data_from_text",
        return_value=TEST_LLM_OUTPUT,
    )

    response = client.post("/extract", json=extract_request_data)

    assert response.status_code == _HTTP_200_OK
    response_data = response.json()
    assert response_data["data"] == TEST_LLM_OUTPUT
    assert response_data["validated"] is False

    mock_extract_structured_data.assert_called_once_with(
        input_text=TEST_TEXT,
        schema=TEST_SCHEMA,
        llm=mock_llm,
        skip_validation=True,
        terminology=None,
    )


def test_extract_endpoint_value_error(
    client: TestClient,
    extract_request_data: dict[str, Any],
    mocker: MockerFixture,
) -> None:
    mocker.patch(
        "sdeul.api.create_llm_instance",
        side_effect=ValueError("Invalid model configuration"),
    )

    response = client.post("/extract", json=extract_request_data)

    assert response.status_code == _HTTP_400_BAD_REQUEST
    assert "Invalid model configuration" in response.json()["detail"]


def test_extract_endpoint_validation_error(
    client: TestClient,
    extract_request_data: dict[str, Any],
    mocker: MockerFixture,
) -> None:
    mock_llm = mocker.MagicMock()
    mocker.patch("sdeul.api.create_llm_instance", return_value=mock_llm)

    validation_error = JsonSchemaValidationError("Invalid data format")
    mocker.patch(
        "sdeul.api.extract_structured_data_from_text",
        side_effect=validation_error,
    )

    response = client.post("/extract", json=extract_request_data)

    assert response.status_code == _HTTP_422_UNPROCESSABLE_ENTITY
    assert "Validation error" in response.json()["detail"]


def test_extract_endpoint_generic_error(
    client: TestClient,
    extract_request_data: dict[str, Any],
    mocker: MockerFixture,
) -> None:
    mock_llm = mocker.MagicMock()
    mocker.patch("sdeul.api.create_llm_instance", return_value=mock_llm)
    mocker.patch(
        "sdeul.api.extract_structured_data_from_text",
        side_effect=RuntimeError("Unexpected error"),
    )

    response = client.post("/extract", json=extract_request_data)

    assert response.status_code == _HTTP_500_INTERNAL_SERVER_ERROR
    assert "Extraction failed" in response.json()["detail"]


def test_extract_endpoint_invalid_request_data(client: TestClient) -> None:
    invalid_data = {
        "text": "",  # Empty text
        "json_schema": {},  # Empty schema
    }

    response = client.post("/extract", json=invalid_data)

    # Empty inputs are valid and should return 200 with empty result
    assert response.status_code == _HTTP_200_OK
    assert response.json()["data"] == {}
    assert response.json()["validated"] is True


def test_extract_endpoint_with_different_models(
    client: TestClient,
    mocker: MockerFixture,
) -> None:
    mock_llm = mocker.MagicMock()
    mock_create_llm_instance = mocker.patch(
        "sdeul.api.create_llm_instance",
        return_value=mock_llm,
    )
    mocker.patch(
        "sdeul.api.extract_structured_data_from_text",
        return_value=TEST_LLM_OUTPUT,
    )

    test_cases = [
        {
            "text": TEST_TEXT,
            "json_schema": TEST_SCHEMA,
            "google_model": "gemini-1.5-flash",
            "google_api_key": "test-key",
        },
        {
            "text": TEST_TEXT,
            "json_schema": TEST_SCHEMA,
            "anthropic_model": "claude-3-5-sonnet-20241022",
            "anthropic_api_key": "test-key",
        },
        {
            "text": TEST_TEXT,
            "json_schema": TEST_SCHEMA,
            "cerebras_model": "llama3.1-70b",
            "cerebras_api_key": "test-key",
        },
        {
            "text": TEST_TEXT,
            "json_schema": TEST_SCHEMA,
            "groq_model": "llama-3.1-70b-versatile",
            "groq_api_key": "test-key",
        },
        {
            "text": TEST_TEXT,
            "json_schema": TEST_SCHEMA,
            "bedrock_model": "anthropic.claude-3-5-sonnet-20240620-v1:0",
            "aws_region": "us-east-1",
        },
        {
            "text": TEST_TEXT,
            "json_schema": TEST_SCHEMA,
            "ollama_model": "llama3.1",
            "ollama_base_url": "http://localhost:11434",
        },
    ]

    for request_data in test_cases:
        response = client.post("/extract", json=request_data)
        assert response.status_code == _HTTP_200_OK
        response_data = response.json()
        assert response_data["data"] == TEST_LLM_OUTPUT
        assert response_data["validated"] is True

    # Verify create_llm_instance was called for each test case
    assert mock_create_llm_instance.call_count == len(test_cases)


def test_validate_endpoint_success(
    client: TestClient,
    validate_request_data: dict[str, Any],
) -> None:
    response = client.post("/validate", json=validate_request_data)

    assert response.status_code == _HTTP_200_OK
    response_data = response.json()
    assert response_data["valid"] is True
    assert response_data["error"] is None


def test_validate_endpoint_validation_error(client: TestClient) -> None:
    invalid_data = {
        "data": {"name": "John", "age": "thirty"},  # age should be integer
        "json_schema": TEST_SCHEMA,
    }

    response = client.post("/validate", json=invalid_data)

    assert response.status_code == _HTTP_200_OK
    response_data = response.json()
    assert response_data["valid"] is False
    assert response_data["error"] is not None


def test_validate_endpoint_generic_error(
    client: TestClient,
    validate_request_data: dict[str, Any],
    mocker: MockerFixture,
) -> None:
    # Mock the validate function to raise an exception
    mocker.patch(
        "sdeul.api.validate",
        side_effect=RuntimeError("Unexpected validation error"),
    )

    response = client.post("/validate", json=validate_request_data)

    assert response.status_code == _HTTP_500_INTERNAL_SERVER_ERROR
    assert "Validation error" in response.json()["detail"]


def test_validate_endpoint_invalid_request_data(client: TestClient) -> None:
    invalid_data = {
        "data": {},
        # Missing json_schema
    }

    response = client.post("/validate", json=invalid_data)

    # Should return 422 for validation error
    assert response.status_code == _HTTP_422_UNPROCESSABLE_ENTITY


@pytest.mark.parametrize(
    ("temperature", "top_p", "top_k", "max_tokens"),
    [
        (0.0, 0.95, 64, 8192),  # Default values
        (1.0, 0.8, 40, 4096),  # Custom values
        (2.0, 1.0, 1, 1),  # Edge values
    ],
)
def test_extract_request_model_parameters(
    temperature: float,
    top_p: float,
    top_k: int,
    max_tokens: int,
) -> None:
    request = ExtractRequest(
        text=TEST_TEXT,
        json_schema=TEST_SCHEMA,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        max_tokens=max_tokens,
        openai_model="gpt-4o-mini",
    )
    assert request.text == TEST_TEXT
    assert request.json_schema == TEST_SCHEMA
    assert request.temperature == temperature
    assert request.top_p == top_p
    assert request.top_k == top_k
    assert request.max_tokens == max_tokens


def test_extract_request_model_invalid_parameters() -> None:
    # Test invalid temperature (> 2.0)
    with pytest.raises(ValidationError, match="less than or equal to 2"):
        ExtractRequest(
            text=TEST_TEXT,
            json_schema=TEST_SCHEMA,
            openai_model="gpt-4o-mini",
            temperature=3.0,
        )

    # Test invalid top_p (> 1.0)
    with pytest.raises(ValidationError, match="less than or equal to 1"):
        ExtractRequest(
            text=TEST_TEXT,
            json_schema=TEST_SCHEMA,
            openai_model="gpt-4o-mini",
            top_p=1.5,
        )

    # Test invalid top_k (< 1)
    with pytest.raises(ValidationError, match="greater than or equal to 1"):
        ExtractRequest(
            text=TEST_TEXT,
            json_schema=TEST_SCHEMA,
            openai_model="gpt-4o-mini",
            top_k=0,
        )

    # Test invalid max_tokens (< 1)
    with pytest.raises(ValidationError, match="greater than or equal to 1"):
        ExtractRequest(
            text=TEST_TEXT,
            json_schema=TEST_SCHEMA,
            openai_model="gpt-4o-mini",
            max_tokens=0,
        )


def test_extract_response_model() -> None:
    response_data = {
        "data": TEST_LLM_OUTPUT,
        "validated": True,
    }

    response = ExtractResponse(**response_data)
    assert response.data == TEST_LLM_OUTPUT
    assert response.validated is True


def test_validate_request_model() -> None:
    request_data = {
        "data": TEST_LLM_OUTPUT,
        "json_schema": TEST_SCHEMA,
    }

    request = ValidateRequest(**request_data)
    assert request.data == TEST_LLM_OUTPUT
    assert request.json_schema == TEST_SCHEMA


def test_validate_response_model() -> None:
    # Test successful validation
    success_response = ValidateResponse(valid=True, error=None)
    assert success_response.valid is True
    assert success_response.error is None

    # Test failed validation
    error_message = "Validation failed"
    error_response = ValidateResponse(valid=False, error=error_message)
    assert error_response.valid is False
    assert error_response.error == error_message


def test_run_server(mocker: MockerFixture) -> None:
    mock_uvicorn_run = mocker.patch("sdeul.api.uvicorn.run")

    run_server()

    mock_uvicorn_run.assert_called_once_with(
        "sdeul.api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )


def test_extract_endpoint_with_all_parameters(
    client: TestClient,
    mocker: MockerFixture,
) -> None:
    """Test extract endpoint with all available parameters."""
    mock_llm = mocker.MagicMock()
    mock_create_llm_instance = mocker.patch(
        "sdeul.api.create_llm_instance",
        return_value=mock_llm,
    )
    mocker.patch(
        "sdeul.api.extract_structured_data_from_text",
        return_value=TEST_LLM_OUTPUT,
    )

    request_data = {
        "text": TEST_TEXT,
        "json_schema": TEST_SCHEMA,
        "skip_validation": True,
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": 50,
        "repeat_penalty": 1.2,
        "repeat_last_n": 100,
        "n_ctx": 4096,
        "max_tokens": 2048,
        "seed": 42,
        "timeout": 30,
        "max_retries": 3,
        "openai_model": "gpt-4",
        "openai_api_key": "test-key",
        "openai_api_base": "https://api.openai.com/v1",
        "openai_organization": "org-123",
    }

    response = client.post("/extract", json=request_data)

    assert response.status_code == _HTTP_200_OK
    response_data = response.json()
    assert response_data["data"] == TEST_LLM_OUTPUT
    assert response_data["validated"] is False  # skip_validation is True

    # Verify create_llm_instance was called with all parameters
    mock_create_llm_instance.assert_called_once()
    call_kwargs = mock_create_llm_instance.call_args[1]

    # Check key parameters
    assert call_kwargs["model_name"] == "gpt-4"
    assert call_kwargs["provider"] == "openai"
    assert call_kwargs["openai_api_key"] == "test-key"
    assert call_kwargs["temperature"] == TEST_TEMPERATURE
    assert call_kwargs["top_p"] == TEST_TOP_P
    assert call_kwargs["max_tokens"] == TEST_MAX_TOKENS
    assert call_kwargs["timeout"] == TEST_TIMEOUT
    assert call_kwargs["max_retries"] == TEST_MAX_RETRIES


def test_extract_endpoint_with_no_model_specified(
    client: TestClient,
    mocker: MockerFixture,
) -> None:
    """Test extract endpoint with no specific model, using auto-detection."""
    mock_llm = mocker.MagicMock()
    mock_create_llm_instance = mocker.patch(
        "sdeul.api.create_llm_instance",
        return_value=mock_llm,
    )
    mocker.patch(
        "sdeul.api.extract_structured_data_from_text",
        return_value=TEST_LLM_OUTPUT,
    )

    request_data = {
        "text": TEST_TEXT,
        "json_schema": TEST_SCHEMA,
        "temperature": 0.7,
        "max_tokens": 2048,
    }

    response = client.post("/extract", json=request_data)

    assert response.status_code == _HTTP_200_OK
    response_data = response.json()
    assert response_data["data"] == TEST_LLM_OUTPUT
    assert response_data["validated"] is True

    # Verify create_llm_instance was called with correct parameters
    mock_create_llm_instance.assert_called_once()
    call_kwargs = mock_create_llm_instance.call_args[1]

    # Check that no provider was explicitly set (auto-detection mode)
    assert call_kwargs["provider"] is None
    assert call_kwargs["model_name"] is None
    assert call_kwargs["temperature"] == TEST_TEMPERATURE
    assert call_kwargs["max_tokens"] == TEST_MAX_TOKENS
