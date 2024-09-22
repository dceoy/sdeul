#!/usr/bin/env python

import json
import logging
import os
from pathlib import Path
from typing import Generator

import pytest
from botocore.exceptions import NoCredentialsError
from pytest_mock import MockerFixture

from sdeul.utility import (
    has_aws_credentials,
    log_execution_time,
    override_env_vars,
    read_json_file,
    read_text_file,
    set_logging_config,
    write_file,
)


def test_log_execution_time_success(caplog: pytest.LogCaptureFixture) -> None:
    @log_execution_time
    def sample_function() -> str:
        return "Success"

    with caplog.at_level(logging.INFO):
        result = sample_function()
    assert result == "Success"
    assert "sample_function` is executed." in caplog.text
    assert "sample_function` succeeded in" in caplog.text


def test_log_execution_time_failure(caplog: pytest.LogCaptureFixture) -> None:
    @log_execution_time
    def failing_function() -> None:
        raise ValueError("Test error")

    with caplog.at_level(logging.ERROR):
        with pytest.raises(ValueError):
            failing_function()
    assert "failing_function` failed after" in caplog.text


@pytest.mark.parametrize(
    "debug, info, expected_level",
    [
        (True, False, logging.DEBUG),
        (False, True, logging.INFO),
        (False, False, logging.WARNING),
    ],
)
def test_set_logging_config(
    debug: bool, info: bool, expected_level: int, mocker: MockerFixture
) -> None:
    logging_format = "%(asctime)s [%(levelname)-8s] <%(name)s> %(message)s"
    mock_logging_basic_config = mocker.patch("logging.basicConfig")
    set_logging_config(debug=debug, info=info, format=logging_format)
    mock_logging_basic_config.assert_called_once_with(
        format=logging_format, level=expected_level
    )


def test_read_json_file(tmp_path: Path, mocker: MockerFixture) -> None:
    test_data = {"key": "value"}
    json_file = tmp_path / "test.json"
    json_file.write_text(json.dumps(test_data))
    mock_logger = mocker.patch("logging.getLogger")

    result = read_json_file(str(json_file))
    assert result == test_data
    mock_logger.return_value.info.assert_called_once()
    mock_logger.return_value.debug.assert_called_once()


def test_read_text_file(tmp_path: Path, mocker: MockerFixture) -> None:
    test_content = "Hello, World!"
    text_file = tmp_path / "test.txt"
    text_file.write_text(test_content)
    mock_logger = mocker.patch("logging.getLogger")

    result = read_text_file(str(text_file))
    assert result == test_content
    mock_logger.return_value.info.assert_called_once()
    mock_logger.return_value.debug.assert_called_once()


def test_write_file(tmp_path: Path, mocker: MockerFixture) -> None:
    test_content = "Hello, World!"
    output_file = tmp_path / "output.txt"
    mock_logger = mocker.patch("logging.getLogger")

    write_file(str(output_file), test_content)
    assert output_file.read_text() == test_content
    mock_logger.return_value.info.assert_called_once()


@pytest.mark.parametrize("expected", [(True), (False)])
def test_has_aws_credentials(expected: bool, mocker: MockerFixture) -> None:
    sts_client = mocker.MagicMock()
    mocker.patch("sdeul.utility.boto3.client", return_value=sts_client)
    if expected:
        sts_client.get_caller_identity.return_value = {"Account": "123456789012"}
    else:
        sts_client.get_caller_identity.side_effect = NoCredentialsError()
    assert has_aws_credentials() == expected


def test_override_env_vars(mocker: MockerFixture) -> None:
    kwargs = {"FOO": "foo", "BAR": None, "BAZ": "baz"}
    override_env_vars(**kwargs)
    for k, v in kwargs.items():
        if v is None:
            assert k not in os.environ
        else:
            assert os.environ.get(k) == v


@pytest.fixture(autouse=True)
def cleanup_env() -> Generator[None, None, None]:
    initial_env = os.environ.copy()
    yield
    os.environ.clear()
    os.environ.update(initial_env)