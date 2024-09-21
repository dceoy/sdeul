#!/usr/bin/env python

from json.decoder import JSONDecodeError
from pathlib import Path
from unittest.mock import call

import pytest
from jsonschema.exceptions import ValidationError
from pytest_mock import MockFixture

from sdeul.validation import _validate_json_file, validate_json_files_using_json_schema


def test_validate_json_files_using_json_schema_success(
    tmp_path: Path,
    mocker: MockFixture,
) -> None:
    json_schema_file_path = "schema.json"
    json_schema = {"key": "string"}
    json_file_paths = [str(tmp_path / f"test_{i}.json") for i in range(3)]
    for p in json_file_paths:
        Path(p).write_text("{}")
    mock_logger = mocker.patch(
        "logging.getLogger",
        return_value=mocker.MagicMock(),
    )
    mock_read_json_file = mocker.patch(
        "sdeul.validation.read_json_file",
        return_value=json_schema,
    )
    mock__validate_json_file = mocker.patch(
        "sdeul.validation._validate_json_file", return_value=None
    )
    mock_sys_exit = mocker.patch("sys.exit")

    validate_json_files_using_json_schema(
        json_file_paths=json_file_paths,
        json_schema_file_path=json_schema_file_path,
    )
    mock_read_json_file.assert_called_once_with(path=json_schema_file_path)
    mock__validate_json_file.assert_has_calls(
        [call(path=p, json_schema=json_schema) for p in json_file_paths]
    )
    mock_logger.return_value.error.assert_not_called()
    mock_sys_exit.assert_not_called()


def test_validate_json_files_using_json_schema_file_not_found(
    mocker: MockFixture,
) -> None:
    mocker.patch("sdeul.validation.read_json_file")
    with pytest.raises(FileNotFoundError):
        validate_json_files_using_json_schema(
            json_file_paths=["non_existent.json"],
            json_schema_file_path="schema.json",
        )


def test_validate_json_files_using_json_schema_invalid_files(
    tmp_path: Path,
    mocker: MockFixture,
) -> None:
    json_file_paths = [str(tmp_path / f"test_{i}.json") for i in range(3)]
    error_messages = ["Test error 0", "Test error 1", None]
    for p in json_file_paths:
        Path(p).write_text("{}")
    mock_logger = mocker.patch(
        "logging.getLogger",
        return_value=mocker.MagicMock(),
    )
    mocker.patch("sdeul.validation.read_json_file", return_value={})
    mocker.patch("sdeul.validation._validate_json_file", side_effect=error_messages)
    mock_sys_exit = mocker.patch("sys.exit")

    validate_json_files_using_json_schema(
        json_file_paths=json_file_paths,
        json_schema_file_path="schema.json",
    )
    mock_logger.return_value.error.assert_called_once()
    mock_sys_exit.assert_called_once_with(sum((e is not None) for e in error_messages))


def test_validate_json_file_valid(
    capsys: pytest.CaptureFixture[str],
    mocker: MockFixture,
) -> None:
    json_file_path = "valid.json"
    json_data = {"key": "value"}
    json_schema = {"key": "string"}
    mocker.patch("sdeul.validation.read_json_file", return_value=json_data)
    mock_validate = mocker.patch("sdeul.validation.validate")

    result = _validate_json_file(path=json_file_path, json_schema=json_schema)
    mock_validate.assert_called_once_with(instance=json_data, schema=json_schema)
    assert f"{json_file_path}:\tvalid" in capsys.readouterr().out
    assert result is None


def test_validate_json_file_json_decode_error(
    capsys: pytest.CaptureFixture[str],
    mocker: MockFixture,
) -> None:
    json_file_path = "undecodable.json"
    error_message = "JSON decode error"
    mocker.patch("sdeul.validation.read_json_file")
    mocker.patch(
        "sdeul.validation.validate",
        side_effect=JSONDecodeError(error_message, "", 0),
    )

    result = _validate_json_file(path=json_file_path, json_schema={})
    assert result == error_message
    assert (
        f"{json_file_path}:\tJSONDecodeError ({error_message})"
        in capsys.readouterr().out
    )


def test_validate_json_file_validation_error(
    capsys: pytest.CaptureFixture[str],
    mocker: MockFixture,
) -> None:
    json_file_path = "invalid.json"
    error_message = "JSON schema validation error"
    mocker.patch("sdeul.validation.read_json_file")
    mocker.patch(
        "sdeul.validation.validate",
        side_effect=ValidationError(error_message),
    )

    result = _validate_json_file(path=json_file_path, json_schema={})
    assert result == error_message
    assert (
        f"{json_file_path}:\tValidationError ({error_message})"
        in capsys.readouterr().out
    )
