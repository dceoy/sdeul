#!/usr/bin/env python

from json.decoder import JSONDecodeError
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from jsonschema.exceptions import ValidationError
from pytest_mock import MockFixture

from sdeul.validation import _validate_json_file, validate_json_files_using_json_schema


@pytest.fixture
def mock_read_json_file(mocker: MockFixture) -> MagicMock:
    return mocker.patch("sdeul.validation.read_json_file")


@pytest.fixture
def mock_validate(mocker: MockFixture) -> MagicMock:
    return mocker.patch("sdeul.validation.validate")


@pytest.fixture
def mock_logger(mocker: MockFixture) -> MagicMock:
    return mocker.patch("logging.getLogger")


def test_validate_json_files_using_json_schema_success(
    mock_read_json_file: MagicMock,
    mock_validate: MagicMock,
    mock_logger: MagicMock,
    tmp_path: Path,
) -> None:
    json_file = tmp_path / "test.json"
    json_file.write_text("{}")

    mock_read_json_file.return_value = {}

    validate_json_files_using_json_schema([str(json_file)], "schema.json")

    mock_read_json_file.assert_called()
    mock_validate.assert_called()


def test_validate_json_files_using_json_schema_file_not_found(
    mock_read_json_file: MagicMock,
) -> None:
    with pytest.raises(FileNotFoundError):
        validate_json_files_using_json_schema(["non_existent.json"], "schema.json")


def test_validate_json_files_using_json_schema_invalid_files(
    mock_read_json_file: MagicMock,
    mock_validate: MagicMock,
    mock_logger: MagicMock,
    tmp_path: Path,
    capsys: pytest.CaptureFixture,
) -> None:
    json_file = tmp_path / "invalid.json"
    json_file.write_text("{")

    mock_read_json_file.side_effect = [{}, JSONDecodeError("", "", 0)]

    with pytest.raises(SystemExit) as e:
        validate_json_files_using_json_schema([str(json_file)], "schema.json")

    assert e.value.code == 1
    captured = capsys.readouterr()
    assert "JSONDecodeError" in captured.out


def test_validate_json_file_valid(
    mock_read_json_file: MagicMock,
    mock_validate: MagicMock,
    mock_logger: MagicMock,
    capsys: pytest.CaptureFixture,
) -> None:
    mock_read_json_file.return_value = {}

    result = _validate_json_file("valid.json", {})

    assert result is None
    captured = capsys.readouterr()
    assert "valid.json:\tvalid" in captured.out


def test_validate_json_file_json_decode_error(
    mock_read_json_file: MagicMock,
    mock_validate: MagicMock,
    mock_logger: MagicMock,
    capsys: pytest.CaptureFixture,
) -> None:
    mock_read_json_file.side_effect = JSONDecodeError("Test error", "", 0)

    result = _validate_json_file("invalid.json", {})

    assert result == "Test error"
    captured = capsys.readouterr()
    assert "invalid.json:\tJSONDecodeError (Test error)" in captured.out


def test_validate_json_file_validation_error(
    mock_read_json_file: MagicMock,
    mock_validate: MagicMock,
    mock_logger: MagicMock,
    capsys: pytest.CaptureFixture,
) -> None:
    mock_read_json_file.return_value = {}
    mock_validate.side_effect = ValidationError("Test validation error")

    result = _validate_json_file("invalid_schema.json", {})

    assert result == "Test validation error"
    captured = capsys.readouterr()
    assert (
        "invalid_schema.json:\tValidationError (Test validation error)" in captured.out
    )
