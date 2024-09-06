#!/usr/bin/env python

import json
from pathlib import Path

import pytest
from jsonschema import ValidationError
from pytest_mock import MockerFixture

from sdeul.extraction import (_create_llm_chain, _parse_llm_output,
                              extract_json_from_text)

TEST_TEXT = "This is a test input text."
TEST_SCHEMA = {
    "type": "object",
    "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
}
TEST_LLM_OUTPUT = """
```json
{
    "name": "John Doe",
    "age": 30
}
```
"""


def test_extract_json_from_text(tmp_path: Path, mocker: MockerFixture) -> None:
    mocker.patch("sdeul.extraction.read_json_file", return_value=TEST_SCHEMA)
    mocker.patch("sdeul.extraction.read_text_file", return_value=TEST_TEXT)
    mock_llm = mocker.Mock()
    mock_llm.invoke.return_value = TEST_LLM_OUTPUT
    mocker.patch("sdeul.extraction._create_llm_chain", return_value=mock_llm)
    mocker.patch("sdeul.extraction.validate")
    output_file = tmp_path / "output.json"
    extract_json_from_text(
        text_file_path="dummy.txt",
        json_schema_file_path="dummy_schema.json",
        output_json_file_path=str(output_file),
        openai_model_name="gpt-4",
    )
    assert output_file.exists()
    with open(output_file, "r") as f:
        output_data = json.load(f)
    assert output_data == {"name": "John Doe", "age": 30}


def test_parse_llm_output() -> None:
    result = _parse_llm_output(TEST_LLM_OUTPUT)
    assert result == {"name": "John Doe", "age": 30}


def test_parse_llm_output_invalid() -> None:
    with pytest.raises(RuntimeError):
        _parse_llm_output("Invalid output without JSON")


def test_create_llm_chain(mocker: MockerFixture) -> None:
    mock_llm = mocker.Mock()
    chain = _create_llm_chain(TEST_SCHEMA, mock_llm)
    assert chain is not None


def test_extract_json_from_text_validation_error(mocker: MockerFixture) -> None:
    mocker.patch("sdeul.extraction.read_json_file", return_value=TEST_SCHEMA)
    mocker.patch("sdeul.extraction.read_text_file", return_value=TEST_TEXT)
    mock_llm = mocker.Mock()
    mock_llm.invoke.return_value = '{"invalid": "data"}'
    mocker.patch("sdeul.extraction._create_llm_chain", return_value=mock_llm)
    mocker.patch(
        "sdeul.extraction.validate", side_effect=ValidationError("Validation failed")
    )
    with pytest.raises(ValidationError):
        extract_json_from_text(
            text_file_path="dummy.txt",
            json_schema_file_path="dummy_schema.json",
            openai_model_name="gpt-4",
        )
