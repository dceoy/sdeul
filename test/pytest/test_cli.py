#!/usr/bin/env python

import pytest
from docopt import DocoptExit
from pytest_mock import MockerFixture

from sdeul.cli import main


def test_main_extract(mocker: MockerFixture) -> None:
    mocker.patch(
        "sdeul.cli.docopt",
        return_value={
            "extract": True,
            "validate": False,
            "--debug": False,
            "--info": True,
            "--output-json": "output.json",
            "--pretty-json": True,
            "--skip-validation": False,
            "--temperature": "0",
            "--top-p": "0.1",
            "--max-tokens": "8000",
            "--n-ctx": "1024",
            "--seed": "-1",
            "--n-batch": "8",
            "--n-gpu-layers": "-1",
            "--openai-model": "gpt-4o-mini",
            "--google-model": None,
            "--groq-model": None,
            "--bedrock-model": None,
            "--model-gguf": None,
            "--openai-api-key": None,
            "--openai-api-base": None,
            "--openai-organization": None,
            "--google-api-key": None,
            "--groq-api-key": None,
            "<text_path>": "input.txt",
            "<json_schema_path>": "schema.json",
            "<json_path>": None,
        },
    )
    mock_set_logging_config = mocker.patch("sdeul.cli.set_logging_config")
    mock_extract_json_from_text = mocker.patch("sdeul.cli.extract_json_from_text")
    main()
    mock_set_logging_config.assert_called_once_with(debug=False, info=True)
    mock_extract_json_from_text.assert_called_once_with(
        text_file_path="input.txt",
        json_schema_file_path="schema.json",
        model_file_path=None,
        bedrock_model_id=None,
        groq_model_name=None,
        groq_api_key=None,
        google_model_name=None,
        google_api_key=None,
        openai_model_name="gpt-4o-mini",
        openai_api_key=None,
        openai_api_base=None,
        openai_organization=None,
        output_json_file_path="output.json",
        pretty_json=True,
        skip_validation=False,
        temperature=0,
        top_p=0.1,
        max_tokens=8000,
        n_ctx=1024,
        seed=-1,
        n_batch=8,
        n_gpu_layers=-1,
    )


def test_main_validate(mocker: MockerFixture) -> None:
    mocker.patch(
        "sdeul.cli.docopt",
        return_value={
            "extract": False,
            "validate": True,
            "--debug": True,
            "--info": False,
            "--output-json": None,
            "--pretty-json": False,
            "--skip-validation": False,
            "--temperature": "0",
            "--top-p": "0.1",
            "--max-tokens": "8000",
            "--n-ctx": "1024",
            "--seed": "-1",
            "--n-batch": "8",
            "--n-gpu-layers": "-1",
            "--openai-model": None,
            "--google-model": None,
            "--groq-model": None,
            "--bedrock-model": None,
            "--model-gguf": None,
            "--openai-api-key": None,
            "--openai-api-base": None,
            "--openai-organization": None,
            "--google-api-key": None,
            "--groq-api-key": None,
            "<text_path>": None,
            "<json_schema_path>": "schema.json",
            "<json_path>": ["file1.json", "file2.json"],
        },
    )
    mock_set_logging_config = mocker.patch("sdeul.cli.set_logging_config")
    mock_validate_json_files_using_json_schema = mocker.patch(
        "sdeul.cli.validate_json_files_using_json_schema"
    )
    main()
    mock_set_logging_config.assert_called_once_with(debug=True, info=False)
    mock_validate_json_files_using_json_schema.assert_called_once_with(
        json_file_paths=["file1.json", "file2.json"],
        json_schema_file_path="schema.json",
    )


def test_main_not_implemented(mocker: MockerFixture) -> None:
    mocker.patch(
        "sdeul.cli.docopt",
        return_value={
            "extract": False,
            "validate": False,
            "--debug": True,
            "--info": False,
        },
    )
    mocker.patch("sdeul.cli.set_logging_config")
    with pytest.raises(NotImplementedError):
        main()


def test_main_invalid_command(mocker: MockerFixture) -> None:
    mock_docopt = mocker.patch("sdeul.cli.docopt")
    mock_docopt.side_effect = DocoptExit("Invalid command")
    with pytest.raises(DocoptExit):
        main()


def test_main_no_docstring(mocker: MockerFixture) -> None:
    mocker.patch("sdeul.cli.__doc__", new=None)
    with pytest.raises(ValueError, match="No docstring found"):
        main()
