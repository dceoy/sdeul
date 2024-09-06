#!/usr/bin/env python

import pytest
from docopt import DocoptExit
from pytest_mock import MockerFixture

from sdeul.cli import main


def test_main_extract(mocker: MockerFixture) -> None:
    mock_docopt = mocker.patch("sdeul.cli.docopt")
    mock_set_logging_config = mocker.patch("sdeul.cli.set_logging_config")
    mock_extract_json_from_text = mocker.patch("sdeul.cli.extract_json_from_text")

    mock_docopt.return_value = {
        "extract": True,
        "--openai-model": "gpt-4",
        "<text_path>": "input.txt",
        "<json_schema_path>": "schema.json",
        "--output-json": "output.json",
        "--pretty-json": True,
        "--skip-validation": False,
        "--temperature": "0.5",
        "--top-p": "0.9",
        "--max-tokens": "1000",
        "--n-ctx": "2048",
        "--seed": "42",
        "--n-batch": "16",
        "--n-gpu-layers": "32",
        "--debug": True,
        "--info": False,
    }

    main()

    mock_set_logging_config.assert_called_once_with(debug=True, info=False)
    mock_extract_json_from_text.assert_called_once_with(
        text_file_path="input.txt",
        json_schema_file_path="schema.json",
        model_file_path=None,
        bedrock_model_id=None,
        groq_model_name=None,
        groq_api_key=None,
        google_model_name=None,
        google_api_key=None,
        openai_model_name="gpt-4",
        openai_api_key=None,
        openai_api_base=None,
        openai_organization=None,
        output_json_file_path="output.json",
        pretty_json=True,
        skip_validation=False,
        temperature=0.5,
        top_p=0.9,
        max_tokens=1000,
        n_ctx=2048,
        seed=42,
        n_batch=16,
        n_gpu_layers=32,
    )


def test_main_validate(mocker: MockerFixture) -> None:
    mock_docopt = mocker.patch("sdeul.cli.docopt")
    mock_set_logging_config = mocker.patch("sdeul.cli.set_logging_config")
    mock_validate_json_files_using_json_schema = mocker.patch(
        "sdeul.cli.validate_json_files_using_json_schema"
    )

    mock_docopt.return_value = {
        "validate": True,
        "<json_schema_path>": "schema.json",
        "<json_path>": ["file1.json", "file2.json"],
        "--debug": False,
        "--info": True,
    }

    main()

    mock_set_logging_config.assert_called_once_with(debug=False, info=True)
    mock_validate_json_files_using_json_schema.assert_called_once_with(
        json_file_paths=["file1.json", "file2.json"],
        json_schema_file_path="schema.json",
    )


def test_main_no_model_specified(mocker: MockerFixture) -> None:
    mock_docopt = mocker.patch("sdeul.cli.docopt")
    mock_docopt.return_value = {
        "extract": True,
        "--openai-model": None,
        "--google-model": None,
        "--groq-model": None,
        "--model-gguf": None,
    }

    with pytest.raises(
        ValueError, match="Either one of the following options is required:"
    ):
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
