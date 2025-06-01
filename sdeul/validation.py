"""JSON validation functions using JSON Schema.

This module provides functionality for validating JSON files against JSON Schema
specifications. It includes batch validation capabilities and detailed error
reporting for invalid files.

Functions:
    validate_json_files_using_json_schema: Validate multiple JSON files
    _validate_json_file: Internal function for validating a single JSON file
"""

import logging
import sys
from json.decoder import JSONDecodeError
from pathlib import Path
from typing import Any

from jsonschema import validate
from jsonschema.exceptions import ValidationError
from rich import print

from .utility import log_execution_time, read_json_file


@log_execution_time
def validate_json_files_using_json_schema(
    json_file_paths: list[str],
    json_schema_file_path: str,
) -> None:
    """Validate multiple JSON files against a JSON Schema.

    Validates each JSON file in the provided list against the given JSON schema.
    Reports validation results for each file and exits with a status code
    indicating the number of invalid files.

    Args:
        json_file_paths: List of paths to JSON files to validate.
        json_schema_file_path: Path to the JSON schema file for validation.

    Raises:
        FileNotFoundError: If any of the JSON files or the schema file
            doesn't exist.

    Note:
        This function calls sys.exit() with the number of invalid files
        as the exit code. An exit code of 0 indicates all files are valid.
    """
    logger = logging.getLogger(validate_json_files_using_json_schema.__name__)
    schema = read_json_file(path=json_schema_file_path)
    n_input = len(json_file_paths)
    logger.info("Start validating %d JSON files.", n_input)
    for p in json_file_paths:
        if not Path(p).is_file():
            error_message = f"File not found: {p}"
            raise FileNotFoundError(error_message)
    n_invalid = sum(
        (_validate_json_file(path=p, json_schema=schema) is not None)
        for p in json_file_paths
    )
    logger.debug("n_invalid: %d", n_invalid)
    if n_invalid:
        logger.error("%d/%d files are invalid.", n_invalid, n_input)
        sys.exit(n_invalid)


def _validate_json_file(path: str, json_schema: dict[str, Any]) -> str | None:
    """Validate a single JSON file against a JSON schema.

    Args:
        path: Path to the JSON file to validate.
        json_schema: JSON schema to validate against.

    Returns:
        str | None: Error message if validation fails, None if successful.
    """
    logger = logging.getLogger(_validate_json_file.__name__)
    try:
        validate(instance=read_json_file(path=path), schema=json_schema)
    except JSONDecodeError as e:
        logger.info(e)
        print(f"{path}:\tJSONDecodeError ({e.msg})")
        return e.msg
    except ValidationError as e:
        logger.info(e)
        print(f"{path}:\tValidationError ({e.message})")
        return e.message
    else:
        print(f"{path}:\tvalid")
        return None
