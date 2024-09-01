#!/usr/bin/env python

import json
import logging
from json.decoder import JSONDecodeError
from pathlib import Path
from typing import Any, Dict, List, Union

from jsonschema import validate
from jsonschema.exceptions import ValidationError


def validate_json_files_using_json_schema(
    json_file_paths: List[str], json_schema_file_path: str
) -> None:
    """Validate JSON files using JSON Schema."""
    logger = logging.getLogger(__name__)
    schema = read_json_schema_file(path=json_schema_file_path)
    n_input = len(json_file_paths)
    logger.info(f"Start validating {n_input} JSON files.")
    for p in json_file_paths:
        if not Path(p).is_file():
            raise FileNotFoundError(f"File not found: {p}")
    n_invalid = sum(
        (_validate_json_file(path=p, json_schema=schema) is not None)
        for p in json_file_paths
    )
    logger.debug(f"n_invalid: {n_invalid}")
    if n_invalid:
        logger.error(f"{n_invalid}/{n_input} files are invalid.")
        exit(n_invalid)


def _validate_json_file(path: str, json_schema: Dict[str, Any]) -> Union[None, str]:
    logger = logging.getLogger(__name__)
    try:
        validate(instance=_read_json_file(path=path), schema=json_schema)
    except JSONDecodeError as e:
        logger.info(e)
        print(f"{path}:\tJSONDecodeError ({e.msg})", flush=True)
        return e.msg
    except ValidationError as e:
        logger.info(e)
        print(f"{path}:\tValidationError ({e.message})", flush=True)
        return e.message
    else:
        print(f"{path}:\tvalid", flush=True)
        return None


def read_json_schema_file(path: str) -> Dict[str, Any]:
    return _read_json_file(path=path)


def _read_json_file(path: str):
    logger = logging.getLogger(__name__)
    logger.info(f"Read a JSON file: {path}")
    with open(path, "r") as f:
        data = json.load(f)
    logger.debug(f"data: {data}")
    return data
