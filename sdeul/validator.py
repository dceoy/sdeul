#!/usr/bin/env python

import json
import logging
from typing import Dict, List, Union

from jsonschema import validate
from jsonschema.exceptions import ValidationError


def validate_json_files_using_json_schema(
    json_file_paths: List[str], json_schema_file_path: str
) -> None:
    '''Validate JSON files using JSON Schema.'''
    logger = logging.getLogger(__name__)
    logger.info(f'Read a JSON Schema file: {json_schema_file_path}')
    schema = _read_json_file(path=json_schema_file_path)
    n_input = len(json_file_paths)
    logger.info(f'Start validating {n_input} JSON files.')
    n_invalid = sum(
        int(_validate_json_file(path=p, json_schema=schema) is not None)
        for p in json_file_paths
    )
    if n_invalid:
        logger.error(f'Invalid JSON files: {n_invalid}/{n_input}')
        exit(n_invalid)


def _validate_json_file(
    path: str, json_schema: dict
) -> Union[None, ValidationError]:
    logger = logging.getLogger(__name__)
    logger.info(f'Read a JSON file: {path}')
    data = _read_json_file(path=path)
    try:
        validate(instance=data, schema=json_schema)
    except ValidationError as e:
        print(f'{path}:\t{e.message}', flush=True)
        return e
    else:
        print(f'{path}:\tvalid', flush=True)
        return None


def _read_json_file(path: str) -> Union[List, Dict]:
    with open(path, 'r') as f:
        return json.load(f)
