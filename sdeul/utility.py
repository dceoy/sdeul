#!/usr/bin/env python
"""Utility functions."""

import json
import logging
import os
import time
from functools import wraps
from pathlib import Path
from typing import Any, Callable

import boto3
from botocore.exceptions import NoCredentialsError
from mypy_boto3_sts.client import STSClient
from rich import print


def log_execution_time(func: Callable[..., Any]) -> Callable[..., Any]:
    """Decorator to log the execution time of a function.

    Args:
        func: The function to be decorated.

    Returns:
        The decorated function.
    """

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        logger = logging.getLogger(log_execution_time.__name__)
        logger.debug("Call function `%s` with parameters: %s", func.__name__, vars())
        start_time = time.time()
        logger.info("Function `%s` started.", func.__name__)
        try:
            result = func(*args, **kwargs)
        except Exception:
            s = time.time() - start_time
            logger.exception("Function `%s` failed after %.3fs.", func.__name__, s)
            raise
        else:
            s = time.time() - start_time
            logger.info("Function `%s` succeeded in %.3fs.", func.__name__, s)
            return result

    return wrapper


def configure_logging(
    debug: bool = False,
    info: bool = False,
    format: str = "%(asctime)s [%(levelname)-8s] <%(name)s> %(message)s",
) -> None:
    """Configure the logging module.

    Args:
        debug: Enable the debug level.
        info: Enable the info level.
        format: The format of the log message.
    """
    if debug:
        lv = logging.DEBUG
    elif info:
        lv = logging.INFO
    else:
        lv = logging.WARNING
    logging.basicConfig(format=format, level=lv)


def read_json_file(path: str) -> Any:
    """Read a JSON file.

    Args:
        path: The path to the JSON file.

    Returns:
        The data in the JSON file.
    """
    logger = logging.getLogger(read_json_file.__name__)
    logger.info("Read a JSON file: %s", path)
    with Path(path).open(mode="r", encoding="utf-8") as f:
        data = json.load(f)
    logger.debug("data: %s", data)
    return data


def read_text_file(path: str) -> str:
    """Read a text file.

    Args:
        path: The path to the text file.

    Returns:
        The data in the text file.
    """
    logger = logging.getLogger(read_text_file.__name__)
    logger.info("Read a text file: %s", path)
    with Path(path).open(mode="r", encoding="utf-8") as f:
        data = f.read()
    logger.debug("data: %s", data)
    return data


def write_file(path: str, data: str) -> None:
    """Write data in a file.

    Args:
        path: The path to the file.
        data: The data to be written in the file.
    """
    logger = logging.getLogger(write_file.__name__)
    logger.info("Write data in a file: %s", path)
    with Path(path).open(mode="w", encoding="utf-8") as f:
        f.write(data)


def has_aws_credentials() -> bool:
    """Check if the AWS credentials are available.

    Returns:
        True if the AWS credentials are available, False otherwise.
    """
    logger = logging.getLogger(has_aws_credentials.__name__)
    sts: STSClient = boto3.client("sts")  # pyright: ignore[reportUnknownMemberType]
    try:
        caller_identity = sts.get_caller_identity()
    except NoCredentialsError as e:
        logger.debug("caller_identity: %s", e)
        return False
    else:
        logger.debug("caller_identity: %s", caller_identity)
        return True


def override_env_vars(**kwargs: str | None) -> None:
    """Override the environment variables.

    Args:
        kwargs: The key-value pairs of the environment variables to be overridden.
    """
    logger = logging.getLogger(override_env_vars.__name__)
    for k, v in kwargs.items():
        if v is not None:
            logger.info("Override the environment variable: %s=%s", k, v)
            os.environ[k] = v
        else:
            logger.info("Skip to override environment variable: %s", k)


def write_or_print_json_data(
    data: Any,
    output_json_file_path: str | None = None,
    compact_json: bool = False,
) -> None:
    """Write or print JSON data.

    Args:
        data: Data to output as JSON.
        output_json_file_path: Path to the output JSON file.
        compact_json: Flag to output the JSON in compact format.
    """
    output_json_string = json.dumps(obj=data, indent=(None if compact_json else 2))
    if output_json_file_path:
        write_file(path=output_json_file_path, data=output_json_string)
    else:
        print(output_json_string)
