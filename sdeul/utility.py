"""Utility functions for file I/O, logging, and AWS operations.

This module provides utility functions for common operations including:
- Execution time logging decorator
- Logging configuration
- File reading and writing operations
- AWS credentials checking
- Environment variable management
- JSON data output formatting
"""

import json
import logging
import os
import time
from functools import wraps
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, TypeVar

import boto3
from botocore.exceptions import NoCredentialsError
from rich import print

if TYPE_CHECKING:
    from mypy_boto3_sts.client import STSClient

T = TypeVar("T")


def log_execution_time(func: Callable[..., T]) -> Callable[..., T]:
    """Decorator to log the execution time of a function.

    This decorator logs the start, completion, and execution time of a function.
    It also logs exceptions if they occur during execution.

    Args:
        func: The function to be decorated.

    Returns:
        The decorated function with execution time logging.
    """

    @wraps(func)
    def wrapper(*args: object, **kwargs: object) -> T:
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
    """Configure the logging module with the specified level and format.

    Sets up the logging configuration based on the specified debug and info flags.
    The logging level is determined by the flags: DEBUG > INFO > WARNING.

    Args:
        debug: If True, sets logging level to DEBUG.
        info: If True and debug is False, sets logging level to INFO.
        format: The format string for log messages.
    """
    if debug:
        lv = logging.DEBUG
    elif info:
        lv = logging.INFO
    else:
        lv = logging.WARNING
    logging.basicConfig(format=format, level=lv)


def read_json_file(path: str) -> Any:  # noqa: ANN401
    """Read and parse a JSON file.

    Reads a JSON file from the specified path and returns the parsed data.
    Logs the operation and the loaded data at appropriate levels.

    Args:
        path: The path to the JSON file.

    Returns:
        The parsed JSON data as a Python object.
    """
    logger = logging.getLogger(read_json_file.__name__)
    logger.info("Read a JSON file: %s", path)
    with Path(path).open(mode="r", encoding="utf-8") as f:
        data = json.load(f)
    logger.debug("data: %s", data)
    return data


def read_text_file(path: str) -> str:
    """Read the contents of a text file.

    Reads a text file from the specified path and returns its contents as a string.
    Uses UTF-8 encoding for reading the file.

    Args:
        path: The path to the text file.

    Returns:
        The contents of the text file as a string.
    """
    logger = logging.getLogger(read_text_file.__name__)
    logger.info("Read a text file: %s", path)
    with Path(path).open(mode="r", encoding="utf-8") as f:
        data = f.read()
    logger.debug("data: %s", data)
    return data


def write_file(path: str, data: str) -> None:
    """Write string data to a file.

    Writes the provided string data to a file at the specified path.
    Uses UTF-8 encoding and creates parent directories if they don't exist.

    Args:
        path: The path where the file should be written.
        data: The string data to write to the file.
    """
    logger = logging.getLogger(write_file.__name__)
    logger.info("Write data in a file: %s", path)
    with Path(path).open(mode="w", encoding="utf-8") as f:
        f.write(data)


def has_aws_credentials() -> bool:
    """Check if AWS credentials are available and valid.

    Attempts to call AWS STS get_caller_identity to verify that valid AWS
    credentials are configured in the environment.

    Returns:
        True if AWS credentials are available and valid, False otherwise.
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
    """Override environment variables with provided values.

    Sets environment variables from the provided keyword arguments, but only
    for non-None values. Logs each operation for debugging purposes.

    Args:
        **kwargs: Key-value pairs where keys are environment variable names
            and values are the values to set. None values are ignored.
    """
    logger = logging.getLogger(override_env_vars.__name__)
    for k, v in kwargs.items():
        if v is not None:
            logger.info("Override the environment variable: %s=%s", k, v)
            os.environ[k] = v
        else:
            logger.info("Skip to override environment variable: %s", k)


def write_or_print_json_data(
    data: Any,  # noqa: ANN401
    output_json_file_path: str | None = None,
    compact_json: bool = False,
) -> None:
    """Write JSON data to a file or print to stdout.

    Serializes the provided data to JSON format and either writes it to a file
    or prints it to stdout. The JSON can be formatted as pretty-printed or compact.

    Args:
        data: The data to serialize and output as JSON.
        output_json_file_path: Optional path to write the JSON file. If None,
            the JSON is printed to stdout.
        compact_json: If True, outputs JSON in compact format without indentation.
            If False, outputs pretty-printed JSON with 2-space indentation.
    """
    output_json_string = json.dumps(
        obj=data, indent=(None if compact_json else 2), ensure_ascii=False
    )
    if output_json_file_path:
        write_file(path=output_json_file_path, data=output_json_string)
    else:
        print(output_json_string)
