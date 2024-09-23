#!/usr/bin/env python

import json
import logging
import os
import time
from functools import wraps
from typing import Any, Callable

import boto3
from botocore.exceptions import NoCredentialsError
from mypy_boto3_sts.client import STSClient


def log_execution_time(func: Callable[..., Any]) -> Callable[..., Any]:
    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        logger = logging.getLogger(log_execution_time.__name__)
        logger.debug(f"Call function `{func.__name__}` with parameters: {vars()}")
        start_time = time.time()
        logger.info(f"Function `{func.__name__}` started.")
        try:
            result = func(*args, **kwargs)
        except Exception as e:
            s = time.time() - start_time
            logger.error(f"Function `{func.__name__}` failed after {s:.3f}s.")
            raise e
        else:
            s = time.time() - start_time
            logger.info(f"Function `{func.__name__}` succeeded in {s:.3f}s.")
            return result

    return wrapper


def configure_logging(
    debug: bool = False,
    info: bool = False,
    format: str = "%(asctime)s [%(levelname)-8s] <%(name)s> %(message)s",
) -> None:
    if debug:
        lv = logging.DEBUG
    elif info:
        lv = logging.INFO
    else:
        lv = logging.WARNING
    logging.basicConfig(format=format, level=lv)


def read_json_file(path: str) -> Any:
    logger = logging.getLogger(read_json_file.__name__)
    logger.info(f"Read a JSON file: {path}")
    with open(path, "r") as f:
        data = json.load(f)
    logger.debug(f"data: {data}")
    return data


def read_text_file(path: str) -> str:
    logger = logging.getLogger(read_text_file.__name__)
    logger.info(f"Read a text file: {path}")
    with open(path, "r") as f:
        data = f.read()
    logger.debug(f"data: {data}")
    return data


def write_file(path: str, data: str) -> None:
    logger = logging.getLogger(write_file.__name__)
    logger.info(f"Write data in a file: {path}")
    with open(path, "w") as f:
        f.write(data)


def has_aws_credentials() -> bool:
    logger = logging.getLogger(has_aws_credentials.__name__)
    sts: STSClient = boto3.client("sts")  # pyright: ignore[reportUnknownMemberType]
    try:
        caller_identity = sts.get_caller_identity()
    except NoCredentialsError as e:
        logger.debug(f"caller_identity: {e}")
        return False
    else:
        logger.debug(f"caller_identity: {caller_identity}")
        return True


def override_env_vars(**kwargs: str | None) -> None:
    logger = logging.getLogger(override_env_vars.__name__)
    for k, v in kwargs.items():
        if v is not None:
            logger.info(f"Override the environment variable: {k}={v}")
            os.environ[k] = v
        else:
            logger.info(f"Skip to override environment variable: {k}")
