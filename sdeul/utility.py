#!/usr/bin/env python

import json
import logging
import time
from functools import wraps
from typing import Any, Callable


def log_execution_time(func: Callable[..., Any]) -> Callable[..., Any]:
    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        logger = logging.getLogger(log_execution_time.__name__)
        start_time = time.time()
        logger.info(f"`{func.__name__}` is executed.")
        try:
            result = func(*args, **kwargs)
        except Exception as e:
            s = time.time() - start_time
            logger.error(f"`{func.__name__}` failed after {s:.3f}s.")
            raise e
        else:
            s = time.time() - start_time
            logger.info(f"`{func.__name__}` succeeded in {s:.3f}s.")
            return result

    return wrapper


def set_logging_config(debug: bool = False, info: bool = False) -> None:
    if debug:
        lv = logging.DEBUG
    elif info:
        lv = logging.INFO
    else:
        lv = logging.WARNING
    logging.basicConfig(
        format="%(asctime)s [%(levelname)-8s] <%(name)s> %(message)s",
        level=lv,
    )


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
