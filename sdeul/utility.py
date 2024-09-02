#!/usr/bin/env python

import logging
import time
from functools import wraps
from typing import Any, Callable


def log_execution_time(func: Callable) -> Callable:
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


def set_logging_config(debug=None, info=None):
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
