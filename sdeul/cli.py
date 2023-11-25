#!/usr/bin/env python
"""
Structural Data Extractor using LLMs

Usage:
    sdeul extract [--debug|--info] (--few-shot-json=<path>)
        [--json-schema=<path>] [--output-json=<path>] <llama_model_path>
        <input_text_path>
    sdeul -h|--help
    sdeul --version

Commands:
    extract                 Extract data as JSON

Options:
    --debug, --info         Execute a command with debug|info messages
    --few-shot-json=<path>  Specify JSON file paths for few-shot learning
    --json-schema=<path>    Specify JSON schema file path for output
    --output-json=<path>    Output JSON file path
    -h, --help              Print help and exit
    --version               Print version and exit

Arguments:
    <llama_model_path>      Llama model path
    <input_text_path>       Input text file path
"""

import logging
import os

from docopt import docopt

from . import __version__
from .model import extract_json_from_text


def main():
    args = docopt(__doc__, version=__version__)
    _set_log_config(debug=args['--debug'], info=args['--info'])
    logger = logging.getLogger(__name__)
    logger.debug(f'args:{os.linesep}{args}')
    print(args)
    if args['extract']:
        assert (args['--few-shot-json'] or args['--json-schema']), \
            'Either --few-shot-json or --json-schema must be provided.'
        extract_json_from_text(
            input_text_path=args['<input_text_path>'],
            llama_model_path=args['<llama_model_path>'],
            few_shot_json_paths=args['--few-shot-json'],
            json_schema_path=args['--json-schema'],
            output_json_path=args['--output-json']
        )


def _set_log_config(debug=None, info=None):
    if debug:
        lv = logging.DEBUG
    elif info:
        lv = logging.INFO
    else:
        lv = logging.WARNING
    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S', level=lv
    )
