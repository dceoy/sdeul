#!/usr/bin/env python
'''
Structural Data Extractor using LLMs

Usage:
    sdeul extract [--debug|--info] [--output-json=<path>] [--pretty-json]
        [--skip-validation] [--temperature=<float>] [--top-p=<float>]
        [--max-tokens=<int>] [--n-ctx=<int>] [--seed=<int>]
        [--openai-model=<name>|--llama-model-gguf=<path>]
        [--openai-api-key=<str>] [--openai-organization=<str>]
        <json_schema_path> <text_path>
    sdeul validate [--debug|--info] <json_schema_path> <json_path>...
    sdeul -h|--help
    sdeul --version

Commands:
    extract                 Extract data as JSON
    validate                Validate JSON files using JSON Schema

Options:
    --debug, --info         Execute a command with debug|info messages
    --output-json=<path>    Output JSON file path
    --pretty-json           Output JSON data with pretty format
    --skip-validation       Skip output validation using JSON Schema
    --temperature=<float>   Specify the temperature for sampling [default: 0]
    --top-p=<float>         Specify the top-p value for sampling [default: 0.1]
    --max-tokens=<int>      Specify the max tokens to generate [default: 8192]
    --n-ctx=<int>           Specify the token context window [default: 1024]
    --seed=<int>            Specify the random seed [default: -1]
    --openai-model=<name>   Use the OpenAI model (e.g., gpt-3.5-turbo)
                            This mode requires the environment variable:
                            - OPENAI_API_KEY (OpenAI API key)
                            - OPENAI_ORGANIZATION (OpenAI organization ID)
    --llama-model-gguf=<path>
                            Use the LLaMA model GGUF file
    --openai-api-key=<str>  Override the OpenAI API key ($OPENAI_API_KEY)
    --openai-organization=<str>
                            Override the OpenAI organization ID
                            ($OPENAI_ORGANIZATION)
    -h, --help              Print help and exit
    --version               Print version and exit

Arguments:
    <json_schema_path>      JSON Schema file path
    <text_path>             Input text file path
    <json_path>             JSON file path
'''

import logging
import os
import signal

from docopt import docopt

from . import __version__
from .extraction import extract_json_from_text
from .validation import validate_json_files_using_json_schema


def main():
    args = docopt(__doc__, version=__version__)
    _set_log_config(debug=args['--debug'], info=args['--info'])
    logger = logging.getLogger(__name__)
    logger.debug(f'args:{os.linesep}{args}')
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    if args['extract']:
        if not (args['--llama-model-gguf'] or args['--openai-model']):
            raise ValueError(
                'Either --llama-model-gguf or --openai-model must be specified'
            )
        else:
            extract_json_from_text(
                text_file_path=args['<text_path>'],
                json_schema_file_path=args['<json_schema_path>'],
                llama_model_file_path=args['--llama-model-gguf'],
                openai_model_name=args['--openai-model'],
                openai_api_key=args['--openai-api-key'],
                openai_organization=args['--openai-organization'],
                output_json_file_path=args['--output-json'],
                pretty_json=args['--pretty-json'],
                skip_validation=args['--skip-validation'],
                temperature=float(args['--temperature']),
                top_p=float(args['--top-p']),
                max_tokens=int(args['--max-tokens']),
                n_ctx=int(args['--n-ctx']), seed=int(args['--seed'])
            )
    elif args['validate']:
        validate_json_files_using_json_schema(
            json_file_paths=args['<json_path>'],
            json_schema_file_path=args['<json_schema_path>']
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
