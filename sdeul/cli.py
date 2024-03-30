#!/usr/bin/env python
'''
Structural Data Extractor using LLMs

Usage:
    sdeul extract [--debug|--info] [--output-json=<path>] [--pretty-json]
        [--skip-validation] [--temperature=<float>] [--top-p=<float>]
        [--max-tokens=<int>] [--n-ctx=<int>] [--seed=<int>] [--n-batch=<int>]
        [--n-gpu-layers=<int>]
        [--openai-model=<name>|--google-model=<name>|--model-gguf=<path>]
        [--openai-api-key=<str>] [--openai-organization=<str>]
        [--google-api-key=<str>] <json_schema_path> <text_path>
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
    --n-batch=<int>         Specify the number of batch tokens [default: 8]
    --n-gpu-layers=<int>    Specify the number of GPU layers [default: -1]
    --openai-model=<name>   Use the OpenAI model (e.g., gpt-3.5-turbo)
                            This option requires the environment variables:
                            - OPENAI_API_KEY (OpenAI API key)
                            - OPENAI_ORGANIZATION (OpenAI organization ID)
    --google-model=<name>   Use the Google model (e.g., gemini-pro)
                            This option requires the environment variable:
                            - GOOGLE_API_KEY (Google API key)
    --model-gguf=<path>     Use the model GGUF file for llama.cpp
    --openai-api-key=<str>  Override the OpenAI API key ($OPENAI_API_KEY)
    --openai-organization=<str>
                            Override the OpenAI organization ID
                            ($OPENAI_ORGANIZATION)
    --google-api-key=<str>  Override the Google API key ($GOOGLE_API_KEY)
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
        either_required_args = [
            '--openai-model', '--google-model', '--model-gguf'
        ]
        if not [s for s in either_required_args if args[s]]:
            raise ValueError(
                'Either one of the following options is required: '
                + ', '.join(either_required_args)
            )
        else:
            extract_json_from_text(
                text_file_path=args['<text_path>'],
                json_schema_file_path=args['<json_schema_path>'],
                model_file_path=args['--model-gguf'],
                google_model_name=args['--google-model'],
                google_api_key=args['--google-api-key'],
                openai_model_name=args['--openai-model'],
                openai_api_key=args['--openai-api-key'],
                openai_organization=args['--openai-organization'],
                output_json_file_path=args['--output-json'],
                pretty_json=args['--pretty-json'],
                skip_validation=args['--skip-validation'],
                temperature=float(args['--temperature']),
                top_p=float(args['--top-p']),
                max_tokens=int(args['--max-tokens']),
                n_ctx=int(args['--n-ctx']), seed=int(args['--seed']),
                n_batch=int(args['--n-batch']),
                n_gpu_layers=int(args['--n-gpu-layers'])
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
