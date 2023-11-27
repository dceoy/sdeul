#!/usr/bin/env python

from llama_cpp import Llama


def extract_json_from_text(
    text_file_path, llama_model_file_path, few_shot_json_file_paths=None,
    json_schema_file_path=None, output_json_file_path=None
):
    llm = Llama(model_path=llama_model_file_path)
    print(llm)


def _read_text_file(path):
    with open(path, 'r') as f:
        return f.read()
