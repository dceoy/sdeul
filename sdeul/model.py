#!/usr/bin/env python

from llama_cpp import Llama


def convert_convert_text_to_json(input_text_path, llama_model_path,
                                 few_shot_json_paths=None,
                                 json_schema_path=None, output_json_path=None):
    llm = Llama(model_path=llama_model_path)
    print(llm)
