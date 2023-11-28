#!/usr/bin/env python

import json
import logging
from typing import Optional

from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains.llm import LLMChain
from langchain.llms import LlamaCpp
from langchain.prompts import PromptTemplate

from .validator import read_json_schema_file

_EXTRACTION_TEMPLATE = '''Extract and save the relevant entities mentioned \
in the following input text together with their properties.

Only extract the properties in the provided JSON schema.

If a property is not present and is not required in the schema, \
do not include it in the output.

JSON schema:
```json
{schema}
```

Input text:
```
{input_text}
```
'''


def extract_json_from_text(
    text_file_path: str, json_schema_file_path: str,
    llama_model_file_path: str, output_json_file_path: Optional[str] = None
) -> None:
    '''Extract JSON from input text.'''
    logger = logging.getLogger(__name__)
    logger.info(f'Read a Llama 2 model file: {llama_model_file_path}')
    llm = LlamaCpp(
        model_path=llama_model_file_path, temperature=0.75, max_tokens=16384,
        top_p=1, verbose=True, n_ctx=1024,
        callback_manager=CallbackManager([StreamingStdOutCallbackHandler()])
    )

    logger.info(f'Read a JSON Schema file: {json_schema_file_path}')
    schema = read_json_schema_file(path=json_schema_file_path)
    logger.info(f'Read a text file: {text_file_path}')
    input_text = _read_text_file(path=text_file_path)

    prompt = PromptTemplate(
        template=_EXTRACTION_TEMPLATE, input_variables=['input_text'],
        partial_variables={'schema': json.dumps(schema, indent=2)}
    )
    llm_chain = LLMChain(prompt=prompt, llm=llm)

    output = llm_chain.run(input_text=input_text)
    if output_json_file_path:
        logger.info(f'Write a JSON file: {output_json_file_path}')
        with open(output_json_file_path, 'w') as f:
            f.write(output)
    else:
        print(output)


def _read_text_file(path: str) -> str:
    with open(path, 'r') as f:
        return f.read()
