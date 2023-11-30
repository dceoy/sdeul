#!/usr/bin/env python

import json
import logging
from json.decoder import JSONDecodeError
from typing import Any, Dict, List, Optional, Union

from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import LLMChain
from langchain.llms import LlamaCpp
from langchain.prompts import PromptTemplate
from langchain.schema import StrOutputParser

from .validator import read_json_schema_file

_EXTRACTION_TEMPLATE = '''\
Instructions:
- Extract and save the relevant entities mentioned in the provided input text together with their properties.
- Only extract the properties defined by the provided JSON schema, and generate them in JSON format.
- If a property is not present and is not required in the schema, do not include it in the output.

Provided JSON schema:
```json
{schema}
```

Provided input text:
```
{input_text}
```
'''     # noqa: E501


def extract_json_from_text(
    text_file_path: str, json_schema_file_path: str,
    llama_model_file_path: str, output_json_file_path: Optional[str] = None
) -> None:
    '''Extract JSON from input text.'''
    logger = logging.getLogger(__name__)
    llm = _read_llm_file(path=llama_model_file_path)
    schema = read_json_schema_file(path=json_schema_file_path)
    input_text = _read_text_file(path=text_file_path)
    llm_chain = _create_llm_chain(schema=schema, llm=llm)

    output_string = llm_chain.invoke({'input_text': input_text})
    logger.info(f'LLM output: {output_string}')

    output_data = _parse_llm_output(output_string=str(output_string))
    if output_json_file_path:
        _write_json_file(path=output_json_file_path, data=output_data)
    else:
        print(output_data)


def _write_json_file(
    path: str, data: Union[List[Any], Dict[Any, Any]]
) -> None:
    logger = logging.getLogger(__name__)
    logger.info(f'Write a JSON file: {path}')
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)


def _parse_llm_output(output_string: str) -> Union[List[Any], Dict[Any, Any]]:
    assert output_string, 'LLM output is empty.'
    logger = logging.getLogger(__name__)
    json_string = None
    for r in output_string.splitlines(keepends=True):
        if json_string is None:
            if r.startswith('```json'):
                json_string = ''
            else:
                pass
        elif not r.startswith('```'):
            json_string += r
        else:
            break
    logger.debug(f'json_string: {json_string}')
    try:
        output_data = json.loads(json_string or output_string)
    except JSONDecodeError as e:
        logger.error(f'Failed to parse the LLM output: {json_string}')
        raise e
    else:
        logger.debug(f'output_data: {output_data}')
        return output_data


def _create_llm_chain(schema: Dict[str, Any], llm: LlamaCpp) -> LLMChain:
    logger = logging.getLogger(__name__)
    prompt = PromptTemplate(
        template=_EXTRACTION_TEMPLATE, input_variables=['input_text'],
        partial_variables={'schema': json.dumps(schema)}
    )
    chain = prompt | llm | StrOutputParser()
    logger.info(f'LLM chain: {chain}')
    return chain


def _read_text_file(path: str) -> str:
    logger = logging.getLogger(__name__)
    logger.info(f'Read a text file: {path}')
    with open(path, 'r') as f:
        data = f.read()
    logger.debug(f'data: {data}')
    return data


def _read_llm_file(path: str, token_wise_streaming: bool = False) -> LlamaCpp:
    logger = logging.getLogger(__name__)
    logger.info(f'Read a Llama 2 model file: {path}')
    llm = LlamaCpp(
        model_path=path, temperature=0.75, max_tokens=8192, top_p=1,
        n_ctx=1024,
        verbose=(
            token_wise_streaming or logging.getLogger().level <= logging.INFO
        ),
        callback_manager=(
            CallbackManager([StreamingStdOutCallbackHandler()])
            if token_wise_streaming else None
        )
    )
    logger.debug(f'llm: {llm}')
    return llm
