#!/usr/bin/env python
"""Functions for LLM."""

import ctypes
import json
import logging
import os
import sys
from typing import Any

from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.schema import StrOutputParser
from langchain_aws import ChatBedrockConverse
from langchain_community.llms import LlamaCpp
from langchain_core.exceptions import OutputParserException
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from llama_cpp import llama_log_callback, llama_log_set

from .utility import has_aws_credentials, override_env_vars

_DEFAULT_MODEL_NAMES = {
    "openai": "o3-mini",
    "google": "gemini-2.0-flash",
    "groq": "llama-3.3-70b-versatile",
    "bedrock": "us.anthropic.claude-3-7-sonnet-20250219-v1:0",
}


class JsonCodeOutputParser(StrOutputParser):
    """Detect and parse the JSON code block in the output of an LLM call."""

    def parse(self, text: str) -> Any:
        """Parse the output text.

        Args:
            text: The output text.

        Returns:
            The parsed output.

        Raises:
            OutputParserException: The JSON code block is not detected or invalid.
        """
        logger = logging.getLogger(f"{self.__class__.__name__}.{self.parse.__name__}")
        logger.debug("text: %s", text)
        json_code = self._detect_json_code_block(text=text)
        logger.debug("json_code: %s", json_code)
        try:
            data = json.loads(s=json_code)
        except json.JSONDecodeError as e:
            m = f"Invalid JSON code: {json_code}"
            raise OutputParserException(m, llm_output=text) from e
        else:
            logger.info("Parsed data: %s", data)
            return data

    @staticmethod
    def _detect_json_code_block(text: str) -> str:
        """Detect the JSON code block in the output text.

        Args:
            text: The output text.

        Returns:
            The detected JSON code.

        Raises:
            OutputParserException: The JSON code block is not detected.
        """
        if "```json" in text:
            return text.split("```json", 1)[1].split("```", 1)[0].strip()
        elif "```" in text:
            return text.split("```", 1)[1].split("```", 1)[0].strip()
        elif text.rstrip().startswith(("[", "{", '"')):
            return text.strip()
        else:
            m = f"JSON code block not detected in the text: {text}"
            raise OutputParserException(m, llm_output=text)


def create_llm_instance(
    ollama_model_name: str | None = None,
    ollama_base_url: str | None = None,
    llamacpp_model_file_path: str | None = None,
    groq_model_name: str | None = None,
    groq_api_key: str | None = None,
    bedrock_model_id: str | None = None,
    google_model_name: str | None = None,
    google_api_key: str | None = None,
    openai_model_name: str | None = None,
    openai_api_key: str | None = None,
    openai_api_base: str | None = None,
    openai_organization: str | None = None,
    temperature: float = 0.8,
    top_p: float = 0.95,
    max_tokens: int = 8192,
    n_ctx: int = 512,
    seed: int = -1,
    n_batch: int = 8,
    n_gpu_layers: int = -1,
    token_wise_streaming: bool = False,
    timeout: int | None = None,
    max_retries: int = 2,
    aws_credentials_profile_name: str | None = None,
    aws_region: str | None = None,
    bedrock_endpoint_base_url: str | None = None,
) -> (
    ChatOllama
    | LlamaCpp
    | ChatGroq
    | ChatBedrockConverse
    | ChatGoogleGenerativeAI
    | ChatOpenAI
):
    """Create an instance of a Language Learning Model (LLM).

    Args:
        ollama_model_name (str | None): Name of the Ollama model to use.
            Defaults to None.
        ollama_base_url (str | None): Base URL for the Ollama API.
            Defaults to None.
        llamacpp_model_file_path (str | None): Path to the llama.cpp model file.
            Defaults to None.
        groq_model_name (str | None): Name of the Groq model to use.
            Defaults to None.
        groq_api_key (str | None): API key for Groq. Defaults to None.
        bedrock_model_id (str | None): ID of the Amazon Bedrock model to use.
            Defaults to None.
        google_model_name (str | None): Name of the Google Generative AI model
            to use. Defaults to None.
        google_api_key (str | None): API key for Google Generative AI.
            Defaults to None.
        openai_model_name (str | None): Name of the OpenAI model to use.
            Defaults to None.
        openai_api_key (str | None): API key for OpenAI. Defaults to None.
        openai_api_base (str | None): Base URL for OpenAI API. Defaults to None.
        openai_organization (str | None): OpenAI organization ID. Defaults to None.
        temperature (float): Sampling temperature for the model. Defaults to 0.8.
        top_p (float): Top-p value for sampling. Defaults to 0.95.
        max_tokens (int): Maximum number of tokens to generate. Defaults to 8192.
        n_ctx (int): Token context window size. Defaults to 512.
        seed (int): Random seed for reproducibility. Defaults to -1.
        n_batch (int): Number of batch tokens. Defaults to 8.
        n_gpu_layers (int): Number of GPU layers to use. Defaults to -1.
        token_wise_streaming (bool): Whether to enable token-wise streaming.
            Defaults to False.
        timeout (int | None): Timeout for the API calls in seconds.
            Defaults to None.
        max_retries (int): Maximum number of retries for API calls. Defaults to 2.
        aws_credentials_profile_name (str | None): AWS credentials profile name.
            Defaults to None.
        aws_region (str | None): AWS region for Bedrock. Defaults to None.
        bedrock_endpoint_base_url (str | None): Base URL for Amazon Bedrock
            endpoint. Defaults to None.

    Returns:
        ChatOllama | LlamaCpp | ChatGroq | ChatBedrockConverse |
        ChatGoogleGenerativeAI | ChatOpenAI:
            An instance of the selected LLM.

    Raises:
        RuntimeError: If the model cannot be determined.
    """
    logger = logging.getLogger(create_llm_instance.__name__)
    override_env_vars(
        GROQ_API_KEY=groq_api_key,
        GOOGLE_API_KEY=google_api_key,
        OPENAI_API_KEY=openai_api_key,
    )
    if ollama_model_name:
        logger.info("Use Ollama: %s", ollama_model_name)
        logger.info("Ollama base URL: %s", ollama_base_url)
        return ChatOllama(
            model=ollama_model_name,
            base_url=ollama_base_url,
            temperature=temperature,
            top_p=top_p,
            num_ctx=n_ctx,
            seed=seed,
        )
    elif llamacpp_model_file_path:
        logger.info("Use local LLM: %s", llamacpp_model_file_path)
        return _read_llm_file(
            path=llamacpp_model_file_path,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            n_ctx=n_ctx,
            seed=seed,
            n_batch=n_batch,
            n_gpu_layers=n_gpu_layers,
            token_wise_streaming=token_wise_streaming,
        )
    elif groq_model_name or (
        (not any([bedrock_model_id, google_model_name, openai_model_name]))
        and os.environ.get("GROQ_API_KEY")
    ):
        logger.info("Use GROQ: %s", groq_model_name)
        m = groq_model_name or _DEFAULT_MODEL_NAMES["groq"]
        return ChatGroq(
            model=m,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            max_retries=max_retries,
            stop_sequences=None,
        )
    elif bedrock_model_id or (
        (not any([google_model_name, openai_model_name])) and has_aws_credentials()
    ):
        logger.info("Use Amazon Bedrock: %s", bedrock_model_id)
        m = bedrock_model_id or _DEFAULT_MODEL_NAMES["bedrock"]
        return ChatBedrockConverse(
            model=m,
            temperature=temperature,
            max_tokens=max_tokens,
            region_name=aws_region,
            base_url=bedrock_endpoint_base_url,
            credentials_profile_name=aws_credentials_profile_name,
        )
    elif google_model_name or (
        (not openai_model_name) and os.environ.get("GOOGLE_API_KEY")
    ):
        logger.info("Use Google Generative AI: %s", google_model_name)
        m = google_model_name or _DEFAULT_MODEL_NAMES["google"]
        return ChatGoogleGenerativeAI(
            model=m,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            timeout=timeout,
            max_retries=max_retries,
        )
    elif openai_model_name or os.environ.get("OPENAI_API_KEY"):
        logger.info("Use OpenAI: %s", openai_model_name)
        logger.info("OpenAI API base: %s", openai_api_base)
        logger.info("OpenAI organization: %s", openai_organization)
        m = openai_model_name or _DEFAULT_MODEL_NAMES["openai"]
        return ChatOpenAI(
            model=m,
            base_url=openai_api_base,
            organization=openai_organization,
            temperature=temperature,
            top_p=top_p,
            seed=seed,
            max_completion_tokens=max_tokens,
            timeout=timeout,
            max_retries=max_retries,
        )
    else:
        error_message = "The model cannot be determined."
        raise RuntimeError(error_message)


def _read_llm_file(
    path: str,
    temperature: float = 0.8,
    top_p: float = 0.95,
    max_tokens: int = 256,
    n_ctx: int = 512,
    seed: int = -1,
    n_batch: int = 8,
    n_gpu_layers: int = -1,
    token_wise_streaming: bool = False,
) -> LlamaCpp:
    logger = logging.getLogger(_read_llm_file.__name__)
    llama_log_set(_llama_log_callback, ctypes.c_void_p(0))
    logger.info("Read the model file: %s", path)
    llm = LlamaCpp(
        model_path=path,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        n_ctx=n_ctx,
        seed=seed,
        n_batch=n_batch,
        n_gpu_layers=n_gpu_layers,
        verbose=(token_wise_streaming or logger.level <= logging.DEBUG),
        callback_manager=(
            CallbackManager([StreamingStdOutCallbackHandler()])
            if token_wise_streaming
            else None
        ),
    )
    logger.debug("llm: %s", llm)
    return llm


@llama_log_callback
def _llama_log_callback(level: int, text: bytes, user_data: ctypes.c_void_p) -> None:  # noqa: ARG001
    if logging.root.level < logging.WARNING:
        print(text.decode("utf-8"), end="", flush=True, file=sys.stderr)  # noqa: T201
