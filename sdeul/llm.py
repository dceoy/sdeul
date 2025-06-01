"""Functions for Language Learning Model (LLM) integration and management.

This module provides functionality for creating and managing various LLM instances
including OpenAI, Google Generative AI, Groq, Amazon Bedrock, Ollama, and local
models via llama.cpp. It also includes custom output parsers for JSON extraction.

Classes:
    JsonCodeOutputParser: Custom parser for extracting JSON from LLM responses

Functions:
    create_llm_instance: Factory function for creating LLM instances
    _read_llm_file: Helper function for loading local llama.cpp models
    _llama_log_callback: Callback function for llama.cpp logging
"""

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

from .constants import DEFAULT_MODEL_NAMES
from .utility import has_aws_credentials, override_env_vars


class JsonCodeOutputParser(StrOutputParser):
    """Parser for extracting and validating JSON from LLM text output.

    This parser detects JSON code blocks in LLM responses and parses them into
    Python objects. It handles various JSON formatting patterns including
    markdown code blocks and plain JSON text.
    """

    def parse(self, text: str) -> Any:  # noqa: ANN401
        """Parse JSON from LLM output text.

        Extracts JSON code blocks from the input text and parses them into
        Python objects. Handles various JSON formatting patterns.

        Args:
            text: The raw text output from an LLM that may contain JSON.

        Returns:
            The parsed JSON data as a Python object (dict, list, etc.).

        Raises:
            OutputParserException: If no valid JSON code block is detected
                or if the detected JSON is malformed.
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
        """Detect and extract JSON code from text output.

        Attempts to identify JSON content in various formats including
        markdown code blocks (```json), generic code blocks (```),
        and plain JSON text starting with brackets or quotes.

        Args:
            text: The text output that may contain JSON code.

        Returns:
            The extracted JSON code as a string.

        Raises:
            OutputParserException: If no valid JSON code block is detected.
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
    temperature: float = 0.0,
    top_p: float = 0.95,
    top_k: int = 64,
    repeat_penalty: float = 1.1,
    repeat_last_n: int = 64,
    n_ctx: int = 8192,
    max_tokens: int = 8192,
    seed: int = -1,
    n_batch: int = 8,
    n_threads: int | None = None,
    n_gpu_layers: int | None = -1,
    f16_kv: bool = True,
    use_mlock: bool = True,
    use_mmap: bool = True,
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
        ollama_base_url (str | None): Base URL for the Ollama API.
        llamacpp_model_file_path (str | None): Path to the llama.cpp model file.
        groq_model_name (str | None): Name of the Groq model to use.
        groq_api_key (str | None): API key for Groq.
        bedrock_model_id (str | None): ID of the Amazon Bedrock model to use.
        google_model_name (str | None): Name of the Google Generative AI model
            to use.
        google_api_key (str | None): API key for Google Generative AI.
        openai_model_name (str | None): Name of the OpenAI model to use.
        openai_api_key (str | None): API key for OpenAI.
        openai_api_base (str | None): Base URL for OpenAI API.
        openai_organization (str | None): OpenAI organization ID.
        temperature (float): Sampling temperature for the model.
        top_p (float): Top-p value for sampling.
        top_k (int): Top-k value for sampling.
        repeat_penalty (float): Penalty for repeating tokens.
        repeat_last_n (int): Number of tokens to look back when applying the repeat
            penalty.
        n_ctx (int): Token context window size.
        max_tokens (int): Maximum number of tokens to generate.
        seed (int): Random seed for reproducibility.
        n_batch (int): Number of tokens to process in parallel for llama.cpp.
        n_threads (int): Number of threads to use for llama.cpp.
        n_gpu_layers (int): Number of GPU layers to use for llama.cpp.
        f16_kv (bool): Whether to use half-precision for key/value cache of llama.cpp.
        use_mlock (bool): Whether to force the system to keep the model in RAM for
            llama.cpp.
        use_mmap (bool): Whether to keep the model loaded in RAM for llama.cpp.
        token_wise_streaming (bool): Whether to enable token-wise streaming.
        timeout (int | None): Timeout for the API calls in seconds.
        max_retries (int): Maximum number of retries for API calls.
        aws_credentials_profile_name (str | None): AWS credentials profile name.
        aws_region (str | None): AWS region for Bedrock.
        bedrock_endpoint_base_url (str | None): Base URL for Amazon Bedrock
            endpoint.

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
            top_k=top_k,
            repeat_penalty=repeat_penalty,
            repeat_last_n=repeat_last_n,
            num_ctx=n_ctx,
            seed=seed,
        )
    elif llamacpp_model_file_path:
        logger.info("Use local LLM: %s", llamacpp_model_file_path)
        return _read_llm_file(
            path=llamacpp_model_file_path,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repeat_penalty=repeat_penalty,
            last_n_tokens_size=repeat_last_n,
            n_ctx=n_ctx,
            max_tokens=max_tokens,
            seed=seed,
            n_batch=n_batch,
            n_threads=n_threads,
            n_gpu_layers=n_gpu_layers,
            f16_kv=f16_kv,
            use_mlock=use_mlock,
            use_mmap=use_mmap,
            token_wise_streaming=token_wise_streaming,
        )
    elif groq_model_name or (
        (not any([bedrock_model_id, google_model_name, openai_model_name]))
        and os.environ.get("GROQ_API_KEY")
    ):
        logger.info("Use GROQ: %s", groq_model_name)
        m = groq_model_name or DEFAULT_MODEL_NAMES["groq"]
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
        m = bedrock_model_id or DEFAULT_MODEL_NAMES["bedrock"]
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
        m = google_model_name or DEFAULT_MODEL_NAMES["google"]
        return ChatGoogleGenerativeAI(
            model=m,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_tokens=max_tokens,
            timeout=timeout,
            max_retries=max_retries,
        )
    elif openai_model_name or os.environ.get("OPENAI_API_KEY"):
        logger.info("Use OpenAI: %s", openai_model_name)
        logger.info("OpenAI API base: %s", openai_api_base)
        logger.info("OpenAI organization: %s", openai_organization)
        m = openai_model_name or DEFAULT_MODEL_NAMES["openai"]
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
    top_k: int = 40,
    repeat_penalty: float = 1.1,
    last_n_tokens_size: int = 64,
    n_ctx: int = 512,
    max_tokens: int = 256,
    seed: int = -1,
    n_batch: int = 8,
    n_threads: int | None = None,
    n_gpu_layers: int | None = None,
    f16_kv: bool = True,
    use_mlock: bool = False,
    use_mmap: bool = True,
    token_wise_streaming: bool = False,
) -> LlamaCpp:
    """Load a local LLM model file using llama.cpp.

    Args:
        path: Path to the model file (GGUF format).
        temperature: Sampling temperature for randomness in generation.
        top_p: Top-p value for nucleus sampling.
        top_k: Top-k value for sampling.
        repeat_penalty: Penalty applied to repeated tokens.
        last_n_tokens_size: Number of tokens to consider for repeat penalty.
        n_ctx: Token context window size.
        max_tokens: Maximum number of tokens to generate.
        seed: Random seed for reproducible generation.
        n_batch: Number of tokens to process in parallel.
        n_threads: Number of threads to use for processing.
        n_gpu_layers: Number of layers to offload to GPU.
        f16_kv: Whether to use half-precision for key/value cache.
        use_mlock: Whether to force system to keep model in RAM.
        use_mmap: Whether to keep the model loaded in RAM
        token_wise_streaming: Whether to enable token-wise streaming output.

    Returns:
        LlamaCpp: Configured LlamaCpp model instance.
    """
    logger = logging.getLogger(_read_llm_file.__name__)
    llama_log_set(_llama_log_callback, ctypes.c_void_p(0))
    logger.info("Read the model file: %s", path)
    llm = LlamaCpp(
        model_path=path,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        repeat_penalty=repeat_penalty,
        last_n_tokens_size=last_n_tokens_size,
        n_ctx=n_ctx,
        max_tokens=max_tokens,
        seed=seed,
        n_batch=n_batch,
        n_threads=n_threads,
        n_gpu_layers=n_gpu_layers,
        f16_kv=f16_kv,
        use_mlock=use_mlock,
        use_mmap=use_mmap,
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
    """Callback function for handling llama.cpp logging output.

    This function is used as a callback for llama.cpp to redirect its log
    messages to stderr when debug logging is enabled.

    Args:
        level: Log level from llama.cpp (unused).
        text: Log message as bytes.
        user_data: User data pointer (unused).
    """
    if logging.root.level < logging.WARNING:
        print(text.decode("utf-8"), end="", flush=True, file=sys.stderr)  # noqa: T201
