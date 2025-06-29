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
from langchain_anthropic import ChatAnthropic
from langchain_aws import ChatBedrockConverse
from langchain_community.llms import LlamaCpp
from langchain_core.exceptions import OutputParserException
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from llama_cpp import llama_log_callback, llama_log_set

from .constants import (
    DEFAULT_CONTEXT_WINDOW,
    DEFAULT_F16_KV,
    DEFAULT_MAX_RETRIES,
    DEFAULT_MAX_TOKENS,
    DEFAULT_N_BATCH,
    DEFAULT_N_GPU_LAYERS,
    DEFAULT_N_THREADS,
    DEFAULT_REPEAT_LAST_N,
    DEFAULT_REPEAT_PENALTY,
    DEFAULT_SEED,
    DEFAULT_TEMPERATURE,
    DEFAULT_TIMEOUT,
    DEFAULT_TOKEN_WISE_STREAMING,
    DEFAULT_TOP_K,
    DEFAULT_TOP_P,
    DEFAULT_USE_MLOCK,
    DEFAULT_USE_MMAP,
)
from .utility import override_env_vars


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
            text (str): The raw text output from an LLM that may contain JSON.

        Returns:
            Any: The parsed JSON data as a Python object (dict, list, etc.).

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
            text (str): The text output that may contain JSON code.

        Returns:
            str: The extracted JSON code as a string.

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


# Factory functions for creating LLM instances


def _create_ollama_llm(
    model_name: str,
    base_url: str | None,
    **kwargs: Any,  # noqa: ANN401
) -> ChatOllama:
    """Create an Ollama LLM instance.

    Returns:
        ChatOllama: Configured Ollama LLM instance.
    """
    logger = logging.getLogger(_create_ollama_llm.__name__)
    logger.info("Use Ollama: %s", model_name)
    logger.info("Ollama base URL: %s", base_url)
    return ChatOllama(
        model=model_name,
        base_url=base_url,
        temperature=kwargs["temperature"],
        top_p=kwargs["top_p"],
        top_k=kwargs["top_k"],
        repeat_penalty=kwargs["repeat_penalty"],
        repeat_last_n=kwargs["repeat_last_n"],
        num_ctx=kwargs["n_ctx"],
        seed=kwargs["seed"],
    )


def _create_llamacpp_llm(
    model_file_path: str,
    **kwargs: Any,  # noqa: ANN401
) -> LlamaCpp:
    """Create a LlamaCpp LLM instance.

    Returns:
        LlamaCpp: Configured LlamaCpp LLM instance.
    """
    logger = logging.getLogger(_create_llamacpp_llm.__name__)
    logger.info("Use local LLM: %s", model_file_path)
    return _read_llm_file(
        path=model_file_path,
        temperature=kwargs["temperature"],
        top_p=kwargs["top_p"],
        top_k=kwargs["top_k"],
        repeat_penalty=kwargs["repeat_penalty"],
        last_n_tokens_size=kwargs["repeat_last_n"],
        n_ctx=kwargs["n_ctx"],
        max_tokens=kwargs["max_tokens"],
        seed=kwargs["seed"],
        n_batch=kwargs["n_batch"],
        n_threads=kwargs["n_threads"],
        n_gpu_layers=kwargs["n_gpu_layers"],
        f16_kv=kwargs["f16_kv"],
        use_mlock=kwargs["use_mlock"],
        use_mmap=kwargs["use_mmap"],
        token_wise_streaming=kwargs["token_wise_streaming"],
    )


def _create_groq_llm(
    model_name: str | None,
    **kwargs: Any,  # noqa: ANN401
) -> ChatGroq:
    """Create a Groq LLM instance.

    Returns:
        ChatGroq: Configured Groq LLM instance.
    """
    from .constants import DEFAULT_MODEL_NAMES  # noqa: PLC0415

    logger = logging.getLogger(_create_groq_llm.__name__)
    model = model_name or DEFAULT_MODEL_NAMES["groq"]
    logger.info("Use GROQ: %s", model)
    return ChatGroq(
        model=model,
        temperature=kwargs["temperature"],
        max_tokens=kwargs["max_tokens"],
        timeout=kwargs["timeout"],
        max_retries=kwargs["max_retries"],
        stop_sequences=None,
    )


def _create_bedrock_llm(
    model_id: str | None,
    aws_region: str | None,
    endpoint_url: str | None,
    profile_name: str | None,
    **kwargs: Any,  # noqa: ANN401
) -> ChatBedrockConverse:
    """Create an Amazon Bedrock LLM instance.

    Returns:
        ChatBedrockConverse: Configured Bedrock LLM instance.
    """
    from .constants import DEFAULT_MODEL_NAMES  # noqa: PLC0415

    logger = logging.getLogger(_create_bedrock_llm.__name__)
    model = model_id or DEFAULT_MODEL_NAMES["bedrock"]
    logger.info("Use Amazon Bedrock: %s", model)
    return ChatBedrockConverse(
        model=model,
        temperature=kwargs["temperature"],
        max_tokens=kwargs["max_tokens"],
        region_name=aws_region,
        base_url=endpoint_url,
        credentials_profile_name=profile_name,
    )


def _create_google_llm(
    model_name: str | None,
    **kwargs: Any,  # noqa: ANN401
) -> ChatGoogleGenerativeAI:
    """Create a Google Generative AI LLM instance.

    Returns:
        ChatGoogleGenerativeAI: Configured Google LLM instance.
    """
    from .constants import DEFAULT_MODEL_NAMES  # noqa: PLC0415

    logger = logging.getLogger(_create_google_llm.__name__)
    model = model_name or DEFAULT_MODEL_NAMES["google"]
    logger.info("Use Google Generative AI: %s", model)
    return ChatGoogleGenerativeAI(
        model=model,
        temperature=kwargs["temperature"],
        top_p=kwargs["top_p"],
        top_k=kwargs["top_k"],
        max_tokens=kwargs["max_tokens"],
        timeout=kwargs["timeout"],
        max_retries=kwargs["max_retries"],
    )


def _create_anthropic_llm(
    model_name: str | None,
    api_base: str | None,
    **kwargs: Any,  # noqa: ANN401
) -> ChatAnthropic:
    """Create an Anthropic LLM instance.

    Returns:
        ChatAnthropic: Configured Anthropic LLM instance.
    """
    from .constants import DEFAULT_MODEL_NAMES  # noqa: PLC0415

    logger = logging.getLogger(_create_anthropic_llm.__name__)
    model = model_name or DEFAULT_MODEL_NAMES["anthropic"]
    logger.info("Use Anthropic: %s", model)
    logger.info("Anthropic API base: %s", api_base)
    return ChatAnthropic(
        model_name=model,
        base_url=api_base,
        temperature=kwargs["temperature"],
        top_p=kwargs["top_p"],
        top_k=kwargs["top_k"],
        max_tokens_to_sample=kwargs["max_tokens"],
        timeout=kwargs["timeout"],
        max_retries=kwargs["max_retries"],
        stop=None,
    )


def _create_openai_llm(
    model_name: str | None,
    api_base: str | None,
    organization: str | None,
    **kwargs: Any,  # noqa: ANN401
) -> ChatOpenAI:
    """Create an OpenAI LLM instance.

    Returns:
        ChatOpenAI: Configured OpenAI LLM instance.
    """
    from .constants import DEFAULT_MODEL_NAMES  # noqa: PLC0415

    logger = logging.getLogger(_create_openai_llm.__name__)
    model = model_name or DEFAULT_MODEL_NAMES["openai"]
    logger.info("Use OpenAI: %s", model)
    logger.info("OpenAI API base: %s", api_base)
    logger.info("OpenAI organization: %s", organization)
    return ChatOpenAI(
        model=model,
        base_url=api_base,
        organization=organization,
        temperature=kwargs["temperature"],
        top_p=kwargs["top_p"],
        seed=kwargs["seed"],
        max_completion_tokens=kwargs["max_tokens"],
        timeout=kwargs["timeout"],
        max_retries=kwargs["max_retries"],
    )


def _should_use_groq(
    groq_model_name: str | None,
    bedrock_model_id: str | None,
    google_model_name: str | None,
    anthropic_model_name: str | None,
    openai_model_name: str | None,
) -> bool:
    """Determine if Groq should be used based on model parameters and environment.

    Returns:
        bool: True if Groq should be used, False otherwise.
    """
    if groq_model_name:
        return True

    other_models_specified = any([
        bedrock_model_id,
        google_model_name,
        anthropic_model_name,
        openai_model_name,
    ])
    has_groq_key = os.environ.get("GROQ_API_KEY") is not None

    return not other_models_specified and has_groq_key


def _should_use_bedrock(
    bedrock_model_id: str | None,
    google_model_name: str | None,
    anthropic_model_name: str | None,
    openai_model_name: str | None,
) -> bool:
    """Determine if Bedrock should be used based on model parameters and environment.

    Returns:
        bool: True if Bedrock should be used, False otherwise.
    """
    if bedrock_model_id:
        return True

    other_models_specified = any([
        google_model_name,
        anthropic_model_name,
        openai_model_name,
    ])

    from .utility import has_aws_credentials  # noqa: PLC0415

    return not other_models_specified and has_aws_credentials()


def _should_use_google(
    google_model_name: str | None,
    anthropic_model_name: str | None,
    openai_model_name: str | None,
) -> bool:
    """Determine if Google should be used based on model parameters and environment.

    Returns:
        bool: True if Google should be used, False otherwise.
    """
    if google_model_name:
        return True

    other_models_specified = any([anthropic_model_name, openai_model_name])
    has_google_key = os.environ.get("GOOGLE_API_KEY") is not None

    return not other_models_specified and has_google_key


def _should_use_anthropic(
    anthropic_model_name: str | None,
    openai_model_name: str | None,
) -> bool:
    """Determine if Anthropic should be used based on model parameters.

    Returns:
        bool: True if Anthropic should be used, False otherwise.
    """
    if anthropic_model_name:
        return True

    has_anthropic_key = os.environ.get("ANTHROPIC_API_KEY") is not None
    return not openai_model_name and has_anthropic_key


def create_llm_instance(  # noqa: PLR0911
    ollama_model_name: str | None = None,
    ollama_base_url: str | None = None,
    llamacpp_model_file_path: str | None = None,
    groq_model_name: str | None = None,
    groq_api_key: str | None = None,
    bedrock_model_id: str | None = None,
    google_model_name: str | None = None,
    google_api_key: str | None = None,
    anthropic_model_name: str | None = None,
    anthropic_api_key: str | None = None,
    anthropic_api_base: str | None = None,
    openai_model_name: str | None = None,
    openai_api_key: str | None = None,
    openai_api_base: str | None = None,
    openai_organization: str | None = None,
    temperature: float = DEFAULT_TEMPERATURE,
    top_p: float = DEFAULT_TOP_P,
    top_k: int = DEFAULT_TOP_K,
    repeat_penalty: float = DEFAULT_REPEAT_PENALTY,
    repeat_last_n: int = DEFAULT_REPEAT_LAST_N,
    n_ctx: int = DEFAULT_CONTEXT_WINDOW,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    seed: int = DEFAULT_SEED,
    n_batch: int = DEFAULT_N_BATCH,
    n_threads: int | None = DEFAULT_N_THREADS,
    n_gpu_layers: int | None = DEFAULT_N_GPU_LAYERS,
    f16_kv: bool = DEFAULT_F16_KV,
    use_mlock: bool = DEFAULT_USE_MLOCK,
    use_mmap: bool = DEFAULT_USE_MMAP,
    token_wise_streaming: bool = DEFAULT_TOKEN_WISE_STREAMING,
    timeout: int | None = DEFAULT_TIMEOUT,
    max_retries: int = DEFAULT_MAX_RETRIES,
    aws_credentials_profile_name: str | None = None,
    aws_region: str | None = None,
    bedrock_endpoint_base_url: str | None = None,
) -> (
    ChatOllama
    | LlamaCpp
    | ChatGroq
    | ChatBedrockConverse
    | ChatGoogleGenerativeAI
    | ChatAnthropic
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
        anthropic_model_name (str | None): Name of the Anthropic model to use.
        anthropic_api_key (str | None): API key for Anthropic.
        anthropic_api_base (str | None): Base URL for Anthropic API.
        openai_model_name (str | None): Name of the OpenAI model to use.
        openai_api_key (str | None): API key for OpenAI.
        openai_api_base (str | None): Base URL for OpenAI API.
        openai_organization (str | None): OpenAI organization ID.
        temperature (float): Sampling temperature for the model.
        top_p (float): Top-p value for sampling.
        top_k (int): Top-k value for sampling.
        repeat_penalty (float): Penalty for repeating tokens.
        repeat_last_n (int): Number of tokens to look back when applying
            the repeat penalty.
        n_ctx (int): Token context window size.
        max_tokens (int): Maximum number of tokens to generate.
        seed (int): Random seed for reproducibility.
        n_batch (int): Number of tokens to process in parallel for llama.cpp.
        n_threads (int | None): Number of threads to use for llama.cpp.
        n_gpu_layers (int | None): Number of GPU layers to use for llama.cpp.
        f16_kv (bool): Whether to use half-precision for key/value cache
            of llama.cpp.
        use_mlock (bool): Whether to force the system to keep the model in RAM
            for llama.cpp.
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
        ChatGoogleGenerativeAI | ChatAnthropic | ChatOpenAI: An instance of the
        selected LLM.

    Raises:
        ValueError: If no valid model configuration is provided or if the model
            cannot be determined.
    """
    override_env_vars(
        GROQ_API_KEY=groq_api_key,
        GOOGLE_API_KEY=google_api_key,
        ANTHROPIC_API_KEY=anthropic_api_key,
        OPENAI_API_KEY=openai_api_key,
    )

    # Pack parameters for factory functions
    llm_kwargs = {
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "repeat_penalty": repeat_penalty,
        "repeat_last_n": repeat_last_n,
        "n_ctx": n_ctx,
        "max_tokens": max_tokens,
        "seed": seed,
        "n_batch": n_batch,
        "n_threads": n_threads,
        "n_gpu_layers": n_gpu_layers,
        "f16_kv": f16_kv,
        "use_mlock": use_mlock,
        "use_mmap": use_mmap,
        "token_wise_streaming": token_wise_streaming,
        "timeout": timeout,
        "max_retries": max_retries,
    }

    # Guard clauses for model selection in priority order
    if ollama_model_name:
        return _create_ollama_llm(
            ollama_model_name,
            ollama_base_url,
            **llm_kwargs,
        )

    if llamacpp_model_file_path:
        return _create_llamacpp_llm(
            llamacpp_model_file_path,
            **llm_kwargs,
        )

    if _should_use_groq(
        groq_model_name,
        bedrock_model_id,
        google_model_name,
        anthropic_model_name,
        openai_model_name,
    ):
        return _create_groq_llm(groq_model_name, **llm_kwargs)

    if _should_use_bedrock(
        bedrock_model_id,
        google_model_name,
        anthropic_model_name,
        openai_model_name,
    ):
        return _create_bedrock_llm(
            bedrock_model_id,
            aws_region,
            bedrock_endpoint_base_url,
            aws_credentials_profile_name,
            **llm_kwargs,
        )

    if _should_use_google(
        google_model_name,
        anthropic_model_name,
        openai_model_name,
    ):
        return _create_google_llm(google_model_name, **llm_kwargs)

    if _should_use_anthropic(anthropic_model_name, openai_model_name):
        return _create_anthropic_llm(
            anthropic_model_name,
            anthropic_api_base,
            **llm_kwargs,
        )

    if openai_model_name or os.environ.get("OPENAI_API_KEY"):
        return _create_openai_llm(
            openai_model_name,
            openai_api_base,
            openai_organization,
            **llm_kwargs,
        )

    error_message = "The model cannot be determined."
    raise ValueError(error_message)


def _read_llm_file(
    path: str,
    temperature: float = DEFAULT_TEMPERATURE,
    top_p: float = DEFAULT_TOP_P,
    top_k: int = DEFAULT_TOP_K,
    repeat_penalty: float = DEFAULT_REPEAT_PENALTY,
    last_n_tokens_size: int = DEFAULT_REPEAT_LAST_N,
    n_ctx: int = DEFAULT_CONTEXT_WINDOW,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    seed: int = DEFAULT_SEED,
    n_batch: int = DEFAULT_N_BATCH,
    n_threads: int | None = DEFAULT_N_THREADS,
    n_gpu_layers: int | None = DEFAULT_N_GPU_LAYERS,
    f16_kv: bool = DEFAULT_F16_KV,
    use_mlock: bool = DEFAULT_USE_MLOCK,
    use_mmap: bool = DEFAULT_USE_MMAP,
    token_wise_streaming: bool = DEFAULT_TOKEN_WISE_STREAMING,
) -> LlamaCpp:
    """Load a local LLM model file using llama.cpp.

    Args:
        path (str): Path to the model file (GGUF format).
        temperature (float): Sampling temperature for randomness in generation.
        top_p (float): Top-p value for nucleus sampling.
        top_k (int): Top-k value for sampling.
        repeat_penalty (float): Penalty applied to repeated tokens.
        last_n_tokens_size (int): Number of tokens to consider for repeat penalty.
        n_ctx (int): Token context window size.
        max_tokens (int): Maximum number of tokens to generate.
        seed (int): Random seed for reproducible generation.
        n_batch (int): Number of tokens to process in parallel.
        n_threads (int | None): Number of threads to use for processing.
        n_gpu_layers (int | None): Number of layers to offload to GPU.
        f16_kv (bool): Whether to use half-precision for key/value cache.
        use_mlock (bool): Whether to force system to keep model in RAM.
        use_mmap (bool): Whether to keep the model loaded in RAM.
        token_wise_streaming (bool): Whether to enable token-wise streaming
            output.

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
        level (int): Log level from llama.cpp (unused).
        text (bytes): Log message as bytes.
        user_data (ctypes.c_void_p): User data pointer (unused).
    """
    if logging.root.level < logging.WARNING:
        print(text.decode("utf-8"), end="", flush=True, file=sys.stderr)  # noqa: T201
