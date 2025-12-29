# pyright: reportMissingImports=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportUntypedBaseClass=false
"""Functions for Language Learning Model (LLM) integration and management.

This module provides functionality for creating and managing various LLM instances
including OpenAI, Google Generative AI, Groq, Amazon Bedrock, and Ollama. It also
includes custom output parsers for JSON extraction.

Classes:
    JsonCodeOutputParser: Custom parser for extracting JSON from LLM responses

Functions:
    create_llm_instance: Factory function for creating LLM instances
"""

import json
import logging
import os
from typing import Any

from langchain_anthropic import ChatAnthropic
from langchain_aws import ChatBedrockConverse
from langchain_cerebras import ChatCerebras
from langchain_core.exceptions import OutputParserException
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

from .constants import (
    DEFAULT_CONTEXT_WINDOW,
    DEFAULT_MAX_RETRIES,
    DEFAULT_MAX_TOKENS,
    DEFAULT_REPEAT_LAST_N,
    DEFAULT_REPEAT_PENALTY,
    DEFAULT_SEED,
    DEFAULT_TEMPERATURE,
    DEFAULT_TIMEOUT,
    DEFAULT_TOP_K,
    DEFAULT_TOP_P,
)
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


def _infer_provider(
    *,
    provider: str | None,
    ollama_base_url: str | None,
    cerebras_api_key: str | None,
    groq_api_key: str | None,
    google_api_key: str | None,
    anthropic_api_key: str | None,
    openai_api_key: str | None,
) -> str | None:
    if provider:
        return provider

    checks = (
        ("ollama", lambda: bool(os.environ.get("OLLAMA_BASE_URL") or ollama_base_url)),
        (
            "cerebras",
            lambda: bool(os.environ.get("CEREBRAS_API_KEY") or cerebras_api_key),
        ),
        ("groq", lambda: bool(os.environ.get("GROQ_API_KEY") or groq_api_key)),
        ("bedrock", has_aws_credentials),
        ("google", lambda: bool(os.environ.get("GOOGLE_API_KEY") or google_api_key)),
        (
            "anthropic",
            lambda: bool(os.environ.get("ANTHROPIC_API_KEY") or anthropic_api_key),
        ),
        ("openai", lambda: bool(os.environ.get("OPENAI_API_KEY") or openai_api_key)),
    )
    for name, predicate in checks:
        if predicate():
            return name
    return None


def _create_ollama_llm(
    model_name: str,
    base_url: str | None,
    keep_alive: str | int | None = None,
    **kwargs: Any,  # noqa: ANN401
) -> ChatOllama:
    """Create an Ollama LLM instance.

    Returns:
        ChatOllama: Configured Ollama LLM instance.
    """
    logger = logging.getLogger(_create_ollama_llm.__name__)
    logger.info("Use Ollama: %s", model_name)
    logger.info("Ollama base URL: %s", base_url)
    logger.info("Ollama keep_alive: %s", keep_alive)
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
        keep_alive=keep_alive,
    )


def _create_cerebras_llm(
    model_name: str,
    **kwargs: Any,  # noqa: ANN401
) -> ChatCerebras:
    """Create a Cerebras LLM instance.

    Returns:
        ChatCerebras: Configured Cerebras LLM instance.
    """
    logger = logging.getLogger(_create_cerebras_llm.__name__)
    logger.info("Use Cerebras: %s", model_name)
    return ChatCerebras(
        model=model_name,
        temperature=kwargs["temperature"],
        max_tokens=kwargs["max_tokens"],
        timeout=kwargs["timeout"],
        max_retries=kwargs["max_retries"],
        stop_sequences=None,
    )


def _create_groq_llm(
    model_name: str,
    **kwargs: Any,  # noqa: ANN401
) -> ChatGroq:
    """Create a Groq LLM instance.

    Returns:
        ChatGroq: Configured Groq LLM instance.
    """
    logger = logging.getLogger(_create_groq_llm.__name__)
    logger.info("Use GROQ: %s", model_name)
    return ChatGroq(
        model=model_name,
        temperature=kwargs["temperature"],
        max_tokens=kwargs["max_tokens"],
        timeout=kwargs["timeout"],
        max_retries=kwargs["max_retries"],
        stop_sequences=None,
    )


def _create_bedrock_llm(
    model_id: str,
    aws_region: str | None,
    endpoint_url: str | None,
    profile_name: str | None,
    **kwargs: Any,  # noqa: ANN401
) -> ChatBedrockConverse:
    """Create an Amazon Bedrock LLM instance.

    Returns:
        ChatBedrockConverse: Configured Bedrock LLM instance.
    """
    logger = logging.getLogger(_create_bedrock_llm.__name__)
    logger.info("Use Amazon Bedrock: %s", model_id)
    return ChatBedrockConverse(
        model=model_id,
        temperature=kwargs["temperature"],
        max_tokens=kwargs["max_tokens"],
        region_name=aws_region,
        base_url=endpoint_url,
        credentials_profile_name=profile_name,
    )


def _create_google_llm(
    model_name: str,
    **kwargs: Any,  # noqa: ANN401
) -> ChatGoogleGenerativeAI:
    """Create a Google Generative AI LLM instance.

    Returns:
        ChatGoogleGenerativeAI: Configured Google LLM instance.
    """
    logger = logging.getLogger(_create_google_llm.__name__)
    logger.info("Use Google Generative AI: %s", model_name)
    return ChatGoogleGenerativeAI(
        model=model_name,
        temperature=kwargs["temperature"],
        top_p=kwargs["top_p"],
        top_k=kwargs["top_k"],
        max_tokens=kwargs["max_tokens"],
        timeout=kwargs["timeout"],
        max_retries=kwargs["max_retries"],
    )


def _create_anthropic_llm(
    model_name: str,
    api_base: str | None,
    **kwargs: Any,  # noqa: ANN401
) -> ChatAnthropic:
    """Create an Anthropic LLM instance.

    Returns:
        ChatAnthropic: Configured Anthropic LLM instance.
    """
    logger = logging.getLogger(_create_anthropic_llm.__name__)
    logger.info("Use Anthropic: %s", model_name)
    logger.info("Anthropic API base: %s", api_base)
    return ChatAnthropic(
        model_name=model_name,
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
    model_name: str,
    api_base: str | None,
    organization: str | None,
    **kwargs: Any,  # noqa: ANN401
) -> ChatOpenAI:
    """Create an OpenAI LLM instance.

    Returns:
        ChatOpenAI: Configured OpenAI LLM instance.
    """
    logger = logging.getLogger(_create_openai_llm.__name__)
    logger.info("Use OpenAI: %s", model_name)
    logger.info("OpenAI API base: %s", api_base)
    logger.info("OpenAI organization: %s", organization)
    return ChatOpenAI(
        model=model_name,
        base_url=api_base,
        organization=organization,
        temperature=kwargs["temperature"],
        top_p=kwargs["top_p"],
        seed=kwargs["seed"],
        max_completion_tokens=kwargs["max_tokens"],
        timeout=kwargs["timeout"],
        max_retries=kwargs["max_retries"],
    )


def create_llm_instance(
    model_name: str | None = None,
    provider: str | None = None,
    ollama_base_url: str | None = None,
    ollama_keep_alive: str | int | None = None,
    cerebras_api_key: str | None = None,
    groq_api_key: str | None = None,
    google_api_key: str | None = None,
    anthropic_api_key: str | None = None,
    anthropic_api_base: str | None = None,
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
    timeout: int | None = DEFAULT_TIMEOUT,
    max_retries: int = DEFAULT_MAX_RETRIES,
    aws_credentials_profile_name: str | None = None,
    aws_region: str | None = None,
    bedrock_endpoint_base_url: str | None = None,
) -> (
    ChatOllama
    | ChatCerebras
    | ChatGroq
    | ChatBedrockConverse
    | ChatGoogleGenerativeAI
    | ChatAnthropic
    | ChatOpenAI
):
    """Create an instance of a Language Learning Model (LLM).

    Args:
        model_name (str | None): Name or ID of the model to use.
        provider (str | None): LLM provider to use (openai, google, anthropic,
            cerebras, groq, bedrock, ollama). If not specified, will be
            inferred from API keys and environment.
        ollama_base_url (str | None): Base URL for the Ollama API.
        ollama_keep_alive (str | int | None): Duration to keep model loaded
            in memory. Can be a duration string (e.g., "5m", "10m") or integer
            (seconds). Use -1 to keep loaded indefinitely.
        cerebras_api_key (str | None): API key for Cerebras.
        groq_api_key (str | None): API key for Groq.
        google_api_key (str | None): API key for Google Generative AI.
        anthropic_api_key (str | None): API key for Anthropic.
        anthropic_api_base (str | None): Base URL for Anthropic API.
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
        timeout (int | None): Timeout for the API calls in seconds.
        max_retries (int): Maximum number of retries for API calls.
        aws_credentials_profile_name (str | None): AWS credentials profile name.
        aws_region (str | None): AWS region for Bedrock.
        bedrock_endpoint_base_url (str | None): Base URL for Amazon Bedrock
            endpoint.

    Returns:
        ChatOllama | ChatCerebras | ChatGroq | ChatBedrockConverse |
        ChatGoogleGenerativeAI | ChatAnthropic | ChatOpenAI: An instance of the
        selected LLM.

    Raises:
        ValueError: If no valid model configuration is provided or if the model
            cannot be determined.
    """
    override_env_vars(
        CEREBRAS_API_KEY=cerebras_api_key,
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
        "timeout": timeout,
        "max_retries": max_retries,
    }

    provider = _infer_provider(
        provider=provider,
        ollama_base_url=ollama_base_url,
        cerebras_api_key=cerebras_api_key,
        groq_api_key=groq_api_key,
        google_api_key=google_api_key,
        anthropic_api_key=anthropic_api_key,
        openai_api_key=openai_api_key,
    )

    provider_specs: dict[str, tuple[Any, str, str | None, str, dict[str, Any]]] = {
        "ollama": (
            _create_ollama_llm,
            "model_name",
            model_name,
            "Model name is required when using Ollama.",
            {"base_url": ollama_base_url, "keep_alive": ollama_keep_alive},
        ),
        "cerebras": (
            _create_cerebras_llm,
            "model_name",
            model_name,
            "Model name is required when using Cerebras API.",
            {},
        ),
        "groq": (
            _create_groq_llm,
            "model_name",
            model_name,
            "Model name is required when using Groq API.",
            {},
        ),
        "bedrock": (
            _create_bedrock_llm,
            "model_id",
            model_name,
            "Model ID is required when using Amazon Bedrock.",
            {
                "aws_region": aws_region,
                "endpoint_url": bedrock_endpoint_base_url,
                "profile_name": aws_credentials_profile_name,
            },
        ),
        "google": (
            _create_google_llm,
            "model_name",
            model_name,
            "Model name is required when using Google API.",
            {},
        ),
        "anthropic": (
            _create_anthropic_llm,
            "model_name",
            model_name,
            "Model name is required when using Anthropic API.",
            {"api_base": anthropic_api_base},
        ),
        "openai": (
            _create_openai_llm,
            "model_name",
            model_name,
            "Model name is required when using OpenAI API.",
            {
                "api_base": openai_api_base,
                "organization": openai_organization,
            },
        ),
    }

    if not provider or provider not in provider_specs:
        error_message = "The model cannot be determined."
        raise ValueError(error_message)

    factory, model_key, model_value, error_message, provider_kwargs = provider_specs[
        provider
    ]
    if not model_value:
        raise ValueError(error_message)

    return factory(
        **{model_key: model_value},
        **provider_kwargs,
        **llm_kwargs,
    )
