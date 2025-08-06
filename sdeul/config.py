"""Configuration dataclasses for SDEUL.

This module defines configuration dataclasses that group related parameters
for better organization and maintainability. These dataclasses follow
Kent Beck's tidying principles by extracting related parameters into
cohesive structures.
"""

from dataclasses import dataclass

from .constants import (
    DEFAULT_API_HOST,
    DEFAULT_API_PORT,
    DEFAULT_API_RELOAD,
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


@dataclass
class LLMConfig:
    """Configuration for Language Learning Model parameters.

    Groups core LLM sampling and generation parameters that are
    common across most LLM providers.
    """

    temperature: float = DEFAULT_TEMPERATURE
    top_p: float = DEFAULT_TOP_P
    top_k: int = DEFAULT_TOP_K
    max_tokens: int = DEFAULT_MAX_TOKENS
    seed: int = DEFAULT_SEED
    timeout: int | None = DEFAULT_TIMEOUT
    max_retries: int = DEFAULT_MAX_RETRIES


@dataclass
class LlamaCppConfig:
    """Configuration for LlamaCpp-specific parameters.

    Groups parameters that are specific to local LlamaCpp model execution,
    including hardware optimization settings.
    """

    repeat_penalty: float = DEFAULT_REPEAT_PENALTY
    repeat_last_n: int = DEFAULT_REPEAT_LAST_N
    n_ctx: int = DEFAULT_CONTEXT_WINDOW
    n_batch: int = DEFAULT_N_BATCH
    n_threads: int | None = DEFAULT_N_THREADS
    n_gpu_layers: int | None = DEFAULT_N_GPU_LAYERS
    f16_kv: bool = DEFAULT_F16_KV
    use_mlock: bool = DEFAULT_USE_MLOCK
    use_mmap: bool = DEFAULT_USE_MMAP
    token_wise_streaming: bool = DEFAULT_TOKEN_WISE_STREAMING


@dataclass
class ModelConfig:
    """Configuration for model selection and API credentials.

    Groups model names and API keys for different LLM providers.
    Organizes the selection of which LLM provider to use.
    """

    # Model names
    ollama_model: str | None = None
    llamacpp_model_file: str | None = None
    cerebras_model: str | None = None
    groq_model: str | None = None
    bedrock_model: str | None = None
    google_model: str | None = None
    anthropic_model: str | None = None
    openai_model: str | None = None

    # Base URLs and endpoints
    ollama_base_url: str | None = None
    openai_api_base: str | None = None
    anthropic_api_base: str | None = None
    bedrock_endpoint_url: str | None = None

    # API credentials
    openai_api_key: str | None = None
    openai_organization: str | None = None
    google_api_key: str | None = None
    anthropic_api_key: str | None = None
    cerebras_api_key: str | None = None
    groq_api_key: str | None = None
    aws_credentials_profile: str | None = None
    aws_region: str | None = None


@dataclass
class ProcessingConfig:
    """Configuration for data processing options.

    Groups parameters related to input/output processing,
    validation, and formatting options.
    """

    output_json_file: str | None = None
    compact_json: bool = False
    skip_validation: bool = False
    debug: bool = False
    info: bool = False


@dataclass
class ServerConfig:
    """Configuration for the FastAPI server.

    Groups server-related configuration parameters.
    """

    host: str = DEFAULT_API_HOST
    port: int = DEFAULT_API_PORT
    reload: bool = DEFAULT_API_RELOAD


@dataclass
class ExtractConfig:
    """Complete configuration for the extract command.

    Aggregates all configuration groups for the main extraction functionality.
    This provides a single configuration object that encompasses all settings.
    """

    llm: LLMConfig
    llamacpp: LlamaCppConfig
    model: ModelConfig
    processing: ProcessingConfig

    def __init__(
        self,
        llm: LLMConfig | None = None,
        llamacpp: LlamaCppConfig | None = None,
        model: ModelConfig | None = None,
        processing: ProcessingConfig | None = None,
    ) -> None:
        """Initialize ExtractConfig with default values if not provided."""
        self.llm = llm or LLMConfig()
        self.llamacpp = llamacpp or LlamaCppConfig()
        self.model = model or ModelConfig()
        self.processing = processing or ProcessingConfig()
