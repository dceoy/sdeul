"""Tests for the config module."""

# pyright: reportMissingImports=false

from sdeul.config import (
    ExtractConfig,
    LLMConfig,
    ModelConfig,
    ProcessingConfig,
)


def test_extract_config_init_with_defaults() -> None:
    """Test ExtractConfig initialization with default values."""
    config = ExtractConfig()

    assert isinstance(config.llm, LLMConfig)
    assert isinstance(config.model, ModelConfig)
    assert isinstance(config.processing, ProcessingConfig)


def test_extract_config_init_with_provided_configs() -> None:
    """Test ExtractConfig initialization with provided config objects."""
    llm_config = LLMConfig(temperature=0.5)
    model_config = ModelConfig(openai_model="gpt-4")
    processing_config = ProcessingConfig(compact_json=True)

    config = ExtractConfig(
        llm=llm_config,
        model=model_config,
        processing=processing_config,
    )

    assert config.llm == llm_config
    assert config.model == model_config
    assert config.processing == processing_config


def test_extract_config_init_partial_configs() -> None:
    """Test ExtractConfig initialization with some provided and some default configs."""
    llm_config = LLMConfig(temperature=0.8)
    model_config = ModelConfig(openai_model="gpt-3.5-turbo")

    config = ExtractConfig(
        llm=llm_config,
        model=model_config,
    )

    assert config.llm == llm_config
    assert config.model == model_config
    assert isinstance(config.processing, ProcessingConfig)
