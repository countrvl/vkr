"""Вспомогательные компоненты инференса для домена генерации кода."""

from .anthropic_backend import AnthropicBackend
from .api_backend import APIBackend
from .base import GenerationResult, InferenceBackend, extract_code
from .ollama_backend import OllamaBackend
from .runner import ExperimentRunner

__all__ = [
    "AnthropicBackend",
    "APIBackend",
    "ExperimentRunner",
    "GenerationResult",
    "InferenceBackend",
    "OllamaBackend",
    "extract_code",
]
