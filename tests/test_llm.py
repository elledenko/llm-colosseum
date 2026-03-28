"""Tests for the LLM client factory module."""
import pytest

from agent.llm import (
    _parse_model_str,
    SUPPORTED_TEXT_PROVIDERS,
    SUPPORTED_MULTIMODAL_PROVIDERS,
    get_client,
    get_client_multimodal,
)


class TestParseModelStr:
    """Tests for _parse_model_str helper."""

    def test_single_part_uses_default_provider(self):
        provider, model = _parse_model_str("gpt-4o", default_provider="openai")
        assert provider == "openai"
        assert model == "gpt-4o"

    def test_two_parts(self):
        provider, model = _parse_model_str("anthropic:claude-3-haiku-20240307")
        assert provider == "anthropic"
        assert model == "claude-3-haiku-20240307"

    def test_three_parts_joins_model_name(self):
        """Model names with colons should be preserved."""
        provider, model = _parse_model_str("ollama:llama:7b")
        assert provider == "ollama"
        assert model == "llama:7b"

    def test_default_provider_for_multimodal(self):
        provider, model = _parse_model_str("mistral", default_provider="ollama")
        assert provider == "ollama"
        assert model == "mistral"

    def test_empty_model_name(self):
        provider, model = _parse_model_str("openai:")
        assert provider == "openai"
        assert model == ""


class TestGetClient:
    """Tests for get_client factory."""

    def test_unknown_provider_raises_with_list(self):
        with pytest.raises(ValueError, match="Unknown text provider 'fakeprovider'"):
            get_client("fakeprovider:some-model")

    def test_error_message_includes_supported_providers(self):
        with pytest.raises(ValueError, match="Supported:"):
            get_client("invalid:model")


class TestGetClientMultimodal:
    """Tests for get_client_multimodal factory."""

    def test_unknown_provider_raises_with_list(self):
        with pytest.raises(ValueError, match="Unknown multimodal provider"):
            get_client_multimodal("fakeprovider:some-model")

    def test_error_message_includes_supported_providers(self):
        with pytest.raises(ValueError, match="Supported:"):
            get_client_multimodal("invalid:model")


class TestProviderSets:
    """Tests for the provider sets."""

    def test_multimodal_is_subset_of_text_plus_extras(self):
        """All multimodal providers should also be available as text providers."""
        assert SUPPORTED_MULTIMODAL_PROVIDERS.issubset(SUPPORTED_TEXT_PROVIDERS)

    def test_known_providers_in_text(self):
        for provider in ["openai", "anthropic", "mistral", "groq", "ollama"]:
            assert provider in SUPPORTED_TEXT_PROVIDERS

    def test_known_providers_in_multimodal(self):
        for provider in ["openai", "anthropic", "mistral", "ollama"]:
            assert provider in SUPPORTED_MULTIMODAL_PROVIDERS
