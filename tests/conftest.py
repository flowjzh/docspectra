"""Shared engine-test fixtures: the provider client seam and chain
pinning — every cloud engine's suite needs both, and only the provider
name differs."""

import pytest

from docspectra.parser import configure


@pytest.fixture
def pin_chain():
    """Return ``pin(mapping)`` — pin the test's chains; the ambient
    config is restored afterwards."""
    def pin(mapping: dict) -> None:
        configure(mapping)
    yield pin
    configure(None)


@pytest.fixture
def patch_provider_client(monkeypatch):
    """Return ``patch(provider, fake)`` — point that provider module's
    ``_AsyncClient`` seam at the fake for the test."""
    def patch(provider: str, fake) -> None:
        monkeypatch.setattr(
            f'docspectra.providers.{provider}._AsyncClient',
            lambda **kw: fake)
    return patch
