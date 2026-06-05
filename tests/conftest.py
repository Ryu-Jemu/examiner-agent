"""테스트 공통 픽스처.

- 가짜 API 키 + SEARCH_BACKEND=local + 임시 CHROMA_DIR 환경을 구성한다.
- 실제 네트워크/키 없이 단위 테스트가 동작하도록 한다.
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _env(tmp_path, monkeypatch):
    """모든 테스트에 깨끗한 설정 환경을 주입한다."""
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-anthropic-key-not-real")
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key-not-real")
    monkeypatch.setenv("SEARCH_BACKEND", "local")
    monkeypatch.setenv("EMBEDDING_BACKEND", "gemini")
    monkeypatch.setenv("CHROMA_DIR", str(tmp_path / ".chroma"))
    monkeypatch.setenv("MAX_LOOPS", "2")
    monkeypatch.setenv("RETRIEVE_K", "4")

    from factchecker.config import reset_settings

    reset_settings()
    yield
    reset_settings()


@pytest.fixture
def fake_embeddings():
    """결정론적 가짜 임베딩(네트워크/키 불필요)."""
    from langchain_core.embeddings import DeterministicFakeEmbedding

    return DeterministicFakeEmbedding(size=64)
