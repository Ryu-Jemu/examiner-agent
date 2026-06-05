"""config 레이어 테스트: 키 검증, 기본값, 토글 파싱."""

from __future__ import annotations

import pytest


def test_missing_embedding_key_raises(monkeypatch):
    """EMBEDDING_BACKEND=gemini(기본) 인데 GOOGLE_API_KEY 없음 → 임베딩 검증 발동."""
    from factchecker.config import ConfigError, reset_settings

    monkeypatch.setenv("GOOGLE_API_KEY", "")
    reset_settings()
    from factchecker.config import get_settings

    with pytest.raises(ConfigError, match="EMBEDDING_BACKEND=gemini"):
        get_settings()


def test_placeholder_key_raises(monkeypatch):
    from factchecker.config import ConfigError, get_settings, reset_settings

    monkeypatch.setenv("GOOGLE_API_KEY", "YOUR-API-KEY-HERE")
    reset_settings()
    with pytest.raises(ConfigError):
        get_settings()


def test_missing_llm_key_raises(monkeypatch):
    """LLM_API_KEY 없음 → 채팅 LLM 키 검증 발동(임베딩 검증이 가리지 않게 hf)."""
    from factchecker.config import ConfigError, get_settings, reset_settings

    monkeypatch.setenv("LLM_API_KEY", "")
    monkeypatch.setenv("EMBEDDING_BACKEND", "hf")
    reset_settings()
    with pytest.raises(ConfigError, match="LLM_API_KEY"):
        get_settings()


def test_missing_llm_model_raises(monkeypatch):
    """LLM_MODEL 없음 → 모델 ID 검증 발동."""
    from factchecker.config import ConfigError, get_settings, reset_settings

    monkeypatch.setenv("LLM_MODEL", "")
    monkeypatch.setenv("EMBEDDING_BACKEND", "hf")
    reset_settings()
    with pytest.raises(ConfigError, match="LLM_MODEL"):
        get_settings()


def test_defaults():
    from factchecker.config import get_settings

    s = get_settings()
    assert s.llm_model == "test-model"      # conftest 주입
    assert s.llm_max_tokens == 4096         # 기본값
    assert s.search_backend == "local"
    assert s.embedding_backend == "gemini"
    assert s.max_loops == 2
    assert s.retrieve_k == 3
    assert s.max_claims == 2


def test_invalid_backend_falls_back(monkeypatch):
    from factchecker.config import get_settings, reset_settings

    monkeypatch.setenv("SEARCH_BACKEND", "garbage")
    monkeypatch.setenv("EMBEDDING_BACKEND", "nonsense")
    reset_settings()
    s = get_settings()
    assert s.search_backend == "local"
    assert s.embedding_backend == "gemini"


def test_paths_are_under_repo():
    from factchecker.config import DATA_DIR, get_settings

    s = get_settings()
    assert s.evidence_corpus_path == DATA_DIR / "evidence_corpus" / "corpus.json"
    assert s.technique_library_path.exists()
    assert s.testset_path.exists()
