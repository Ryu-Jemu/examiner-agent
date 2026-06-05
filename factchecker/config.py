"""중앙 설정 레이어.

모든 환경변수를 한 곳에서 읽고 검증한다. 필요한 API 키가 없으면 스택트레이스 대신
친절한 한국어 오류(`ConfigError`)로 빠르게 실패한다. 경로는 모두 패키지 기준
`pathlib` 로 해석하여 하드코딩된 절대경로가 없도록 한다(크로스 OS 재현성).

LLM 공급자(provider)는 환경변수로 선택한다:
- anthropic (기본): Claude. langchain-anthropic ChatAnthropic 사용. ANTHROPIC_API_KEY 필요.
- gemini: Google Gemini. langchain-google-genai 사용. GOOGLE_API_KEY 필요.
임베딩은 채팅 공급자와 독립이다. Anthropic 에는 임베딩 API 가 없으므로, 임베딩은
gemini(GOOGLE_API_KEY 필요) 또는 hf(로컬, 키 불필요)를 사용한다.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

# --- 경로 (패키지 기준, 하드코딩 절대경로 없음) ---
PACKAGE_ROOT = Path(__file__).resolve().parent  # .../factchecker
REPO_ROOT = PACKAGE_ROOT.parent                 # 저장소 루트
DATA_DIR = REPO_ROOT / "data"

PLACEHOLDER_KEY = "YOUR-API-KEY-HERE"


class ConfigError(RuntimeError):
    """설정 누락/오류를 사용자 친화적으로 알리기 위한 예외."""


@dataclass(frozen=True)
class Settings:
    """불변 설정 스냅샷."""

    llm_provider: str                 # "anthropic" | "gemini"
    # Anthropic (Claude)
    anthropic_api_key: str
    anthropic_model: str
    anthropic_max_tokens: int
    # Google Gemini
    google_api_key: str
    gemini_model: str
    # 임베딩
    embedding_backend: str            # "gemini" | "hf"
    gemini_embedding_model: str
    hf_embedding_model: str
    # 검색
    search_backend: str               # "local" | "ddg" | "tavily"
    tavily_api_key: str | None
    # 그래프/실행
    max_loops: int
    retrieve_k: int
    confidence_delta_threshold: float
    max_claims: int
    llm_temperature: float
    llm_throttle_seconds: float
    llm_max_attempts: int
    chroma_dir: Path

    @property
    def evidence_corpus_path(self) -> Path:
        return DATA_DIR / "evidence_corpus" / "corpus.json"

    @property
    def technique_library_path(self) -> Path:
        return DATA_DIR / "technique_library" / "techniques.json"

    @property
    def testset_path(self) -> Path:
        return DATA_DIR / "testset" / "testset.json"


def _missing(value: str) -> bool:
    return value == "" or value == PLACEHOLDER_KEY


def _build_settings(require_api_key: bool = True) -> Settings:
    # 저장소의 .env 만 로드한다. 상위 디렉터리의 .env 를 walk-up 으로 픽업하면
    # 환경마다 결과가 달라지므로(재현성 저해) 명시적 경로만 사용한다.
    # 셸에 직접 export 된 환경변수는 os.getenv 로 그대로 인식된다.
    load_dotenv(REPO_ROOT / ".env")

    def _int(name: str, default: int) -> int:
        try:
            return int(os.getenv(name, str(default)))
        except (TypeError, ValueError):
            return default

    def _float(name: str, default: float) -> float:
        try:
            return float(os.getenv(name, str(default)))
        except (TypeError, ValueError):
            return default

    provider = (os.getenv("LLM_PROVIDER") or "anthropic").strip().lower()
    if provider not in ("anthropic", "gemini"):
        provider = "anthropic"

    embedding_backend = (os.getenv("EMBEDDING_BACKEND") or "gemini").strip().lower()
    if embedding_backend not in ("gemini", "hf"):
        embedding_backend = "gemini"

    anthropic_key = (os.getenv("ANTHROPIC_API_KEY") or "").strip()
    google_key = (os.getenv("GOOGLE_API_KEY") or "").strip()

    # 필요한 키만 검증한다(공급자/임베딩 백엔드에 따라 다름).
    if require_api_key:
        if provider == "anthropic" and _missing(anthropic_key):
            raise ConfigError(
                "\n[설정 오류] ANTHROPIC_API_KEY 가 설정되지 않았습니다 "
                "(LLM_PROVIDER=anthropic).\n"
                "  1) .env.example 을 .env 로 복사하세요:  cp .env.example .env\n"
                "  2) https://console.anthropic.com/ 에서 키를 발급해\n"
                "     .env 의 ANTHROPIC_API_KEY 값을 실제 키로 교체하세요.\n"
                "  (제출/커밋 파일에는 키를 'YOUR-API-KEY-HERE' 로 비워 두세요.)\n"
            )
        if provider == "gemini" and _missing(google_key):
            raise ConfigError(
                "\n[설정 오류] GOOGLE_API_KEY 가 설정되지 않았습니다 "
                "(LLM_PROVIDER=gemini).\n"
                "  https://aistudio.google.com/apikey 에서 키를 발급해 .env 에 넣으세요.\n"
            )
        if embedding_backend == "gemini" and _missing(google_key):
            raise ConfigError(
                "\n[설정 오류] EMBEDDING_BACKEND=gemini 인데 GOOGLE_API_KEY 가 없습니다.\n"
                "  Anthropic 에는 임베딩 API 가 없어 임베딩은 Gemini 를 씁니다.\n"
                "  해결: (a) .env 에 GOOGLE_API_KEY 를 넣거나,\n"
                "        (b) 키 없이 쓰려면 EMBEDDING_BACKEND=hf 로 바꾸세요"
                "(로컬 임베딩, torch 설치 필요).\n"
            )

    chroma_override = (os.getenv("CHROMA_DIR") or "").strip()
    chroma_dir = Path(chroma_override) if chroma_override else (DATA_DIR / ".chroma")

    search_backend = (os.getenv("SEARCH_BACKEND") or "local").strip().lower()
    if search_backend not in ("local", "ddg", "tavily"):
        search_backend = "local"

    return Settings(
        llm_provider=provider,
        anthropic_api_key=anthropic_key,
        anthropic_model=(os.getenv("ANTHROPIC_MODEL") or "claude-opus-4-8").strip(),
        anthropic_max_tokens=_int("ANTHROPIC_MAX_TOKENS", 4096),
        google_api_key=google_key,
        gemini_model=(os.getenv("GEMINI_MODEL") or "gemini-2.5-flash-lite").strip(),
        embedding_backend=embedding_backend,
        gemini_embedding_model=(
            os.getenv("GEMINI_EMBEDDING_MODEL") or "models/gemini-embedding-001"
        ).strip(),
        hf_embedding_model=(os.getenv("HF_EMBEDDING_MODEL") or "BAAI/bge-m3").strip(),
        search_backend=search_backend,
        tavily_api_key=(os.getenv("TAVILY_API_KEY") or "").strip() or None,
        max_loops=_int("MAX_LOOPS", 2),
        retrieve_k=_int("RETRIEVE_K", 4),
        confidence_delta_threshold=_float("CONFIDENCE_DELTA_THRESHOLD", 0.05),
        max_claims=_int("MAX_CLAIMS", 3),
        llm_temperature=_float("LLM_TEMPERATURE", 0.0),
        llm_throttle_seconds=_float("LLM_THROTTLE_SECONDS", 0.0),
        llm_max_attempts=_int("LLM_MAX_ATTEMPTS", 5),
        chroma_dir=chroma_dir,
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """싱글턴 설정. 테스트는 `reset_settings()` 로 캐시를 비운다."""
    return _build_settings(require_api_key=True)


def reset_settings() -> None:
    """테스트 헬퍼: 캐시된 설정을 비운다."""
    get_settings.cache_clear()
