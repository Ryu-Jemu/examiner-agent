"""중앙 설정 레이어.

모든 환경변수를 한 곳에서 읽고 검증한다. 필요한 API 키가 없으면 스택트레이스 대신
친절한 한국어 오류(`ConfigError`)로 빠르게 실패한다. 경로는 모두 패키지 기준
`pathlib` 로 해석하여 하드코딩된 절대경로가 없도록 한다(크로스 OS 재현성).

구성:
- 채팅 LLM: 외부 LLM 1종. `LLM_API_KEY` 로 호출하며 모델 ID 는 `LLM_MODEL` 로 지정한다
  (실제 모델 ID 는 커밋되지 않는 .env 에만 둔다).
- 임베딩(RAG): 채팅과 독립. 기본은 Gemini 임베딩(`GOOGLE_API_KEY` 필요), 또는 로컬 hf.
  → 하이브리드: 채팅은 외부 LLM, 임베딩은 Gemini.
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

    # 채팅 LLM
    llm_api_key: str
    llm_model: str
    llm_max_tokens: int
    allow_user_key: bool              # BYOK: 요청마다 사용자가 키 입력(서버 키 생략 허용)
    # 임베딩 (RAG)
    google_api_key: str               # EMBEDDING_BACKEND=gemini 일 때 사용
    embedding_backend: str            # "gemini" | "hf"
    gemini_embedding_model: str
    hf_embedding_model: str
    # 검색
    search_backend: str               # "local" | "ddg" | "tavily"
    tavily_api_key: str | None
    # 그래프/실행
    max_loops: int
    retrieve_k: int
    retrieve_min_relevance: float     # 코사인 관련성 임계값(이하 스니펫은 무관으로 제외)
    confidence_delta_threshold: float
    max_claims: int
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

    embedding_backend = (os.getenv("EMBEDDING_BACKEND") or "gemini").strip().lower()
    if embedding_backend not in ("gemini", "hf"):
        embedding_backend = "gemini"

    llm_api_key = (os.getenv("LLM_API_KEY") or "").strip()
    llm_model = (os.getenv("LLM_MODEL") or "").strip()
    google_key = (os.getenv("GOOGLE_API_KEY") or "").strip()
    allow_user_key = (os.getenv("ALLOW_USER_KEY") or "").strip().lower() in (
        "1", "true", "yes", "on",
    )

    # 필요한 값만 검증한다.
    if require_api_key:
        # BYOK(allow_user_key) 면 서버 키는 생략 가능(요청마다 사용자가 입력).
        if not allow_user_key and _missing(llm_api_key):
            raise ConfigError(
                "\n[설정 오류] LLM_API_KEY 가 설정되지 않았습니다.\n"
                "  1) .env.example 을 .env 로 복사하세요:  cp .env.example .env\n"
                "  2) .env 의 LLM_API_KEY 값을 발급받은 실제 키로 교체하세요.\n"
                "  (BYOK 배포는 ALLOW_USER_KEY=true 로 두고 키는 사용자가 입력합니다.)\n"
            )
        if _missing(llm_model):
            raise ConfigError(
                "\n[설정 오류] LLM_MODEL 이 설정되지 않았습니다.\n"
                "  .env 의 LLM_MODEL 에 사용할 모델 ID 를 입력하세요.\n"
            )
        if llm_model.startswith(("sk-", "AIza", "Bearer ")):
            raise ConfigError(
                "\n[설정 오류] LLM_MODEL 에 API 키로 보이는 값이 들어 있습니다.\n"
                "  LLM_MODEL 에는 '모델 ID'(모델 이름)를, API 키는 LLM_API_KEY\n"
                "  (BYOK 배포는 화면 입력)에 넣어야 합니다. 두 값을 바꿔 넣지 마세요.\n"
            )
        if embedding_backend == "gemini" and _missing(google_key):
            raise ConfigError(
                "\n[설정 오류] EMBEDDING_BACKEND=gemini 인데 GOOGLE_API_KEY 가 없습니다.\n"
                "  임베딩(RAG)은 Gemini 임베딩을 사용합니다.\n"
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
        llm_api_key=llm_api_key,
        llm_model=llm_model,
        llm_max_tokens=_int("LLM_MAX_TOKENS", 4096),
        allow_user_key=allow_user_key,
        google_api_key=google_key,
        embedding_backend=embedding_backend,
        gemini_embedding_model=(
            os.getenv("GEMINI_EMBEDDING_MODEL") or "models/gemini-embedding-001"
        ).strip(),
        hf_embedding_model=(os.getenv("HF_EMBEDDING_MODEL") or "BAAI/bge-m3").strip(),
        search_backend=search_backend,
        tavily_api_key=(os.getenv("TAVILY_API_KEY") or "").strip() or None,
        max_loops=_int("MAX_LOOPS", 2),
        retrieve_k=_int("RETRIEVE_K", 3),
        retrieve_min_relevance=_float("RETRIEVE_MIN_RELEVANCE", 0.32),
        confidence_delta_threshold=_float("CONFIDENCE_DELTA_THRESHOLD", 0.05),
        max_claims=_int("MAX_CLAIMS", 2),
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
