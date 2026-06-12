"""환경변수 로드/검증과 설정 스냅샷."""

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

PACKAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_ROOT.parent
DATA_DIR = REPO_ROOT / "data"

PLACEHOLDER_KEY = "YOUR-API-KEY-HERE"


class ConfigError(RuntimeError):
    """설정이 없거나 잘못됐을 때 사용자에게 보여줄 오류."""


@dataclass(frozen=True)
class Settings:
    llm_api_key: str
    llm_model: str
    llm_max_tokens: int
    allow_user_key: bool              # BYOK: 요청마다 사용자가 키 입력
    google_api_key: str               # 임베딩(Gemini)용
    gemini_embedding_model: str
    max_loops: int
    retrieve_k: int
    retrieve_min_relevance: float
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


def _missing(value: str) -> bool:
    return value == "" or value == PLACEHOLDER_KEY


def _build_settings(require_api_key: bool = True) -> Settings:
    # 저장소 루트의 .env 만 읽는다(상위 디렉터리 walk-up 은 재현성을 해침).
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

    llm_api_key = (os.getenv("LLM_API_KEY") or "").strip()
    llm_model = (os.getenv("LLM_MODEL") or "").strip()
    google_key = (os.getenv("GOOGLE_API_KEY") or "").strip()
    allow_user_key = (os.getenv("ALLOW_USER_KEY") or "").strip().lower() in (
        "1", "true", "yes", "on",
    )

    if require_api_key:
        # BYOK 면 서버 키는 생략 가능(요청마다 사용자가 입력).
        if not allow_user_key and _missing(llm_api_key):
            raise ConfigError(
                "[설정 오류] LLM_API_KEY 가 없습니다. .env.example 을 .env 로 복사한 뒤 "
                "실제 키를 넣으세요. (BYOK 배포는 ALLOW_USER_KEY=true)"
            )
        if _missing(llm_model):
            raise ConfigError("[설정 오류] LLM_MODEL 이 없습니다. .env 에 모델 ID 를 넣으세요.")
        if llm_model.startswith(("sk-", "AIza", "Bearer ")):
            raise ConfigError(
                "[설정 오류] LLM_MODEL 에 API 키로 보이는 값이 들어 있습니다. "
                "모델 ID 와 LLM_API_KEY 를 바꿔 넣지 마세요."
            )
        if _missing(google_key):
            raise ConfigError(
                "[설정 오류] GOOGLE_API_KEY 가 없습니다. 임베딩(RAG)은 Gemini 임베딩을 "
                "사용하므로 .env 에 GOOGLE_API_KEY 를 넣어야 합니다."
            )

    chroma_override = (os.getenv("CHROMA_DIR") or "").strip()
    chroma_dir = Path(chroma_override) if chroma_override else (DATA_DIR / ".chroma")

    return Settings(
        llm_api_key=llm_api_key,
        llm_model=llm_model,
        llm_max_tokens=_int("LLM_MAX_TOKENS", 4096),
        allow_user_key=allow_user_key,
        google_api_key=google_key,
        gemini_embedding_model=(
            os.getenv("GEMINI_EMBEDDING_MODEL") or "models/gemini-embedding-001"
        ).strip(),
        max_loops=_int("MAX_LOOPS", 2),
        retrieve_k=_int("RETRIEVE_K", 3),
        # 코퍼스 실측 보정값(gemini-embedding-001·cosine): 주제 일치 스니펫
        # 0.717 이상 / 무관 0.676 이하로 분리. 임베딩 모델 교체 시 재보정.
        retrieve_min_relevance=_float("RETRIEVE_MIN_RELEVANCE", 0.70),
        confidence_delta_threshold=_float("CONFIDENCE_DELTA_THRESHOLD", 0.05),
        max_claims=_int("MAX_CLAIMS", 2),
        llm_throttle_seconds=_float("LLM_THROTTLE_SECONDS", 0.0),
        llm_max_attempts=_int("LLM_MAX_ATTEMPTS", 5),
        chroma_dir=chroma_dir,
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return _build_settings(require_api_key=True)


def reset_settings() -> None:
    get_settings.cache_clear()
