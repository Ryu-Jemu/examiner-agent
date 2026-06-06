"""LLM·임베딩 팩토리 + 안전한 구조화 출력 헬퍼.

LLM 구조화 출력은 가끔 None/누락/파싱 실패가 발생한다. 모든 노드가 안전하게
degrade 하도록, 구조화 호출 실패 시 호출자가 정한 기본값을 돌려주는 헬퍼를 둔다.

채팅 모델은 `LLM_MODEL` 환경변수의 모델 ID 로 `init_chat_model` 을 통해 로드한다
(공급자는 모델 ID 로부터 자동 추론). 임베딩은 Gemini(또는 로컬 hf)를 사용한다.
"""

from __future__ import annotations

import contextvars
import logging
import re
import time
from functools import lru_cache
from typing import Type, TypeVar

from pydantic import BaseModel

from .config import get_settings

logger = logging.getLogger("factchecker.llm")

# 요청 범위 사용자 API 키(BYOK). 서버가 요청마다 set/reset 한다. graph.invoke 이전에
# 설정하면 LangGraph 가 병렬 노드 워커로 컨텍스트를 복사할 때 키가 함께 전파된다.
_user_api_key: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "factchecker_user_api_key", default=None
)


def set_request_api_key(key: str | None):
    """요청 범위 키 설정. 반환된 토큰을 finally 에서 reset_request_api_key 로 되돌린다."""
    return _user_api_key.set((key or "").strip() or None)


def reset_request_api_key(token) -> None:
    try:
        _user_api_key.reset(token)
    except (ValueError, LookupError):  # 다른 컨텍스트의 토큰이면 무시
        pass


T = TypeVar("T", bound=BaseModel)

# 재시도(백오프) 대상: 레이트리밋(429/quota) + 일시적 네트워크 오류
_RATE_LIMIT_MARKERS = (
    "429",
    "resourceexhausted",
    "resource exhausted",
    "quota",
    "rate limit",
    "rate_limit",
    "too many requests",
    "overloaded",   # 일부 공급자의 과부하(529)
    "529",
    # 일시적 네트워크/서버 오류 — 즉시 degrade 대신 백오프 재시도
    "connection error",
    "connection reset",
    "timed out",
    "timeout",
    "temporarily unavailable",
    "service unavailable",
    "502",
    "503",
    "504",
)

_last_call_ts = 0.0


def _is_rate_limit(exc: Exception) -> bool:
    return any(m in str(exc).lower() for m in _RATE_LIMIT_MARKERS)


# 로그·오류 메시지에서 API 키로 보이는 토큰을 마스킹(키가 서버 로그로 새지 않도록)
_KEY_RE = re.compile(r"(sk-ant-[\w\-]+|sk-[A-Za-z0-9]{16,}|AIza[\w\-]{20,})")


def _redact(text: str) -> str:
    return _KEY_RE.sub("[REDACTED]", text or "")


def _throttle() -> None:
    """설정된 최소 호출 간격을 유지(무료 등급 RPM 초과 예방)."""
    global _last_call_ts
    interval = get_settings().llm_throttle_seconds
    if interval <= 0:
        return
    now = time.monotonic()
    wait = _last_call_ts + interval - now
    if wait > 0:
        time.sleep(wait)
    _last_call_ts = time.monotonic()


@lru_cache(maxsize=8)
def _build_llm(model: str, api_key: str, max_tokens: int):
    """모델 ID 로 채팅 모델을 만든다(키별 캐시). `init_chat_model` 이 공급자를 자동 선택.

    sampling 파라미터(temperature 등)는 일부 모델이 거부할 수 있어 전달하지 않고,
    응답 상한만 `LLM_MAX_TOKENS` 로 둔다. 키별 캐시는 LRU 8개로 제한해 BYOK 다중 키의
    메모리 보관을 한정한다.
    """
    from langchain.chat_models import init_chat_model

    return init_chat_model(
        model=model, api_key=api_key, max_tokens=max_tokens, timeout=120
    )


def get_llm():
    """현재 요청에 적용할 채팅 모델을 반환한다.

    키 우선순위: 요청 범위 사용자 키(BYOK) > 환경설정 `LLM_API_KEY`.
    """
    settings = get_settings()
    key = _user_api_key.get(None) or settings.llm_api_key
    return _build_llm(settings.llm_model, key, settings.llm_max_tokens)


@lru_cache(maxsize=2)
def get_embeddings():
    """임베딩 객체(RAG). 기본 Gemini 임베딩, 선택적으로 로컬 HuggingFace."""
    settings = get_settings()
    if settings.embedding_backend == "hf":
        try:
            from langchain_huggingface import HuggingFaceEmbeddings  # 선택 의존성
        except ModuleNotFoundError as exc:
            from .config import ConfigError

            raise ConfigError(
                "\n[설정 오류] EMBEDDING_BACKEND=hf 인데 로컬 임베딩 라이브러리가 없습니다.\n"
                "  해결: pip install -e \".[local-embed]\"  "
                "(langchain-huggingface + sentence-transformers)\n"
                "  또는 .env 에서 EMBEDDING_BACKEND=gemini 로 변경하세요.\n"
            ) from exc

        return HuggingFaceEmbeddings(model_name=settings.hf_embedding_model)

    from langchain_google_genai import GoogleGenerativeAIEmbeddings

    return GoogleGenerativeAIEmbeddings(
        model=settings.gemini_embedding_model,
        google_api_key=settings.google_api_key,
    )


def structured_invoke(
    prompt: str,
    schema: Type[T],
    *,
    default: T,
) -> T:
    """`schema` 로 구조화된 응답을 받되, 실패하면 `default` 를 돌려준다.

    - 레이트리밋(429/529/quota) 은 지수 백오프로 재시도한다.
    - 그 외 실패는 즉시 기본값으로 degrade 해 노드가 크래시하지 않게 한다.
    """
    max_attempts = max(1, get_settings().llm_max_attempts)
    for attempt in range(max_attempts):
        try:
            _throttle()
            llm = get_llm()
            structured = llm.with_structured_output(schema)
            result = structured.invoke(prompt)
            if result is None:
                logger.warning("구조화 출력이 None → 기본값 (schema=%s)", schema.__name__)
                return default
            if not isinstance(result, schema):
                try:
                    return schema.model_validate(result)
                except Exception:  # noqa: BLE001
                    logger.warning("구조화 출력 타입 불일치 → 기본값 (%s)", schema.__name__)
                    return default
            return result
        except Exception as exc:  # noqa: BLE001
            if _is_rate_limit(exc) and attempt < max_attempts - 1:
                wait = min(60, 5 * (2 ** attempt))  # 5,10,20,40,60s
                logger.warning(
                    "레이트리밋 감지 → %ds 후 재시도 (%d/%d)",
                    wait, attempt + 1, max_attempts,
                )
                time.sleep(wait)
                continue
            logger.warning(
                "구조화 호출 실패(%s) → 기본값: %s",
                schema.__name__, _redact(str(exc)),
            )
            return default
    return default
