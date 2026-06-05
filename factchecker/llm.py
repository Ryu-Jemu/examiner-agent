"""LLM·임베딩 팩토리 + 안전한 구조화 출력 헬퍼.

Gemini 구조화 출력은 가끔 None/누락/파싱 실패가 발생한다. 모든 노드가 안전하게
degrade 하도록, 구조화 호출 실패 시 호출자가 정한 기본값을 돌려주는 헬퍼를 둔다.
"""

from __future__ import annotations

import logging
import time
from functools import lru_cache
from typing import Type, TypeVar

from pydantic import BaseModel

from .config import get_settings

logger = logging.getLogger("factchecker.llm")

T = TypeVar("T", bound=BaseModel)

# 무료 등급 레이트리밋(429/quota) 식별용 마커
_RATE_LIMIT_MARKERS = (
    "429",
    "resourceexhausted",
    "resource exhausted",
    "quota",
    "rate limit",
    "rate_limit",
    "too many requests",
    "overloaded",   # Anthropic 529
    "529",
)

_last_call_ts = 0.0


def _is_rate_limit(exc: Exception) -> bool:
    return any(m in str(exc).lower() for m in _RATE_LIMIT_MARKERS)


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


@lru_cache(maxsize=4)
def get_llm(temperature: float | None = None):
    """설정된 공급자의 LangChain 채팅 모델을 반환한다(캐시).

    - anthropic: ChatAnthropic(Claude). Opus 4.8/4.7 은 temperature 파라미터를 받지
      않으므로(400) 전달하지 않는다. max_tokens 는 설정값을 사용한다.
    - gemini: ChatGoogleGenerativeAI. temperature 를 전달한다.
    """
    settings = get_settings()

    if settings.llm_provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            model=settings.anthropic_model,
            api_key=settings.anthropic_api_key,
            max_tokens=settings.anthropic_max_tokens,
            timeout=120,
            # temperature 는 의도적으로 전달하지 않음(Opus 4.8/4.7 에서 400).
        )

    from langchain_google_genai import ChatGoogleGenerativeAI

    temp = settings.llm_temperature if temperature is None else temperature
    return ChatGoogleGenerativeAI(
        model=settings.gemini_model,
        google_api_key=settings.google_api_key,
        temperature=temp,
    )


@lru_cache(maxsize=2)
def get_embeddings():
    """임베딩 객체. 기본 Gemini, 선택적으로 로컬 HuggingFace."""
    settings = get_settings()
    if settings.embedding_backend == "hf":
        from langchain_huggingface import HuggingFaceEmbeddings  # 선택 의존성

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
    temperature: float | None = None,
) -> T:
    """`schema` 로 구조화된 응답을 받되, 실패하면 `default` 를 돌려준다.

    - 레이트리밋(429/quota) 은 지수 백오프로 재시도한다(무료 등급 대응).
    - 그 외 실패는 즉시 기본값으로 degrade 해 노드가 크래시하지 않게 한다.
    """
    max_attempts = max(1, get_settings().llm_max_attempts)
    for attempt in range(max_attempts):
        try:
            _throttle()
            llm = get_llm(temperature=temperature)
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
            logger.warning("구조화 호출 실패(%s) → 기본값: %s", schema.__name__, exc)
            return default
    return default
