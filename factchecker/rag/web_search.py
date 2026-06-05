"""선택적 라이브 웹 검색 (기본 비활성).

SEARCH_BACKEND=ddg  → DuckDuckGo (키 불필요, `ddgs` 패키지 필요)
SEARCH_BACKEND=tavily → Tavily (TAVILY_API_KEY 필요)

재현성 요건상 기본값은 local 이며, 이 모듈은 절대 크래시하지 않는다(실패 시 빈 결과).
"""

from __future__ import annotations

import hashlib
import logging

from ..config import get_settings

logger = logging.getLogger("factchecker.rag.web")


def _stable_id(prefix: str, text: str) -> str:
    h = hashlib.sha1(text.encode("utf-8")).hexdigest()[:10]
    return f"{prefix}_{h}"


def _search_ddg(query: str, k: int) -> list[dict]:
    try:
        from ddgs import DDGS  # 패키지명: ddgs (구 duckduckgo-search)
    except Exception:  # noqa: BLE001
        try:
            from duckduckgo_search import DDGS  # 구버전 호환
        except Exception as exc:  # noqa: BLE001
            logger.warning("ddgs/duckduckgo-search 미설치 → 웹 검색 생략: %s", exc)
            return []

    results = []
    try:
        with DDGS() as ddgs:
            for hit in ddgs.text(query, max_results=k):
                body = hit.get("body") or hit.get("snippet") or ""
                if not body:
                    continue
                results.append(
                    {
                        "snippet_id": _stable_id("web", body),
                        "snippet": body,
                        "source": hit.get("title", "DuckDuckGo"),
                        "url": hit.get("href") or hit.get("url"),
                    }
                )
    except Exception as exc:  # noqa: BLE001 (레이트리밋 등 모든 예외 → 빈 결과)
        logger.warning("DuckDuckGo 검색 실패(무시): %s", exc)
        return []
    return results


def _search_tavily(query: str, k: int) -> list[dict]:
    settings = get_settings()
    if not settings.tavily_api_key:
        logger.warning("TAVILY_API_KEY 없음 → Tavily 검색 생략")
        return []
    try:
        from langchain_community.tools.tavily_search import TavilySearchResults

        tool = TavilySearchResults(max_results=k, api_key=settings.tavily_api_key)
        hits = tool.invoke({"query": query})
    except Exception as exc:  # noqa: BLE001
        logger.warning("Tavily 검색 실패(무시): %s", exc)
        return []

    results = []
    for hit in hits or []:
        content = hit.get("content", "") if isinstance(hit, dict) else str(hit)
        if not content:
            continue
        results.append(
            {
                "snippet_id": _stable_id("web", content),
                "snippet": content,
                "source": (hit.get("url") if isinstance(hit, dict) else "Tavily") or "Tavily",
                "url": hit.get("url") if isinstance(hit, dict) else None,
            }
        )
    return results


def search_web(query: str, k: int = 4) -> list[dict]:
    """설정된 백엔드로 웹 검색. 항상 list[dict] 반환(실패 시 빈 목록)."""
    settings = get_settings()
    backend = settings.search_backend
    if backend == "ddg":
        return _search_ddg(query, k)
    if backend == "tavily":
        return _search_tavily(query, k)
    return []
