"""조작 기법 라이브러리 회수(4종뿐이라 유사도 순으로 전부 회수해 프롬프트에 제공)."""

import logging

logger = logging.getLogger("factchecker.rag.technique")


def retrieve_techniques(input_text: str, *, store=None, k: int = 4) -> list[dict]:
    """기법 라이브러리 항목 목록을 반환한다. 각 항목: {library_entry_id, tag, content}."""
    if store is None:
        from .vectorstore import get_or_build_techniques

        store = get_or_build_techniques()

    try:
        docs = store.similarity_search(input_text or " ", k=k)
    except Exception as exc:
        logger.warning("기법 유사도 검색 실패: %s", exc)
        docs = []

    results = []
    for doc in docs:
        meta = doc.metadata or {}
        results.append(
            {
                "library_entry_id": meta.get("library_entry_id", ""),
                "tag": meta.get("tag", ""),
                "content": doc.page_content,
            }
        )
    return results


def format_library_block(entries: list[dict]) -> str:
    """프롬프트용 라이브러리 블록 문자열."""
    if not entries:
        return "(기법 라이브러리가 비어 있습니다)"
    lines = []
    for e in entries:
        lines.append(f"- id={e['library_entry_id']}\n{e['content']}")
    return "\n\n".join(lines)
