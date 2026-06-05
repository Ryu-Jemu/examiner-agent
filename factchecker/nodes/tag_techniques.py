"""Node 5: tag_techniques (RAG 기법 라이브러리, 병렬 분기).

입력 본문만 보고 조작 기법을 태깅한다(진위 판정과 독립 → 메인 체인과 병렬 실행).
`technique_tags` 는 add 리듀서를 가지므로 병렬 쓰기에도 안전하다.
"""

from __future__ import annotations

import logging

from .. import prompts
from ..llm import structured_invoke
from ..models import TechniqueTagList
from ..rag.technique_retriever import format_library_block, retrieve_techniques
from ..state import FactCheckState

logger = logging.getLogger("factchecker.nodes.technique")


def tag_techniques(state: FactCheckState) -> dict:
    input_text = (state.get("input_text") or "").strip()
    if not input_text:
        return {"technique_tags": []}

    try:
        entries = retrieve_techniques(input_text)
    except Exception as exc:  # noqa: BLE001
        logger.warning("기법 라이브러리 회수 실패: %s", exc)
        entries = []

    valid_ids = {e["library_entry_id"] for e in entries}
    prompt = prompts.render(
        "tag_techniques",
        input_text=input_text,
        library_block=format_library_block(entries),
    )
    result = structured_invoke(
        prompt, TechniqueTagList, default=TechniqueTagList(technique_tags=[])
    )

    # library_entry_id 가 비거나 잘못된 경우 tag 이름으로 보정 시도
    tag_to_id = {e["tag"]: e["library_entry_id"] for e in entries}
    tags = []
    for t in result.technique_tags:
        if t.library_entry_id not in valid_ids:
            t.library_entry_id = tag_to_id.get(t.tag.value, t.library_entry_id)
        tags.append(t)

    logger.info("기법 태그 %d개", len(tags))
    return {"technique_tags": tags}  # add 리듀서(병렬 안전)
