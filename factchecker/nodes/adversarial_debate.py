"""Node 3: adversarial_debate (검사 ⇄ 변호).

각 주장에 대해 검사·변호가 *해당 주장의 회수 스니펫만* 보고 논거를 만든다.
환각 차단을 위해 코드 레벨에서 cited_snippet_ids 를 실제 풀 id 의 부분집합으로
사후 필터링한다(프롬프트 규칙 + 코드 가드 이중 방어).
"""

from __future__ import annotations

import logging

from .. import prompts
from ..llm import structured_invoke
from ..models import ArgumentPair, Claim, EvidenceItem, SideArgument
from ..state import FactCheckState
from .formatting import evidence_for_claim, format_evidence_block

logger = logging.getLogger("factchecker.nodes.debate")


def _filter_citations(side: SideArgument, valid_ids: set[str]) -> SideArgument:
    """지어낸 인용 id 를 제거한다(환각 가드)."""
    kept, dropped = [], []
    for sid in side.cited_snippet_ids:
        (kept if sid in valid_ids else dropped).append(sid)
    if dropped:
        logger.warning("환각 인용 제거: %s (유효 id 아님)", dropped)
    return SideArgument(summary=side.summary, cited_snippet_ids=kept)


def adversarial_debate(state: FactCheckState) -> dict:
    claims: list[Claim] = state.get("claims", []) or []
    pool: list[EvidenceItem] = state.get("evidence_pool", []) or []
    loop = state.get("loop_count", 0)

    pairs: list[ArgumentPair] = []
    skipped = 0
    for claim in claims:
        if not claim.checkable:
            continue
        ev = evidence_for_claim(pool, claim.claim_id)
        # 비용 절감: 증거가 없으면 양측이 인용할 게 없으므로 LLM 호출을 건너뛴다
        # (판사가 증거 0 → "불충분(판단 불가)"으로 처리). 빈 논거쌍만 남긴다.
        if not ev:
            skipped += 1
            empty = SideArgument(summary="(인용할 증거 없음)", cited_snippet_ids=[])
            pairs.append(
                ArgumentPair(
                    claim_id=claim.claim_id, loop=loop,
                    prosecution=empty, defense=empty,
                )
            )
            continue
        valid_ids = {e.snippet_id for e in ev}
        evidence_block = format_evidence_block(ev)

        pro_prompt = prompts.render(
            "prosecution", claim_text=claim.text, evidence_block=evidence_block
        )
        def_prompt = prompts.render(
            "defense", claim_text=claim.text, evidence_block=evidence_block
        )
        empty = SideArgument(summary="(논거 생성 실패)", cited_snippet_ids=[])
        prosecution = structured_invoke(pro_prompt, SideArgument, default=empty)
        defense = structured_invoke(def_prompt, SideArgument, default=empty)

        prosecution = _filter_citations(prosecution, valid_ids)
        defense = _filter_citations(defense, valid_ids)

        pairs.append(
            ArgumentPair(
                claim_id=claim.claim_id,
                loop=loop,
                prosecution=prosecution,
                defense=defense,
            )
        )

    logger.info(
        "적대적 논거 %d쌍 생성(loop=%d, 증거없음 %d건 LLM 생략)", len(pairs), loop, skipped
    )
    return {"arguments": pairs}  # add 리듀서 → 누적
