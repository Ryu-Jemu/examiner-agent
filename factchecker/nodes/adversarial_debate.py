"""검사 ⇄ 변호 적대적 논거 생성 노드. 인용 id 는 실제 풀의 부분집합으로 사후 필터링한다."""

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
    for claim in claims:
        if not claim.checkable:
            continue
        ev = evidence_for_claim(pool, claim.claim_id)
        # 비용 절감: 증거가 없으면 양측이 인용할 게 없으므로 LLM 호출을 건너뛴다
        # (판사가 증거 0 → "불충분(판단 불가)"으로 처리). 빈 논거쌍만 남긴다.
        if not ev:
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

    return {"arguments": pairs}  # add 리듀서 → 누적
