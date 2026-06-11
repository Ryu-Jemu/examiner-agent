"""검사 ⇄ 변호 3턴 순차 토론 노드.

턴 구성(주장당): ① 검사 발언 → ② 변호가 검사 논거를 읽고 반박 → ③ 검사
재반박. 각 측은 자기 전속 리서처가 회수한 증거 풀(스탠스 태깅)만 받고,
인용 id 는 자기 풀의 부분집합으로 사후 필터링한다(환각 가드).
대화 전문은 debate_transcript 로 누적되어 프런트 대화창에 표시된다.
"""

import logging

from .. import prompts
from ..llm import structured_invoke
from ..models import (
    STANCE_DEFENSE,
    STANCE_PROSECUTION,
    ArgumentPair,
    Claim,
    DebateTurn,
    EvidenceItem,
    SideArgument,
)
from ..state import FactCheckState
from .formatting import (
    evidence_for_claim,
    evidence_for_side,
    format_evidence_block,
)

logger = logging.getLogger("factchecker.nodes.debate")


def _filter_citations(side: SideArgument, valid_ids: set[str]) -> SideArgument:
    """지어낸 인용 id 를 제거한다(환각 가드)."""
    kept, dropped = [], []
    for sid in side.cited_snippet_ids:
        (kept if sid in valid_ids else dropped).append(sid)
    if dropped:
        logger.warning("환각 인용 제거: %s (유효 id 아님)", dropped)
    return SideArgument(summary=side.summary, cited_snippet_ids=kept)


def _side_block(side: SideArgument) -> str:
    ids = ", ".join(side.cited_snippet_ids) or "없음"
    return f"{side.summary} (인용: {ids})"


def adversarial_debate(state: FactCheckState) -> dict:
    claims: list[Claim] = state.get("claims", []) or []
    pool: list[EvidenceItem] = state.get("evidence_pool", []) or []
    loop = state.get("loop_count", 0)

    pairs: list[ArgumentPair] = []
    transcript: list[DebateTurn] = []
    for claim in claims:
        if not claim.checkable:
            continue
        ev_all = evidence_for_claim(pool, claim.claim_id)
        # 비용 절감: 증거가 없으면 양측이 인용할 게 없으므로 LLM 호출을 건너뛴다
        # (판사가 증거 0 → "불충분(판단 불가)"으로 처리). 빈 논거쌍만 남긴다.
        if not ev_all:
            empty = SideArgument(
                summary="(인용할 증거 없음)", cited_snippet_ids=[]
            )
            pairs.append(
                ArgumentPair(
                    claim_id=claim.claim_id, loop=loop,
                    prosecution=empty, defense=empty,
                )
            )
            for turn, role in enumerate(("검사", "변호")):
                transcript.append(
                    DebateTurn(
                        claim_id=claim.claim_id, loop=loop, turn=turn,
                        role=role, text="(인용할 증거가 없습니다)",
                    )
                )
            continue

        # 양측 전속 리서처 풀(자기 스탠스 + both + 미태깅)
        ev_pro = evidence_for_side(pool, claim.claim_id, STANCE_PROSECUTION)
        ev_def = evidence_for_side(pool, claim.claim_id, STANCE_DEFENSE)
        pro_ids = {e.snippet_id for e in ev_pro}
        def_ids = {e.snippet_id for e in ev_def}
        pro_block = format_evidence_block(ev_pro)
        def_block = format_evidence_block(ev_def)
        empty = SideArgument(summary="(논거 생성 실패)", cited_snippet_ids=[])

        # ① 검사 발언
        prosecution = structured_invoke(
            prompts.render(
                "prosecution",
                claim_text=claim.text,
                evidence_block=pro_block,
            ),
            SideArgument,
            default=empty,
        )
        prosecution = _filter_citations(prosecution, pro_ids)

        # ② 변호 반박(검사 논거를 입력으로 받음)
        defense = structured_invoke(
            prompts.render(
                "defense",
                claim_text=claim.text,
                opponent_block=_side_block(prosecution),
                evidence_block=def_block,
            ),
            SideArgument,
            default=empty,
        )
        defense = _filter_citations(defense, def_ids)

        # ③ 검사 재반박(변호 반박을 입력으로 받음)
        rebuttal = structured_invoke(
            prompts.render(
                "prosecution_rebuttal",
                claim_text=claim.text,
                own_block=_side_block(prosecution),
                opponent_block=_side_block(defense),
                evidence_block=pro_block,
            ),
            SideArgument,
            default=empty,
        )
        rebuttal = _filter_citations(rebuttal, pro_ids)

        pairs.append(
            ArgumentPair(
                claim_id=claim.claim_id,
                loop=loop,
                prosecution=prosecution,
                defense=defense,
                prosecution_rebuttal=rebuttal,
            )
        )
        for turn, (role, side) in enumerate(
            (("검사", prosecution), ("변호", defense), ("검사", rebuttal))
        ):
            transcript.append(
                DebateTurn(
                    claim_id=claim.claim_id,
                    loop=loop,
                    turn=turn,
                    role=role,
                    text=side.summary,
                    cited_snippet_ids=side.cited_snippet_ids,
                )
            )

    # arguments / debate_transcript 모두 add 리듀서 → 라운드 누적
    return {"arguments": pairs, "debate_transcript": transcript}
