"""Node 1: extract_claims — 입력에서 검증 가능한 사실주장을 원자 단위로 분해."""

from __future__ import annotations

import logging

from .. import prompts
from ..config import get_settings
from ..llm import structured_invoke
from ..models import ClaimList
from ..state import FactCheckState

logger = logging.getLogger("factchecker.nodes.extract")


def extract_claims(state: FactCheckState) -> dict:
    input_text = (state.get("input_text") or "").strip()

    # 루프 제어 필드를 깨끗하게 초기화(리듀서가 깔끔히 시작되도록).
    base_update = {
        "loop_count": 0,
        "last_confidence": 0.0,
        "prev_pool_size": 0,
    }

    if not input_text:
        logger.info("입력 텍스트가 비어 있음 → 빈 주장 목록")
        return {"claims": [], **base_update}

    prompt = prompts.render("extract_claims", input_text=input_text)
    result = structured_invoke(prompt, ClaimList, default=ClaimList(claims=[]))

    settings = get_settings()
    claims = result.claims[: settings.max_claims]  # 비용/시간 상한
    # claim_id 재정렬(모델이 빠뜨리거나 중복할 수 있으므로 0..N-1 보장)
    for idx, c in enumerate(claims):
        c.claim_id = idx

    checkable = [c for c in claims if c.checkable]
    logger.info("주장 %d개 추출(검증대상 %d개)", len(claims), len(checkable))
    return {"claims": claims, **base_update}
