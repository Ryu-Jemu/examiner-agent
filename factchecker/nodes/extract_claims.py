"""입력에서 검증 가능한 사실주장을 원자 단위로 분해하는 노드."""

from .. import prompts
from ..config import get_settings
from ..llm import structured_invoke
from ..models import ClaimList
from ..state import FactCheckState


def extract_claims(state: FactCheckState) -> dict:
    input_text = (state.get("input_text") or "").strip()

    # 루프 제어 필드를 깨끗하게 초기화한다.
    base_update = {"loop_count": 0, "last_confidence": 0.0, "prev_pool_size": 0}

    if not input_text:
        return {"claims": [], **base_update}

    prompt = prompts.render("extract_claims", input_text=input_text)
    result = structured_invoke(prompt, ClaimList, default=ClaimList(claims=[]))

    settings = get_settings()
    claims = result.claims[: settings.max_claims]  # 비용/시간 상한
    # 모델이 claim_id 를 빠뜨리거나 중복할 수 있어 0..N-1 로 재정렬
    for idx, c in enumerate(claims):
        c.claim_id = idx

    return {"claims": claims, **base_update}
