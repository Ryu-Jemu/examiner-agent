"""테스트 공통 설정: 실제 API 키 없이도 돌도록 더미 환경변수를 주입한다."""

import os

os.environ.setdefault("LLM_API_KEY", "test-key")
os.environ.setdefault("LLM_MODEL", "test-model")
os.environ.setdefault("GOOGLE_API_KEY", "test-key")
# 임계값 테스트가 셸 환경·.env 에 좌우되지 않도록 고정(보정 기본값과 동일).
os.environ["RETRIEVE_MIN_RELEVANCE"] = "0.70"

from factchecker.config import reset_settings  # noqa: E402

reset_settings()
