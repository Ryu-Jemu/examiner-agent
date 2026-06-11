"""테스트 공통 설정: 실제 API 키 없이도 돌도록 더미 환경변수를 주입한다."""

import os

os.environ.setdefault("LLM_API_KEY", "test-key")
os.environ.setdefault("LLM_MODEL", "test-model")
os.environ.setdefault("GOOGLE_API_KEY", "test-key")

from factchecker.config import reset_settings  # noqa: E402

reset_settings()
