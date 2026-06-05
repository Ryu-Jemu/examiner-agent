"""프롬프트 템플릿 로더.

`prompts/<name>.txt` 파일을 읽어 `.format(**kwargs)` 로 채운다. 모든 파일은
UTF-8 로 읽어 한국어 깨짐을 방지한다.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

_PROMPT_DIR = Path(__file__).resolve().parent


@lru_cache(maxsize=32)
def _read_template(name: str) -> str:
    path = _PROMPT_DIR / f"{name}.txt"
    if not path.exists():
        raise FileNotFoundError(f"프롬프트 템플릿을 찾을 수 없습니다: {path}")
    return path.read_text(encoding="utf-8")


def render(name: str, **kwargs) -> str:
    """이름으로 템플릿을 읽고 변수들을 채워 반환한다."""
    template = _read_template(name)
    if not kwargs:
        return template
    return template.format(**kwargs)
