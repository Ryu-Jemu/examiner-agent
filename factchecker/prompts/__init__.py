"""prompts/<name>.txt 를 읽어 .format(**kwargs) 로 채우는 템플릿 로더."""

from pathlib import Path

_PROMPT_DIR = Path(__file__).resolve().parent


def _read_template(name: str) -> str:
    path = _PROMPT_DIR / f"{name}.txt"
    if not path.exists():
        raise FileNotFoundError(f"프롬프트 템플릿을 찾을 수 없습니다: {path}")
    return path.read_text(encoding="utf-8")


def render(name: str, **kwargs) -> str:
    template = _read_template(name)
    if not kwargs:
        return template
    return template.format(**kwargs)
