#!/usr/bin/env python3
"""라이브 웹 앱. `python server.py` → http://127.0.0.1:8000 (배포는 DEPLOY.md)."""

import logging
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from fastapi import FastAPI  # noqa: E402
from fastapi.responses import HTMLResponse, JSONResponse  # noqa: E402
from pydantic import BaseModel  # noqa: E402

logger = logging.getLogger("server")

_INDEX = _ROOT / "web" / "index.html"


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, "") or default)
    except ValueError:
        return default


_MAX_INPUT_CHARS = _int_env("MAX_INPUT_CHARS", 2000)

app = FastAPI(title="Rumor Verification Agent", version="0.1.0")

_cors_raw = os.getenv("CORS_ORIGINS", "")
_cors = [o.strip() for o in _cors_raw.split(",") if o.strip()]
if _cors:
    from fastapi.middleware.cors import CORSMiddleware

    app.add_middleware(
        CORSMiddleware,
        allow_origins=_cors,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )


class CheckRequest(BaseModel):
    # 길이/공백 검증은 핸들러에서 처리해 일관된 {"error": ...} JSON 으로 응답한다.
    text: str = ""
    api_key: str | None = None  # BYOK: 사용자가 입력한 외부 LLM API 키


def _allow_user_key() -> bool:
    try:
        from factchecker.config import get_settings

        return bool(get_settings().allow_user_key)
    except Exception:
        return False


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return _INDEX.read_text(encoding="utf-8")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/api/config")
def config() -> dict:
    """프런트엔드가 키 입력란 표시 여부를 판단하는 데 사용."""
    return {"require_user_key": _allow_user_key()}


@app.post("/api/factcheck")
def factcheck(req: CheckRequest):
    text = req.text.strip()
    if not text:
        return JSONResponse({"error": "검증할 텍스트를 입력하세요."}, status_code=400)
    if len(text) > _MAX_INPUT_CHARS:
        return JSONResponse(
            {"error": f"입력이 너무 깁니다(최대 {_MAX_INPUT_CHARS}자)."}, status_code=413
        )

    # BYOK 모드면 사용자가 키를 입력해야 사용할 수 있다.
    user_key = (req.api_key or "").strip() or None
    if _allow_user_key():
        if not user_key:
            return JSONResponse(
                {"error": "API 키를 입력하세요. 이 서비스는 사용자가 발급한 외부 LLM "
                          "API 키로 동작하며, 키는 서버에 저장되지 않습니다."},
                status_code=400,
            )
    else:
        user_key = None  # 비-BYOK: 서버 환경설정 키 사용(요청 키 무시)

    from factchecker.config import ConfigError

    try:
        from factchecker.runner import run_factcheck

        payload = run_factcheck(text, api_key=user_key).model_dump(mode="json")
    except ConfigError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)
    except Exception:
        # 외부 API 오류 메시지(요청 ID·부분 키 등)가 클라이언트로 새지 않도록
        # 상세는 서버 로그로만 남기고 일반 메시지를 반환한다.
        logger.exception("factcheck failed")
        return JSONResponse(
            {"error": "검증 중 오류가 발생했습니다. 잠시 후 다시 시도해 주세요."},
            status_code=500,
        )

    return JSONResponse(payload)


def main() -> None:
    import uvicorn

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    host = os.getenv("HOST", "127.0.0.1")
    port = _int_env("PORT", 8000)

    # 시작 전 설정을 점검해 키 누락 시 안내(서버는 그래도 띄운다).
    try:
        from factchecker.config import get_settings

        get_settings()
        print("[설정 확인] OK — 라이브 검증 준비됨")
    except Exception as exc:
        print(str(exc))
        print("[안내] .env(또는 환경변수) 설정 후 새로고침하면 동작합니다.\n")

    print(f"http://{host}:{port}")
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
