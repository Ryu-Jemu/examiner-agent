#!/usr/bin/env python3
"""백엔드 연결 라이브 웹 앱.

  python server.py            # 로컬: http://127.0.0.1:8000

정적 데모(demo.html)와 달리, 이 서버는 입력한 임의 주장을 실제 LangGraph 에이전트로
실행한다(RAG 증거 수집 → 적대적 토론 → 판정 → 기법 태깅 → 반론 카드). API 키는 서버
측 `.env`(또는 배포 플랫폼 환경변수)에서만 읽으며 프런트엔드(HTML)로는 절대 전달되지
않는다. 외부 배포는 DEPLOY.md 참고.

엔드포인트:
  GET  /                  라이브 프런트엔드(web/index.html)
  GET  /health            상태 점검(배포 헬스체크용)
  POST /api/factcheck     {"text": "..."} → 최종 리포트 JSON

배포/운영 환경변수(서비스 안정성·최적화):
  HOST(기본 127.0.0.1) / PORT(기본 8000)   배포 시 HOST=0.0.0.0
  MAX_INPUT_CHARS(기본 2000)               과대 입력 차단(비용·안정성)
  MAX_CONCURRENCY(기본 2)                   동시 검증 상한(초과 시 429)
  CACHE_SIZE(기본 64)                       동일 입력 결과 캐시(0=비활성)
  CORS_ORIGINS(기본 없음)                   교차 출처 허용 목록(콤마 구분; 미설정=동일 출처만)
"""

from __future__ import annotations

import logging
import os
import sys
import threading
from collections import OrderedDict
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
_MAX_CONCURRENCY = max(1, _int_env("MAX_CONCURRENCY", 2))
_CACHE_SIZE = max(0, _int_env("CACHE_SIZE", 64))

# 동시 실행 상한(무거운 LLM/임베딩 호출이 무한정 쌓이지 않게 — 안정성·비용 보호).
_sem = threading.BoundedSemaphore(_MAX_CONCURRENCY)
# 동일 입력 결과 캐시(반복 질의 비용 절감). LLM 산출은 비결정적이라 "같은 입력=같은 답"
# 수준의 근사 캐시이며 데모/비용 최적화 목적이다.
_cache: "OrderedDict[str, dict]" = OrderedDict()
_cache_lock = threading.Lock()

app = FastAPI(title="단톡방 루머 적대적 팩트체커", version="0.1.0")

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
    # 길이/공백 검증은 핸들러에서 처리해 일관된 {"error": ...} JSON 으로 응답한다
    # (Pydantic 제약을 쓰면 422 detail 구조가 나와 프런트가 일관되게 표시하지 못함).
    text: str = ""
    api_key: str | None = None  # BYOK: 사용자가 입력한 외부 LLM API 키


def _allow_user_key() -> bool:
    """BYOK 모드 여부(설정 오류 시 False 로 간주)."""
    try:
        from factchecker.config import get_settings

        return bool(get_settings().allow_user_key)
    except Exception:  # noqa: BLE001
        return False


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return _INDEX.read_text(encoding="utf-8")


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "max_concurrency": _MAX_CONCURRENCY}


@app.get("/api/config")
def config() -> dict:
    """프런트엔드가 키 입력란 표시·필수 여부를 판단하는 데 사용."""
    return {"require_user_key": _allow_user_key()}


@app.post("/api/factcheck")
def factcheck(req: CheckRequest):
    """입력 주장을 실제 에이전트로 검증하고 최종 리포트(JSON)를 반환한다."""
    text = req.text.strip()
    if not text:
        return JSONResponse({"error": "검증할 텍스트를 입력하세요."}, status_code=400)
    if len(text) > _MAX_INPUT_CHARS:
        return JSONResponse(
            {"error": f"입력이 너무 깁니다(최대 {_MAX_INPUT_CHARS}자)."}, status_code=413
        )

    # BYOK: 사용자 키 전용 모드면 키가 있어야 서비스 사용 가능(캐시 응답 포함).
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

    if _CACHE_SIZE:
        with _cache_lock:
            hit = _cache.get(text)
            if hit is not None:
                _cache.move_to_end(text)
                return JSONResponse({**hit, "cached": True})

    # 동시 검증 상한 초과 시 즉시 429(대기열 무한 적체 방지 → 서비스 안정성).
    if not _sem.acquire(blocking=False):
        return JSONResponse(
            {"error": "현재 다른 검증을 처리 중입니다. 잠시 후 다시 시도해 주세요."},
            status_code=429,
        )
    try:
        from factchecker.config import ConfigError

        try:
            from factchecker.runner import run_factcheck

            payload = run_factcheck(text, api_key=user_key).model_dump(mode="json")
        except ConfigError as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)
        except Exception:  # noqa: BLE001
            # 외부 API 오류 메시지(엔드포인트·요청 ID·부분 키 등)가 클라이언트로
            # 새지 않도록, 상세는 서버 로그로만 남기고 일반 메시지를 반환한다.
            logger.exception("factcheck failed")
            return JSONResponse(
                {"error": "검증 중 오류가 발생했습니다. 잠시 후 다시 시도해 주세요."},
                status_code=500
            )
    finally:
        _sem.release()

    if _CACHE_SIZE:
        with _cache_lock:
            _cache[text] = payload
            _cache.move_to_end(text)
            while len(_cache) > _CACHE_SIZE:
                _cache.popitem(last=False)

    return JSONResponse(payload)


def main() -> None:
    import uvicorn

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    host = os.getenv("HOST", "127.0.0.1")
    port = _int_env("PORT", 8000)

    # 시작 전 설정을 한 번 점검해 키 누락 시 친절히 안내(서버는 그래도 띄움).
    try:
        from factchecker.config import get_settings

        get_settings()
        print("[설정 확인] OK — 라이브 검증 준비됨")
    except Exception as exc:  # noqa: BLE001
        print(str(exc))
        print("[안내] .env(또는 환경변수) 설정 후 새로고침하면 동작합니다. 서버는 계속 띄웁니다.\n")

    print(
        f"➜ http://{host}:{port}  "
        f"(동시 {_MAX_CONCURRENCY} · 입력≤{_MAX_INPUT_CHARS}자 · 캐시 {_CACHE_SIZE})"
    )
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
