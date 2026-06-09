# Rumor Verification Agent — 라이브 웹 앱 컨테이너
# 키는 이미지에 포함하지 않는다. 런타임에 환경변수(--env-file/-e 또는 플랫폼 시크릿)로 주입.
# 임베딩(RAG)은 Gemini 를 사용하므로 런타임에 GOOGLE_API_KEY 가 필요하다.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HOST=0.0.0.0 \
    PORT=8000

WORKDIR /app

# 의존성 먼저 설치(레이어 캐시 최적화)
COPY requirements.txt pyproject.toml README.md ./
RUN pip install --no-cache-dir -r requirements.txt

# 애플리케이션 코드
COPY factchecker ./factchecker
COPY data ./data
COPY web ./web
COPY server.py ./
RUN pip install --no-cache-dir -e .

EXPOSE 8000

# 헬스체크(배포 플랫폼이 활용)
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,os;urllib.request.urlopen(f'http://127.0.0.1:{os.getenv(\"PORT\",\"8000\")}/health').read()" || exit 1

CMD ["python", "server.py"]
