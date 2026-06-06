# 외부 배포 가이드 (DEPLOY.md)

라이브 웹 앱(`server.py` + `web/index.html`)을 외부에서 접속 가능하게 배포하는
**단계별** 절차입니다. 채점자·외부 사용자가 브라우저로 접속해 임의 주장을 실시간
검증할 수 있게 합니다.

> **핵심 원칙**
> 1. **키는 절대 코드/이미지/Git 에 넣지 않는다.** 런타임 환경변수(또는 플랫폼 시크릿)로만 주입.
> 2. **재현성**: 의존성 핀 + 동일 절차로 어느 환경에서나 동일하게 기동.
> 3. **안정성**: 동시성 상한·입력 길이 캡·헬스체크·임베딩 한도 대비를 항상 설정.

---

## 0. 사전 점검(공통)

```bash
# 새 환경에서 재현성 확인
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install -e .
pytest -q                                   # 44개 통과(키·네트워크 불필요)
```

배포 전 반드시 결정할 2가지:

| 항목 | 선택지 | 권장 |
|---|---|---|
| **채팅 LLM** | `LLM_API_KEY` + `LLM_MODEL` | 필수(외부 LLM 키) |
| **임베딩(RAG)** | `gemini`(키 필요·무료 일일한도 있음) / `hf`(로컬·키 불필요) | **배포는 `hf` 권장**(임베딩 한도에 따른 장애 차단) |

> `hf` 사용 시: `pip install -e ".[local-embed]"` 후 `EMBEDDING_BACKEND=hf`.
> 인덱스(`data/.chroma`)는 임베딩 백엔드별로 벡터 차원이 다르다. 매니페스트가 임베딩
> 백엔드를 추적하므로 **백엔드를 바꾸면 다음 실행에서 자동 재빌드**된다(별도 명령 불필요).
> 영속 볼륨을 쓰면 전환 직후 첫 요청에서 재빌드 지연이 한 번 생길 수 있다.

운영 환경변수(서비스 안정성·최적화):

| 변수 | 기본 | 의미 |
|---|---|---|
| `HOST` / `PORT` | `127.0.0.1` / `8000` | 외부 공개 시 `HOST=0.0.0.0`, `PORT`는 플랫폼 지정값 |
| `MAX_CONCURRENCY` | `2` | 동시 검증 상한(초과 요청은 429) |
| `MAX_INPUT_CHARS` | `2000` | 과대 입력 차단 |
| `CACHE_SIZE` | `64` | 동일 입력 결과 캐시(비용 절감, `0`=비활성) |
| `CORS_ORIGINS` | (없음) | 프런트를 **다른 도메인**에서 서빙할 때만 콤마로 허용 출처 지정 |

---

## 옵션 A — 단일 서버(VM/온프레미스) 프로덕션 실행

리눅스 VM(클라우드 인스턴스 등)에 직접 띄우는 방법.

1. **코드·환경 준비**
   ```bash
   git clone <YOUR-REPO-URL> && cd examiner-agent
   python3.12 -m venv .venv && source .venv/bin/activate
   pip install -r requirements.txt && pip install -e ".[local-embed]"
   ```
2. **시크릿 주입**(`.env` — Git 에 올리지 않음)
   ```bash
   cp .env.example .env
   #  LLM_API_KEY=...     LLM_MODEL=...     EMBEDDING_BACKEND=hf
   ```
3. **인덱스 사전 빌드**(첫 요청 지연·임베딩 버스트 방지)
   ```bash
   python -m factchecker.rag.ingest --force
   ```
4. **프로덕션 기동**(uvicorn 내장 멀티워커 — 추가 의존성 불필요)
   ```bash
   MAX_CONCURRENCY=2 uvicorn server:app --host 0.0.0.0 --port 8000 \
     --workers 2 --timeout-keep-alive 75
   #  또는 가장 간단히:  HOST=0.0.0.0 python server.py
   ```
   > 검증 1건은 외부 LLM 다회 호출로 수십 초가 걸린다. 워커 수는 1~2 로 작게 시작하고,
   > 동시성은 `MAX_CONCURRENCY` 로 제한해 한도 소진·지연을 막는다.
5. **HTTPS·도메인**: 앞단에 리버스 프록시(Nginx/Caddy)를 두고 TLS 종단.
   Caddy 예: `your.domain { reverse_proxy 127.0.0.1:8000 }` (자동 HTTPS).
6. **상시 실행**: `systemd` 서비스 또는 `tmux`/`pm2` 로 데몬화. 방화벽에서 80/443만 개방.

> 워커 수(`-w`)는 CPU·메모리와 임베딩/LLM 한도를 고려해 작게 시작(1~2). 각 검증은
> 수십 초·외부 호출 비용이 있으므로 과도한 동시성은 한도 소진·지연을 유발한다.

---

## 옵션 B — Docker (권장: 어디서나 동일 재현)

저장소에 `Dockerfile`·`.dockerignore` 포함. **키는 이미지에 넣지 않고 런타임 주입.**

기본 이미지는 **로컬 임베딩(hf) 자체완결형**이다(`[local-embed]` 설치 + `EMBEDDING_BACKEND=hf`).
임베딩 키·한도에 의존하지 않아 가장 안정적이며, 런타임엔 **채팅 LLM 키만** 필요하다.

1. **이미지 빌드**
   ```bash
   docker build -t factchecker:latest .
   # gemini 임베딩 경량 이미지(선택): --build-arg EMBED_EXTRA="" 로 빌드
   ```
2. **실행**(`.env` 를 런타임에 주입 — 이미지엔 미포함)
   ```bash
   docker run --rm -p 8000:8000 \
     -e LLM_API_KEY=... -e LLM_MODEL=... -e MAX_CONCURRENCY=2 \
     factchecker:latest
   #  → http://localhost:8000 , 헬스체크 /health
   #  (.env 파일을 쓰려면 --env-file .env)
   ```
3. **인덱스 영속화**(재시작마다 재빌드 방지):
   ```bash
   docker run --rm -p 8000:8000 -e LLM_API_KEY=... -e LLM_MODEL=... \
     -v factchecker_chroma:/app/data/.chroma factchecker:latest
   ```
   > 임베딩 백엔드를 바꾸면(예: hf→gemini) 매니페스트가 이를 감지해 볼륨 인덱스를 **자동
   > 재빌드**한다(첫 요청 1회 지연). gemini 로 운영하려면 `-e EMBEDDING_BACKEND=gemini
   > -e GOOGLE_API_KEY=...` 를 주입하고, 이미지에 google 임베딩 의존성이 포함되어 있어야 한다.
4. 첫 요청은 인덱스 빌드(및 hf 모델 다운로드)로 지연될 수 있다 — `/health` 로 기동을 확인하고,
   인덱스를 영속화하면 이후 재시작은 빠르다.

---

## 옵션 C — Render 상시 배포 (Blueprint + BYOK 사용자 키) ⭐

저장소에 **`render.yaml`(Blueprint)** 가 포함되어 있어 거의 자동으로 배포된다.
**BYOK** 구성: 서버에는 채팅 LLM 키를 두지 않고, **서비스를 쓰는 사람이 화면에서
자기 API 키를 입력**한다 → 소유자 키로 비용이 발생하지 않는다.

1. GitHub 에 저장소 push (커밋은 직접). **`.env` 는 절대 커밋하지 않는다**(gitignore 확인).
2. Render → **New → Blueprint** → 이 저장소 선택 → `render.yaml` 자동 인식.
3. 배포 프롬프트에서 **대시보드 입력값**(blueprint 의 `sync:false`)을 채운다:
   - `LLM_MODEL` — 사용할 모델 ID(산출물엔 비노출, 여기서만 입력).
   - `GOOGLE_API_KEY` — **임베딩(RAG)용 무료 키**(채팅 키 아님). 무료 등급 메모리에 맞춰
     기본 임베딩은 Gemini 다.
   - *(주의) `LLM_API_KEY` 는 입력하지 않는다 — 각 사용자가 화면에서 입력한다.*
4. **Apply/Deploy** → 빌드 시 인덱스가 미리 빌드되고(`ingest`), `/health` 통과 후 기동.
5. 발급된 `https://<app>.onrender.com` 접속 → 화면 상단 **"외부 LLM API 키"** 란에
   사용자가 자기 키를 입력하고 검증. (키는 세션 보관·서버 미저장·HTTPS 전송.)

> **임베딩 선택(메모리 트레이드오프):** 무료 등급(512MB)에서는 로컬(hf) 임베딩의
> torch 가 메모리를 초과할 수 있어 **Gemini 임베딩이 기본**이다(`render.yaml`).
> 키·할당량 의존을 없애려면 `EMBEDDING_BACKEND=hf` + `pip install -e ".[local-embed]"`
> 로 바꾸되 **메모리 여유(≥1GB, 유료 등급)** 를 확보한다. 백엔드를 바꿔도 인덱스는 자동 재빌드.

> **채점자 안내:** BYOK 인스턴스는 **사용자가 서버 설정 모델의 제공자 키**를 입력해야
> 동작한다. 제출 시 "어느 제공자의 키가 필요한지"를 함께 안내하라(산출물엔 모델명 비노출).
> 동일 패턴이 Railway/Fly.io 에도 적용된다(저장소 연결→빌드/스타트→시크릿→배포).

---

## 옵션 D — 빠른 임시 공개(데모/채점용 터널)

서버를 PaaS 에 올리지 않고 로컬 실행분을 잠깐 외부 공개.

```bash
# 로컬 서버 실행
HOST=127.0.0.1 python server.py
# 다른 터미널에서 터널(둘 중 하나)
cloudflared tunnel --url http://localhost:8000     # 무료, 임시 https 주소 발급
#  또는
ngrok http 8000
```
발급된 공개 URL 을 제출/공유한다. **임시 데모용**이며 로컬 PC 가 켜져 있어야 동작한다.

---

## 운영 체크리스트(재현성·안정성)

- [ ] **키 비노출**: `.env`·키가 Git/이미지/로그에 없는지 확인(`.gitignore`·`.dockerignore` 적용).
- [ ] **헬스체크**: `curl https://<host>/health` → `{"status":"ok"}`.
- [ ] **스모크 테스트**: `curl -X POST https://<host>/api/factcheck -H 'Content-Type: application/json' -d '{"text":"산업화 이후 지구 평균 기온이 상승해 왔다."}'` → 등급/근거/반론 JSON.
- [ ] **임베딩 한도**: `gemini` 무료 한도 소진 가능성 → 배포는 `hf` 권장. 한도 도달 시 그래프는
      증거 없이 "불충분"으로 안전 degrade(장애 대신 보수적 판정).
- [ ] **동시성·입력 캡**: `MAX_CONCURRENCY`·`MAX_INPUT_CHARS` 로 폭주/남용 방지.
- [ ] **비용 상한**: 검증 1건 = 외부 LLM 다회 호출. 공개 배포 시 캐시·동시성·(필요시) 인증/레이트리밋 고려.
- [ ] **인덱스 영속화**: 재시작마다 재임베딩하지 않도록 볼륨/사전 빌드 사용.
- [ ] **재현성**: 동일 `requirements.txt`·이미지 태그로 빌드 → 위 스모크 테스트 통과.

---

## 트러블슈팅

| 증상 | 원인 | 조치 |
|---|---|---|
| 결과가 자꾸 "불충분(0.0)" | 임베딩 한도(429)로 증거 회수 실패 | `EMBEDDING_BACKEND=hf` 로 전환(인덱스는 자동 재빌드) |
| 첫 요청이 매우 느림 | 인덱스 빌드/모델 다운로드 | 사전 `ingest`·인덱스 볼륨 영속화 |
| 백엔드 전환 후 첫 요청 지연 | 매니페스트가 임베딩 변경을 감지해 인덱스 자동 재빌드 | 정상 동작(1회). 이후 로드는 빠름 |
| 429 "다른 검증 처리 중" | 동시성 상한 도달 | 정상 보호 동작. 필요 시 `MAX_CONCURRENCY` 상향 |
| `EMBEDDING_BACKEND=hf` 인데 모듈 오류 | 로컬 임베딩 미설치 | `pip install -e ".[local-embed]"` |
| 브라우저 CORS 오류 | 프런트를 다른 도메인에서 서빙 | `CORS_ORIGINS` 에 해당 출처 추가(같은 서버가 HTML 도 서빙하면 불필요) |
| 키 관련 오류 메시지 | 환경변수 미설정 | 플랫폼 시크릿/`.env` 에 `LLM_API_KEY`·`LLM_MODEL` 설정 |
