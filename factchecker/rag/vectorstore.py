"""Chroma 벡터스토어 빌드/로드(멱등). 소스 JSON 해시가 같으면 로드, 다르면 재빌드한다."""

import hashlib
import json
import logging
import threading
from functools import lru_cache
from pathlib import Path
from typing import Any

from ..config import get_settings

logger = logging.getLogger("factchecker.rag.vectorstore")

# 로드/빌드 직렬화 락(병렬 분기의 chromadb 클라이언트 동시 생성 경합 방지)
_BUILD_LOCK = threading.RLock()

EVIDENCE_COLLECTION = "evidence"
TECHNIQUE_COLLECTION = "techniques"
_MANIFEST_NAME = "manifest.json"
_EMBED_CHUNK = 6  # 인덱스 빌드 시 임베딩 청크 크기(무료 등급 RPM 대응)
_RL_MARKERS = ("429", "resource exhausted", "resourceexhausted", "quota",
               "rate limit", "rate_limit", "too many requests", "overloaded")


def _is_rate_limit_text(s: str) -> bool:
    s = s.lower()
    return any(m in s for m in _RL_MARKERS)


@lru_cache(maxsize=4)
def _get_client(persist: str):
    # 경로별로 PersistentClient 하나만 공유
    import chromadb

    Path(persist).mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=persist)


def _content_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _embedding_id(settings) -> str:
    # 임베딩 모델+거리 메트릭 식별자(모델 변경 시 매니페스트로 재빌드 유도)
    return f"gemini:{settings.gemini_embedding_model}|cosine"


def _read_manifest(chroma_dir: Path) -> dict[str, str]:
    mpath = chroma_dir / _MANIFEST_NAME
    if mpath.exists():
        try:
            return json.loads(mpath.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def _write_manifest(chroma_dir: Path, manifest: dict[str, str]) -> None:
    chroma_dir.mkdir(parents=True, exist_ok=True)
    (chroma_dir / _MANIFEST_NAME).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _load_json(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"데이터 파일을 찾을 수 없습니다: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _evidence_documents(rows: list[dict[str, Any]]):
    from langchain_core.documents import Document

    docs, ids = [], []
    for row in rows:
        meta = {
            "snippet_id": row["id"],
            "source": row.get("source", ""),
            "source_type": row.get("source_type", "출처불명"),
            "url": row.get("url") or "",
            "date": row.get("date") or "",
        }
        docs.append(Document(page_content=row["text"], metadata=meta))
        ids.append(row["id"])
    return docs, ids


def _technique_documents(rows: list[dict[str, Any]]):
    from langchain_core.documents import Document

    docs, ids = [], []
    for row in rows:
        cues = " / ".join(row.get("식별_단서", []))
        examples = " / ".join(row.get("예시", []))
        content = (
            f"[기법] {row['tag']}\n"
            f"[정의] {row.get('정의', '')}\n"
            f"[식별 단서] {cues}\n"
            f"[예시] {examples}"
        )
        meta = {"library_entry_id": row["id"], "tag": row["tag"]}
        docs.append(Document(page_content=content, metadata=meta))
        ids.append(row["id"])
    return docs, ids


def _build_collection(
    *, collection: str, source_path: Path, docs, ids, chroma_dir: Path, embeddings,
    reset: bool = False,
):
    """인덱스를 안정 id로 upsert 빌드. reset=True(벡터 차원 변경 또는 force)면 컬렉션 선삭제."""
    import time

    from langchain_chroma import Chroma

    client = _get_client(str(chroma_dir))
    if reset:
        try:
            client.delete_collection(collection)
            logger.info("컬렉션 '%s' 초기화(차원 변경 또는 force)", collection)
        except Exception as exc:
            logger.debug("컬렉션 삭제 생략(%s): %s", collection, exc)
    store = Chroma(
        collection_name=collection, embedding_function=embeddings, client=client,
        collection_metadata={"hnsw:space": "cosine"},  # 관련성 점수 [0,1] 포터블
    )

    # 소량 청크 + 백오프 추가(무료 등급 임베딩 RPM 대응, 청크별 upsert로 자가 복구)
    chunk = _EMBED_CHUNK
    added = 0
    for i in range(0, len(docs), chunk):
        cdocs, cids = docs[i:i + chunk], ids[i:i + chunk]
        for attempt in range(6):
            try:
                store.add_documents(cdocs, ids=cids)
                added += len(cdocs)
                break
            except Exception as exc:
                if _is_rate_limit_text(str(exc)) and attempt < 5:
                    wait = min(60, 5 * (2 ** attempt))
                    logger.warning(
                        "임베딩 레이트리밋, %ds 후 재시도(%d/%d, %d/%d개)",
                        wait, attempt + 1, 6, added, len(docs),
                    )
                    time.sleep(wait)
                    continue
                raise
    return store


def _load_collection(*, collection: str, chroma_dir: Path, embeddings):
    from langchain_chroma import Chroma

    return Chroma(
        collection_name=collection,
        embedding_function=embeddings,
        client=_get_client(str(chroma_dir)),
    )


def _get_or_build(**kwargs):
    with _BUILD_LOCK:  # 병렬 분기 chromadb 경합 방지
        return _get_or_build_impl(**kwargs)


def _get_or_build_impl(
    *, collection: str, source_path: Path, doc_fn, embeddings=None, force: bool = False
):
    settings = get_settings()
    chroma_dir = settings.chroma_dir
    if embeddings is None:
        from ..llm import get_embeddings

        embeddings = get_embeddings()

    rows = _load_json(source_path)
    expected = len(rows)
    source_hash = _content_hash(source_path)
    embed_id = _embedding_id(settings)
    want = f"{source_hash}|{embed_id}"  # 소스 해시 + 임베딩 백엔드/모델

    manifest = _read_manifest(chroma_dir)
    cached = manifest.get(collection)
    cached_embed = cached.split("|", 1)[1] if cached and "|" in cached else None
    # 백엔드/모델 변경 또는 구 매니페스트 형식이면 컬렉션 재생성(차원 고정 Chroma 대응)
    backend_changed = cached is not None and cached_embed != embed_id
    must_reset = bool(force or backend_changed)

    if not force and cached == want:
        try:
            store = _load_collection(
                collection=collection, chroma_dir=chroma_dir, embeddings=embeddings
            )
            count = store._collection.count()  # noqa: SLF001
            if count == expected:  # 부분/빈 인덱스는 거부하고 재빌드
                return store
            logger.warning(
                "컬렉션 '%s' 개수 불일치(%d != %d), 재빌드", collection, count, expected
            )
        except Exception as exc:
            logger.warning("캐시 로드 실패(%s), 재빌드: %s", collection, exc)

    if backend_changed:
        logger.info(
            "컬렉션 '%s' 임베딩 변경 감지(%s -> %s), 재빌드", collection, cached_embed, embed_id
        )

    docs, ids = doc_fn(rows)
    store = _build_collection(
        collection=collection,
        source_path=source_path,
        docs=docs,
        ids=ids,
        chroma_dir=chroma_dir,
        embeddings=embeddings,
        reset=must_reset,
    )
    manifest[collection] = want
    _write_manifest(chroma_dir, manifest)
    return store


def get_or_build_evidence(embeddings=None, force: bool = False):
    settings = get_settings()
    return _get_or_build(
        collection=EVIDENCE_COLLECTION,
        source_path=settings.evidence_corpus_path,
        doc_fn=_evidence_documents,
        embeddings=embeddings,
        force=force,
    )


def get_or_build_techniques(embeddings=None, force: bool = False):
    settings = get_settings()
    return _get_or_build(
        collection=TECHNIQUE_COLLECTION,
        source_path=settings.technique_library_path,
        doc_fn=_technique_documents,
        embeddings=embeddings,
        force=force,
    )
