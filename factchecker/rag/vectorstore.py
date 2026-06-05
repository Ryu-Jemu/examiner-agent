"""Chroma 벡터스토어 빌드/로드 (멱등).

설계 원칙(재현성):
- 인덱스(.chroma/)는 커밋하지 않는다. 소스 JSON만 커밋하고 로컬에서 재빌드한다.
- 소스 JSON의 콘텐츠 해시를 매니페스트에 저장해, 해시가 같으면 로드·다르면 재빌드한다.
- 스니펫 1개 = Document 1개(snippet_id ↔ Document 1:1) → "실제 id만 인용" 가드의 전제.
- 명시적 ids 로 결정론적 upsert, MMR 등 랜덤성 없는 similarity_search 사용.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

from ..config import get_settings

logger = logging.getLogger("factchecker.rag.vectorstore")

EVIDENCE_COLLECTION = "evidence"
TECHNIQUE_COLLECTION = "techniques"
_MANIFEST_NAME = "manifest.json"


def _content_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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
    *, collection: str, source_path: Path, docs, ids, chroma_dir: Path, embeddings
):
    """기존 컬렉션을 비우고 from_documents 로 결정론적으로 재빌드."""
    from langchain_chroma import Chroma

    persist = str(chroma_dir)
    chroma_dir.mkdir(parents=True, exist_ok=True)

    # 기존 컬렉션 제거(베스트 에포트) → 중복/오래된 벡터 방지
    try:
        existing = Chroma(
            collection_name=collection,
            embedding_function=embeddings,
            persist_directory=persist,
        )
        existing.delete_collection()
    except Exception as exc:  # noqa: BLE001
        logger.debug("기존 컬렉션 삭제 생략(%s): %s", collection, exc)

    store = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        ids=ids,
        collection_name=collection,
        persist_directory=persist,
    )
    logger.info("컬렉션 '%s' 빌드 완료: %d개 문서", collection, len(docs))
    return store


def _load_collection(*, collection: str, chroma_dir: Path, embeddings):
    from langchain_chroma import Chroma

    return Chroma(
        collection_name=collection,
        embedding_function=embeddings,
        persist_directory=str(chroma_dir),
    )


def _get_or_build(
    *, collection: str, source_path: Path, doc_fn, embeddings=None, force: bool = False
):
    settings = get_settings()
    chroma_dir = settings.chroma_dir
    if embeddings is None:
        from ..llm import get_embeddings

        embeddings = get_embeddings()

    source_hash = _content_hash(source_path)
    manifest = _read_manifest(chroma_dir)
    cached_hash = manifest.get(collection)

    if not force and cached_hash == source_hash:
        try:
            store = _load_collection(
                collection=collection, chroma_dir=chroma_dir, embeddings=embeddings
            )
            if store._collection.count() > 0:  # noqa: SLF001
                logger.info("컬렉션 '%s' 캐시 로드(해시 일치)", collection)
                return store
        except Exception as exc:  # noqa: BLE001
            logger.warning("캐시 로드 실패(%s) → 재빌드: %s", collection, exc)

    rows = _load_json(source_path)
    docs, ids = doc_fn(rows)
    store = _build_collection(
        collection=collection,
        source_path=source_path,
        docs=docs,
        ids=ids,
        chroma_dir=chroma_dir,
        embeddings=embeddings,
    )
    manifest[collection] = source_hash
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
