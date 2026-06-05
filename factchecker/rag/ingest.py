"""인덱스 빌드 스크립트.

  python -m factchecker.rag.ingest         # 캐시 활용(해시 변경 시에만 재빌드)
  python -m factchecker.rag.ingest --force # 강제 재빌드

그래프 첫 실행 시 lazy 자동 빌드도 되지만, 미리 빌드해 두고 싶을 때 사용한다.
"""

from __future__ import annotations

import argparse
import sys

from ..config import ConfigError
from .vectorstore import (
    EVIDENCE_COLLECTION,
    TECHNIQUE_COLLECTION,
    get_or_build_evidence,
    get_or_build_techniques,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="증거 코퍼스/기법 라이브러리 인덱스 빌드")
    parser.add_argument("--force", action="store_true", help="해시와 무관하게 강제 재빌드")
    args = parser.parse_args(argv)

    try:
        ev = get_or_build_evidence(force=args.force)
        tech = get_or_build_techniques(force=args.force)
    except ConfigError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    ev_count = ev._collection.count()      # noqa: SLF001
    tech_count = tech._collection.count()  # noqa: SLF001
    print(f"[완료] '{EVIDENCE_COLLECTION}' 컬렉션: {ev_count}개 스니펫")
    print(f"[완료] '{TECHNIQUE_COLLECTION}' 컬렉션: {tech_count}개 기법")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
