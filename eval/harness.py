"""평가 하니스.

  python -m eval.harness            # 테스트셋 1회 실행 + 지표 표
  python -m eval.harness --runs 2   # 2회 실행해 라벨 안정성(결정론) 점검

재현성을 위해 SEARCH_BACKEND=local 을 강제하고 temperature 는 설정 기본값(0.0)을 쓴다.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# `pip install -e .` 없이도 동작하도록 저장소 루트를 경로에 추가
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from eval.metrics import aggregate, format_report, score_case  # noqa: E402


def _force_local_backend() -> None:
    """평가는 항상 로컬 코퍼스만 사용(웹 변동성 배제)."""
    os.environ["SEARCH_BACKEND"] = "local"
    from factchecker.config import reset_settings

    reset_settings()


def load_testset() -> list[dict]:
    from factchecker.config import get_settings

    path = get_settings().testset_path
    return json.loads(Path(path).read_text(encoding="utf-8"))


def run_once(testset: list[dict]):
    from factchecker.runner import run_factcheck

    cases = []
    for meta in testset:
        report = run_factcheck(meta["input_text"])
        pred_label = report.overall_grade.value
        confidence = report.overall_confidence
        pred_tech = {t.tag.value for t in report.technique_tags}
        cases.append(score_case(meta, pred_label, confidence, pred_tech))
    return cases


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="팩트체커 평가 하니스")
    parser.add_argument("--runs", type=int, default=1, help="반복 실행 횟수(결정론 점검)")
    args = parser.parse_args(argv)

    from factchecker.config import ConfigError

    try:
        _force_local_backend()
        testset = load_testset()
    except ConfigError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    all_runs = []
    for r in range(args.runs):
        print(f"\n>>> 실행 {r + 1}/{args.runs} (케이스 {len(testset)}개) …")
        try:
            cases = run_once(testset)
        except ConfigError as exc:
            print(str(exc), file=sys.stderr)
            return 1
        rep = aggregate(cases)
        print(format_report(rep))
        all_runs.append({c.case_id: c.pred_label for c in cases})

    # 결정론(라벨 안정성) 점검
    if args.runs > 1:
        base = all_runs[0]
        flips = []
        for other in all_runs[1:]:
            for cid, label in base.items():
                if other.get(cid) != label:
                    flips.append((cid, label, other.get(cid)))
        if flips:
            print("\n[경고] 실행 간 라벨이 바뀐 케이스:")
            for cid, a, b in flips:
                print(f"  - {cid}: {a} → {b}")
        else:
            print("\n[확인] 모든 실행에서 판정 라벨이 동일합니다(결정론적).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
