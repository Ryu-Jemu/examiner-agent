"""CLI 진입 로직 (콘솔 스크립트 `factcheck` 및 루트 cli.py 가 호출)."""

from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="단톡방 루머 적대적 팩트체커 (CLI)",
    )
    parser.add_argument("text", nargs="*", help="검증할 루머/메시지 텍스트")
    parser.add_argument(
        "--stdin", action="store_true", help="표준 입력에서 텍스트를 읽음"
    )
    args = parser.parse_args(argv)

    if args.stdin:
        text = sys.stdin.read().strip()
    else:
        text = " ".join(args.text).strip()

    if not text:
        parser.print_help()
        print("\n[안내] 검증할 텍스트를 입력하세요. 예) python cli.py \"백신 맞으면 자석 붙는대요\"")
        return 2

    # 무거운 임포트는 여기서(설정 검증이 먼저 깔끔히 실패하도록)
    from .config import ConfigError

    try:
        from .report_format import report_to_text
        from .runner import run_factcheck

        report = run_factcheck(text)
    except ConfigError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    print(report_to_text(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
