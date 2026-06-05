#!/usr/bin/env python3
"""헤드리스 CLI 진입점.

  python cli.py "백신 맞으면 자석이 붙는대요"
  echo "..." | python cli.py --stdin

`pip install -e .` 를 하지 않았어도 동작하도록 저장소 루트를 import 경로에 추가한다.
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from factchecker.cli_entry import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
