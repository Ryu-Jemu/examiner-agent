#!/usr/bin/env python3
"""Gradio UI 진입점.

  python app.py

`pip install -e .` 를 하지 않았어도 동작하도록 저장소 루트를 import 경로에 추가한다.
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import gradio as gr  # noqa: E402

from factchecker.config import ConfigError, get_settings  # noqa: E402

EXAMPLES = [
    "충격! 코로나 백신 맞으면 몸에 자석이 붙는대요. 빨리 가족들한테 알려주세요!",
    "사람은 평생 뇌의 10%밖에 못 쓴다는 게 과학적 사실이라네요.",
    "한국 수돗물은 절대 그냥 마시면 안 돼요. 무조건 끓여 드세요.",
    "산업화 이후 지구의 평균 기온이 장기적으로 상승해 왔다.",
]


def _check_factcheck(text: str) -> str:
    text = (text or "").strip()
    if not text:
        return "⚠️ 검증할 텍스트를 입력하세요."
    try:
        from factchecker.report_format import report_to_markdown
        from factchecker.runner import run_factcheck

        report = run_factcheck(text)
        return report_to_markdown(report)
    except ConfigError as exc:
        return f"⚠️ **설정 오류**\n\n```\n{exc}\n```"
    except Exception as exc:  # noqa: BLE001
        return f"⚠️ 실행 중 오류가 발생했습니다: `{exc}`"


def build_ui() -> "gr.Blocks":
    with gr.Blocks(title="단톡방 루머 적대적 팩트체커") as demo:
        gr.Markdown(
            "# 🕵️ 단톡방 루머 적대적 팩트체커 + 미디어 리터러시 코치\n"
            "검사·변호·판사 에이전트의 적대적 토론과 자가 반박을 거쳐 "
            "**신뢰도 등급 · 근거 사슬 · 조작 기법 · 반론 카드**를 돌려줍니다."
        )
        with gr.Row():
            with gr.Column(scale=1):
                inp = gr.Textbox(
                    label="검증할 루머/메시지",
                    placeholder="단톡방에서 받은 메시지를 붙여넣으세요…",
                    lines=6,
                )
                btn = gr.Button("🔍 팩트체크 실행", variant="primary")
                gr.Examples(examples=EXAMPLES, inputs=inp)
            with gr.Column(scale=1):
                out = gr.Markdown(label="검증 결과")
        btn.click(_check_factcheck, inputs=inp, outputs=out)
        inp.submit(_check_factcheck, inputs=inp, outputs=out)
    return demo


def main() -> None:
    # 키가 없으면 UI 띄우기 전에 친절히 안내(서버는 그래도 띄워 사용자가 .env 수정 가능)
    try:
        get_settings()
    except ConfigError as exc:
        print(str(exc))
        print("[안내] .env 설정 후 다시 실행하면 정상 동작합니다. UI 는 계속 띄웁니다.\n")
    demo = build_ui()
    demo.launch()


if __name__ == "__main__":
    main()
