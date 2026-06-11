"""extract_claims — 텍스트/이미지 입력 분기와 claim_id 재정렬."""

import importlib

from factchecker.models import Claim, ClaimList, ClaimType

ec = importlib.import_module("factchecker.nodes.extract_claims")


def _fake(claims):
    def fake_invoke(payload, schema, *, default):
        fake_invoke.payload = payload
        return ClaimList(claims=claims)

    return fake_invoke


def test_text_path_uses_plain_prompt(monkeypatch):
    fake = _fake([
        Claim(claim_id=7, text="주장", claim_type=ClaimType.FACT,
              checkable=True),
    ])
    monkeypatch.setattr(ec, "structured_invoke", fake)
    update = ec.extract_claims({"input_text": "루머 본문"})
    assert isinstance(fake.payload, str)  # 기존 텍스트 경로 그대로
    assert "루머 본문" in fake.payload
    assert update["claims"][0].claim_id == 0  # 재정렬


def test_image_path_builds_multimodal_message(monkeypatch):
    fake = _fake([])
    monkeypatch.setattr(ec, "structured_invoke", fake)
    data_url = "data:image/jpeg;base64,AAAA"
    ec.extract_claims({"input_text": "", "input_image": data_url})
    [msg] = fake.payload  # HumanMessage 리스트
    kinds = [b["type"] for b in msg.content]
    assert kinds == ["text", "image_url"]
    assert msg.content[1]["image_url"]["url"] == data_url
    assert "첨부 이미지" in msg.content[0]["text"]


def test_empty_input_returns_no_claims(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("빈 입력에는 LLM 을 호출하면 안 된다")

    monkeypatch.setattr(ec, "structured_invoke", boom)
    update = ec.extract_claims({"input_text": "  "})
    assert update["claims"] == []
