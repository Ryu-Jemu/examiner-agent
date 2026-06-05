"""평가 지표 계산.

- 판정 정확도: 5등급 정확 일치 + 관대(±1등급). "불충분" 정답도 정답으로 인정.
- 기법 태깅: 4기법에 대한 micro Precision/Recall/F1.
- 신뢰도 보정: 자신 있게 틀린 비율(오답인데 confidence>=0.7) + 경량 ECE.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# 진실성 축 순서(±1 관대 정확도 계산용)
GRADE_ORDER = [
    "사실",
    "대체로 사실",
    "불충분(판단 불가)",
    "대체로 거짓",
    "거짓·오도",
]
_GRADE_IDX = {g: i for i, g in enumerate(GRADE_ORDER)}


@dataclass
class CaseScore:
    case_id: str
    gold_label: str
    pred_label: str
    confidence: float
    exact: bool
    lenient: bool
    gold_techniques: set[str]
    pred_techniques: set[str]
    tech_tp: int = 0
    tech_fp: int = 0
    tech_fn: int = 0


@dataclass
class EvalReport:
    cases: list[CaseScore] = field(default_factory=list)

    # 판정
    n: int = 0
    exact_acc: float = 0.0
    lenient_acc: float = 0.0
    insufficient_correct: int = 0

    # 기법
    tech_precision: float = 0.0
    tech_recall: float = 0.0
    tech_f1: float = 0.0

    # 보정
    overconfident_wrong_rate: float = 0.0
    ece: float = 0.0


def score_case(case_meta: dict, pred_label: str, confidence: float,
               pred_techniques: set[str]) -> CaseScore:
    gold = case_meta["gold_label"]
    gold_tech = set(case_meta.get("gold_techniques", []))

    exact = pred_label == gold
    gi = _GRADE_IDX.get(gold)
    pi = _GRADE_IDX.get(pred_label)
    lenient = exact or (gi is not None and pi is not None and abs(gi - pi) <= 1)

    tp = len(gold_tech & pred_techniques)
    fp = len(pred_techniques - gold_tech)
    fn = len(gold_tech - pred_techniques)

    return CaseScore(
        case_id=case_meta.get("id", "?"),
        gold_label=gold,
        pred_label=pred_label,
        confidence=confidence,
        exact=exact,
        lenient=lenient,
        gold_techniques=gold_tech,
        pred_techniques=pred_techniques,
        tech_tp=tp,
        tech_fp=fp,
        tech_fn=fn,
    )


def _ece(cases: list[CaseScore], n_bins: int = 5) -> float:
    """경량 Expected Calibration Error (정확도=exact 기준)."""
    if not cases:
        return 0.0
    bins = [[] for _ in range(n_bins)]
    for c in cases:
        idx = min(int(c.confidence * n_bins), n_bins - 1)
        bins[idx].append(c)
    total = len(cases)
    ece = 0.0
    for b in bins:
        if not b:
            continue
        avg_conf = sum(c.confidence for c in b) / len(b)
        acc = sum(1 for c in b if c.exact) / len(b)
        ece += (len(b) / total) * abs(avg_conf - acc)
    return ece


def aggregate(cases: list[CaseScore]) -> EvalReport:
    rep = EvalReport(cases=cases)
    rep.n = len(cases)
    if rep.n == 0:
        return rep

    rep.exact_acc = sum(1 for c in cases if c.exact) / rep.n
    rep.lenient_acc = sum(1 for c in cases if c.lenient) / rep.n
    rep.insufficient_correct = sum(
        1 for c in cases if c.gold_label == "불충분(판단 불가)" and c.exact
    )

    tp = sum(c.tech_tp for c in cases)
    fp = sum(c.tech_fp for c in cases)
    fn = sum(c.tech_fn for c in cases)
    rep.tech_precision = tp / (tp + fp) if (tp + fp) else 0.0
    rep.tech_recall = tp / (tp + fn) if (tp + fn) else 0.0
    denom = rep.tech_precision + rep.tech_recall
    rep.tech_f1 = (2 * rep.tech_precision * rep.tech_recall / denom) if denom else 0.0

    wrong = [c for c in cases if not c.exact]
    overconf = [c for c in wrong if c.confidence >= 0.7]
    rep.overconfident_wrong_rate = len(overconf) / rep.n
    rep.ece = _ece(cases)
    return rep


def format_report(rep: EvalReport) -> str:
    lines = []
    lines.append("=" * 64)
    lines.append("평가 결과 (테스트셋)")
    lines.append("=" * 64)
    lines.append(f"케이스 수: {rep.n}")
    lines.append("")
    lines.append("[판정 정확도]")
    lines.append(f"  정확 일치(5등급): {rep.exact_acc:.1%}")
    lines.append(f"  관대(±1등급)    : {rep.lenient_acc:.1%}")
    lines.append(f"  '불충분' 정답 맞춘 수: {rep.insufficient_correct}")
    lines.append("")
    lines.append("[기법 태깅 (micro)]")
    lines.append(f"  Precision: {rep.tech_precision:.1%}")
    lines.append(f"  Recall   : {rep.tech_recall:.1%}")
    lines.append(f"  F1       : {rep.tech_f1:.1%}")
    lines.append("")
    lines.append("[신뢰도 보정]")
    lines.append(f"  자신 있게 틀린 비율(conf≥0.7 & 오답): {rep.overconfident_wrong_rate:.1%}")
    lines.append(f"  ECE: {rep.ece:.3f}")
    lines.append("")
    lines.append("[케이스별]")
    lines.append(f"  {'id':<24}{'gold':<16}{'pred':<16}{'conf':<6}{'judge'}")
    for c in rep.cases:
        mark = "✓" if c.exact else ("~" if c.lenient else "✗")
        lines.append(
            f"  {c.case_id:<24}{c.gold_label:<16}{c.pred_label:<16}"
            f"{c.confidence:<6.2f}{mark}"
        )
    lines.append("=" * 64)
    return "\n".join(lines)
