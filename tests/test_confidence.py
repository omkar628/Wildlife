from backend.detector.parser import ParsedDetection
from backend.services.confidence import filter_detection


def _det(confidence: float) -> ParsedDetection:
    return ParsedDetection(0, "tiger", confidence, 0, 0, 10, 10)


def test_below_detect_min_is_dropped():
    decision = filter_detection(_det(0.10), auto_accept=0.60, detect_min=0.15)
    assert decision.keep is False
    assert decision.needs_review is False


def test_between_thresholds_goes_to_review():
    decision = filter_detection(_det(0.47), auto_accept=0.60, detect_min=0.15)
    assert decision.keep is True
    assert decision.accepted is False
    assert decision.needs_review is True


def test_at_threshold_is_auto_accepted():
    decision = filter_detection(_det(0.60), auto_accept=0.60, detect_min=0.15)
    assert decision.accepted is True
    assert decision.needs_review is False


def test_thresholds_are_not_hardcoded():
    decision = filter_detection(_det(0.70), auto_accept=0.80, detect_min=0.50)
    assert decision.needs_review is True
    decision = filter_detection(_det(0.70), auto_accept=0.65, detect_min=0.50)
    assert decision.accepted is True
