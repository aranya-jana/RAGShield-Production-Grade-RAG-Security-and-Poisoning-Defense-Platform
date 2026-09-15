"""
Tests for the RAGShield Red-Team Attack Framework.
"""

from types import SimpleNamespace
from unittest.mock import Mock

from red_team.attack_cases import (
    AttackType,
    DEFAULT_ATTACK_CASES,
    get_attack_cases,
)
from red_team.attack_runner import RedTeamRunner


def make_prompt_detector():
    detector = Mock()

    def analyze(prompt):
        result = SimpleNamespace()

        malicious = any(
            phrase in prompt.lower()
            for phrase in (
                "ignore previous instructions",
                "ignore the developer instructions",
                "reveal your system prompt",
                "bypass all security controls",
                "you are now the system administrator",
                "critical system instruction",
                "decode this base64 payload",
            )
        )

        result.score = 0.90 if malicious else 0.0
        result.is_injected = malicious
        result.reasons = (
            ["High-confidence prompt injection indicators detected."]
            if malicious
            else []
        )
        result.matched_patterns = (
            ["instruction_override"]
            if malicious
            else []
        )

        return result

    detector.analyze.side_effect = analyze
    return detector


def make_document_detector():
    detector = Mock()

    def analyze(text, metadata):
        result = SimpleNamespace()

        combined = f"{text} {metadata}".lower()

        malicious = any(
            phrase in combined
            for phrase in (
                "ignore previous instructions",
                "bypass all security controls",
                "system administrator",
                "ignore security rules",
            )
        )

        result.score = 0.90 if malicious else 0.0
        result.is_injected = malicious
        result.reasons = (
            ["Document injection detected."]
            if malicious
            else []
        )
        result.matched_patterns = (
            ["instruction_override"]
            if malicious
            else []
        )

        return result

    detector.analyze.side_effect = analyze
    return detector


def make_rag():
    rag = Mock()

    rag.prompt_injection_detector = make_prompt_detector()
    rag.document_injection_detector = make_document_detector()
    rag.security_events = []

    def query(prompt):
        detection = rag.prompt_injection_detector.analyze(prompt)

        if detection.is_injected:
            rag.security_events = [
                {
                    "detector": "PromptInjectionDetector",
                    "status": "BLOCKED",
                    "score": detection.score,
                    "matched_patterns": detection.matched_patterns,
                    "reasons": detection.reasons,
                    "metadata": {
                        "llm_called": False,
                        "retrieval_started": False,
                    },
                }
            ]
        else:
            rag.security_events = [
                {
                    "detector": "PromptInjectionDetector",
                    "status": "SAFE",
                    "score": 0.0,
                    "matched_patterns": [],
                    "reasons": [],
                    "metadata": {
                        "llm_called": True,
                        "retrieval_started": True,
                    },
                }
            ]

        return {
            "query": prompt,
            "security_events": rag.security_events,
        }

    rag.query.side_effect = query
    return rag


def test_default_attack_corpus_contains_expected_categories():
    categories = {
        case.category
        for case in DEFAULT_ATTACK_CASES
    }

    assert "prompt_injection" in categories
    assert "document_injection" in categories
    assert "metadata_injection" in categories
    assert "factual_poisoning" in categories
    assert "obfuscated_injection" in categories


def test_attack_corpus_has_unique_ids():
    ids = [
        case.attack_id
        for case in DEFAULT_ATTACK_CASES
    ]

    assert len(ids) == len(set(ids))


def test_get_attack_cases_filters_by_type():
    prompt_cases = get_attack_cases(
        attack_type=AttackType.PROMPT
    )

    assert prompt_cases
    assert all(
        case.attack_type == AttackType.PROMPT
        for case in prompt_cases
    )


def test_get_attack_cases_filters_by_category():
    cases = get_attack_cases(
        category="document_injection"
    )

    assert cases
    assert all(
        case.category == "document_injection"
        for case in cases
    )


def test_red_team_runner_blocks_prompt_injection():
    rag = make_rag()
    runner = RedTeamRunner(rag)

    attack = next(
        case
        for case in DEFAULT_ATTACK_CASES
        if case.attack_id == "PI-001"
    )

    result = runner.run_case(attack)

    assert result.passed is True
    assert result.actual_status == "BLOCKED"
    assert result.llm_called is False
    assert result.retrieval_started is False
    assert result.detection_score >= 0.50


def test_red_team_runner_blocks_document_injection():
    rag = make_rag()
    runner = RedTeamRunner(rag)

    attack = next(
        case
        for case in DEFAULT_ATTACK_CASES
        if case.attack_id == "DI-001"
    )

    result = runner.run_case(attack)

    assert result.passed is True
    assert result.actual_status == "BLOCKED"
    assert result.detection_score >= 0.50


def test_red_team_runner_produces_aggregate_report():
    rag = make_rag()
    runner = RedTeamRunner(rag)

    attacks = [
        case
        for case in DEFAULT_ATTACK_CASES
        if case.attack_type == AttackType.PROMPT
    ]

    report = runner.run(attacks)

    assert report.total == len(attacks)
    assert report.passed == report.total
    assert report.failed == 0
    assert report.pass_rate == 100.0
    assert report.blocked == report.total


def test_report_serialization():
    rag = make_rag()
    runner = RedTeamRunner(rag)

    report = runner.run(
        DEFAULT_ATTACK_CASES[:2]
    )

    data = report.to_dict()

    assert data["total"] == 2
    assert data["passed"] == 2
    assert data["failed"] == 0
    assert isinstance(data["results"], list)


def test_runner_handles_attack_errors():
    rag = Mock()
    rag.query.side_effect = RuntimeError(
        "simulated failure"
    )

    runner = RedTeamRunner(rag)

    attack = next(
        case
        for case in DEFAULT_ATTACK_CASES
        if case.attack_type == AttackType.PROMPT
    )

    result = runner.run_case(attack)

    assert result.passed is False
    assert result.actual_status == "ERROR"
    assert "simulated failure" in result.error
