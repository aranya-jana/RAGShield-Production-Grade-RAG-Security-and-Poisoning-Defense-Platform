"""
Tests for the security audit logger.
"""

import json

from src.audit_logger import SecurityAuditLogger


# =====================================================================
# CREATE LOGGER
# =====================================================================


def test_audit_logger_creates_log_directory(tmp_path):
    log_path = (
        tmp_path
        / "logs"
        / "security_audit.jsonl"
    )

    logger = SecurityAuditLogger(
        str(log_path)
    )

    assert log_path.parent.exists()
    assert logger.get_events() == []


# =====================================================================
# WRITE EVENT
# =====================================================================


def test_security_event_is_written(tmp_path):
    log_path = (
        tmp_path
        / "security_audit.jsonl"
    )

    logger = SecurityAuditLogger(
        str(log_path)
    )

    event = logger.log_event(
        event_type="document_blocked",
        query="What is cloud computing?",
        source="malicious.md",
        document_type="poisoned",
        detector="PoisonDetector",
        score=1.0,
        status="BLOCKED",
        reasons=[
            "Attempt to ignore previous instructions"
        ],
    )

    assert log_path.exists()

    assert event["event_type"] == "document_blocked"
    assert event["source"] == "malicious.md"
    assert event["detector"] == "PoisonDetector"
    assert event["score"] == 1.0
    assert event["status"] == "BLOCKED"


# =====================================================================
# READ EVENT
# =====================================================================


def test_security_event_can_be_read(tmp_path):
    log_path = (
        tmp_path
        / "security_audit.jsonl"
    )

    logger = SecurityAuditLogger(
        str(log_path)
    )

    logger.log_event(
        event_type="document_blocked",
        source="attack.md",
        detector="PoisonDetector",
        score=1.0,
        status="BLOCKED",
    )

    events = logger.get_events()

    assert len(events) == 1

    assert events[0]["source"] == "attack.md"

    assert (
        events[0]["detector"]
        == "PoisonDetector"
    )


# =====================================================================
# MULTIPLE EVENTS
# =====================================================================


def test_multiple_events_are_stored(tmp_path):
    log_path = (
        tmp_path
        / "security_audit.jsonl"
    )

    logger = SecurityAuditLogger(
        str(log_path)
    )

    logger.log_event(
        event_type="document_safe",
        source="trusted.md",
        detector="None",
        score=0.0,
        status="SAFE",
    )

    logger.log_event(
        event_type="document_blocked",
        source="attack.md",
        detector="PoisonDetector",
        score=1.0,
        status="BLOCKED",
    )

    events = logger.get_events()

    assert len(events) == 2

    # Newest event should be first.
    assert (
        events[0]["source"]
        == "attack.md"
    )

    assert (
        events[1]["source"]
        == "trusted.md"
    )


# =====================================================================
# LIMIT
# =====================================================================


def test_event_limit_is_respected(tmp_path):
    log_path = (
        tmp_path
        / "security_audit.jsonl"
    )

    logger = SecurityAuditLogger(
        str(log_path)
    )

    for index in range(5):

        logger.log_event(
            event_type="test",
            source=f"document_{index}.md",
        )

    events = logger.get_events(
        limit=2
    )

    assert len(events) == 2

    assert (
        events[0]["source"]
        == "document_4.md"
    )

    assert (
        events[1]["source"]
        == "document_3.md"
    )


# =====================================================================
# COUNT
# =====================================================================


def test_event_count(tmp_path):
    log_path = (
        tmp_path
        / "security_audit.jsonl"
    )

    logger = SecurityAuditLogger(
        str(log_path)
    )

    assert logger.count() == 0

    logger.log_event(
        event_type="test"
    )

    logger.log_event(
        event_type="test"
    )

    assert logger.count() == 2


# =====================================================================
# CLEAR
# =====================================================================


def test_clear_removes_audit_log(tmp_path):
    log_path = (
        tmp_path
        / "security_audit.jsonl"
    )

    logger = SecurityAuditLogger(
        str(log_path)
    )

    logger.log_event(
        event_type="test"
    )

    assert log_path.exists()

    logger.clear()

    assert not log_path.exists()

    assert logger.get_events() == []


# =====================================================================
# JSONL FORMAT
# =====================================================================


def test_audit_file_uses_json_lines_format(tmp_path):
    log_path = (
        tmp_path
        / "security_audit.jsonl"
    )

    logger = SecurityAuditLogger(
        str(log_path)
    )

    logger.log_event(
        event_type="document_blocked",
        source="attack.md",
        score=1.0,
        status="BLOCKED",
    )

    logger.log_event(
        event_type="document_safe",
        source="trusted.md",
        score=0.0,
        status="SAFE",
    )

    lines = log_path.read_text(
        encoding="utf-8"
    ).strip().splitlines()

    assert len(lines) == 2

    first = json.loads(lines[0])
    second = json.loads(lines[1])

    assert (
        first["event_type"]
        == "document_blocked"
    )

    assert (
        second["event_type"]
        == "document_safe"
    )


# =====================================================================
# EMPTY LIMIT
# =====================================================================


def test_zero_limit_returns_empty_list(tmp_path):
    log_path = (
        tmp_path
        / "security_audit.jsonl"
    )

    logger = SecurityAuditLogger(
        str(log_path)
    )

    logger.log_event(
        event_type="test"
    )

    assert logger.get_events(
        limit=0
    ) == []


def test_negative_limit_returns_empty_list(tmp_path):
    log_path = (
        tmp_path
        / "security_audit.jsonl"
    )

    logger = SecurityAuditLogger(
        str(log_path)
    )

    logger.log_event(
        event_type="test"
    )

    assert logger.get_events(
        limit=-1
    ) == []