"""
RAGShield Red-Team Attack Runner.

Executes deterministic adversarial scenarios against a RAGSystem
instance and reports whether each expected security decision occurred.

The runner is intentionally lightweight: it does not require an
external service, database, or network connection.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional
import time

from .attack_cases import AttackCase, AttackType


@dataclass
class AttackResult:
    """Result of one red-team attack."""

    attack_id: str
    name: str
    category: str
    attack_type: str
    expected_status: str
    actual_status: str
    passed: bool
    detection_score: float
    matched_patterns: List[str] = field(default_factory=list)
    reasons: List[str] = field(default_factory=list)
    llm_called: Optional[bool] = None
    retrieval_started: Optional[bool] = None
    latency_ms: float = 0.0
    error: Optional[str] = None


@dataclass
class AttackReport:
    """Aggregate report for a red-team run."""

    results: List[AttackResult] = field(default_factory=list)

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def passed(self) -> int:
        return sum(result.passed for result in self.results)

    @property
    def failed(self) -> int:
        return self.total - self.passed

    @property
    def pass_rate(self) -> float:
        if self.total == 0:
            return 0.0
        return round((self.passed / self.total) * 100.0, 2)

    @property
    def blocked(self) -> int:
        return sum(
            result.actual_status == "BLOCKED"
            for result in self.results
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the report into JSON-friendly data."""

        return {
            "total": self.total,
            "passed": self.passed,
            "failed": self.failed,
            "pass_rate": self.pass_rate,
            "blocked": self.blocked,
            "results": [
                {
                    "attack_id": result.attack_id,
                    "name": result.name,
                    "category": result.category,
                    "attack_type": result.attack_type,
                    "expected_status": result.expected_status,
                    "actual_status": result.actual_status,
                    "passed": result.passed,
                    "detection_score": result.detection_score,
                    "matched_patterns": result.matched_patterns,
                    "reasons": result.reasons,
                    "llm_called": result.llm_called,
                    "retrieval_started": result.retrieval_started,
                    "latency_ms": result.latency_ms,
                    "error": result.error,
                }
                for result in self.results
            ],
        }


class RedTeamRunner:
    """Run deterministic attacks against a RAGSystem instance."""

    def __init__(self, rag_system):
        self.rag_system = rag_system

    def run(
        self,
        attacks: Iterable[AttackCase],
    ) -> AttackReport:
        """Execute all supplied attacks and return an aggregate report."""

        report = AttackReport()

        for attack in attacks:
            report.results.append(
                self.run_case(attack)
            )

        return report

    def run_case(
        self,
        attack: AttackCase,
    ) -> AttackResult:
        """Execute one attack case."""

        started = time.perf_counter()

        try:
            if attack.attack_type == AttackType.PROMPT:
                result = self._run_prompt_attack(attack)
            elif attack.attack_type == AttackType.DOCUMENT:
                result = self._run_document_attack(attack)
            else:
                raise ValueError(
                    f"Unsupported attack type: {attack.attack_type}"
                )

            result.latency_ms = round(
                (time.perf_counter() - started) * 1000.0,
                3,
            )

            return result

        except Exception as exc:
            return AttackResult(
                attack_id=attack.attack_id,
                name=attack.name,
                category=attack.category,
                attack_type=attack.attack_type.value,
                expected_status=attack.expected_status,
                actual_status="ERROR",
                passed=False,
                detection_score=0.0,
                latency_ms=round(
                    (time.perf_counter() - started) * 1000.0,
                    3,
                ),
                error=str(exc),
            )

    def _run_prompt_attack(
        self,
        attack: AttackCase,
    ) -> AttackResult:
        """Run a malicious user query through RAGSystem.query()."""

        self.rag_system.query(attack.payload)

        security_events = list(
            getattr(
                self.rag_system,
                "security_events",
                [],
            )
            or []
        )

        injection_events = [
            event
            for event in security_events
            if event.get("detector")
            == "PromptInjectionDetector"
        ]

        blocked = any(
            event.get("status") == "BLOCKED"
            for event in injection_events
        )

        detected_event = (
            injection_events[0]
            if injection_events
            else {}
        )

        actual_status = (
            "BLOCKED"
            if blocked
            else "ALLOWED"
        )

        llm_called = any(
            bool(
                event.get("metadata", {}).get(
                    "llm_called",
                    False,
                )
            )
            for event in security_events
        )

        retrieval_started = any(
            bool(
                event.get("metadata", {}).get(
                    "retrieval_started",
                    False,
                )
            )
            for event in security_events
        )

        return AttackResult(
            attack_id=attack.attack_id,
            name=attack.name,
            category=attack.category,
            attack_type=attack.attack_type.value,
            expected_status=attack.expected_status,
            actual_status=actual_status,
            passed=actual_status == attack.expected_status,
            detection_score=float(
                detected_event.get("score", 0.0)
            ),
            matched_patterns=list(
                detected_event.get(
                    "matched_patterns",
                    [],
                )
                or []
            ),
            reasons=list(
                detected_event.get(
                    "reasons",
                    [],
                )
                or []
            ),
            llm_called=llm_called,
            retrieval_started=retrieval_started,
        )

    def _run_document_attack(
        self,
        attack: AttackCase,
    ) -> AttackResult:
        """
        Run document content and metadata through the existing
        document injection detector.
        """

        detector = getattr(
            self.rag_system,
            "document_injection_detector",
            None,
        )

        if detector is None:
            raise AttributeError(
                "RAGSystem does not expose document_injection_detector."
            )

        metadata = dict(
            attack.metadata
            or {}
        )

        detection = detector.analyze(
            text=attack.payload,
            metadata=metadata,
        )

        actual_status = (
            "BLOCKED"
            if detection.is_injected
            else "ALLOWED"
        )

        return AttackResult(
            attack_id=attack.attack_id,
            name=attack.name,
            category=attack.category,
            attack_type=attack.attack_type.value,
            expected_status=attack.expected_status,
            actual_status=actual_status,
            passed=actual_status == attack.expected_status,
            detection_score=float(
                detection.score
            ),
            matched_patterns=list(
                detection.matched_patterns
            ),
            reasons=list(
                detection.reasons
            ),
            llm_called=False,
            retrieval_started=False,
        )


__all__ = [
    "AttackResult",
    "AttackReport",
    "RedTeamRunner",
]