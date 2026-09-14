"""
Security audit logging for the RAG poisoning protection system.

Stores security events in a JSON Lines (.jsonl) file so that every
security decision can be reviewed later.

Each line in the audit file represents one security event.
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


logger = logging.getLogger(__name__)


class SecurityAuditLogger:
    """
    Persistent security audit logger.

    Events are stored as JSON Lines:

        {"timestamp": "...", "event_type": "...", ...}
        {"timestamp": "...", "event_type": "...", ...}

    JSONL is convenient for:
    - appending new events
    - reading logs incrementally
    - debugging
    - exporting to SIEM/logging systems later
    """

    def __init__(
        self,
        log_path: str = "./data/security_audit.jsonl",
    ):
        self.log_path = Path(log_path)

        # Ensure the parent directory exists.
        self.log_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

    # ------------------------------------------------------------------
    # WRITE EVENT
    # ------------------------------------------------------------------

    def log_event(
        self,
        event_type: str,
        query: Optional[str] = None,
        source: Optional[str] = None,
        document_type: Optional[str] = None,
        detector: Optional[str] = None,
        score: Optional[float] = None,
        status: Optional[str] = None,
        reasons: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Write one security event to the audit log.

        Returns the event dictionary that was written.
        """

        event = {
            "timestamp": datetime.now(
                timezone.utc
            ).isoformat(),

            "event_type": event_type,

            "query": query,

            "source": source,

            "document_type": document_type,

            "detector": detector,

            "score": score,

            "status": status,

            "reasons": reasons or [],

            "metadata": metadata or {},
        }

        try:
            with self.log_path.open(
                "a",
                encoding="utf-8",
            ) as file:

                file.write(
                    json.dumps(
                        event,
                        ensure_ascii=False,
                    )
                    + "\n"
                )

        except OSError as exc:

            logger.error(
                "Failed to write security audit event: %s",
                exc,
            )

            raise

        return event

    # ------------------------------------------------------------------
    # READ EVENTS
    # ------------------------------------------------------------------

    def get_events(
        self,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        Read the most recent audit events.

        Events are returned newest first.
        """

        if limit <= 0:
            return []

        if not self.log_path.exists():
            return []

        events = []

        try:

            with self.log_path.open(
                "r",
                encoding="utf-8",
            ) as file:

                for line in file:

                    line = line.strip()

                    if not line:
                        continue

                    try:

                        event = json.loads(line)

                        events.append(event)

                    except json.JSONDecodeError:

                        logger.warning(
                            "Skipping invalid audit log line."
                        )

        except OSError as exc:

            logger.error(
                "Failed to read security audit log: %s",
                exc,
            )

            return []

        # Newest events first.
        events.reverse()

        return events[:limit]

    # ------------------------------------------------------------------
    # CLEAR LOG
    # ------------------------------------------------------------------

    def clear(self):
        """
        Delete the current audit log.

        Primarily useful for tests and development.
        """

        if self.log_path.exists():

            self.log_path.unlink()

    # ------------------------------------------------------------------
    # COUNT
    # ------------------------------------------------------------------

    def count(self) -> int:
        """
        Return the number of valid audit events.
        """

        return len(
            self.get_events(
                limit=10_000_000
            )
        )