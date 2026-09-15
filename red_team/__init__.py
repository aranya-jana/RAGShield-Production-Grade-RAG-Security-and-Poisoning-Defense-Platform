"""
RAGShield Red-Team Attack Framework.

Provides predefined adversarial attack cases and a runner that can
exercise RAGShield prompt and document security boundaries.
"""

from .attack_cases import (
    AttackCase,
    AttackType,
    DEFAULT_ATTACK_CASES,
    get_attack_cases,
)
from .attack_runner import (
    AttackResult,
    AttackReport,
    RedTeamRunner,
)

__all__ = [
    "AttackCase",
    "AttackType",
    "DEFAULT_ATTACK_CASES",
    "get_attack_cases",
    "AttackResult",
    "AttackReport",
    "RedTeamRunner",
]