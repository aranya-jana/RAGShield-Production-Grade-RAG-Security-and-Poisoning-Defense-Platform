"""
FastAPI backend for the RAG Poisoning Detection POC.

Provides:

GET     /
GET     /health
GET     /info
POST    /setup
POST    /attack
POST    /query
GET     /audit
DELETE  /audit

The API exposes the RAG security pipeline to the React frontend.

Audit architecture:
    - SecurityAuditLogger is the single persistent audit source.
    - RAGSystem writes pipeline-level security events.
    - /attack writes API-level attack events.
    - /audit reads the persistent JSONL audit trail.
    - DELETE /audit clears the persistent audit trail.
"""

from __future__ import annotations

import logging
import time
import uuid
from typing import List, Optional

from src.request_size_limit import RequestSizeLimitMiddleware
from fastapi import Depends, FastAPI, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from src.security_middleware import SecurityHeadersMiddleware
from src.security_telemetry import (
    get_request_id,
    safe_content_metadata,
    security_event_metadata,
)

try:
    from src.rate_limiter import RateLimiter
except ImportError:
    from rate_limiter import RateLimiter
# ============================================================================
# LOGGING
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

logger = logging.getLogger(__name__)


# ============================================================================
# PROJECT IMPORTS
# ============================================================================

try:
    from src.config import Config
    from src.llm_factory import (
        create_embeddings,
        create_llm,
    )
    from src.rag_system import RAGSystem
    from src.audit_logger import SecurityAuditLogger
    from src.auth import (
        AuthenticatedPrincipal,
        AuthenticationError,
        AuthorizationError,
        TokenValidationError,
        RBAC,
        TokenManager,
        UserStore,
    )

except ImportError:
    from config import Config
    from llm_factory import (
        create_embeddings,
        create_llm,
    )
    from rag_system import RAGSystem
    from audit_logger import SecurityAuditLogger
    from auth import (
        AuthenticatedPrincipal,
        AuthenticationError,
        AuthorizationError,
        TokenValidationError,
        RBAC,
        TokenManager,
        UserStore,
    )


# ============================================================================
# FASTAPI APP
# ============================================================================

app = FastAPI(
    title="RAG Poisoning Detection API",
    description=(
        "Security-focused Retrieval-Augmented Generation API "
        "with instruction poisoning and factual contradiction detection."
    ),
    version="1.0.0",
)

# ============================================================================
# HTTP SECURITY HEADERS
# ============================================================================

app.add_middleware(
    SecurityHeadersMiddleware,
)

app.add_middleware(
    RequestSizeLimitMiddleware,
    max_body_bytes=5 * 1024 * 1024,
    path_limits={
        "/query": 64 * 1024,
        "/attack": 256 * 1024,
        "/setup": 5 * 1024 * 1024,
    },
)

# ============================================================================
# AUTHENTICATION / RBAC
# ============================================================================

bearer_scheme = HTTPBearer(
    auto_error=False,
)

# Local-development authentication stores.
#
# Production deployments should replace UserStore with persistent storage
# and provide a stable RAGSHIELD_AUTH_SECRET.

user_store = UserStore()

token_manager = TokenManager()

rbac = RBAC()

# ============================================================================
# API RATE LIMITING
# ============================================================================

# Endpoint-specific limits for authenticated expensive operations.
#
# These limits are intentionally conservative for the local deployment.
# A multi-worker production deployment should use a shared limiter such
# as Redis instead of process-local state.

rate_limiter = RateLimiter(
    max_clients=10_000,
    cleanup_interval_seconds=60,
)

RATE_LIMITS = {
    "query": {
        "limit": 10,
        "window_seconds": 60,
    },
    "attack": {
        "limit": 20,
        "window_seconds": 60,
    },
    "setup": {
        "limit": 5,
        "window_seconds": 60,
    },
}


def get_current_principal(
    credentials: Optional[
        HTTPAuthorizationCredentials
    ] = Depends(bearer_scheme),
) -> AuthenticatedPrincipal:
    """
    Authenticate the current API request.

    Requires:

        Authorization: Bearer <token>
    """

    if credentials is None:
        add_audit_event(
            event_type="AUTHENTICATION FAILURE",
            status="BLOCKED",
            detector="Authentication",
            reasons=["Authentication credentials missing."],
            telemetry_metadata=security_event_metadata(
                category="authentication",
                reason_code="missing_credentials",
                severity="WARNING",
            ),
        )
        raise HTTPException(
            status_code=401,
            detail="Authentication required.",
            headers={
                "WWW-Authenticate": "Bearer",
            },
        )

    if credentials.scheme.lower() != "bearer":
        add_audit_event(
            event_type="AUTHENTICATION FAILURE",
            status="BLOCKED",
            detector="Authentication",
            reasons=["Unsupported authentication scheme."],
            telemetry_metadata=security_event_metadata(
                category="authentication",
                reason_code="unsupported_scheme",
                severity="WARNING",
            ),
        )
        raise HTTPException(
            status_code=401,
            detail="Bearer authentication required.",
            headers={
                "WWW-Authenticate": "Bearer",
            },
        )

    try:
        return token_manager.validate(
            credentials.credentials
        )

    except (
        AuthenticationError,
        AuthorizationError,
        TokenValidationError,
        ValueError,
    ) as exc:
        logger.warning(
            "Authentication failed: %s",
            exc,
        )

        add_audit_event(
            event_type="AUTHENTICATION FAILURE",
            status="BLOCKED",
            detector="Authentication",
            reasons=["Authentication token rejected."],
            telemetry_metadata=security_event_metadata(
                category="authentication",
                reason_code="invalid_token",
                severity="WARNING",
            ),
        )

        raise HTTPException(
            status_code=401,
            detail="Invalid or expired authentication token.",
            headers={
                "WWW-Authenticate": "Bearer",
            },
        )


def require_permission(
    permission: str,
):
    """
    Create a FastAPI dependency requiring an RBAC permission.
    """

    def dependency(
        principal: AuthenticatedPrincipal = Depends(
            get_current_principal
        ),
    ) -> AuthenticatedPrincipal:

        try:
            rbac.require_permission(
                principal,
                permission,
            )

        except AuthorizationError as exc:
            username = getattr(
                principal,
                "username",
                "unknown",
            )

            logger.warning(
                "Authorization denied for %s: %s",
                username,
                exc,
            )

            add_audit_event(
                event_type="AUTHORIZATION FAILURE",
                status="BLOCKED",
                detector="RBAC",
                reasons=["Required permission denied."],
                telemetry_metadata=security_event_metadata(
                    category="authorization",
                    reason_code="permission_denied",
                    severity="WARNING",
                    username=username,
                    permission=permission,
                ),
            )

            raise HTTPException(
                status_code=403,
                detail="Insufficient permissions.",
            )

        return principal

    return dependency


def require_admin(
    principal: AuthenticatedPrincipal = Depends(
        get_current_principal
    ),
) -> AuthenticatedPrincipal:
    """
    Require the admin role for destructive administrative operations.
    """

    try:
        rbac.require_role(
            principal,
            "admin",
        )

    except AuthorizationError as exc:
        username = getattr(
            principal,
            "username",
            "unknown",
        )

        logger.warning(
            "Admin authorization denied for %s: %s",
            username,
            exc,
        )

        add_audit_event(
            event_type="AUTHORIZATION FAILURE",
            status="BLOCKED",
            detector="RBAC",
            reasons=["Administrator role required."],
            telemetry_metadata=security_event_metadata(
                category="authorization",
                reason_code="admin_role_denied",
                severity="HIGH",
                username=username,
                permission="admin",
            ),
        )

        raise HTTPException(
            status_code=403,
            detail="Administrator privileges required.",
        )

    return principal

def enforce_rate_limit(
    scope: str,
):
    """
    Create a FastAPI dependency for an endpoint-specific rate limit.

    Rate limiting is keyed by the authenticated username.

    No token, password, query, attack payload, or request body is stored
    by the rate limiter.
    """

    if scope not in RATE_LIMITS:
        raise ValueError(
            f"Unknown rate-limit scope: {scope}"
        )

    settings = RATE_LIMITS[scope]

    def dependency(
        principal: AuthenticatedPrincipal = Depends(
            get_current_principal
        ),
    ) -> AuthenticatedPrincipal:

        username = getattr(
            principal,
            "username",
            None,
        )

        if not username:
            raise HTTPException(
                status_code=401,
                detail="Authentication required.",
                headers={
                    "WWW-Authenticate": "Bearer",
                },
            )

        decision = rate_limiter.check(
            client_key=f"user:{username}",
            scope=scope,
            limit=settings["limit"],
            window_seconds=settings[
                "window_seconds"
            ],
        )

        if not decision.allowed:

            logger.warning(
                "Rate limit exceeded: scope=%s user=%s",
                scope,
                username,
            )

            add_audit_event(
                event_type="RATE LIMIT VIOLATION",
                status="BLOCKED",
                detector="RateLimiter",
                reasons=["Endpoint rate limit exceeded."],
                telemetry_metadata=security_event_metadata(
                    category="rate_limit",
                    reason_code="limit_exceeded",
                    severity="HIGH",
                    username=username,
                    scope=scope,
                    limit=decision.limit,
                    remaining=decision.remaining,
                    retry_after_seconds=decision.retry_after_seconds,
                ),
            )

            raise HTTPException(
                status_code=429,
                detail=(
                    "Rate limit exceeded. "
                    "Please retry later."
                ),
                headers={
                    "Retry-After": str(
                        decision.retry_after_seconds
                    ),
                    "X-RateLimit-Limit": str(
                        decision.limit
                    ),
                    "X-RateLimit-Remaining": "0",
                },
            )

        return principal

    return dependency

# ============================================================================
# CORS
# ============================================================================

# React/Vite normally runs on port 5173.
#
# The browser treats:
#
#   http://localhost:5173
#
# and:
#
#   http://127.0.0.1:8000
#
# as different origins.
#
# Therefore CORS explicitly allows the frontend origins.

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# GLOBAL RAG INSTANCE
# ============================================================================

rag_system: Optional[RAGSystem] = None


# ============================================================================
# PERSISTENT AUDIT STORAGE
# ============================================================================

# SecurityAuditLogger is the single source of truth for audit events.
#
# File:
#
#     data/security_audit.jsonl
#
# JSONL is appropriate for this POC because:
#
#   - events are append-only
#   - each line is an independent JSON object
#   - logs survive API restarts
#   - the React dashboard can consume them through /audit
#
# This can later be replaced by:
#
#   SQLite
#   PostgreSQL
#   Elasticsearch
#   SIEM
#
# without changing the frontend architecture significantly.

audit_logger = SecurityAuditLogger(
    log_path="./data/security_audit.jsonl"
)


# ============================================================================
# REQUEST MODELS
# ============================================================================

class SetupRequest(BaseModel):
    """Request body for corpus setup."""

    include_poison: bool = False

    payload: Optional[str] = None


class QueryRequest(BaseModel):
    """Request body for protected RAG queries."""

    query: str = Field(
        ...,
        min_length=1,
        description="Question to send through the protected RAG pipeline.",
    )


class AttackRequest(BaseModel):
    """Request body for attack analysis."""

    payload: str = Field(
        ...,
        min_length=1,
        description="Poisoning payload to analyze.",
    )


# ============================================================================
# RESPONSE MODELS
# ============================================================================

class DocumentResponse(BaseModel):
    """Safe/blocked document information returned to the UI."""

    source: str

    document_type: str

    content: str

    poison_score: Optional[float] = None

    poison_detected: bool = False

    contradiction_score: Optional[float] = None

    contradiction_detected: bool = False

    reasons: List[str] = []

    status: str = "SAFE"


class SecurityEvent(BaseModel):
    """Single document security event."""

    source: str

    document_type: str

    detector: str

    score: float

    is_poisoned: bool

    is_contradictory: bool

    reasons: List[str]

    status: str


class SecurityResponse(BaseModel):
    """Overall security decision."""

    status: str

    poison_detected: bool

    contradiction_detected: bool

    blocked_count: int

    events: List[SecurityEvent]


class QueryResponse(BaseModel):
    """Protected query response."""

    query: str

    answer: str

    security: SecurityResponse

    retrieved_documents: List[DocumentResponse]

    blocked_documents: List[DocumentResponse]


class AttackResponse(BaseModel):
    """Attack detector response."""

    status: str

    message: str

    payload: str

    poison_score: float

    poison_detected: bool

    contradiction_score: float

    contradiction_detected: bool

    blocked_by: List[str]

    reasons: List[str]


class AuditEvent(BaseModel):
    """Audit log event exposed to the frontend."""

    event_id: str

    timestamp: str

    event_type: str

    status: str

    query: Optional[str] = None

    payload: Optional[str] = None

    source: Optional[str] = None

    detector: Optional[str] = None

    score: float = 0.0

    reasons: List[str] = []

    request_id: Optional[str] = None

    username: Optional[str] = None

    category: Optional[str] = None

    reason_code: Optional[str] = None

    severity: Optional[str] = None

    endpoint: Optional[str] = None


class AuditResponse(BaseModel):
    """Audit log response."""

    total: int

    events: List[AuditEvent]


# ============================================================================
# HELPERS
# ============================================================================

def get_rag_system() -> RAGSystem:
    """
    Create and return the global RAG system.

    The embeddings factory requires Config, so we explicitly pass:

        create_embeddings(config)

    Ollama is the default local LLM provider.
    """

    global rag_system

    if rag_system is None:

        logger.info(
            "Initializing RAG system..."
        )

        config = Config()

        # IMPORTANT:
        # create_embeddings requires config.
        embeddings = create_embeddings(
            config
        )

        # Local Ollama.
        llm = create_llm(
            config,
            provider="ollama",
        )

        rag_system = RAGSystem(
            config=config,
            embeddings=embeddings,
            llm=llm,
        )

        # Start with clean trusted corpus.
        rag_system.setup_vector_database(
            include_poison=False
        )

        logger.info(
            "RAG system initialized."
        )

    return rag_system


def document_to_response(
    document,
    poison_score: Optional[float] = None,
    poison_detected: bool = False,
    contradiction_score: Optional[float] = None,
    contradiction_detected: bool = False,
    reasons: Optional[List[str]] = None,
    status: str = "SAFE",
) -> DocumentResponse:
    """
    Convert a LangChain Document into an API response object.
    """

    metadata = getattr(
        document,
        "metadata",
        {},
    ) or {}

    content = getattr(
        document,
        "page_content",
        "",
    )

    source = metadata.get(
        "source",
        "unknown",
    )

    # RAGSystem historically used both:
    #
    #     document_type
    #
    # and:
    #
    #     type
    #
    # Support both.

    document_type = metadata.get(
        "document_type",
        metadata.get(
            "type",
            "benign",
        ),
    )

    return DocumentResponse(
        source=str(source),

        document_type=str(
            document_type
        ),

        content=content,

        poison_score=poison_score,

        poison_detected=poison_detected,

        contradiction_score=contradiction_score,

        contradiction_detected=contradiction_detected,

        reasons=reasons or [],

        status=status,
    )


def add_audit_event(
    *,
    event_type: str,
    status: str,
    query: Optional[str] = None,
    payload: Optional[str] = None,
    source: Optional[str] = None,
    detector: Optional[str] = None,
    score: float = 0.0,
    reasons: Optional[List[str]] = None,
    telemetry_metadata: Optional[dict] = None,
) -> dict:
    """
    Write an event to the persistent security audit log.

    Raw request contents are never persisted by API telemetry.

    When ``query`` or ``payload`` is supplied, only:
        - SHA-256 content digest
        - content length

    are retained in the ``request_content`` metadata.

    Structured telemetry metadata is merged into the same event.
    """

    metadata = {
        "api_event": True,
    }

    request_content = None

    if payload is not None:
        request_content = safe_content_metadata(payload)
    elif query is not None:
        request_content = safe_content_metadata(query)

    if request_content is not None:
        metadata["request_content"] = request_content

    if telemetry_metadata:
        metadata.update(
            telemetry_metadata
        )

    request_id = get_request_id()

    if request_id:
        metadata["request_id"] = request_id

    event = audit_logger.log_event(
        event_type=event_type,
        query=None,
        source=source,
        detector=detector,
        score=float(score),
        status=status,
        reasons=reasons or [],
        metadata=metadata,
    )

    if not event.get("event_id"):
        event["event_id"] = str(
            uuid.uuid4()
        )

    return event


def build_security_events(
    result: dict,
) -> List[SecurityEvent]:
    """
    Convert RAGSystem security telemetry into frontend events.

    The current RAGSystem stores detailed security information in:

        rag_system.security_events

    We first use that telemetry.

    If it is unavailable, we reconstruct the events from the
    returned safe/blocked document lists.
    """

    events: List[SecurityEvent] = []

    # ------------------------------------------------------------------
    # Preferred source: structured telemetry from RAGSystem
    # ------------------------------------------------------------------

    rag = get_rag_system()

    raw_events = getattr(
        rag,
        "security_events",
        None,
    )

    if raw_events:

        for event in raw_events:

            if isinstance(
                event,
                SecurityEvent,
            ):
                events.append(
                    event
                )
                continue

            if isinstance(
                event,
                dict,
            ):

                events.append(
                    SecurityEvent(
                        source=str(
                            event.get(
                                "source",
                                "unknown",
                            )
                        ),

                        document_type=str(
                            event.get(
                                "document_type",
                                event.get(
                                    "type",
                                    "benign",
                                ),
                            )
                        ),

                        detector=str(
                            event.get(
                                "detector",
                                "None",
                            )
                        ),

                        score=float(
                            event.get(
                                "score",
                                0.0,
                            )
                            or 0.0
                        ),

                        is_poisoned=bool(
                            event.get(
                                "is_poisoned",
                                False,
                            )
                        ),

                        is_contradictory=bool(
                            event.get(
                                "is_contradictory",
                                False,
                            )
                        ),

                        reasons=list(
                            event.get(
                                "reasons",
                                [],
                            )
                            or []
                        ),

                        status=str(
                            event.get(
                                "status",
                                "SAFE",
                            )
                        ),
                    )
                )

        if events:
            return events

    # ------------------------------------------------------------------
    # Fallback reconstruction
    # ------------------------------------------------------------------

    safe_documents = result.get(
        "source_documents",
        [],
    )

    blocked_documents = result.get(
        "blocked_documents",
        [],
    )

    for document in safe_documents:

        metadata = getattr(
            document,
            "metadata",
            {},
        ) or {}

        events.append(
            SecurityEvent(
                source=str(
                    metadata.get(
                        "source",
                        "unknown",
                    )
                ),

                document_type=str(
                    metadata.get(
                        "document_type",
                        metadata.get(
                            "type",
                            "benign",
                        ),
                    )
                ),

                detector="None",

                score=0.0,

                is_poisoned=False,

                is_contradictory=False,

                reasons=[],

                status="SAFE",
            )
        )

    for document in blocked_documents:

        metadata = getattr(
            document,
            "metadata",
            {},
        ) or {}

        events.append(
            SecurityEvent(
                source=str(
                    metadata.get(
                        "source",
                        "unknown",
                    )
                ),

                document_type=str(
                    metadata.get(
                        "document_type",
                        metadata.get(
                            "type",
                            "poisoned",
                        ),
                    )
                ),

                detector="SecurityLayer",

                score=1.0,

                is_poisoned=True,

                is_contradictory=False,

                reasons=[
                    "Document blocked by security layer."
                ],

                status="BLOCKED",
            )
        )

    return events


# ============================================================================
# ROOT
# ============================================================================

@app.get("/")
def root():
    """Root endpoint."""

    return {
        "name": "RAG Poisoning Detection API",

        "version": "1.0.0",

        "status": "running",

        "docs": "/docs",

        "audit": "/audit",
    }


# ============================================================================
# HEALTH
# ============================================================================

@app.get("/health")
def health():
    """
    API health check.

    Used by the React frontend to display:

        API Online
        API Offline
    """

    return {
        "status": "healthy",

        "service": "rag-poisoning-api",

        "timestamp": time.time(),
    }


# ============================================================================
# INFO
# ============================================================================

@app.get("/info")
def info():
    """Return current RAG configuration."""

    try:

        config = Config()

        return {
            "name": "RAG Poisoning Detection",

            "version": "1.0.0",

            "llm_provider": "ollama",

            "llm_model": getattr(
                config,
                "ollama_model",
                "phi4-mini:latest",
            ),

            "embedding_model": getattr(
                config,
                "embedding_model",
                "sentence-transformers/all-MiniLM-L6-v2",
            ),
        }

    except Exception as exc:

        logger.exception(
            "Unable to load API information."
        )

        raise HTTPException(
            status_code=500,
            detail="Internal server error.",
        )


# ============================================================================
# SETUP
# ============================================================================

@app.post("/setup")
def setup(
    request: SetupRequest,
    principal: AuthenticatedPrincipal = Depends(
        require_permission("manage_documents")
    ),
    _rate_limit_principal: AuthenticatedPrincipal = Depends(
        enforce_rate_limit("setup")
    ),
):
    """
    Reset the vector database and optionally inject a poisoned document.

    Starting a new experiment clears the persistent audit trail so the
    following audit events belong to the new experiment.
    """

    try:

        rag = get_rag_system()

        rag.setup_vector_database(
            include_poison=request.include_poison,

            payload=request.payload,
        )

        # Clear previous RAG query telemetry.
        if hasattr(
            rag,
            "security_events",
        ):
            rag.security_events = []

        # Clear the persistent audit trail.
        audit_logger.clear()

        # If a poisoned document was requested, the setup operation itself
        # is useful to record in the audit trail.

        if request.include_poison:

            message = (
                "RAG corpus reset and poisoned "
                "document injected."
            )

            add_audit_event(
                event_type="CORPUS SETUP",
                status="POISONED",
                payload=request.payload,
                detector="Setup",
                score=0.0,
                reasons=[
                    "Poisoned document intentionally injected for POC testing."
                ],
            )

        else:

            message = (
                "RAG corpus reset successfully."
            )

            add_audit_event(
                event_type="CORPUS SETUP",
                status="SAFE",
                detector="Setup",
                score=0.0,
                reasons=[
                    "Trusted corpus initialized with benign documents."
                ],
            )

        logger.info(
            message
        )

        return {
            "status": "ok",

            "message": message,

            "trusted_documents": 3,

            "poisoned_document": (
                request.include_poison
            ),
        }

    except Exception as exc:

        logger.exception(
            "Setup failed."
        )

        raise HTTPException(
            status_code=500,
            detail="Internal server error.",
        )


# ============================================================================
# ATTACK
# ============================================================================

@app.post(
    "/attack",
    response_model=AttackResponse,
)
def attack(
    request: AttackRequest,
    principal: AuthenticatedPrincipal = Depends(
        require_permission("run_red_team")
    ),
    _rate_limit_principal: AuthenticatedPrincipal = Depends(
        enforce_rate_limit("attack")
    ),
):
    """
    Analyze a poisoning payload.

    The attack is NOT sent to the LLM.

    It is evaluated by:

        PoisonDetector

        ContradictionDetector
    """

    try:

        rag = get_rag_system()

        payload = request.payload.strip()

        if not payload:

            raise HTTPException(
                status_code=400,
                detail="Payload cannot be empty.",
            )

        # --------------------------------------------------------------
        # PoisonDetector
        # --------------------------------------------------------------

        poison_result = (
            rag.poison_detector.analyze(
                payload
            )
        )

        # --------------------------------------------------------------
        # ContradictionDetector
        # --------------------------------------------------------------

        contradiction_result = (
            rag.contradiction_detector.analyze(
                candidate_text=payload,

                trusted_documents=(
                    rag.trusted_documents
                ),
            )
        )

        blocked_by = []

        reasons = []

        # --------------------------------------------------------------
        # Poison result
        # --------------------------------------------------------------

        if poison_result.is_poisoned:

            blocked_by.append(
                "PoisonDetector"
            )

            reasons.extend(
                poison_result.reasons
            )

        # --------------------------------------------------------------
        # Contradiction result
        # --------------------------------------------------------------

        if (
            contradiction_result.is_contradictory
        ):

            blocked_by.append(
                "ContradictionDetector"
            )

            reasons.extend(
                contradiction_result.reasons
            )

        # Remove duplicate reasons.
        reasons = list(
            dict.fromkeys(
                reasons
            )
        )

        status = (
            "BLOCKED"
            if blocked_by
            else "SAFE"
        )

        if blocked_by:

            message = (
                "Poisoning payload detected "
                "and blocked by the security "
                "layer(s)."
            )

        else:

            message = (
                "Payload passed the current "
                "security checks."
            )

        # --------------------------------------------------------------
        # Persistent audit event
        # --------------------------------------------------------------

        if blocked_by:

            for detector in blocked_by:

                if detector == "PoisonDetector":

                    detector_score = float(
                        poison_result.score
                    )

                    detector_reasons = (
                        poison_result.reasons
                    )

                    event_type = "POISON ATTACK"

                else:

                    detector_score = float(
                        contradiction_result.score
                    )

                    detector_reasons = (
                        contradiction_result.reasons
                    )

                    event_type = "FACTUAL ATTACK"

                add_audit_event(
                    event_type=event_type,

                    status="BLOCKED",

                    payload=payload,

                    detector=detector,

                    score=detector_score,

                    reasons=detector_reasons,
                )

        else:

            add_audit_event(
                event_type="ATTACK ANALYSIS",

                status="SAFE",

                payload=payload,

                detector="None",

                score=0.0,

                reasons=[],
            )

        return AttackResponse(
            status=status,

            message=message,

            payload=payload,

            poison_score=float(
                poison_result.score
            ),

            poison_detected=bool(
                poison_result.is_poisoned
            ),

            contradiction_score=float(
                contradiction_result.score
            ),

            contradiction_detected=bool(
                contradiction_result.is_contradictory
            ),

            blocked_by=blocked_by,

            reasons=reasons,
        )

    except HTTPException:
        raise

    except Exception as exc:

        logger.exception(
            "Attack analysis failed."
        )

        raise HTTPException(
            status_code=500,
            detail="Internal server error.",
        )


# ============================================================================
# QUERY
# ============================================================================

@app.post(
    "/query",
    response_model=QueryResponse,
)
def query(
    request: QueryRequest,
    principal: AuthenticatedPrincipal = Depends(
        require_permission("query")
    ),
    _rate_limit_principal: AuthenticatedPrincipal = Depends(
        enforce_rate_limit("query")
    ),
):
    """
    Execute a protected RAG query.

    Documents are retrieved first.

    Security detectors inspect the retrieved documents.

    Unsafe documents are blocked.

    Only safe context reaches the LLM.
    """

    try:

        rag = get_rag_system()

        query_text = request.query.strip()

        if not query_text:

            raise HTTPException(
                status_code=400,
                detail="Query cannot be empty.",
            )

        logger.info(
            "Protected query: %s",
            query_text,
        )

        # --------------------------------------------------------------
        # Execute protected RAG
        # --------------------------------------------------------------

        result = rag.query(
            query_text
        )

        # --------------------------------------------------------------
        # Security events
        # --------------------------------------------------------------

        security_events = (
            build_security_events(
                result
            )
        )

        # --------------------------------------------------------------
        # Overall security state
        # --------------------------------------------------------------

        poison_detected = any(
            event.is_poisoned
            for event in security_events
        )

        contradiction_detected = any(
            event.is_contradictory
            for event in security_events
        )

        blocked_count = sum(
            1
            for event in security_events
            if event.status == "BLOCKED"
        )

        security_status = (
            "BLOCKED"
            if blocked_count > 0
            else "SAFE"
        )

        security = SecurityResponse(
            status=security_status,

            poison_detected=poison_detected,

            contradiction_detected=(
                contradiction_detected
            ),

            blocked_count=blocked_count,

            events=security_events,
        )

        # --------------------------------------------------------------
        # Safe documents
        # --------------------------------------------------------------

        safe_documents = result.get(
            "source_documents",
            [],
        )

        blocked_documents = result.get(
            "blocked_documents",
            [],
        )

        safe_response_documents = []

        blocked_response_documents = []

        # --------------------------------------------------------------
        # Convert safe documents
        # --------------------------------------------------------------

        for document in safe_documents:

            metadata = getattr(
                document,
                "metadata",
                {},
            ) or {}

            source = metadata.get(
                "source",
                "unknown",
            )

            matching_event = next(
                (
                    event
                    for event in security_events
                    if event.source == source
                    and event.status == "SAFE"
                ),
                None,
            )

            safe_response_documents.append(
                document_to_response(
                    document=document,

                    poison_score=(
                        matching_event.score
                        if matching_event
                        and matching_event.detector
                        == "PoisonDetector"
                        else None
                    ),

                    poison_detected=(
                        matching_event.is_poisoned
                        if matching_event
                        else False
                    ),

                    contradiction_score=(
                        matching_event.score
                        if matching_event
                        and matching_event.detector
                        == "ContradictionDetector"
                        else None
                    ),

                    contradiction_detected=(
                        matching_event.is_contradictory
                        if matching_event
                        else False
                    ),

                    reasons=(
                        matching_event.reasons
                        if matching_event
                        else []
                    ),

                    status="SAFE",
                )
            )

        # --------------------------------------------------------------
        # Convert blocked documents
        # --------------------------------------------------------------

        for document in blocked_documents:

            metadata = getattr(
                document,
                "metadata",
                {},
            ) or {}

            source = metadata.get(
                "source",
                "unknown",
            )

            matching_events = [
                event
                for event in security_events
                if event.source == source
                and event.status == "BLOCKED"
            ]

            is_poisoned = any(
                event.is_poisoned
                for event in matching_events
            )

            is_contradictory = any(
                event.is_contradictory
                for event in matching_events
            )

            poison_score = next(
                (
                    event.score
                    for event in matching_events
                    if event.is_poisoned
                ),
                None,
            )

            contradiction_score = next(
                (
                    event.score
                    for event in matching_events
                    if event.is_contradictory
                ),
                None,
            )

            blocked_reasons = []

            for event in matching_events:

                blocked_reasons.extend(
                    event.reasons
                )

            blocked_reasons = list(
                dict.fromkeys(
                    blocked_reasons
                )
            )

            detectors = [
                event.detector
                for event in matching_events
                if event.detector
                and event.detector != "None"
            ]

            detector = (
                detectors[0]
                if detectors
                else "SecurityLayer"
            )

            blocked_score = next(
                (
                    event.score
                    for event in matching_events
                ),
                1.0,
            )

            blocked_response_documents.append(
                DocumentResponse(
                    source=str(
                        source
                    ),

                    document_type=str(
                        metadata.get(
                            "document_type",
                            metadata.get(
                                "type",
                                "poisoned",
                            ),
                        )
                    ),

                    content=getattr(
                        document,
                        "page_content",
                        "",
                    ),

                    poison_score=(
                        poison_score
                        if poison_score is not None
                        else (
                            blocked_score
                            if detector
                            == "PoisonDetector"
                            else None
                        )
                    ),

                    poison_detected=(
                        is_poisoned
                    ),

                    contradiction_score=(
                        contradiction_score
                        if contradiction_score is not None
                        else (
                            blocked_score
                            if detector
                            == "ContradictionDetector"
                            else None
                        )
                    ),

                    contradiction_detected=(
                        is_contradictory
                    ),

                    reasons=blocked_reasons,

                    status="BLOCKED",
                )
            )

        # --------------------------------------------------------------
        # Answer
        # --------------------------------------------------------------

        answer = str(
            result.get(
                "result",
                "",
            )
        )

        # --------------------------------------------------------------
        # API-level query audit
        # --------------------------------------------------------------

        if blocked_count > 0:

            for event in security_events:

                if event.status != "BLOCKED":
                    continue

                add_audit_event(
                    event_type="QUERY SECURITY",

                    status="BLOCKED",

                    query=query_text,

                    source=event.source,

                    detector=event.detector,

                    score=event.score,

                    reasons=event.reasons,
                )

        else:

            add_audit_event(
                event_type="QUERY",

                status="SAFE",

                query=query_text,

                detector="None",

                score=0.0,

                reasons=[],
            )

        return QueryResponse(
            query=query_text,

            answer=answer,

            security=security,

            retrieved_documents=(
                safe_response_documents
            ),

            blocked_documents=(
                blocked_response_documents
            ),
        )

    except HTTPException:
        raise

    except Exception as exc:

        logger.exception(
            "Query failed."
        )

        raise HTTPException(
            status_code=500,
            detail="Internal server error.",
        )


# ============================================================================
# AUDIT
# ============================================================================

@app.get(
    "/audit",
    response_model=AuditResponse,
)
def get_audit(
    principal: AuthenticatedPrincipal = Depends(
        require_permission("read_audit")
    ),
):
    """
    Return the persistent security audit trail.

    Events are returned newest first.
    """

    try:

        raw_events = audit_logger.get_events(
            limit=500
        )

        events: List[AuditEvent] = []

        for index, event in enumerate(
            raw_events
        ):

            metadata = (
                event.get(
                    "metadata",
                    {}
                )
                or {}
            )

            # SecurityAuditLogger events may not have an event_id.
            # Generate a stable-looking fallback for API presentation.

            event_id = event.get(
                "event_id"
            )

            if not event_id:

                timestamp = str(
                    event.get(
                        "timestamp",
                        "",
                    )
                )

                event_id = (
                    f"{timestamp}-{index}"
                )

            payload = event.get(
                "payload"
            )

            if payload is None:

                payload = metadata.get(
                    "payload"
                )

            events.append(
                AuditEvent(
                    event_id=str(
                        event_id
                    ),

                    timestamp=str(
                        event.get(
                            "timestamp",
                            "",
                        )
                    ),

                    event_type=str(
                        event.get(
                            "event_type",
                            "UNKNOWN",
                        )
                    ),

                    status=str(
                        event.get(
                            "status",
                            "UNKNOWN",
                        )
                    ),

                    query=event.get(
                        "query"
                    ),

                    payload=payload,

                    source=event.get(
                        "source"
                    ),

                    detector=event.get(
                        "detector"
                    ),

                    score=float(
                        event.get(
                            "score",
                            0.0,
                        )
                        or 0.0
                    ),

                    reasons=list(
                        event.get(
                            "reasons",
                            [],
                        )
                        or []
                    ),

                    request_id=metadata.get(
                        "request_id"
                    ),

                    username=metadata.get(
                        "username"
                    ),

                    category=metadata.get(
                        "category"
                    ),

                    reason_code=metadata.get(
                        "reason_code"
                    ),

                    severity=metadata.get(
                        "severity"
                    ),

                    endpoint=metadata.get(
                        "path"
                    ),
                )
            )

        return AuditResponse(
            total=len(events),

            events=events,
        )

    except Exception as exc:

        logger.exception(
            "Unable to read audit log."
        )

        raise HTTPException(
            status_code=500,
            detail="Internal server error.",
        )


# ============================================================================
# CLEAR AUDIT
# ============================================================================

@app.delete("/audit")
def clear_audit(
    principal: AuthenticatedPrincipal = Depends(
        require_admin
    ),
):
    """
    Clear the persistent security audit trail.
    """

    try:

        count = audit_logger.count()

        audit_logger.clear()

        logger.info(
            "Persistent audit log cleared. Removed %d events.",
            count,
        )

        return {
            "status": "ok",

            "message": "Audit log cleared.",

            "removed_events": count,
        }

    except Exception as exc:

        logger.exception(
            "Unable to clear audit log."
        )

        raise HTTPException(
            status_code=500,
            detail="Internal server error.",
        )


# ============================================================================
# STARTUP
# ============================================================================

@app.on_event("startup")
def startup_event():
    """Application startup."""

    logger.info(
        "=================================================="
    )

    logger.info(
        "RAG Poisoning Detection API starting..."
    )

    logger.info(
        "Frontend CORS enabled for localhost:5173"
    )

    logger.info(
        "Persistent audit logging enabled."
    )

    logger.info(
        "API authentication and RBAC enabled."
    )

    logger.info(
        "Audit endpoint available at /audit"
    )

    logger.info(
        "=================================================="
    )