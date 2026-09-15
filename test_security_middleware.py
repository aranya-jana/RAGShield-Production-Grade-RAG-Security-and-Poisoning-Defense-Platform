from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.security_middleware import SecurityHeadersMiddleware


def create_test_app() -> FastAPI:
    app = FastAPI()
    app.add_middleware(SecurityHeadersMiddleware)

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.post("/query")
    def query():
        return {"answer": "safe"}

    @app.post("/attack")
    def attack():
        return {"status": "blocked"}

    @app.post("/setup")
    def setup():
        return {"status": "ready"}

    @app.get("/audit")
    def audit():
        return {"events": []}

    return app


def test_content_type_options_header():
    client = TestClient(create_test_app())

    response = client.get("/health")

    assert response.status_code == 200
    assert response.headers["X-Content-Type-Options"] == "nosniff"


def test_frame_options_header():
    client = TestClient(create_test_app())

    response = client.get("/health")

    assert response.headers["X-Frame-Options"] == "DENY"


def test_referrer_policy_header():
    client = TestClient(create_test_app())

    response = client.get("/health")

    assert response.headers["Referrer-Policy"] == "no-referrer"


def test_permissions_policy_header():
    client = TestClient(create_test_app())

    response = client.get("/health")

    policy = response.headers["Permissions-Policy"]

    assert "camera=()" in policy
    assert "microphone=()" in policy
    assert "geolocation=()" in policy


def test_content_security_policy_header():
    client = TestClient(create_test_app())

    response = client.get("/health")

    policy = response.headers["Content-Security-Policy"]

    assert "default-src 'none'" in policy
    assert "frame-ancestors 'none'" in policy
    assert "base-uri 'none'" in policy
    assert "form-action 'none'" in policy


def test_cross_origin_resource_policy_header():
    client = TestClient(create_test_app())

    response = client.get("/health")

    assert (
        response.headers["Cross-Origin-Resource-Policy"]
        == "same-site"
    )


def test_cross_domain_policy_header():
    client = TestClient(create_test_app())

    response = client.get("/health")

    assert (
        response.headers["X-Permitted-Cross-Domain-Policies"]
        == "none"
    )


def test_sensitive_query_response_is_not_cached():
    client = TestClient(create_test_app())

    response = client.post("/query")

    assert "no-store" in response.headers["Cache-Control"]
    assert "no-cache" in response.headers["Cache-Control"]
    assert response.headers["Pragma"] == "no-cache"


def test_attack_response_is_not_cached():
    client = TestClient(create_test_app())

    response = client.post("/attack")

    assert "no-store" in response.headers["Cache-Control"]
    assert response.headers["Pragma"] == "no-cache"


def test_setup_response_is_not_cached():
    client = TestClient(create_test_app())

    response = client.post("/setup")

    assert "no-store" in response.headers["Cache-Control"]
    assert response.headers["Pragma"] == "no-cache"


def test_audit_response_is_not_cached():
    client = TestClient(create_test_app())

    response = client.get("/audit")

    assert "no-store" in response.headers["Cache-Control"]
    assert response.headers["Pragma"] == "no-cache"


def test_public_health_response_does_not_force_sensitive_cache_policy():
    client = TestClient(create_test_app())

    response = client.get("/health")

    assert "Cache-Control" not in response.headers
    assert "Pragma" not in response.headers


def test_headers_exist_on_404_response():
    client = TestClient(create_test_app())

    response = client.get("/does-not-exist")

    assert response.status_code == 404
    assert response.headers["X-Content-Type-Options"] == "nosniff"
    assert response.headers["X-Frame-Options"] == "DENY"
    assert response.headers["Referrer-Policy"] == "no-referrer"