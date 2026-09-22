# RAGShield

RAGShield is a security-focused Retrieval-Augmented Generation (RAG) platform designed to detect, quarantine, and block poisoned or unsafe documents before they can influence downstream LLM responses.

The project combines document security analysis, provenance tracking, integrity metadata, authentication and authorization, audit telemetry, and security-focused dashboard workflows.

> **Current status:** RAGShield is a locally runnable, production-oriented security implementation. It is not yet a fully production-deployed system. Persistent storage, production identity management, HTTPS/TLS, cloud infrastructure, CI/CD, production monitoring, backup/recovery, and external security testing remain future phases.

---

## Security Capabilities

RAGShield currently provides security controls across the document-to-LLM pipeline.

### Document Security

- Document ingestion and indexing
- Document provenance tracking
- SHA-256 document integrity identifiers
- Document security status
- Risk scoring
- Trust scoring
- Document quarantine
- Prompt-injection / instruction detection
- Poisoning detection
- Factual contradiction detection
- PII detection
- DLP-oriented detection
- Safe-context filtering
- Output validation

### Authentication and Authorization

The backend includes:

- PBKDF2-HMAC-SHA256 password hashing
- Signed bearer-token authentication
- Token expiration
- Token revocation
- Token versioning
- Disabled-account enforcement
- Role-based access control
- Permission checks
- Document access scopes
- Authentication telemetry
- Authorization telemetry
- Login rate limiting
- API request rate limiting
- Request-size limits
- Security response headers
- Local development CORS restrictions

Disabled accounts are revalidated when authenticated requests are processed, preventing an existing token from continuing to authorize requests after the account has been disabled.

Login protection includes separate IP-based and username-based throttling to reduce repeated authentication attempts.

---

## Security Pipeline

The current security-oriented document flow is:

```text
Document Upload
      |
      v
Document Identification
      |
      v
SHA-256 Integrity Metadata
      |
      v
Provenance Extraction
      |
      v
Security Analysis
      |
      +-----------------------------+
      |                             |
      v                             v
Instruction Detection        Poisoning Detection
      |                             |
      +-------------+---------------+
                    |
                    v
             Risk / Trust Scoring
                    |
                    v
             Security Decision
                    |
          +---------+---------+
          |                   |
          v                   v
       Trusted            Quarantine
          |                   |
          v                   v
       Indexing          Blocked from
          |              trusted context
          v
      Safe Retrieval
          |
          v
     Output Validation
          |
          v
      User Response

## Security Command Center

The Security Command Center is the primary security overview on the Dashboard.

It provides a live summary of the current application security posture using backend data.

### Current capabilities

-  Overall operational posture
-  Backend health
-  Highest-risk document
-  Lowest-trust document
-  Quarantined document count
-  Threat-channel breakdown
-  Document security activity
-  Blocked security events
-  Authentication events
-  Authorization events
-  Recent high-severity events
-  24-hour security activity
-  Attention queue
-  Direct navigation to Documents
-  Direct navigation to Audit

### Operational posture

The dashboard evaluates the current security state using available backend signals such as:

-  Backend availability
-  Document risk
-  Document trust
-  Quarantine activity
-  Blocked security events
-  Authentication and authorization activity

The dashboard does not use static demonstration metrics for the live security command-center data.

If the backend is unavailable, the interface displays an unavailable state rather than presenting stale or fabricated security information.

### Attention queue

The attention queue surfaces recent security events that require investigation, particularly:

-  High-severity events
-  Blocked events
-  Recent security activity

Users can navigate from the dashboard to the appropriate security workspace for additional investigation.

---

## Document Security Workspace

The Documents workspace provides security-focused document management and investigation.

### Document discovery

Users can:

-  Search documents
-  Filter indexed documents
-  Filter quarantined documents
-  Select individual documents
-  Navigate through document results
-  Use keyboard-accessible document selection

### Security details

The selected document can expose:

-  Document identifier
-  SHA-256 integrity identifier
-  Provenance metadata
-  Upload timestamp
-  Security status
-  Risk score
-  Trust score
-  Injection status
-  Poisoning status
-  Contradiction status
-  DLP status
-  Applied detectors
-  Security findings

### Risk and trust

Documents have security-oriented risk and trust indicators.

The frontend displays normalized values on a 0â€“100 scale.

These values are presentation-oriented security indicators and should not be interpreted as cryptographic proof of document authenticity.

### Quarantine

Documents identified as unsafe or suspicious can be placed into quarantine.

Quarantined content is kept outside the trusted retrieval path so that potentially unsafe instructions or poisoned content does not automatically become model context.

### Security-safe presentation

The workspace is designed to avoid unnecessarily exposing sensitive data or raw attack material while still providing the information needed for security investigation.

The interface is also responsive across supported desktop and smaller-screen layouts.

---

## Audit and Telemetry

RAGShield records security-relevant telemetry to support investigation and operational visibility.

### Authentication telemetry

Examples include:

-  Login activity
-  Authentication failures
-  Authentication successes
-  Login rate-limit violations
-  Disabled-account access attempts

### Authorization telemetry

Examples include:

-  Permission failures
-  Protected-resource access events
-  Role-related authorization events
-  Document-scope access events

### Security telemetry

Examples include:

-  Blocked security events
-  Document security findings
-  Quarantine events
-  High-severity events
-  Security detection activity

### Rate-limit telemetry

Rate-limit violations are recorded without intentionally exposing:

-  Passwords
-  Bearer tokens
-  Authentication secrets
-  Unnecessary raw query contents
-  Unnecessary raw attack payloads

The goal is to preserve operational security visibility without turning logs into a source of sensitive information.

---

## Technology Stack

### Backend

The backend is implemented with:

-  Python 3.11 / 3.12
-  FastAPI
-  LangChain
-  LangChain Community
-  LangChain OpenAI integration
-  ChromaDB
-  Sentence Transformers
-  Transformers
-  PyTorch
-  llama.cpp
-  HTTPX
-  BeautifulSoup
-  lxml
-  python-dotenv

### Frontend

The frontend is implemented with:

-  React
-  React DOM
-  TypeScript
-  Vite
-  Tailwind CSS
-  Framer Motion
-  Lucide React
-  Recharts

### Development Tooling

The project also uses:

-  uv
-  npm
-  Git
-  pytest
-  Vite
-  TypeScript compiler
-  oxlint

### Local LLM Support

The local configuration supports model backends including:

-  Ollama
-  OpenAI-compatible local endpoints
-  llama.cpp
-  DeepSeek-compatible configuration

---

## Project Structure

```

```
RAG_Poisoning/
|
+-- .github/
|   +-- ...
|
+-- frontend/
|   +-- src/
|   |   +-- App.tsx
|   |   +-- App.css
|   |   +-- ...
|   |
|   +-- package.json
|   +-- package-lock.json
|   +-- vite.config.*
|   +-- tsconfig.*
|   +-- ...
|
+-- src/
|   +-- api.py
|   +-- ...
|
+-- tests/
|   +-- ...
|
+-- red_team/
|   +-- ...
|
+-- data/
|   +-- local vector database/runtime data
|
+-- models/
|   +-- local model files
|
+-- logs/
|   +-- local application logs
|
+-- .env.example
+-- .keys.example
+-- .gitignore
+-- pyproject.toml
+-- uv.lock
+-- setup.sh
+-- README.md
```

Local runtime directories such as `data/`, `logs/`, and `models/` are intentionally excluded from Git tracking.

Generated frontend files such as `frontend/dist/` and installed dependencies such as `frontend/node_modules/` are also excluded from Git.

---

## Requirements

The local development environment requires:

-  Python 3.11 or 3.12
-  Node.js
-  npm
-  Git
-  Ollama

The Python project configuration is maintained in:

```

```
pyproject.toml
```

The locked Python dependency versions are maintained in:

```

```
uv.lock
```

---

## Local Setup

### 1. Clone the repository

```

```
git clone <repository-url>
cd RAG_Poisoning
```

### 2. Create the Python environment

```

```
uv venv
```

Activate it:

```

```
.venv\Scripts\Activate.ps1
```

### 3. Install Python dependencies

```

```
uv sync
```

### 4. Create the local environment file

```

```
Copy-Item .env.example .env
```

### 5. Create the local key configuration if required

```

```
Copy-Item .keys.example .keys
```

Do not commit:

- `.env`
- `.keys`
-  passwords
-  API keys
-  bearer tokens
-  private keys
-  certificates
-  other local secrets

### 6. Start the backend

From the repository root:

```

```
uv run uvicorn src.api:app --reload
```

### 7. Install frontend dependencies

Open another terminal:

```

```
cd frontend
npm install
```

### 8. Start the frontend

```

```
npm run dev
```

The normal Vite development address is:

```

```
http://localhost:5173
```

The backend must be running for features that require live backend data.

---

## Local Model Configuration

The example environment file contains configuration for local model and vector-store operation.

Example configuration:

```

```
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

VECTOR_DB_PATH=./data/chroma_db

LLAMA_MODEL_PATH=./models/llm/Phi-3.5-mini-instruct.Q4_K_M.gguf

SENTENCE_TRANSFORMERS_HOME=./models/embedding

TRANSFORMERS_CACHE=./models/embedding

OLLAMA_BASE_URL=http://localhost:11434

OLLAMA_MODEL=phi4-mini

OPENAI_COMPAT_BASE_URL=http://localhost:8080

OPENAI_COMPAT_MODEL=local-model

DEEPSEEK_MODEL=deepseek-chat

TOP_K_RETRIEVAL=4

LOG_LEVEL=WARN

LOG_FILE=./logs/rag_demo.log
```

### Ollama

The current example configuration uses:

```

```
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=phi4-mini
```

The required Ollama model must be available locally before using an Ollama-backed configuration.

### Local model files

Large model files should remain in the local `models/` directory and must not be committed to Git.

---

## Backend

The FastAPI backend provides the API and security enforcement layer.

The backend is responsible for areas including:

-  Authentication
-  Authorization
-  User state
-  Document processing
-  Document security analysis
-  Document provenance
-  Integrity metadata
-  Quarantine
-  Retrieval
-  Output validation
-  Audit events
-  Security telemetry
-  Rate limiting
-  Request security
-  Health information

Start the development backend with:

```

```
uv run uvicorn src.api:app --reload
```

---

## Frontend

The React frontend provides the user-facing RAGShield security workspace.

Major application areas include:

-  Dashboard
-  Security Command Center
-  Documents
-  Audit
-  User menu
-  Account controls
-  Workspace navigation
-  Document security details
-  Security notifications

The frontend communicates with the backend API for live security and operational data.

---

## Frontend Commands

Enter the frontend directory:

```

```
cd frontend
```

### Install dependencies

```

```
npm install
```

### Development server

```

```
npm run dev
```

### Production build

```

```
npm run build
```

The build command performs TypeScript project compilation through:

```

```
tsc -b
```

and then performs the Vite production build.

There is currently no separate:

```

```
npm run typecheck
```

script.

### Lint

```

```
npm run lint
```

### Preview production build

```

```
npm run preview
```

---

## Backend Testing

Run the complete backend and security regression suite from the repository root:

```

```
uv run pytest -q
```

The validated regression baseline is:

```

```
382 passed
1 skipped
6 warnings
48 subtests passed
```

The exact execution time may vary depending on the local environment.

### Phase G security tests

Focused security-hardening tests can be run with:

```

```
uv run pytest -q test_phase_g_security_hardening.py
```

The validated focused result was:

```

```
2 passed
2 warnings
```

These tests cover security-hardening behavior including:

-  Disabled-user existing-token invalidation
-  Login IP throttling before repeated authentication

---

## Frontend Validation

Run the frontend production build:

```

```
cd frontend
npm run build
```

The validated build completed successfully through:

```

```
tsc -b
vite build
```

The build generates the `dist/` directory.

The generated frontend build output is intentionally ignored by Git.

---

## Security Testing Areas

RAGShield includes security testing around several application boundaries.

### Authentication

Testing includes areas such as:

-  Password verification
-  Token validation
-  Token expiration
-  Token revocation
-  Token versioning
-  Disabled-account handling
-  Login throttling

### Authorization

Testing includes:

-  Role-based permissions
-  Protected endpoints
-  Permission enforcement
-  Document access scopes
-  Unauthorized access handling

### Request Security

Testing includes:

-  Request-size limits
-  CORS restrictions
-  Security response headers
-  Rate limiting
-  Safe authentication telemetry

### Document Security

Testing includes:

-  Prompt/instruction detection
-  Poisoning detection
-  Contradiction detection
-  PII/DLP-oriented detection
-  Quarantine behavior
-  Safe retrieval filtering
-  Provenance metadata
-  Integrity identifiers

### Operational Security

Testing and dashboard visibility include:

-  Security audit events
-  Blocked security events
-  Authentication activity
-  Authorization activity
-  Rate-limit violations
-  Security posture
-  High-severity events
-  Recent security activity

---

## Security Considerations

RAGShield is designed as a security-focused local implementation, but several controls remain development-oriented.

### Local Authentication State

The current authentication implementation uses local application state rather than a production identity provider or persistent production user database.

A production deployment should use a properly managed identity and credential system.

### Authentication Secrets

Production deployments should use stable, securely managed authentication secrets.

Development secrets and local configuration should not be reused as production secrets.

### SHA-256 Integrity Identifiers

SHA-256 values are used as document integrity identifiers.

A SHA-256 value alone does not establish:

-  Cryptographic authenticity
-  Document ownership
-  Author identity
-  Trustworthiness of document content

It identifies the content represented by the hash.

### Process-Local Rate Limiting

The current rate limiter is process-local.

A multi-instance production deployment would require shared rate-limit state or an equivalent distributed security control.

### CORS

The current CORS configuration is intentionally restricted to local development origins:

```

```
http://localhost:5173
http://127.0.0.1:5173
```

Production deployments should explicitly configure the required trusted frontend origins.

### Request Security

The backend includes request-size limits and security response headers intended to reduce common application-level abuse and unsafe request behavior.

These controls should still be reviewed and tested against the actual production deployment architecture.

---

## Current Limitations

The current implementation is local and security-focused.

The following production capabilities are not yet implemented:

-  Persistent production database
-  Production identity provider
-  Production secret-management platform
-  HTTPS
-  TLS certificate management
-  Public production domain
-  Cloud deployment
-  CI/CD pipeline
-  Centralized production monitoring
-  Centralized production logging
-  Production alerting
-  Automated backup infrastructure
-  Disaster recovery
-  Production-scale distributed rate limiting
-  Full external security assessment
-  Production penetration testing
-  Production deployment hardening
-  Public installation documentation
-  Container release artifacts
-  Public release process

These are planned future phases rather than capabilities that should be assumed to exist in the current local implementation.

---

## Development Security Practices

When developing or modifying RAGShield:

1.  Never commit passwords.
2.  Never commit API keys.
3.  Never commit bearer tokens.
4.  Never commit authentication secrets.
5.  Never commit private keys or certificates.
6.  Do not log passwords.
7.  Do not log bearer tokens.
8.  Avoid logging unnecessary raw attack payloads.
9.  Keep local runtime artifacts out of Git.
10.  Run backend regression tests after backend security changes.
11.  Run the frontend production build after frontend changes.
12.  Review staged Git changes before committing.
13.  Run `git diff --check` before committing.
14.  Treat authentication and authorization changes as regression-sensitive.
15.  Treat document-security changes as regression-sensitive.

---

## Git Hygiene

The repository intentionally ignores local and generated files such as:

```

```
.env
.keys
.venv/
data/
logs/
models/
frontend/node_modules/
frontend/dist/
__pycache__/
.pytest_cache/
.cache/
```

Before committing changes, check repository status:

```

```
git status --short
```

Review unstaged changes:

```

```
git diff
```

Check for whitespace errors:

```

```
git diff --check
```

Stage only the intended files:

```

```
git add <files>
```

Review staged changes:

```

```
git diff --cached
```

Then commit only after the staged diff has been reviewed.

---

## Roadmap

### Completed

-  Dashboard live metrics
-  User menu and account controls
-  Workspace architecture
-  Document security workspace
-  Security Command Center
-  Authentication hardening
-  Disabled-account token enforcement
-  Login rate limiting
-  API request rate limiting
-  Request-size limits
-  Security response headers
-  Local CORS restrictions
-  Security telemetry
-  Audit visibility
-  Backend regression testing
-  Frontend production build validation
-  Security-focused documentation

### Future Production Phases

#### Persistent Storage

Introduce a persistent production database and durable application state.

#### Production Authentication and Secrets

Introduce production-grade identity management, credential storage, session management, and secure secret management.

#### HTTPS / Domain / TLS

Deploy the application using HTTPS with properly managed certificates and production domain configuration.

#### Cloud Deployment

Deploy the application using appropriate production infrastructure for the backend, frontend, data, and model services.

#### CI/CD Pipeline

Introduce automated testing, security checks, frontend builds, backend builds, and controlled deployment workflows.

#### Monitoring and Production Logging

Introduce centralized monitoring, structured production logging, alerting, metrics, and operational observability.

#### Backup and Recovery

Introduce automated backups, retention policies, recovery procedures, and disaster-recovery validation.

#### Production Security Testing

Perform comprehensive deployment-specific security testing, penetration testing, and external security assessment.

#### Public Documentation

Provide complete public installation, configuration, architecture, API, security, and troubleshooting documentation.

#### Docker / Container Release

Provide reproducible container-based deployment artifacts.

#### Public GitHub Release

Prepare the repository for a public release with finalized documentation, licensing, release notes, and versioning.

#### Real-World Production Launch

Complete production infrastructure and operational readiness before public deployment.

---

## Architecture Direction

The long-term architecture separates the document security boundary from the generation boundary.

```

```
                  +----------------------+
                  |      User / Client   |
                  +----------+-----------+
                             |
                             v
                  +----------------------+
                  |   Authentication     |
                  |   Authorization      |
                  +----------+-----------+
                             |
                             v
                  +----------------------+
                  |  Document Security   |
                  |                      |
                  |  + Provenance        |
                  |  + Integrity         |
                  |  + Detection         |
                  |  + Risk / Trust      |
                  +----------+-----------+
                             |
                    +--------+--------+
                    |                 |
                    v                 v
                Trusted          Quarantine
                Documents        / Blocked
                    |
                    v
                  +----------------------+
                  |   Retrieval Layer    |
                  +----------+-----------+
                             |
                             v
                  +----------------------+
                  |     LLM / Model      |
                  +----------+-----------+
                             |
                             v
                  +----------------------+
                  | Output Validation    |
                  +----------+-----------+
                             |
                             v
                  +----------------------+
                  |       Response       |
                  +----------------------+
```

The key security objective is to prevent untrusted retrieved content from directly becoming trusted model context.

---

## Security Boundary

RAGShield treats document trust as a security boundary rather than simply a retrieval-quality problem.

The intended flow is:

```

```
Untrusted Document
       |
       v
Security Analysis
       |
       +--------------------+
       |                    |
       v                    v
   Acceptable           Suspicious
       |                    |
       v                    v
   Trusted Index        Quarantine
       |                    |
       v                    v
   Retrieval             Block
       |
       v
      LLM
       |
       v
Output Validation
```

This architecture is intended to reduce the chance that malicious instructions, poisoned content, or unsafe document material becomes trusted model context.

---

## Configuration Files

### `.env.example`

The example environment file provides non-secret configuration defaults for:

-  Embedding model
-  Vector database path
-  Local model paths
-  Sentence Transformers cache
-  Ollama
-  OpenAI-compatible local endpoints
-  DeepSeek model configuration
-  Retrieval count
-  Logging level
-  Log file location

### `.keys.example`

The example key file documents where local API keys can be configured.

Example:

```

```
DEEPSEEK_API_KEY=""
```

The actual `.keys` file is ignored by Git and must never be committed with real credentials.

---

## Local Runtime Data

RAGShield may create local runtime data including:

```

```
data/
logs/
models/
frontend/dist/
frontend/node_modules/
```

These directories are intentionally excluded from source control where appropriate.

Do not use Git to distribute local databases, logs, downloaded models, dependency directories, or generated frontend builds.

---

## Production Readiness Scope

The current project should be described as:

> **A locally runnable, production-oriented, security-focused RAG implementation.**

It should not currently be described as:

-  Fully enterprise production-ready
-  Fully production deployed
-  Cloud production ready
-  Fully audited
-  Fully penetration tested
-  Disaster-recovery ready
-  Enterprise identity integrated
-  Fully distributed

Those descriptions require the future production phases listed in the roadmap.

---

## Project Status

RAGShield currently represents a locally runnable security-focused RAG application with:

-  Document ingestion
-  Document security analysis
-  Document provenance
-  SHA-256 integrity identifiers
-  Risk scoring
-  Trust scoring
-  Document quarantine
-  Prompt-injection detection
-  Poisoning detection
-  Contradiction detection
-  PII/DLP-oriented detection
-  Safe retrieval filtering
-  Output validation
-  Authentication
-  Authorization
-  Role-based access control
-  Permission checks
-  Disabled-account enforcement
-  Token expiration and revocation controls
-  Login rate limiting
-  API rate limiting
-  Request-size limits
-  Security response headers
-  Local CORS restrictions
-  Security telemetry
-  Audit visibility
-  Security Command Center
-  Document Security Workspace
-  Backend regression coverage
-  Frontend production build validation

The project has moved beyond the original proof-of-concept stage into a more structured local security implementation.

However, the system is still intended for local development and security-focused validation until the remaining production phases are completed.

The remaining production work concerns:

-  Persistent storage
-  Production identity
-  Secret management
-  HTTPS/TLS
-  Cloud deployment
-  CI/CD
-  Monitoring
-  Production logging
-  Backup and recovery
-  External security testing
-  Public documentation
-  Container release
-  Public release
-  Real-world production deployment
