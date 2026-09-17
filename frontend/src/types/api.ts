export interface HealthResponse {
  status: string;
  service: string;
  timestamp: number;
}

export interface InfoResponse {
  name: string;
  version: string;
  llm_provider: string;
  llm_model: string;
  embedding_model: string;
}

export interface DocumentResponse {
  source: string;
  document_type: string;
  content: string;
  poison_score: number | null;
  poison_detected: boolean;
  contradiction_score: number | null;
  contradiction_detected: boolean;
  reasons: string[];
  status: string;
}

export interface SecurityEvent {
  source: string;
  document_type: string;
  detector: string;
  score: number;
  is_poisoned: boolean;
  is_contradictory: boolean;
  reasons: string[];
  status: string;
}

export interface SecurityResponse {
  status: string;
  poison_detected: boolean;
  contradiction_detected: boolean;
  blocked_count: number;
  events: SecurityEvent[];
}

export interface QueryRequest {
  query: string;
}

export interface QueryResponse {
  query: string;
  answer: string;
  security: SecurityResponse;
  retrieved_documents: DocumentResponse[];
  blocked_documents: DocumentResponse[];
}

export interface AttackRequest {
  payload: string;
}

export interface AttackResponse {
  status: string;
  message: string;
  payload: string;
  poison_score: number;
  poison_detected: boolean;
  contradiction_score: number;
  contradiction_detected: boolean;
  blocked_by: string[];
  reasons: string[];
}

export interface AuditEvent {
  event_id: string;
  timestamp: string;
  event_type: string;
  status: string;
  query: string | null;
  payload: string | null;
  source: string | null;
  detector: string | null;
  score: number;
  reasons: string[];
  request_id: string | null;
  username: string | null;
  category: string | null;
  reason_code: string | null;
  severity: string | null;
  endpoint: string | null;
}

export interface AuditResponse {
  total: number;
  events: AuditEvent[];
}

export interface SetupRequest {
  include_poison: boolean;
  payload?: string;
}

export interface SetupResponse {
  status: string;
  message: string;
  trusted_documents: number;
  poisoned_document: boolean;
}

export interface ApiError {
  detail?: string;
  message?: string;
}