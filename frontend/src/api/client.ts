import type {
  ApiError,
  AttackRequest,
  AttackResponse,
  AuditResponse,
  HealthResponse,
  InfoResponse,
  QueryRequest,
  QueryResponse,
  SetupRequest,
  SetupResponse,
} from "../types/api";

const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL ??
  "http://127.0.0.1:8000";

const TOKEN_STORAGE_KEY =
  "ragshield_access_token";

export type UploadedDocument = {
  document_id: string;
  source: string;
  status: string;
  indexed: boolean;
  quarantined: boolean;
  size_bytes: number;
  content_sha256: string;
  provenance_version: string;
  poison_score: number;
  poison_detected: boolean;
  contradiction_score: number;
  contradiction_detected: boolean;
  injection_score: number;
  injection_detected: boolean;
  dlp: {
    score: number;
    has_pii: boolean;
    finding_count: number;
    categories: string[];
  };
  risk_score: number;
  trust_score: number;
  classification: string;
  detectors: string[];
  reasons: string[];
  content_preview: string;
};

export type DocumentInventoryItem =
  Omit<
    UploadedDocument,
    | "indexed"
    | "quarantined"
    | "dlp"
    | "content_preview"
  > & {
    extension: string;
    uploaded_at: string;
    metadata_sha256: string;
    dlp_score: number;
    dlp_detected: boolean;
  };

export class ApiClientError extends Error {
  status: number;
  detail: string;

  constructor(
    status: number,
    detail: string,
  ) {
    super(detail);
    this.name = "ApiClientError";
    this.status = status;
    this.detail = detail;
  }
}

function getStoredToken(): string | null {
  return localStorage.getItem(
    TOKEN_STORAGE_KEY,
  );
}

export function setApiToken(
  token: string,
): void {
  const normalizedToken =
    token.trim();

  if (!normalizedToken) {
    localStorage.removeItem(
      TOKEN_STORAGE_KEY,
    );
    return;
  }

  localStorage.setItem(
    TOKEN_STORAGE_KEY,
    normalizedToken,
  );
}

export function clearApiToken(): void {
  localStorage.removeItem(
    TOKEN_STORAGE_KEY,
  );
}

export function hasApiToken(): boolean {
  return Boolean(
    getStoredToken(),
  );
}

async function parseError(
  response: Response,
): Promise<ApiClientError> {
  let detail =
    `Request failed with status ${response.status}.`;

  try {
    const body =
      (await response.json()) as ApiError;

    if (
      typeof body.detail ===
      "string"
    ) {
      detail = body.detail;
    } else if (
      typeof body.message ===
      "string"
    ) {
      detail = body.message;
    }
  } catch {
    // Keep the safe generic message.
  }

  return new ApiClientError(
    response.status,
    detail,
  );
}

async function request<T>(
  path: string,
  options: RequestInit = {},
): Promise<T> {
  const token =
    getStoredToken();

  const headers =
    new Headers(
      options.headers,
    );

  headers.set(
    "Accept",
    "application/json",
  );

  if (options.body) {
    headers.set(
      "Content-Type",
      "application/json",
    );
  }

  if (token) {
    headers.set(
      "Authorization",
      `Bearer ${token}`,
    );
  }

  const response =
    await fetch(
      `${API_BASE_URL}${path}`,
      {
        ...options,
        headers,
      },
    );

  if (!response.ok) {
    throw await parseError(
      response,
    );
  }

  return (await response.json()) as T;
}

export async function getHealth(): Promise<HealthResponse> {
  return request<HealthResponse>(
    "/health",
  );
}

export async function getInfo(): Promise<InfoResponse> {
  return request<InfoResponse>(
    "/info",
  );
}

export async function executeQuery(
  query: string,
): Promise<QueryResponse> {
  const body: QueryRequest = {
    query,
  };

  return request<QueryResponse>(
    "/query",
    {
      method: "POST",
      body: JSON.stringify(body),
    },
  );
}

export async function analyzeAttack(
  payload: string,
): Promise<AttackResponse> {
  const body: AttackRequest = {
    payload,
  };

  return request<AttackResponse>(
    "/attack",
    {
      method: "POST",
      body: JSON.stringify(body),
    },
  );
}

export async function setupCorpus(
  includePoison: boolean,
  payload?: string,
): Promise<SetupResponse> {
  const body: SetupRequest = {
    include_poison:
      includePoison,
    ...(payload
      ? { payload }
      : {}),
  };

  return request<SetupResponse>(
    "/setup",
    {
      method: "POST",
      body: JSON.stringify(body),
    },
  );
}

export async function getAudit(): Promise<AuditResponse> {
  return request<AuditResponse>(
    "/audit",
  );
}

export async function clearAudit(): Promise<{
  status: string;
  message: string;
  removed_events: number;
}> {
  return request<{
    status: string;
    message: string;
    removed_events: number;
  }>(
    "/audit",
    {
      method: "DELETE",
    },
  );
}

export async function listDocuments(): Promise<{
  documents: DocumentInventoryItem[];
}> {
  return request<{
    documents: DocumentInventoryItem[];
  }>(
    "/documents",
    {
      method: "GET",
    },
  );
}

export async function uploadDocument(
  file: File,
): Promise<UploadedDocument> {
  const token =
    getStoredToken();

  if (!token) {
    throw new ApiClientError(
      401,
      "Authentication required.",
    );
  }

  const formData =
    new FormData();

  formData.append(
    "file",
    file,
  );

  const headers =
    new Headers();

  headers.set(
    "Accept",
    "application/json",
  );

  headers.set(
    "Authorization",
    `Bearer ${token}`,
  );

  const response =
    await fetch(
      `${API_BASE_URL}/documents/upload`,
      {
        method: "POST",
        headers,
        body: formData,
      },
    );

  if (!response.ok) {
    throw await parseError(
      response,
    );
  }

  return (await response.json()) as UploadedDocument;
}

export {
  API_BASE_URL,
  TOKEN_STORAGE_KEY,
};
