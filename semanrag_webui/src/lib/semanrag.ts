/** SemanRAG API client — typed fetch wrapper with SSE streaming support. */

const BASE = import.meta.env.VITE_API_BASE ?? "";

function getAuthHeaders(): Record<string, string> {
  const token = localStorage.getItem("semanrag_token");
  if (token) return { Authorization: `Bearer ${token}` };
  const apiKey = localStorage.getItem("semanrag_api_key");
  if (apiKey) return { "X-API-Key": apiKey };
  return {};
}

export class ApiError extends Error {
  constructor(public status: number, message: string) {
    super(message);
    this.name = "ApiError";
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const headers = new Headers(init?.headers);
  if (!headers.has("Content-Type") && !(init?.body instanceof FormData)) {
    headers.set("Content-Type", "application/json");
  }
  for (const [k, v] of Object.entries(getAuthHeaders())) headers.set(k, v);

  const res = await fetch(`${BASE}${path}`, { ...init, headers });
  if (!res.ok) {
    const body = await res.json().catch(() => ({ detail: res.statusText }));
    throw new ApiError(res.status, (body as { detail?: string }).detail ?? "Request failed");
  }
  if (res.status === 204) return undefined as T;
  return res.json() as Promise<T>;
}

// ── Query ────────────────────────────────────────────────────────────

export interface QueryRequest {
  query: string;
  mode?: string;
  top_k?: number;
  stream?: boolean;
  conversation_history?: { role: string; content: string }[];
  snapshot_at?: string | null;
  user_id?: string | null;
  user_groups?: string[];
  response_type?: string;
  enable_rerank?: boolean;
  verifier_enabled?: boolean;
}

export interface QueryResponse {
  answer: string;
  references: Record<string, unknown>[];
  communities_used: string[];
  latency_ms: number;
  tokens_used: Record<string, number>;
}

export interface GroundedClaim {
  claim: string;
  status: "supported" | "partial" | "unsupported";
  evidence?: string;
}

export interface QueryDataResponse {
  answer: string;
  references: Record<string, unknown>[];
  grounded_check: GroundedClaim[];
  latency_ms: number;
}

export interface CompareRequest {
  query: string;
  variant_a: { mode: string; top_k: number; enable_rerank: boolean; response_type: string };
  variant_b: { mode: string; top_k: number; enable_rerank: boolean; response_type: string };
  user_id?: string | null;
  user_groups?: string[];
}

export interface CompareResponse {
  query: string;
  result_a: QueryResponse;
  result_b: QueryResponse;
}

export interface StreamEvent {
  chunk?: string;
  done?: boolean;
  references?: Record<string, unknown>[];
  latency_ms?: number;
}

export const queryApi = {
  query: (body: QueryRequest) =>
    request<QueryResponse>("/query", { method: "POST", body: JSON.stringify(body) }),

  queryData: (body: QueryRequest) =>
    request<QueryDataResponse>("/query/data", { method: "POST", body: JSON.stringify(body) }),

  compare: (body: CompareRequest) =>
    request<CompareResponse>("/query/compare", { method: "POST", body: JSON.stringify(body) }),

  async *stream(body: QueryRequest): AsyncGenerator<StreamEvent> {
    const headers: Record<string, string> = { "Content-Type": "application/json", ...getAuthHeaders() };
    const res = await fetch(`${BASE}/query/stream`, {
      method: "POST",
      headers,
      body: JSON.stringify({ ...body, stream: true }),
    });
    if (!res.ok || !res.body) throw new ApiError(res.status, "Stream failed");
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buf = "";
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });
      const lines = buf.split("\n");
      buf = lines.pop() ?? "";
      for (const line of lines) {
        if (line.startsWith("data: ")) {
          yield JSON.parse(line.slice(6)) as StreamEvent;
        }
      }
    }
  },
};

// ── Graph ────────────────────────────────────────────────────────────

export interface GraphData {
  nodes: Record<string, unknown>[];
  edges: Record<string, unknown>[];
}

export interface CommunityInfo {
  community_id: string;
  summary?: string;
  members: string[];
}

export const graphApi = {
  getGraph: (params?: { snapshot_at?: string; community_level?: number }) => {
    const qs = new URLSearchParams();
    if (params?.snapshot_at) qs.set("snapshot_at", params.snapshot_at);
    if (params?.community_level != null) qs.set("community_level", String(params.community_level));
    return request<GraphData>(`/graph?${qs.toString()}`);
  },
  getLabels: () => request<{ labels: string[] }>("/graph/labels"),
  getNeighborhood: (name: string, hops = 1) =>
    request<GraphData>(`/graph/neighborhood/${encodeURIComponent(name)}?hops=${hops}`),
  getPath: (src: string, tgt: string) =>
    request<{ path: string[]; length: number }>(`/graph/path?src=${encodeURIComponent(src)}&tgt=${encodeURIComponent(tgt)}`),
  getCommunities: () => request<{ communities: CommunityInfo[] }>("/graph/communities"),
  createEntity: (body: { name: string; type?: string; description?: string }) =>
    request<unknown>("/graph/entities", { method: "POST", body: JSON.stringify(body) }),
  updateEntity: (name: string, body: Record<string, unknown>) =>
    request<unknown>(`/graph/entities/${encodeURIComponent(name)}`, { method: "PUT", body: JSON.stringify(body) }),
  deleteEntity: (name: string) =>
    request<unknown>(`/graph/entities/${encodeURIComponent(name)}`, { method: "DELETE" }),
  mergeEntities: (body: { source_entities: string[]; target_entity: string; merge_strategy?: string }) =>
    request<unknown>("/graph/entities/merge", { method: "POST", body: JSON.stringify(body) }),
};

// ── Documents ────────────────────────────────────────────────────────

export interface DocSummary {
  id: string;
  status: string;
  content_length: number;
  chunks_count: number;
  created_at: string;
  file_path: string;
}

export interface DocListResponse {
  documents: DocSummary[];
  total: number;
  offset: number;
  limit: number;
}

export interface DocDetail {
  id: string;
  status: string;
  content: string;
  content_length: number;
  chunks_count: number;
  chunks_list: string[];
  created_at: string;
  updated_at: string;
  file_path: string;
  pii_findings: Record<string, unknown>[];
  prompt_injection_flags: Record<string, unknown>[];
  acl_policy?: { owner: string; visible_to_groups: string[]; visible_to_users: string[]; public: boolean } | null;
  version: number;
  error_message: string;
}

export interface PipelineStatus {
  status_counts: Record<string, number>;
  pending: number;
  processing: number;
  completed: number;
  failed: number;
}

export const documentsApi = {
  list: (params: { offset?: number; limit?: number; status?: string }) => {
    const qs = new URLSearchParams();
    if (params.offset != null) qs.set("offset", String(params.offset));
    if (params.limit != null) qs.set("limit", String(params.limit));
    if (params.status) qs.set("status", params.status);
    return request<DocListResponse>(`/documents?${qs.toString()}`);
  },
  get: (id: string) => request<DocDetail>(`/documents/${encodeURIComponent(id)}`),
  delete: (id: string) => request<unknown>(`/documents/${encodeURIComponent(id)}`, { method: "DELETE" }),
  reingest: (id: string) => request<unknown>(`/documents/${encodeURIComponent(id)}/reingest`, { method: "POST" }),
  upload: async (file: File) => {
    const form = new FormData();
    form.append("file", file);
    const res = await fetch(`${BASE}/documents/upload`, { method: "POST", headers: getAuthHeaders(), body: form });
    if (!res.ok) throw new ApiError(res.status, "Upload failed");
    return res.json() as Promise<unknown>;
  },
  inboxUpload: async (file: File) => {
    const form = new FormData();
    form.append("file", file);
    const res = await fetch(`${BASE}/documents/inbox/upload`, { method: "POST", headers: getAuthHeaders(), body: form });
    if (!res.ok) throw new ApiError(res.status, "Inbox copy failed");
    return res.json() as Promise<{ file: string; status: string; path: string }>;
  },
  inboxList: () => request<{ files: { file: string; size: number }[] }>("/documents/inbox"),
  inboxScan: () => request<{ files: string[]; message: string }>("/documents/inbox/scan", { method: "POST" }),
  pipelineStatus: () => request<PipelineStatus>("/documents/pipeline-status"),
};

// ── Admin ────────────────────────────────────────────────────────────

export const adminApi = {
  getUsers: () => request<{ users: string[] }>("/admin/users"),
  getGroups: () => request<{ groups: string[] }>("/admin/groups"),
  configureBudget: (body: { max_tokens_per_user_per_day: number; max_tokens_per_workspace_per_day: number }) =>
    request<unknown>("/admin/budget", { method: "POST", body: JSON.stringify(body) }),
  getCostReport: () =>
    request<{ user_usage: Record<string, number>; workspace_usage: Record<string, number> }>("/admin/cost-report"),
  getAuditLog: (params: { offset?: number; limit?: number }) => {
    const qs = new URLSearchParams();
    if (params.offset != null) qs.set("offset", String(params.offset));
    if (params.limit != null) qs.set("limit", String(params.limit));
    return request<{ entries: Record<string, unknown>[]; total: number }>(`/admin/audit-log?${qs.toString()}`);
  },
  runEval: (body: { dataset?: string; metrics?: string[] }) =>
    request<unknown>("/admin/eval/run", { method: "POST", body: JSON.stringify(body) }),
  getEvalHistory: () => request<{ runs: Record<string, unknown>[] }>("/admin/eval/history"),
  getPiiReport: () =>
    request<{ total_docs_with_pii: number; documents: Record<string, unknown>[] }>("/admin/pii-report"),
  purgeCache: (scope: string) =>
    request<unknown>("/admin/cache/purge", { method: "POST", body: JSON.stringify({ scope }) }),
};

// ── Feedback ─────────────────────────────────────────────────────────

export interface FeedbackRequest {
  query_id: string;
  thumbs: "up" | "down";
  rating?: { relevance: number; accuracy: number; faithfulness: number };
  comment?: string;
}

export const feedbackApi = {
  submit: (body: FeedbackRequest) =>
    request<unknown>("/feedback", { method: "POST", body: JSON.stringify(body) }),
};

// ── Auth ──────────────────────────────────────────────────────────────

export const authApi = {
  login: (username: string, password: string) =>
    request<{ access_token: string }>("/auth/login", {
      method: "POST",
      body: JSON.stringify({ username, password }),
    }),
};
