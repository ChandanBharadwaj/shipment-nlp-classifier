// api.ts — thin fetch wrapper for the FastAPI backend.
//
// Two responsibilities only:
//   1. Concentrate URL construction so views never hardcode "/admin/api".
//      If the prefix changes, it changes here.
//   2. Inject `actor` from the operator store into every write payload so
//      view code can't accidentally forget the audit field. The backend
//      validates it again — this is convenience, not security.
//
// Errors are normalised: any non-2xx surfaces as `ApiError` with the
// status and the response body (parsed when JSON, raw otherwise) so
// views can show a real message instead of "TypeError: Failed to fetch".
import { useOperatorStore } from "@/stores/operator";
import type {
  Overview, Category, KeywordPage, KeywordByToken,
  Collision, AuditPage, DiscoverResult,
} from "@/types";

const API_PREFIX = "/admin/api";

export class ApiError extends Error {
  status: number;
  body: unknown;
  constructor(status: number, body: unknown, message?: string) {
    super(message ?? `HTTP ${status}`);
    this.name = "ApiError";
    this.status = status;
    this.body = body;
  }
}

async function parseBody(resp: Response): Promise<unknown> {
  const ct = resp.headers.get("content-type") ?? "";
  if (ct.includes("application/json")) {
    try { return await resp.json(); } catch { return null; }
  }
  return await resp.text();
}

async function request<T>(
  method: string,
  path: string,
  opts: { params?: Record<string, unknown>; body?: unknown; injectActor?: boolean } = {},
): Promise<T> {
  const url = new URL(path.startsWith("http") ? path : path, window.location.origin);
  if (opts.params) {
    for (const [k, v] of Object.entries(opts.params)) {
      if (v === undefined || v === null || v === "") continue;
      url.searchParams.set(k, String(v));
    }
  }

  const init: RequestInit = { method, headers: {} };
  if (opts.body !== undefined) {
    let body = opts.body as Record<string, unknown> | unknown;
    if (opts.injectActor && body && typeof body === "object" && !("actor" in (body as object))) {
      const op = useOperatorStore();
      body = { ...(body as Record<string, unknown>), actor: op.name };
    }
    (init.headers as Record<string, string>)["Content-Type"] = "application/json";
    init.body = JSON.stringify(body);
  }

  const resp = await fetch(url.toString(), init);
  const parsed = await parseBody(resp);
  if (!resp.ok) {
    throw new ApiError(resp.status, parsed, `${method} ${path} → ${resp.status}`);
  }
  return parsed as T;
}

// ── Read endpoints ───────────────────────────────────────────────────────

export const api = {
  // Admin reads
  overview:    ()                              => request<Overview>("GET",  `${API_PREFIX}/overview`),
  categories:  ()                              => request<Category[]>("GET",  `${API_PREFIX}/categories`),
  keywords:    (params: Record<string, unknown> = {}) =>
                                                 request<KeywordPage>("GET",  `${API_PREFIX}/keywords`, { params }),
  keywordByToken: (token: string)              => request<KeywordByToken>("GET",  `${API_PREFIX}/keywords/by-token/${encodeURIComponent(token)}`),
  collisions:  ()                              => request<Collision[]>("GET",  `${API_PREFIX}/collisions`),
  collision:   (token: string)                 => request<Collision>("GET",  `${API_PREFIX}/collisions/${encodeURIComponent(token)}`),
  auditLog:    (params: Record<string, unknown> = {}) =>
                                                 request<AuditPage>("GET",  `${API_PREFIX}/audit-log`, { params }),
  discover:    (params: Record<string, unknown> = {}) =>
                                                 request<DiscoverResult>("POST", `${API_PREFIX}/discover`, { params }),

  // Bulk classify (existing endpoint, NOT under /admin)
  classifyBatch: (shipments: unknown[]) =>
    request<unknown>("POST", "/classify/batch", { body: { shipments } }),
};

export default api;
