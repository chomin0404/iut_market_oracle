import type {
  DCFRequest,
  DCFResponse,
  ReverseDCFRequest,
  ReverseDCFResponse,
  BayesianUpdateRequest,
  PosteriorSummary,
  EntropyDetectRequest,
  EntropyReport,
  GraphInput,
  PortfolioMetrics,
  YieldTwinRequest,
  DOERecommendation,
  StrategyRunRequest,
  StrategyTwinReport,
} from "./types";

/** All API routes are under /api/v1 on the FastAPI server.
 *  The Vite dev proxy forwards /api/* → http://localhost:8000/api/*
 *  (path kept as-is, no rewrite).
 */
const BASE = "/api/v1";

async function postJson<T>(path: string, body: unknown): Promise<T> {
  let res: Response;
  try {
    res = await fetch(`${BASE}${path}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
  } catch {
    throw new Error("Network error: backend unreachable");
  }
  if (!res.ok) {
    const err = await res.text().catch(() => "");
    throw new Error(`${res.status} ${res.statusText}: ${err}`);
  }
  return res.json() as Promise<T>;
}

/** Health check — /health is at FastAPI root (not under /api/v1). */
export async function checkHealth(): Promise<{ status: string }> {
  const res = await fetch("/health");
  if (!res.ok) throw new Error("Backend unreachable");
  return res.json() as Promise<{ status: string }>;
}

// ---- Valuation ----

export function postDcf(req: DCFRequest): Promise<DCFResponse> {
  return postJson<DCFResponse>("/valuation/dcf", req);
}

export function postReverseDcf(req: ReverseDCFRequest): Promise<ReverseDCFResponse> {
  return postJson<ReverseDCFResponse>("/valuation/dcf/reverse", req);
}

// ---- Bayesian ----

export function postBayesianUpdate(req: BayesianUpdateRequest): Promise<PosteriorSummary> {
  return postJson<PosteriorSummary>("/bayesian/update", req);
}

// ---- Entropy ----

export function postEntropyDetect(req: EntropyDetectRequest): Promise<EntropyReport> {
  return postJson<EntropyReport>("/entropy/detect", req);
}

// ---- Graph (T300) ----

export function postGraphMetrics(req: GraphInput): Promise<PortfolioMetrics> {
  return postJson<PortfolioMetrics>("/graph/metrics", req);
}

// ---- Yield Twin (T1600) ----

export function postYieldTwinRecommend(req: YieldTwinRequest): Promise<DOERecommendation> {
  return postJson<DOERecommendation>("/yield-twin/recommend", req);
}

// ---- Strategy Twin (T1700) ----

export function postStrategyRun(req: StrategyRunRequest): Promise<StrategyTwinReport> {
  return postJson<StrategyTwinReport>("/strategy/run", req);
}
