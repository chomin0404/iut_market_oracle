import type {
  TwinRunReport,
  ResilienceTwinReport,
  TwinRunRequest,
  DCFRequest,
  DCFResponse,
  ReverseDCFRequest,
  ReverseDCFResponse,
  BayesianUpdateRequest,
  BayesianUpdateResponse,
  EntropyRequest,
  EntropyReport,
} from "./types";

const BASE = "/api";

async function postJson<T>(path: string, body: unknown): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const err = await res.text();
    throw new Error(`${res.status} ${res.statusText}: ${err}`);
  }
  return res.json() as Promise<T>;
}

export async function checkHealth(): Promise<{ status: string }> {
  const res = await fetch(`${BASE}/health`);
  if (!res.ok) throw new Error("Backend unreachable");
  return res.json() as Promise<{ status: string }>;
}

export function postTwinRun(req: TwinRunRequest): Promise<TwinRunReport> {
  return postJson<TwinRunReport>("/gnss/twin/run", req);
}

export interface ResilienceSimParams {
  n_mc: number;
  n_epochs: number;
  n_sats: number;
  doppler_noise_std: number;
  spoof_bias_std: number;
  spoof_diff_std: number;
  graph_sigma: number;
  dirichlet_alpha: number;
  random_seed: number;
}

export function postResilienceSim(
  params: ResilienceSimParams
): Promise<ResilienceTwinReport> {
  return postJson<ResilienceTwinReport>("/gnss/resilience-sim", params);
}

// ---- Valuation ----

export function postDcf(req: DCFRequest): Promise<DCFResponse> {
  return postJson<DCFResponse>("/valuation/dcf", req);
}

export function postReverseDcf(req: ReverseDCFRequest): Promise<ReverseDCFResponse> {
  return postJson<ReverseDCFResponse>("/valuation/reverse-dcf", req);
}

// ---- Bayesian ----

export function postBayesianUpdate(
  req: BayesianUpdateRequest
): Promise<BayesianUpdateResponse> {
  return postJson<BayesianUpdateResponse>("/bayesian/update", req);
}

// ---- Entropy ----

export function postEntropyDetect(req: EntropyRequest): Promise<EntropyReport> {
  return postJson<EntropyReport>("/entropy/detect", req);
}
