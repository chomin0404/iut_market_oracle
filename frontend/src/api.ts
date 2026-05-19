import type {
  TwinRunReport,
  ResilienceTwinReport,
  TwinRunRequest,
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
