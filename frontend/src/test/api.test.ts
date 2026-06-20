import { describe, it, expect, vi, beforeEach } from "vitest";
import { checkHealth, postDcf, postBayesianUpdate } from "../api";
import type { DCFRequest, BayesianUpdateRequest } from "../types";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function mockFetch(status: number, body: unknown, networkError = false) {
  const impl = networkError
    ? () => Promise.reject(new TypeError("Failed to fetch"))
    : () =>
        Promise.resolve(
          new Response(JSON.stringify(body), {
            status,
            headers: { "Content-Type": "application/json" },
          })
        );
  vi.stubGlobal("fetch", vi.fn(impl));
}

beforeEach(() => {
  vi.restoreAllMocks();
});

// ---------------------------------------------------------------------------
// checkHealth
// ---------------------------------------------------------------------------

describe("checkHealth", () => {
  it("resolves with status when API is up", async () => {
    mockFetch(200, { status: "ok" });
    const result = await checkHealth();
    expect(result).toEqual({ status: "ok" });
  });

  it("rejects when status is not ok", async () => {
    mockFetch(503, {});
    await expect(checkHealth()).rejects.toThrow("Backend unreachable");
  });
});

// ---------------------------------------------------------------------------
// postJson — tested via postDcf (representative POST wrapper)
// ---------------------------------------------------------------------------

describe("postDcf (postJson)", () => {
  const minimalReq: DCFRequest = {
    initial_fcf: 1000,
    discount_rate: 0.1,
    growth_rate: 0.05,
    terminal_growth_rate: 0.02,
    forecast_years: 5,
  };

  it("sends POST to /api/v1/valuation/dcf and returns parsed JSON", async () => {
    const payload = { intrinsic_value: 42 };
    mockFetch(200, payload);
    const result = await postDcf(minimalReq);
    expect(result).toEqual(payload);

    const fetchMock = vi.mocked(fetch);
    expect(fetchMock).toHaveBeenCalledOnce();
    const [url, init] = fetchMock.mock.calls[0] as [string, RequestInit];
    expect(url).toBe("/api/v1/valuation/dcf");
    expect(init.method).toBe("POST");
    expect(JSON.parse(init.body as string)).toEqual(minimalReq);
  });

  it("throws 'Network error' on network failure", async () => {
    mockFetch(0, null, true);
    await expect(postDcf(minimalReq)).rejects.toThrow("Network error: backend unreachable");
  });

  it("throws HTTP error message on non-2xx response", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(() =>
        Promise.resolve(
          new Response("Validation failed", {
            status: 422,
            statusText: "Unprocessable Entity",
          })
        )
      )
    );
    await expect(postDcf(minimalReq)).rejects.toThrow("422 Unprocessable Entity");
  });
});

// ---------------------------------------------------------------------------
// postBayesianUpdate
// ---------------------------------------------------------------------------

describe("postBayesianUpdate", () => {
  it("resolves with posterior summary on success", async () => {
    const payload = { mean: 0.6, variance: 0.01, ci_low: 0.5, ci_high: 0.7 };
    mockFetch(200, payload);
    const req: BayesianUpdateRequest = {
      prior: { distribution: "beta", params: { alpha: 1, beta: 1 } },
      evidence: [{ source: "obs", kind: "observation", value: 0.6, weight: 1 }],
    };
    const result = await postBayesianUpdate(req);
    expect(result).toMatchObject({ mean: 0.6 });
  });
});
