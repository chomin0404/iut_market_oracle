import { useState } from "react";
import { postBayesianUpdate } from "../api";
import type { BayesianUpdateResponse } from "../types";

const CARD = {
  background: "#161616",
  border: "1px solid #2a2a2a",
  borderRadius: 8,
  padding: 20,
  marginBottom: 16,
} as const;

const INPUT_STYLE = {
  background: "#111",
  border: "1px solid #333",
  borderRadius: 4,
  color: "#e0e0e0",
  padding: "6px 10px",
  fontSize: 13,
  width: "100%",
  fontFamily: "inherit",
  boxSizing: "border-box" as const,
};

const BTN_STYLE = {
  background: "#f59e0b",
  color: "#000",
  border: "none",
  borderRadius: 4,
  padding: "8px 20px",
  fontSize: 13,
  fontWeight: 700,
  cursor: "pointer",
  fontFamily: "inherit",
};

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div style={{ marginBottom: 10 }}>
      <div style={{ color: "#888", fontSize: 11, marginBottom: 4 }}>{label}</div>
      {children}
    </div>
  );
}

function StatCard({ label, value, color = "#f59e0b" }: { label: string; value: string; color?: string }) {
  return (
    <div
      style={{
        background: "#111",
        border: "1px solid #2a2a2a",
        borderRadius: 6,
        padding: "10px 16px",
        flex: "1 1 140px",
      }}
    >
      <div style={{ color: "#888", fontSize: 11 }}>{label}</div>
      <div style={{ color, fontSize: 17, fontWeight: 700 }}>{value}</div>
    </div>
  );
}

// Posterior distribution visualised as a simple inline bar
function PosteriorBar({
  mean,
  std,
  lo,
  hi,
}: {
  mean: number;
  std: number;
  lo: number;
  hi: number;
}) {
  const range = hi - lo || 1;
  const meanPct = ((mean - lo) / range) * 100;
  const stdLeft = ((mean - std - lo) / range) * 100;
  const stdWidth = (2 * std / range) * 100;

  return (
    <div style={{ marginTop: 16 }}>
      <div style={{ color: "#888", fontSize: 11, marginBottom: 6 }}>
        Posterior distribution (±1σ band, credible interval [{lo.toFixed(3)}, {hi.toFixed(3)}])
      </div>
      <div
        style={{
          position: "relative",
          height: 32,
          background: "#111",
          borderRadius: 4,
          overflow: "hidden",
        }}
      >
        {/* 1σ band */}
        <div
          style={{
            position: "absolute",
            left: `${Math.max(0, stdLeft)}%`,
            width: `${Math.min(100 - Math.max(0, stdLeft), stdWidth)}%`,
            height: "100%",
            background: "#78350f40",
          }}
        />
        {/* Mean line */}
        <div
          style={{
            position: "absolute",
            left: `${meanPct}%`,
            width: 2,
            height: "100%",
            background: "#f59e0b",
          }}
        />
        {/* CI endpoints */}
        <div
          style={{
            position: "absolute",
            left: "0%",
            width: 2,
            height: "100%",
            background: "#444",
          }}
        />
        <div
          style={{
            position: "absolute",
            left: "calc(100% - 2px)",
            width: 2,
            height: "100%",
            background: "#444",
          }}
        />
      </div>
      <div style={{ display: "flex", justifyContent: "space-between", marginTop: 4 }}>
        <span style={{ color: "#555", fontSize: 10 }}>{lo.toFixed(3)}</span>
        <span style={{ color: "#f59e0b", fontSize: 10 }}>mean={mean.toFixed(3)}</span>
        <span style={{ color: "#555", fontSize: 10 }}>{hi.toFixed(3)}</span>
      </div>
    </div>
  );
}

export function BayesianPage() {
  // Prior
  const [distribution, setDistribution] = useState("normal");
  const [priorMean, setPriorMean] = useState("0.0");
  const [priorStd, setPriorStd] = useState("1.0");

  // Evidence (simple list of metric=value pairs)
  const [evidenceInput, setEvidenceInput] = useState(
    "revenue_growth=0.12\nearnings_surprise=0.05\npe_ratio=18.0"
  );
  const [nSamples, setNSamples] = useState("1000");

  const [result, setResult] = useState<BayesianUpdateResponse | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  async function handleSubmit() {
    setLoading(true);
    setErr(null);
    try {
      const evidence = evidenceInput
        .split("\n")
        .map((line) => line.trim())
        .filter(Boolean)
        .map((line) => {
          const [metric, valueStr] = line.split("=");
          return { metric: metric.trim(), value: parseFloat(valueStr.trim()) };
        });

      const res = await postBayesianUpdate({
        prior: {
          distribution,
          params:
            distribution === "normal"
              ? { mean: parseFloat(priorMean), std: parseFloat(priorStd) }
              : { alpha: parseFloat(priorMean), beta: parseFloat(priorStd) },
        },
        evidence,
        n_samples: parseInt(nSamples, 10),
      });
      setResult(res);
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }

  return (
    <div>
      <h2
        style={{ color: "#fff", fontSize: 18, fontWeight: 700, margin: "0 0 20px", letterSpacing: 1 }}
      >
        Bayesian Inference — T200
      </h2>

      <div style={CARD}>
        <div
          style={{ color: "#f59e0b", fontSize: 12, fontWeight: 600, marginBottom: 14, letterSpacing: 1 }}
        >
          PRIOR SPECIFICATION
        </div>
        <div
          style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(180px, 1fr))", gap: 10 }}
        >
          <Field label="Distribution">
            <select
              style={INPUT_STYLE}
              value={distribution}
              onChange={(e) => setDistribution(e.target.value)}
            >
              <option value="normal">Normal</option>
              <option value="beta">Beta</option>
            </select>
          </Field>
          <Field label={distribution === "normal" ? "Prior mean (μ)" : "Alpha (α)"}>
            <input
              style={INPUT_STYLE}
              value={priorMean}
              onChange={(e) => setPriorMean(e.target.value)}
            />
          </Field>
          <Field label={distribution === "normal" ? "Prior std (σ)" : "Beta (β)"}>
            <input
              style={INPUT_STYLE}
              value={priorStd}
              onChange={(e) => setPriorStd(e.target.value)}
            />
          </Field>
          <Field label="MC samples">
            <input
              style={INPUT_STYLE}
              value={nSamples}
              onChange={(e) => setNSamples(e.target.value)}
            />
          </Field>
        </div>

        <div style={{ color: "#f59e0b", fontSize: 12, fontWeight: 600, margin: "14px 0 10px", letterSpacing: 1 }}>
          EVIDENCE (metric=value, one per line)
        </div>
        <textarea
          style={{ ...INPUT_STYLE, height: 100, resize: "vertical" }}
          value={evidenceInput}
          onChange={(e) => setEvidenceInput(e.target.value)}
        />

        <button style={{ ...BTN_STYLE, marginTop: 12 }} onClick={() => void handleSubmit()} disabled={loading}>
          {loading ? "Updating…" : "Run Bayesian Update"}
        </button>

        {err && (
          <div style={{ color: "#f87171", fontSize: 12, marginTop: 10 }}>Error: {err}</div>
        )}

        {result && (
          <div style={{ marginTop: 20 }}>
            <div style={{ display: "flex", gap: 12, flexWrap: "wrap" }}>
              <StatCard label="Posterior Mean" value={result.posterior.mean.toFixed(4)} />
              <StatCard label="Posterior Std" value={result.posterior.std.toFixed(4)} />
              <StatCard label="Bayes Factor" value={result.bayes_factor.toFixed(3)} />
              <StatCard label="Evidence Weight" value={result.evidence_weight.toFixed(3)} />
              <StatCard
                label="95% Credible Interval"
                value={`[${result.posterior.credible_interval[0].toFixed(3)}, ${result.posterior.credible_interval[1].toFixed(3)}]`}
              />
            </div>
            <PosteriorBar
              mean={result.posterior.mean}
              std={result.posterior.std}
              lo={result.posterior.credible_interval[0]}
              hi={result.posterior.credible_interval[1]}
            />
          </div>
        )}
      </div>
    </div>
  );
}
