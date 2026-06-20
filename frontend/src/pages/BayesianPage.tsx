import { useState } from "react";
import { postBayesianUpdate } from "../api";
import type { PosteriorSummary } from "../types";
import {
  Card,
  SectionLabel,
  Field,
  Input,
  Select,
  Textarea,
  Button,
  ErrorBox,
  StatCard,
  StatsRow,
  PageHeading,
  FormGrid,
} from "../components/ui";
import { colors, spacing } from "../styles/tokens";

// ---- Posterior distribution bar ----

function PosteriorBar({ posterior }: { posterior: PosteriorSummary }) {
  const [lo, hi] = posterior.credible_interval_95;
  const mean = posterior.mean;
  const std = Math.sqrt(posterior.variance);
  const range = hi - lo || 1;
  const meanPct = Math.max(0, Math.min(100, ((mean - lo) / range) * 100));
  const stdLeft = Math.max(0, ((mean - std - lo) / range) * 100);
  const stdWidth = Math.min(100 - stdLeft, (2 * std / range) * 100);

  return (
    <div style={{ marginTop: spacing.lg }}>
      <div style={{ color: colors.textMuted, fontSize: 11, marginBottom: 6 }}>
        Posterior ({posterior.n_evidence} evidence items) — 95% CI [{lo.toFixed(4)}, {hi.toFixed(4)}]
      </div>
      <div
        style={{
          position: "relative",
          height: 32,
          background: colors.surface0,
          borderRadius: 4,
          overflow: "hidden",
        }}
      >
        <div
          style={{
            position: "absolute",
            left: `${stdLeft}%`,
            width: `${stdWidth}%`,
            height: "100%",
            background: `${colors.accent.amber}22`,
          }}
        />
        <div
          style={{
            position: "absolute",
            left: `${meanPct}%`,
            width: 2,
            height: "100%",
            background: colors.accent.amber,
          }}
        />
      </div>
      <div style={{ display: "flex", justifyContent: "space-between", marginTop: 4 }}>
        <span style={{ color: colors.textDim, fontSize: 10 }}>{lo.toFixed(4)}</span>
        <span style={{ color: colors.accent.amber, fontSize: 10 }}>
          μ={mean.toFixed(4)}, σ²={posterior.variance.toFixed(6)}
        </span>
        <span style={{ color: colors.textDim, fontSize: 10 }}>{hi.toFixed(4)}</span>
      </div>
    </div>
  );
}

// ---- Evidence format help ----

const EVIDENCE_PLACEHOLDER = `# One item per line: source=value  (kind defaults to "observation")
# Examples:
q1_return=0.07
analyst_target=0.12
backtest_sharpe=1.4`;

// ---- Page ----

export function BayesianPage() {
  const [distribution, setDistribution] = useState("normal");
  const [paramMu, setParamMu] = useState("0.05");
  const [paramSigma, setParamSigma] = useState("0.02");
  const [evidenceText, setEvidenceText] = useState(
    "q1_return=0.07\nanalyst_target=0.12\nmarket_data=0.09"
  );
  const [result, setResult] = useState<PosteriorSummary | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  function buildParams(): Record<string, number> {
    if (distribution === "beta") {
      return { alpha: parseFloat(paramMu), beta: parseFloat(paramSigma) };
    }
    return { mu: parseFloat(paramMu), sigma: parseFloat(paramSigma) };
  }

  function parseEvidence() {
    return evidenceText
      .split("\n")
      .map((line) => line.trim())
      .filter((line) => line && !line.startsWith("#"))
      .map((line) => {
        const eqIdx = line.lastIndexOf("=");
        const source = line.slice(0, eqIdx).trim();
        const value = parseFloat(line.slice(eqIdx + 1).trim());
        return { source, kind: "observation" as const, value, weight: 1.0 };
      });
  }

  async function handleSubmit() {
    setLoading(true);
    setErr(null);
    try {
      const evidence = parseEvidence();
      if (evidence.length === 0) throw new Error("No evidence items parsed.");
      const res = await postBayesianUpdate({
        prior: { distribution, params: buildParams() },
        evidence,
      });
      setResult(res);
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }

  const paramLabel1 = distribution === "beta" ? "Alpha (α)" : "Prior mean (μ)";
  const paramLabel2 = distribution === "beta" ? "Beta (β)" : "Prior std (σ)";

  return (
    <div>
      <PageHeading subtitle="T200 — Bayesian conjugate update (Beta / Normal) from observation evidence">
        Bayesian Inference
      </PageHeading>

      <Card>
        <SectionLabel color={colors.accent.amber}>PRIOR SPECIFICATION</SectionLabel>
        <FormGrid>
          <Field label="Distribution family">
            <Select value={distribution} onChange={(e) => setDistribution(e.target.value)}>
              <option value="normal">Normal</option>
              <option value="beta">Beta</option>
            </Select>
          </Field>
          <Field label={paramLabel1}>
            <Input value={paramMu} onChange={(e) => setParamMu(e.target.value)} />
          </Field>
          <Field label={paramLabel2}>
            <Input value={paramSigma} onChange={(e) => setParamSigma(e.target.value)} />
          </Field>
        </FormGrid>

        <SectionLabel color={colors.accent.amber}>
          EVIDENCE (source=value, one per line)
        </SectionLabel>
        <Textarea
          rows={6}
          value={evidenceText}
          onChange={(e) => setEvidenceText(e.target.value)}
          placeholder={EVIDENCE_PLACEHOLDER}
          style={{ marginBottom: spacing.md }}
        />

        <Button
          accent={colors.accent.amber}
          loading={loading}
          loadingLabel="Updating…"
          onClick={() => void handleSubmit()}
        >
          Run Bayesian Update
        </Button>

        {err && <ErrorBox message={err} />}

        {result && (
          <>
            <StatsRow>
              <StatCard
                label="Posterior Mean"
                value={result.mean.toFixed(5)}
                color={colors.accent.amber}
              />
              <StatCard
                label="Posterior Variance"
                value={result.variance.toFixed(7)}
                color={colors.accent.amber}
              />
              <StatCard
                label="95% CI"
                value={`[${result.credible_interval_95[0].toFixed(4)}, ${result.credible_interval_95[1].toFixed(4)}]`}
                color={colors.accent.amber}
              />
              <StatCard
                label="Evidence Count"
                value={result.n_evidence.toString()}
                color={colors.accent.amber}
              />
            </StatsRow>
            <PosteriorBar posterior={result} />
          </>
        )}
      </Card>
    </div>
  );
}
