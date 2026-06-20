import { useState } from "react";
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
} from "recharts";
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

// ---- Normal PDF approximation ----
// The API returns mean/variance/CI only, so we approximate the posterior as Normal.
function normalPdf(x: number, mu: number, sigma: number): number {
  const z = (x - mu) / sigma;
  return Math.exp(-0.5 * z * z) / (sigma * Math.sqrt(2 * Math.PI));
}

function buildPdfCurve(
  mean: number,
  variance: number,
  lo: number,
  hi: number,
  nPoints = 120
): Array<{ x: number; density: number }> {
  const sigma = Math.sqrt(Math.max(variance, 1e-10));
  const margin = (hi - lo) * 0.15;
  const xMin = lo - margin;
  const xMax = hi + margin;
  const step = (xMax - xMin) / (nPoints - 1);
  return Array.from({ length: nPoints }, (_, i) => {
    const x = xMin + i * step;
    return { x: parseFloat(x.toFixed(6)), density: normalPdf(x, mean, sigma) };
  });
}

// ---- Posterior chart ----

function PosteriorChart({ posterior }: { posterior: PosteriorSummary }) {
  const [lo, hi] = posterior.credible_interval_95;
  const mean = posterior.mean;
  const variance = posterior.variance;
  const curve = buildPdfCurve(mean, variance, lo, hi);

  return (
    <div style={{ marginTop: spacing.lg }}>
      <div style={{ color: colors.textMuted, fontSize: 11, marginBottom: 6 }}>
        Posterior distribution (Normal approximation) — 95% CI [{lo.toFixed(4)}, {hi.toFixed(4)}]
      </div>
      {/* improvement 5: AreaChart instead of custom div bar */}
      <ResponsiveContainer width="100%" height={180}>
        <AreaChart data={curve} margin={{ top: 8, right: 8, left: 0, bottom: 4 }}>
          <defs>
            <linearGradient id="posteriorGrad" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor={colors.accent.amber} stopOpacity={0.35} />
              <stop offset="95%" stopColor={colors.accent.amber} stopOpacity={0.04} />
            </linearGradient>
          </defs>
          <XAxis
            dataKey="x"
            tick={{ fill: colors.textDim, fontSize: 10 }}
            axisLine={false}
            tickLine={false}
            tickFormatter={(v: number) => v.toFixed(3)}
          />
          <YAxis hide />
          <Tooltip
            contentStyle={{
              background: colors.surface0,
              border: `1px solid ${colors.border}`,
              fontSize: 11,
            }}
            labelStyle={{ color: colors.textMuted }}
            formatter={(v: number) => [v.toFixed(5), "density"]}
            labelFormatter={(v: number) => `x = ${Number(v).toFixed(4)}`}
          />
          {/* 95% CI boundaries */}
          <ReferenceLine x={lo} stroke={colors.accent.amberDark} strokeDasharray="3 3" strokeOpacity={0.8} />
          <ReferenceLine x={hi} stroke={colors.accent.amberDark} strokeDasharray="3 3" strokeOpacity={0.8} />
          {/* Posterior mean */}
          <ReferenceLine
            x={mean}
            stroke={colors.accent.amber}
            strokeWidth={1.5}
            label={
              <text fill={colors.accent.amber} fontSize={10} dy={-4}>
                μ
              </text>
            }
          />
          <Area
            type="monotone"
            dataKey="density"
            stroke={colors.accent.amber}
            strokeWidth={1.5}
            fill="url(#posteriorGrad)"
            dot={false}
            name="Density"
          />
        </AreaChart>
      </ResponsiveContainer>
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
      .filter((line) => line && !line.startsWith("#") && line.includes("="))
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
        {/* improvement 3: Enter key submits */}
        <form
          onSubmit={(e) => {
            e.preventDefault();
            void handleSubmit();
          }}
        >
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
            type="submit"
          >
            Run Bayesian Update
          </Button>
        </form>

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
            <PosteriorChart posterior={result} />
          </>
        )}
      </Card>
    </div>
  );
}
