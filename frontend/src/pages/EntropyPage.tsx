/**
 * EntropyPage — POST /api/v1/entropy/detect
 *
 * The entropy endpoint takes a sequence of PosteriorSummary objects
 * (Bayesian posteriors over time) and computes Shannon entropy, KL divergence
 * from the prior, and entropy rate for each step.
 *
 * UI: user defines a prior + number of epochs.  Two synthetic regimes are
 * generated automatically — a stable regime followed by a shifted regime —
 * so the change-point detection can be demonstrated without real data.
 */
import { useState } from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
  Legend,
} from "recharts";
import { postEntropyDetect } from "../api";
import type { EntropyReport, PosteriorSummary } from "../types";
import {
  Card,
  SectionLabel,
  Field,
  Input,
  Select,
  Button,
  ErrorBox,
  StatCard,
  StatsRow,
  PageHeading,
  FormGrid,
} from "../components/ui";
import { colors, spacing } from "../styles/tokens";

// ---- Synthetic posterior generator ----

/**
 * Generate a sequence of PosteriorSummary objects with a simulated regime shift.
 *   - Epochs 0 … shiftAt-1: posteriors close to the prior (stable regime)
 *   - Epochs shiftAt … n-1: posteriors drifted by `drift` (shifted regime)
 */
function generateSyntheticPosteriors(
  nEpochs: number,
  shiftAt: number,
  priorMu: number,
  priorSigma: number,
  drift: number
): PosteriorSummary[] {
  const posteriors: PosteriorSummary[] = [];
  for (let i = 0; i < nEpochs; i++) {
    const inShift = i >= shiftAt;
    const mean = priorMu + (inShift ? drift : 0) + (Math.random() - 0.5) * priorSigma * 0.3;
    const variance = Math.max(1e-6, priorSigma * priorSigma * (inShift ? 0.5 : 1.0));
    const halfWidth = 1.96 * Math.sqrt(variance);
    posteriors.push({
      mean,
      variance,
      credible_interval_95: [mean - halfWidth, mean + halfWidth],
      n_evidence: i + 1,
      updated_at: new Date(Date.now() + i * 1000).toISOString(),
    });
  }
  return posteriors;
}

// ---- Page ----

export function EntropyPage() {
  const [distribution, setDistribution] = useState("normal");
  const [priorMu, setPriorMu] = useState("0.05");
  const [priorSigma, setPriorSigma] = useState("0.02");
  const [nEpochs, setNEpochs] = useState("50");
  const [shiftAt, setShiftAt] = useState("30");
  const [drift, setDrift] = useState("0.08");
  const [klThreshold, setKlThreshold] = useState("0.5");
  const [gradThreshold, setGradThreshold] = useState("0.1");
  const [rollingWindow, setRollingWindow] = useState("3");
  const [experimentId, setExperimentId] = useState("exp-001");

  const [report, setReport] = useState<EntropyReport | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  async function handleSubmit() {
    setLoading(true);
    setErr(null);
    try {
      const mu = parseFloat(priorMu);
      const sigma = parseFloat(priorSigma);
      const posteriors = generateSyntheticPosteriors(
        parseInt(nEpochs, 10),
        parseInt(shiftAt, 10),
        mu,
        sigma,
        parseFloat(drift)
      );
      const prior = {
        distribution,
        params: (distribution === "beta" ? { alpha: mu, beta: sigma } : { mu, sigma }) as Record<string, number>,
      };
      const res = await postEntropyDetect({
        posteriors,
        prior,
        experiment_id: experimentId,
        kl_threshold: parseFloat(klThreshold),
        entropy_gradient_threshold: parseFloat(gradThreshold),
        rolling_window: parseInt(rollingWindow, 10),
      });
      setReport(res);
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }

  const chartData =
    report?.entropy_series.map((h, i) => ({
      t: i,
      entropy: parseFloat(h.toFixed(5)),
      kl: report.kl_series[i] !== undefined ? parseFloat(report.kl_series[i].toFixed(5)) : null,
      rate:
        report.entropy_rate_series[i] !== undefined
          ? parseFloat(report.entropy_rate_series[i].toFixed(5))
          : null,
    })) ?? [];

  const alertEpochs = new Set(report?.alerts.map((a) => a.triggered_at) ?? []);

  return (
    <div>
      <PageHeading subtitle="T1000 — Shannon entropy, KL divergence, entropy rate on Bayesian posterior sequences">
        Entropy Monitor
      </PageHeading>

      <Card>
        <SectionLabel color={colors.accent.purple}>PRIOR &amp; SIMULATION PARAMETERS</SectionLabel>
        <FormGrid>
          <Field label="Prior distribution">
            <Select value={distribution} onChange={(e) => setDistribution(e.target.value)}>
              <option value="normal">Normal</option>
              <option value="beta">Beta</option>
            </Select>
          </Field>
          <Field label={distribution === "beta" ? "Alpha (α)" : "Prior mean (μ)"}>
            <Input value={priorMu} onChange={(e) => setPriorMu(e.target.value)} />
          </Field>
          <Field label={distribution === "beta" ? "Beta (β)" : "Prior std (σ)"}>
            <Input value={priorSigma} onChange={(e) => setPriorSigma(e.target.value)} />
          </Field>
          <Field label="Total epochs">
            <Input type="number" min={5} max={500} value={nEpochs} onChange={(e) => setNEpochs(e.target.value)} />
          </Field>
          <Field label="Regime shift at epoch">
            <Input type="number" min={1} value={shiftAt} onChange={(e) => setShiftAt(e.target.value)} />
          </Field>
          <Field label="Shift drift magnitude">
            <Input value={drift} onChange={(e) => setDrift(e.target.value)} />
          </Field>
        </FormGrid>

        <SectionLabel color={colors.accent.purple}>DETECTION THRESHOLDS</SectionLabel>
        <FormGrid>
          <Field label="KL threshold">
            <Input value={klThreshold} onChange={(e) => setKlThreshold(e.target.value)} />
          </Field>
          <Field label="Entropy gradient threshold">
            <Input value={gradThreshold} onChange={(e) => setGradThreshold(e.target.value)} />
          </Field>
          <Field label="Rolling window">
            <Input type="number" min={1} value={rollingWindow} onChange={(e) => setRollingWindow(e.target.value)} />
          </Field>
          <Field label="Experiment ID">
            <Input value={experimentId} onChange={(e) => setExperimentId(e.target.value)} />
          </Field>
        </FormGrid>

        <Button
          accent={colors.accent.purple}
          loading={loading}
          loadingLabel="Computing…"
          onClick={() => void handleSubmit()}
        >
          Run Entropy Analysis
        </Button>

        {err && <ErrorBox message={err} />}
      </Card>

      {report && (
        <>
          <StatsRow>
            <StatCard label="Epochs" value={report.entropy_series.length.toString()} color={colors.accent.purple} />
            <StatCard
              label="Max Entropy"
              value={Math.max(...report.entropy_series).toFixed(4)}
              color={colors.accent.purple}
            />
            <StatCard
              label="Max KL"
              value={report.kl_series.length > 0 ? Math.max(...report.kl_series).toFixed(4) : "—"}
              color={colors.accent.purple}
            />
            <StatCard label="Alerts" value={report.alerts.length.toString()} color={colors.accent.red} />
          </StatsRow>

          <Card>
            <div style={{ color: colors.textMuted, fontSize: 11, marginBottom: spacing.sm }}>
              Entropy, KL divergence from prior, entropy rate — alert epochs marked in red
            </div>
            <ResponsiveContainer width="100%" height={260}>
              <LineChart data={chartData} margin={{ top: 4, right: 16, left: 0, bottom: 4 }}>
                <XAxis
                  dataKey="t"
                  tick={{ fill: colors.textFaint, fontSize: 10 }}
                  axisLine={false}
                  tickLine={false}
                />
                <YAxis
                  tick={{ fill: colors.textFaint, fontSize: 10 }}
                  axisLine={false}
                  tickLine={false}
                  width={52}
                />
                <Tooltip
                  contentStyle={{
                    background: colors.surface0,
                    border: `1px solid ${colors.border}`,
                    fontSize: 11,
                  }}
                  labelStyle={{ color: colors.textMuted }}
                />
                <Legend wrapperStyle={{ fontSize: 11, color: colors.textMuted }} />
                {Array.from(alertEpochs).map((t) => (
                  <ReferenceLine
                    key={t}
                    x={t}
                    stroke={colors.accent.red}
                    strokeDasharray="3 3"
                    strokeOpacity={0.7}
                  />
                ))}
                <Line
                  type="monotone"
                  dataKey="entropy"
                  stroke={colors.accent.purple}
                  dot={false}
                  strokeWidth={1.5}
                  name="Entropy (nats)"
                />
                <Line
                  type="monotone"
                  dataKey="kl"
                  stroke={colors.accent.blue}
                  dot={false}
                  strokeWidth={1.2}
                  name="KL divergence"
                  strokeDasharray="4 2"
                />
                <Line
                  type="monotone"
                  dataKey="rate"
                  stroke={colors.accent.green}
                  dot={false}
                  strokeWidth={1.2}
                  name="Entropy rate"
                  strokeDasharray="2 3"
                />
              </LineChart>
            </ResponsiveContainer>
          </Card>

          {report.alerts.length > 0 && (
            <Card>
              <SectionLabel color={colors.accent.red}>
                ALERTS ({report.alerts.length})
              </SectionLabel>
              <div style={{ display: "flex", flexDirection: "column", gap: spacing.sm }}>
                {report.alerts.map((a, i) => (
                  <div
                    key={i}
                    style={{
                      background: colors.accent.redDark,
                      border: `1px solid ${colors.accent.orangeDark}`,
                      borderRadius: 6,
                      padding: "8px 14px",
                      fontSize: 12,
                    }}
                  >
                    <span style={{ color: colors.accent.red, fontWeight: 700 }}>t={a.triggered_at}</span>
                    <span style={{ color: colors.textMuted, marginLeft: 10 }}>[{a.alert_type}]</span>
                    <span style={{ color: colors.text, marginLeft: 10 }}>{a.message}</span>
                  </div>
                ))}
              </div>
            </Card>
          )}
        </>
      )}
    </div>
  );
}
