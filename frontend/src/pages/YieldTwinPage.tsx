/**
 * YieldTwinPage — POST /api/v1/yield-twin/recommend
 *
 * T1600: GP surrogate (ARD RBF) + D-Optimal DOE + Expected Improvement fusion.
 * UI: define factors + optional past observations; get next experiment recommendation.
 */
import { useState } from "react";
import { postYieldTwinRecommend } from "../api";
import type { DOERecommendation, FactorSpec, ExperimentPoint } from "../types";
import {
  Card,
  SectionLabel,
  Field,
  Input,
  Button,
  ErrorBox,
  StatCard,
  StatsRow,
  PageHeading,
  FormGrid,
  Textarea,
} from "../components/ui";
import { colors, spacing } from "../styles/tokens";

const DEFAULT_OBSERVATIONS = `# Optional: past experiments (JSON array)
# Each item: { "factors": { "temp": 180, "pressure": 2.5 }, "yield_obs": 0.87 }
# Leave blank or [] to run in pure DOE exploration mode.
[]`;

const ACQUISITION_COLOR: Record<string, string> = {
  doe_explore: colors.accent.amber,
  fused: colors.accent.green,
  ei_exploit: colors.accent.blue,
};

const ACQUISITION_LABEL: Record<string, string> = {
  doe_explore: "DOE Explore",
  fused: "Fused (EI + D-opt)",
  ei_exploit: "EI Exploit",
};

// ---- Factor row ----

interface FactorRowProps {
  factor: FactorSpec;
  index: number;
  onChange: (i: number, field: keyof FactorSpec, value: string) => void;
  onRemove: (i: number) => void;
  canRemove: boolean;
}

function FactorRow({ factor, index, onChange, onRemove, canRemove }: FactorRowProps) {
  return (
    <div
      style={{
        display: "grid",
        gridTemplateColumns: "1fr 100px 100px auto",
        gap: spacing.sm,
        alignItems: "end",
      }}
    >
      <Field label={index === 0 ? "Factor name" : ""}>
        <Input
          value={factor.name}
          onChange={(e) => onChange(index, "name", e.target.value)}
          placeholder="e.g. temperature"
        />
      </Field>
      <Field label={index === 0 ? "Low" : ""}>
        <Input
          type="number"
          value={factor.low}
          onChange={(e) => onChange(index, "low", e.target.value)}
        />
      </Field>
      <Field label={index === 0 ? "High" : ""}>
        <Input
          type="number"
          value={factor.high}
          onChange={(e) => onChange(index, "high", e.target.value)}
        />
      </Field>
      <button
        type="button"
        onClick={() => onRemove(index)}
        disabled={!canRemove}
        style={{
          background: "none",
          border: `1px solid ${colors.border}`,
          color: canRemove ? colors.accent.red : colors.border,
          borderRadius: 4,
          cursor: canRemove ? "pointer" : "not-allowed",
          padding: "6px 10px",
          fontSize: 14,
          marginBottom: 2,
        }}
        title="Remove factor"
      >
        ×
      </button>
    </div>
  );
}

// ---- Page ----

export function YieldTwinPage() {
  const [factors, setFactors] = useState<FactorSpec[]>([
    { name: "temperature", low: 150, high: 220 },
    { name: "pressure", low: 1.0, high: 4.0 },
  ]);
  const [obsText, setObsText] = useState(DEFAULT_OBSERVATIONS);
  const [randomSeed, setRandomSeed] = useState("42");
  const [nCandidates, setNCandidates] = useState("2000");
  const [eiXi, setEiXi] = useState("0.01");

  const [result, setResult] = useState<DOERecommendation | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  function updateFactor(i: number, field: keyof FactorSpec, value: string) {
    setFactors((prev) => {
      const next = [...prev];
      if (field === "name") {
        next[i] = { ...next[i], name: value };
      } else {
        next[i] = { ...next[i], [field]: parseFloat(value) };
      }
      return next;
    });
  }

  function addFactor() {
    setFactors((prev) => [...prev, { name: `factor_${prev.length + 1}`, low: 0, high: 1 }]);
  }

  function removeFactor(i: number) {
    setFactors((prev) => prev.filter((_, idx) => idx !== i));
  }

  function parseObservations(): ExperimentPoint[] {
    const cleaned = obsText
      .split("\n")
      .filter((l) => !l.trim().startsWith("#"))
      .join("\n")
      .trim();
    if (!cleaned || cleaned === "[]") return [];
    const parsed: unknown = JSON.parse(cleaned);
    if (!Array.isArray(parsed)) throw new Error("Observations must be a JSON array.");
    return parsed as ExperimentPoint[];
  }

  async function handleSubmit() {
    setLoading(true);
    setErr(null);
    try {
      const observations = parseObservations();
      const res = await postYieldTwinRecommend({
        factor_specs: factors,
        observations,
        random_seed: parseInt(randomSeed, 10),
        n_candidates: parseInt(nCandidates, 10),
        ei_xi: parseFloat(eiXi),
      });
      setResult(res);
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }

  const acqColor = result ? (ACQUISITION_COLOR[result.acquisition_mode] ?? colors.textMuted) : colors.textMuted;
  const acqLabel = result ? (ACQUISITION_LABEL[result.acquisition_mode] ?? result.acquisition_mode) : "";

  return (
    <div>
      <PageHeading subtitle="T1600 — GP surrogate (ARD RBF) + D-Optimal DOE + Expected Improvement fusion">
        Process Yield Twin
      </PageHeading>

      <Card>
        <SectionLabel color={colors.accent.orange}>FACTOR SPECIFICATIONS</SectionLabel>
        <form
          onSubmit={(e) => {
            e.preventDefault();
            void handleSubmit();
          }}
        >
          <div style={{ display: "flex", flexDirection: "column", gap: spacing.sm, marginBottom: spacing.md }}>
            {factors.map((f, i) => (
              <FactorRow
                key={i}
                factor={f}
                index={i}
                onChange={updateFactor}
                onRemove={removeFactor}
                canRemove={factors.length > 1}
              />
            ))}
          </div>
          <button
            type="button"
            onClick={addFactor}
            style={{
              background: "none",
              border: `1px dashed ${colors.border}`,
              color: colors.textMuted,
              borderRadius: 4,
              cursor: "pointer",
              padding: "6px 14px",
              fontSize: 12,
              marginBottom: spacing.lg,
              width: "100%",
            }}
          >
            + Add factor
          </button>

          <SectionLabel color={colors.accent.orange}>PAST OBSERVATIONS (JSON)</SectionLabel>
          <Field label='[ { "factors": { "temp": 180, "pressure": 2.5 }, "yield_obs": 0.87 }, … ]'>
            <Textarea
              rows={6}
              value={obsText}
              onChange={(e) => setObsText(e.target.value)}
              style={{ fontFamily: "'Consolas', monospace", fontSize: 12 }}
            />
          </Field>

          <SectionLabel color={colors.accent.orange}>SOLVER OPTIONS</SectionLabel>
          <FormGrid>
            <Field label="Random seed">
              <Input type="number" value={randomSeed} onChange={(e) => setRandomSeed(e.target.value)} />
            </Field>
            <Field label="LHS candidates">
              <Input type="number" min={2} max={5000} value={nCandidates} onChange={(e) => setNCandidates(e.target.value)} />
            </Field>
            <Field label="EI exploration bonus (ξ)">
              <Input value={eiXi} onChange={(e) => setEiXi(e.target.value)} />
            </Field>
          </FormGrid>

          <Button accent={colors.accent.orange} loading={loading} loadingLabel="Optimising…" type="submit">
            Get Next Experiment
          </Button>
        </form>

        {err && <ErrorBox message={err} />}
      </Card>

      {result && (
        <>
          <StatsRow>
            <StatCard
              label="Predicted Yield"
              value={`${(result.predicted_yield * 100).toFixed(2)}%`}
              color={colors.accent.orange}
            />
            <StatCard
              label="Predicted Std"
              value={`±${(result.predicted_std * 100).toFixed(2)}%`}
              color={colors.textMuted}
            />
            <StatCard
              label="Expected Improvement"
              value={result.expected_improvement.toFixed(5)}
              color={colors.accent.blue}
            />
            <StatCard
              label="D-Leverage"
              value={result.d_leverage.toFixed(4)}
              color={colors.accent.green}
            />
            <StatCard
              label="Acquisition Mode"
              value={acqLabel}
              color={acqColor}
            />
            <StatCard
              label="Observations"
              value={result.n_observations.toString()}
              color={colors.textMuted}
            />
          </StatsRow>

          <Card>
            <SectionLabel color={colors.accent.orange}>RECOMMENDED FACTOR SETTINGS</SectionLabel>
            <div style={{ display: "flex", flexWrap: "wrap", gap: spacing.md }}>
              {Object.entries(result.factors).map(([name, value]) => (
                <div
                  key={name}
                  style={{
                    background: colors.inputBg,
                    border: `1px solid ${colors.border}`,
                    borderRadius: 6,
                    padding: `${spacing.sm}px ${spacing.lg}px`,
                    minWidth: 140,
                  }}
                >
                  <div style={{ color: colors.textMuted, fontSize: 11 }}>{name}</div>
                  <div style={{ color: colors.accent.orange, fontSize: 18, fontWeight: 700, marginTop: 2 }}>
                    {value.toFixed(4)}
                  </div>
                </div>
              ))}
            </div>
          </Card>
        </>
      )}
    </div>
  );
}
