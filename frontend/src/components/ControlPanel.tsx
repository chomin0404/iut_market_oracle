import { useState } from "react";
import type { ResilienceSimParams } from "../api";
import { colors } from "../styles/tokens";

// ---- Shared UI helpers ----

function Field({
  label,
  children,
}: {
  label: string;
  children: React.ReactNode;
}) {
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
      <label style={{ color: colors.textMuted, fontSize: 12 }}>{label}</label>
      {children}
    </div>
  );
}

const inputStyle: React.CSSProperties = {
  background: colors.inputBg,
  border: `1px solid ${colors.border}`,
  borderRadius: 4,
  color: colors.text,
  padding: "5px 8px",
  fontSize: 13,
  width: "100%",
  boxSizing: "border-box",
};

// ---- Twin Run panel ----

interface TwinRunForm {
  n_sats: number;
  n_epochs: number;
  doppler_noise_std: number;
  spoof_bias_std: number;
  graph_sigma: number;
  inject_spoof: boolean;
  spoof_start_frac: number;
  spoof_end_frac: number;
  seed: number;
}

function buildObservations(form: TwinRunForm) {
  const rng = seededRng(form.seed);
  const epochs = [];
  const spoofStart = Math.floor(form.n_epochs * form.spoof_start_frac);
  const spoofEnd = Math.floor(form.n_epochs * form.spoof_end_frac);

  for (let t = 0; t < form.n_epochs; t++) {
    const isSpoof = form.inject_spoof && t >= spoofStart && t <= spoofEnd;
    const bias = isSpoof ? (rng() - 0.5) * 2 * form.spoof_bias_std : 0;
    const residuals = Array.from({ length: form.n_sats }, () => {
      const noise = (rng() - 0.5) * 2 * form.doppler_noise_std;
      return bias + noise;
    });
    epochs.push({ epoch: t, doppler_residuals: residuals });
  }
  return epochs;
}

function seededRng(seed: number) {
  let s = seed >>> 0;
  return () => {
    s ^= s << 13;
    s ^= s >> 17;
    s ^= s << 5;
    return (s >>> 0) / 4294967296;
  };
}

interface TwinRunPanelProps {
  onSubmit: (
    obs: { epoch: number; doppler_residuals: number[] }[],
    nSats: number,
    noiseStd: number,
    sigma: number
  ) => void;
  loading: boolean;
}

export function TwinRunPanel({ onSubmit, loading }: TwinRunPanelProps) {
  const [form, setForm] = useState<TwinRunForm>({
    n_sats: 6,
    n_epochs: 60,
    doppler_noise_std: 0.3,
    spoof_bias_std: 3.0,
    graph_sigma: 1.5,
    inject_spoof: true,
    spoof_start_frac: 0.4,
    spoof_end_frac: 0.75,
    seed: 42,
  });

  function num(key: keyof TwinRunForm, value: string) {
    setForm((f) => ({ ...f, [key]: parseFloat(value) || 0 }));
  }

  function handleRun() {
    const obs = buildObservations(form);
    onSubmit(obs, form.n_sats, form.doppler_noise_std, form.graph_sigma);
  }

  return (
    <div style={{ background: colors.surface1, borderRadius: 8, padding: 16 }}>
      <h3 style={{ color: colors.text, margin: "0 0 12px", fontSize: 14 }}>
        Twin Run — per-epoch fault analysis
      </h3>
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(3, 1fr)",
          gap: 10,
          marginBottom: 12,
        }}
      >
        <Field label="Satellites">
          <input
            type="number"
            style={inputStyle}
            value={form.n_sats}
            min={5}
            max={20}
            onChange={(e) => num("n_sats", e.target.value)}
          />
        </Field>
        <Field label="Epochs">
          <input
            type="number"
            style={inputStyle}
            value={form.n_epochs}
            min={10}
            max={500}
            onChange={(e) => num("n_epochs", e.target.value)}
          />
        </Field>
        <Field label="Doppler noise σ [Hz]">
          <input
            type="number"
            style={inputStyle}
            value={form.doppler_noise_std}
            step={0.05}
            min={0.05}
            onChange={(e) => num("doppler_noise_std", e.target.value)}
          />
        </Field>
        <Field label="Spoof bias σ [Hz]">
          <input
            type="number"
            style={inputStyle}
            value={form.spoof_bias_std}
            step={0.5}
            min={0.5}
            onChange={(e) => num("spoof_bias_std", e.target.value)}
          />
        </Field>
        <Field label="Graph σ [Hz]">
          <input
            type="number"
            style={inputStyle}
            value={form.graph_sigma}
            step={0.1}
            min={0.1}
            onChange={(e) => num("graph_sigma", e.target.value)}
          />
        </Field>
        <Field label="RNG seed">
          <input
            type="number"
            style={inputStyle}
            value={form.seed}
            onChange={(e) => num("seed", e.target.value)}
          />
        </Field>
      </div>
      <div style={{ display: "flex", alignItems: "center", gap: 12, marginBottom: 12 }}>
        <label style={{ color: colors.textMuted, fontSize: 12, display: "flex", gap: 6, alignItems: "center" }}>
          <input
            type="checkbox"
            checked={form.inject_spoof}
            onChange={(e) => setForm((f) => ({ ...f, inject_spoof: e.target.checked }))}
          />
          Inject spoofing attack
        </label>
        {form.inject_spoof && (
          <>
            <Field label="Start (0–1)">
              <input
                type="number"
                style={{ ...inputStyle, width: 70 }}
                value={form.spoof_start_frac}
                step={0.05}
                min={0}
                max={1}
                onChange={(e) => num("spoof_start_frac", e.target.value)}
              />
            </Field>
            <Field label="End (0–1)">
              <input
                type="number"
                style={{ ...inputStyle, width: 70 }}
                value={form.spoof_end_frac}
                step={0.05}
                min={0}
                max={1}
                onChange={(e) => num("spoof_end_frac", e.target.value)}
              />
            </Field>
          </>
        )}
      </div>
      <button
        onClick={handleRun}
        disabled={loading}
        style={{
          background: loading ? colors.surface1Hover : colors.accent.blue,
          color: loading ? colors.textDim : "#000",
          border: "none",
          borderRadius: 6,
          padding: "8px 24px",
          cursor: loading ? "not-allowed" : "pointer",
          fontSize: 13,
          fontWeight: 600,
        }}
      >
        {loading ? "Running…" : "Run Twin Analysis"}
      </button>
    </div>
  );
}

// ---- Resilience Sim panel ----

interface ResPanelProps {
  onSubmit: (params: ResilienceSimParams) => void;
  loading: boolean;
}

export function ResilienceSimPanel({ onSubmit, loading }: ResPanelProps) {
  const [form, setForm] = useState<ResilienceSimParams>({
    n_mc: 200,
    n_epochs: 80,
    n_sats: 6,
    doppler_noise_std: 0.3,
    spoof_bias_std: 4.0,
    spoof_diff_std: 0.8,
    graph_sigma: 1.5,
    dirichlet_alpha: 2.0,
    random_seed: 42,
  });

  function num(key: keyof ResilienceSimParams, value: string) {
    setForm((f) => ({ ...f, [key]: parseFloat(value) || 0 }));
  }

  return (
    <div style={{ background: colors.surface1, borderRadius: 8, padding: 16 }}>
      <h3 style={{ color: colors.text, margin: "0 0 12px", fontSize: 14 }}>
        Resilience Sim — 4-class Monte Carlo benchmark
      </h3>
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(3, 1fr)",
          gap: 10,
          marginBottom: 12,
        }}
      >
        <Field label="MC runs">
          <input type="number" style={inputStyle} value={form.n_mc} min={4} max={2000}
            onChange={(e) => num("n_mc", e.target.value)} />
        </Field>
        <Field label="Epochs / run">
          <input type="number" style={inputStyle} value={form.n_epochs} min={10} max={500}
            onChange={(e) => num("n_epochs", e.target.value)} />
        </Field>
        <Field label="Satellites">
          <input type="number" style={inputStyle} value={form.n_sats} min={5} max={20}
            onChange={(e) => num("n_sats", e.target.value)} />
        </Field>
        <Field label="Doppler noise σ">
          <input type="number" style={inputStyle} value={form.doppler_noise_std} step={0.05}
            onChange={(e) => num("doppler_noise_std", e.target.value)} />
        </Field>
        <Field label="Spoof bias σ">
          <input type="number" style={inputStyle} value={form.spoof_bias_std} step={0.5}
            onChange={(e) => num("spoof_bias_std", e.target.value)} />
        </Field>
        <Field label="Spoof diff σ">
          <input type="number" style={inputStyle} value={form.spoof_diff_std} step={0.1}
            onChange={(e) => num("spoof_diff_std", e.target.value)} />
        </Field>
        <Field label="Graph σ">
          <input type="number" style={inputStyle} value={form.graph_sigma} step={0.1}
            onChange={(e) => num("graph_sigma", e.target.value)} />
        </Field>
        <Field label="Dirichlet α">
          <input type="number" style={inputStyle} value={form.dirichlet_alpha} step={0.5}
            onChange={(e) => num("dirichlet_alpha", e.target.value)} />
        </Field>
        <Field label="RNG seed">
          <input type="number" style={inputStyle} value={form.random_seed}
            onChange={(e) => num("random_seed", e.target.value)} />
        </Field>
      </div>
      <button
        onClick={() => onSubmit(form)}
        disabled={loading}
        style={{
          background: loading ? colors.surface1Hover : colors.accent.purple,
          color: loading ? colors.textDim : "#000",
          border: "none",
          borderRadius: 6,
          padding: "8px 24px",
          cursor: loading ? "not-allowed" : "pointer",
          fontSize: 13,
          fontWeight: 600,
        }}
      >
        {loading ? "Running…" : "Run MC Benchmark"}
      </button>
    </div>
  );
}
