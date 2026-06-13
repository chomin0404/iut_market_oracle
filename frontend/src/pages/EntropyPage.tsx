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
import type { EntropyReport } from "../types";

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
  background: "#a78bfa",
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

// Default synthetic time series — random walk with a regime shift around index 60
function generateDefaultSeries(): string {
  const series: number[] = [];
  let val = 0;
  for (let i = 0; i < 100; i++) {
    val += (Math.random() - 0.5) * 0.5;
    if (i >= 60) val += 0.08; // regime shift
    series.push(parseFloat(val.toFixed(4)));
  }
  return series.join(",");
}

export function EntropyPage() {
  const [seriesInput, setSeriesInput] = useState(() => generateDefaultSeries());
  const [window_, setWindow] = useState("10");
  const [threshold, setThreshold] = useState("0.5");
  const [experimentId, setExperimentId] = useState("exp-001");

  const [report, setReport] = useState<EntropyReport | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  async function handleSubmit() {
    setLoading(true);
    setErr(null);
    try {
      const series = seriesInput
        .split(",")
        .map((v) => parseFloat(v.trim()))
        .filter((v) => !isNaN(v));
      const result = await postEntropyDetect({
        series,
        window: parseInt(window_, 10),
        threshold: parseFloat(threshold),
        experiment_id: experimentId,
      });
      setReport(result);
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }

  // Build chart data — align all series by index
  const chartData = report
    ? report.entropy_series.map((h, i) => ({
        t: i,
        entropy: parseFloat(h.toFixed(4)),
        kl: report.kl_series[i] !== undefined ? parseFloat(report.kl_series[i].toFixed(4)) : null,
        rate:
          report.entropy_rate_series[i] !== undefined
            ? parseFloat(report.entropy_rate_series[i].toFixed(4))
            : null,
      }))
    : [];

  // Alert epoch indices
  const alertEpochs = new Set(report?.alerts.map((a) => a.triggered_at) ?? []);

  return (
    <div>
      <h2
        style={{ color: "#fff", fontSize: 18, fontWeight: 700, margin: "0 0 20px", letterSpacing: 1 }}
      >
        Entropy Monitor — T1000
      </h2>

      <div style={CARD}>
        <div
          style={{ color: "#a78bfa", fontSize: 12, fontWeight: 600, marginBottom: 14, letterSpacing: 1 }}
        >
          INPUT CONFIGURATION
        </div>

        <div
          style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(160px, 1fr))", gap: 10 }}
        >
          <Field label="Window size">
            <input style={INPUT_STYLE} value={window_} onChange={(e) => setWindow(e.target.value)} />
          </Field>
          <Field label="Alert threshold">
            <input style={INPUT_STYLE} value={threshold} onChange={(e) => setThreshold(e.target.value)} />
          </Field>
          <Field label="Experiment ID">
            <input
              style={INPUT_STYLE}
              value={experimentId}
              onChange={(e) => setExperimentId(e.target.value)}
            />
          </Field>
        </div>

        <Field label="Time series (comma-separated floats)">
          <textarea
            style={{ ...INPUT_STYLE, height: 80, resize: "vertical" }}
            value={seriesInput}
            onChange={(e) => setSeriesInput(e.target.value)}
          />
        </Field>

        <button style={BTN_STYLE} onClick={() => void handleSubmit()} disabled={loading}>
          {loading ? "Computing…" : "Run Entropy Analysis"}
        </button>

        {err && (
          <div style={{ color: "#f87171", fontSize: 12, marginTop: 10 }}>Error: {err}</div>
        )}
      </div>

      {report && (
        <>
          {/* Summary */}
          <div style={{ display: "flex", gap: 12, flexWrap: "wrap", marginBottom: 16 }}>
            {[
              { label: "Epochs", value: report.entropy_series.length.toString() },
              {
                label: "Max Entropy",
                value: Math.max(...report.entropy_series).toFixed(4),
              },
              {
                label: "Max KL",
                value:
                  report.kl_series.length > 0
                    ? Math.max(...report.kl_series).toFixed(4)
                    : "—",
              },
              { label: "Alerts", value: report.alerts.length.toString() },
            ].map((s) => (
              <div
                key={s.label}
                style={{
                  background: "#161616",
                  border: "1px solid #2a2a2a",
                  borderRadius: 6,
                  padding: "10px 16px",
                  flex: "1 1 120px",
                }}
              >
                <div style={{ color: "#888", fontSize: 11 }}>{s.label}</div>
                <div style={{ color: "#a78bfa", fontSize: 17, fontWeight: 700 }}>{s.value}</div>
              </div>
            ))}
          </div>

          {/* Entropy chart */}
          <div style={CARD}>
            <div
              style={{ color: "#888", fontSize: 11, marginBottom: 10 }}
            >
              Shannon entropy, KL divergence, entropy rate — alert epochs marked in red
            </div>
            <ResponsiveContainer width="100%" height={260}>
              <LineChart data={chartData} margin={{ top: 4, right: 16, left: 0, bottom: 4 }}>
                <XAxis
                  dataKey="t"
                  tick={{ fill: "#555", fontSize: 10 }}
                  axisLine={false}
                  tickLine={false}
                  label={{ value: "epoch", fill: "#444", fontSize: 10, position: "insideBottomRight", offset: -4 }}
                />
                <YAxis
                  tick={{ fill: "#555", fontSize: 10 }}
                  axisLine={false}
                  tickLine={false}
                  width={46}
                />
                <Tooltip
                  contentStyle={{ background: "#111", border: "1px solid #333", fontSize: 11 }}
                  labelStyle={{ color: "#888" }}
                />
                <Legend
                  wrapperStyle={{ fontSize: 11, color: "#888" }}
                />
                {/* Alert epoch reference lines */}
                {Array.from(alertEpochs).map((t) => (
                  <ReferenceLine key={t} x={t} stroke="#f87171" strokeDasharray="3 3" strokeOpacity={0.7} />
                ))}
                <Line
                  type="monotone"
                  dataKey="entropy"
                  stroke="#a78bfa"
                  dot={false}
                  strokeWidth={1.5}
                  name="Entropy"
                />
                <Line
                  type="monotone"
                  dataKey="kl"
                  stroke="#60a5fa"
                  dot={false}
                  strokeWidth={1.2}
                  name="KL div"
                  strokeDasharray="4 2"
                />
                <Line
                  type="monotone"
                  dataKey="rate"
                  stroke="#4ade80"
                  dot={false}
                  strokeWidth={1.2}
                  name="Entropy rate"
                  strokeDasharray="2 3"
                />
              </LineChart>
            </ResponsiveContainer>
          </div>

          {/* Alert list */}
          {report.alerts.length > 0 && (
            <div style={CARD}>
              <div
                style={{ color: "#f87171", fontSize: 12, fontWeight: 600, marginBottom: 12, letterSpacing: 1 }}
              >
                ALERTS ({report.alerts.length})
              </div>
              <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
                {report.alerts.map((a, i) => (
                  <div
                    key={i}
                    style={{
                      background: "#1a0000",
                      border: "1px solid #3a1010",
                      borderRadius: 6,
                      padding: "8px 14px",
                      fontSize: 12,
                    }}
                  >
                    <span style={{ color: "#f87171", fontWeight: 600 }}>
                      t={a.triggered_at}
                    </span>
                    <span style={{ color: "#888", marginLeft: 10 }}>
                      [{a.alert_type}]
                    </span>
                    <span style={{ color: "#e0e0e0", marginLeft: 10 }}>{a.message}</span>
                    <span style={{ color: "#666", marginLeft: 10 }}>
                      val={a.metric_value.toFixed(4)} &gt; thr={a.threshold.toFixed(4)}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </>
      )}
    </div>
  );
}
