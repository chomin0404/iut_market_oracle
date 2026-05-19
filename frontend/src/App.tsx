import { useEffect, useState, useCallback } from "react";
import {
  checkHealth,
  postTwinRun,
  postResilienceSim,
  type ResilienceSimParams,
} from "./api";
import type { TwinRunReport, ResilienceTwinReport, RecommendedAction } from "./types";
import { StatusBanner } from "./components/StatusBanner";
import { FaultPosteriorChart } from "./components/FaultPosteriorChart";
import { ResilienceSimChart } from "./components/ResilienceSimChart";
import { TwinRunPanel, ResilienceSimPanel } from "./components/ControlPanel";

// ---- Layout constants ----
const PAGE_BG = "#0a0a0a";
const SECTION_GAP = 20;

// ---- Top header ----
function Header({ healthy }: { healthy: boolean | null }) {
  const dot =
    healthy === null
      ? { color: "#888", label: "checking…" }
      : healthy
        ? { color: "#4ade80", label: "API online" }
        : { color: "#f87171", label: "API offline" };

  return (
    <div
      style={{
        display: "flex",
        alignItems: "center",
        justifyContent: "space-between",
        padding: "12px 24px",
        background: "#111",
        borderBottom: "1px solid #222",
      }}
    >
      <div>
        <span style={{ color: "#fff", fontWeight: 700, fontSize: 16, letterSpacing: 1 }}>
          GNSS SPOOFING DETECTION
        </span>
        <span style={{ color: "#555", fontSize: 12, marginLeft: 12 }}>
          RESILIENCE TWIN DASHBOARD
        </span>
      </div>
      <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
        <span
          style={{
            width: 10,
            height: 10,
            borderRadius: "50%",
            background: dot.color,
            display: "inline-block",
          }}
        />
        <span style={{ color: dot.color, fontSize: 12 }}>{dot.label}</span>
      </div>
    </div>
  );
}

// ---- Summary stats row ----
function StatsRow({ report }: { report: TwinRunReport }) {
  const stats = [
    {
      label: "Mean Genuine",
      value: `${(report.mean_authenticity_genuine * 100).toFixed(1)}%`,
    },
    {
      label: "Mean Nominal",
      value: `${(report.mean_integrity_nominal * 100).toFixed(1)}%`,
    },
    {
      label: "Dominant",
      value: report.dominant_diagnosis,
    },
    {
      label: "Alert Epochs",
      value: report.alert_epochs.length.toString(),
    },
    {
      label: "Spoof Window",
      value: report.spoofing_window
        ? `${report.spoofing_window[0]}–${report.spoofing_window[1]}`
        : "none",
    },
  ];

  return (
    <div
      style={{
        display: "flex",
        gap: 12,
        flexWrap: "wrap",
      }}
    >
      {stats.map((s) => (
        <div
          key={s.label}
          style={{
            background: "#161616",
            border: "1px solid #2a2a2a",
            borderRadius: 6,
            padding: "8px 16px",
            flex: "1 1 100px",
          }}
        >
          <div style={{ color: "#888", fontSize: 11 }}>{s.label}</div>
          <div style={{ color: "#e0e0e0", fontSize: 15, fontWeight: 600 }}>
            {s.value}
          </div>
        </div>
      ))}
    </div>
  );
}

// ---- Main App ----

export default function App() {
  const [healthy, setHealthy] = useState<boolean | null>(null);
  const [twinReport, setTwinReport] = useState<TwinRunReport | null>(null);
  const [twinError, setTwinError] = useState<string | null>(null);
  const [twinLoading, setTwinLoading] = useState(false);

  const [resReport, setResReport] = useState<ResilienceTwinReport | null>(null);
  const [resError, setResError] = useState<string | null>(null);
  const [resLoading, setResLoading] = useState(false);

  // Health check on mount
  useEffect(() => {
    checkHealth()
      .then(() => setHealthy(true))
      .catch(() => setHealthy(false));
  }, []);

  // Twin Run handler
  const handleTwinRun = useCallback(
    async (
      obs: { epoch: number; doppler_residuals: number[] }[],
      nSats: number,
      noiseStd: number,
      sigma: number
    ) => {
      setTwinLoading(true);
      setTwinError(null);
      try {
        const report = await postTwinRun({
          observations: obs,
          n_sats: nSats,
          doppler_noise_std: noiseStd,
          graph_sigma: sigma,
          save: false,
        });
        setTwinReport(report);
      } catch (e) {
        setTwinError(e instanceof Error ? e.message : String(e));
      } finally {
        setTwinLoading(false);
      }
    },
    []
  );

  // Resilience Sim handler
  const handleResSim = useCallback(async (params: ResilienceSimParams) => {
    setResLoading(true);
    setResError(null);
    try {
      const report = await postResilienceSim(params);
      setResReport(report);
    } catch (e) {
      setResError(e instanceof Error ? e.message : String(e));
    } finally {
      setResLoading(false);
    }
  }, []);

  // Derive worst action for banner
  const worstAction: RecommendedAction = twinReport?.worst_action ?? "nominal";
  const bannerReason = twinReport
    ? `Dominant: ${twinReport.dominant_diagnosis} | ${twinReport.alert_epochs.length} alert epoch(s)`
    : "No analysis run yet";

  return (
    <div
      style={{
        background: PAGE_BG,
        minHeight: "100vh",
        color: "#e0e0e0",
        fontFamily: "'Consolas', 'Courier New', monospace",
      }}
    >
      <Header healthy={healthy} />

      <div style={{ maxWidth: 1200, margin: "0 auto", padding: "20px 20px" }}>
        {/* Status Banner */}
        <StatusBanner
          action={worstAction}
          reason={bannerReason}
          runId={twinReport?.run_id}
        />

        {/* Twin Run section */}
        <div style={{ marginBottom: SECTION_GAP }}>
          <TwinRunPanel onSubmit={handleTwinRun} loading={twinLoading} />
        </div>

        {twinError && (
          <div
            style={{
              color: "#f87171",
              background: "#1a0000",
              border: "1px solid #f87171",
              borderRadius: 6,
              padding: "8px 16px",
              marginBottom: SECTION_GAP,
              fontSize: 13,
            }}
          >
            Error: {twinError}
          </div>
        )}

        {twinReport && (
          <div style={{ marginBottom: SECTION_GAP, display: "flex", flexDirection: "column", gap: 12 }}>
            <StatsRow report={twinReport} />
            <FaultPosteriorChart
              epochs={twinReport.epoch_reports}
              alertEpochs={twinReport.alert_epochs}
            />
          </div>
        )}

        {/* Divider */}
        <div style={{ borderTop: "1px solid #222", marginBottom: SECTION_GAP }} />

        {/* Resilience Sim section */}
        <div style={{ marginBottom: SECTION_GAP }}>
          <ResilienceSimPanel onSubmit={handleResSim} loading={resLoading} />
        </div>

        {resError && (
          <div
            style={{
              color: "#f87171",
              background: "#1a0000",
              border: "1px solid #f87171",
              borderRadius: 6,
              padding: "8px 16px",
              marginBottom: SECTION_GAP,
              fontSize: 13,
            }}
          >
            Error: {resError}
          </div>
        )}

        {resReport && (
          <div style={{ marginBottom: SECTION_GAP }}>
            <ResilienceSimChart report={resReport} />
          </div>
        )}
      </div>
    </div>
  );
}
