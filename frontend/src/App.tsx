import { lazy, Suspense, useEffect, useState } from "react";
import { HashRouter, Routes, Route } from "react-router-dom";
import { checkHealth } from "./api";
import { Sidebar } from "./components/Sidebar";
import { OverviewPage } from "./pages/OverviewPage";
import { colors, typography } from "./styles/tokens";

// Lazy-load heavy pages so each gets its own JS chunk
const GnssPage = lazy(() => import("./pages/GnssPage").then((m) => ({ default: m.GnssPage })));
const ValuationPage = lazy(() =>
  import("./pages/ValuationPage").then((m) => ({ default: m.ValuationPage }))
);
const BayesianPage = lazy(() =>
  import("./pages/BayesianPage").then((m) => ({ default: m.BayesianPage }))
);
const EntropyPage = lazy(() =>
  import("./pages/EntropyPage").then((m) => ({ default: m.EntropyPage }))
);

function Header({ healthy }: { healthy: boolean | null }) {
  const dot =
    healthy === null
      ? { color: colors.textMuted, label: "checking…" }
      : healthy
        ? { color: colors.accent.green, label: "API online" }
        : { color: colors.accent.red, label: "API offline" };

  return (
    <div
      style={{
        display: "flex",
        alignItems: "center",
        justifyContent: "flex-end",
        padding: "10px 24px",
        background: colors.surface0,
        borderBottom: `1px solid ${colors.borderSide}`,
        height: 42,
        flexShrink: 0,
      }}
    >
      <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
        <span
          style={{
            width: 8,
            height: 8,
            borderRadius: "50%",
            background: dot.color,
            display: "inline-block",
          }}
        />
        <span style={{ color: dot.color, fontSize: 11 }}>{dot.label}</span>
      </div>
    </div>
  );
}

export default function App() {
  const [healthy, setHealthy] = useState<boolean | null>(null);

  useEffect(() => {
    checkHealth()
      .then(() => setHealthy(true))
      .catch(() => setHealthy(false));
  }, []);

  return (
    <HashRouter>
      <div
        style={{
          display: "flex",
          minHeight: "100vh",
          background: colors.bg,
          color: colors.text,
          fontFamily: typography.fontMono,
        }}
      >
        <Sidebar />

        <div style={{ flex: 1, display: "flex", flexDirection: "column", minWidth: 0 }}>
          <Header healthy={healthy} />

          <main style={{ flex: 1, padding: "24px 28px", overflowY: "auto" }}>
            <Suspense fallback={<div style={{ color: colors.textMuted, padding: 24 }}>Loading…</div>}>
              <Routes>
                <Route path="/" element={<OverviewPage />} />
                <Route path="/gnss" element={<GnssPage />} />
                <Route path="/valuation" element={<ValuationPage />} />
                <Route path="/bayesian" element={<BayesianPage />} />
                <Route path="/entropy" element={<EntropyPage />} />
              </Routes>
            </Suspense>
          </main>
        </div>
      </div>
    </HashRouter>
  );
}
