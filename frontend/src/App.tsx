import { useEffect, useState } from "react";
import { HashRouter, Routes, Route } from "react-router-dom";
import { checkHealth } from "./api";
import { Sidebar } from "./components/Sidebar";
import { OverviewPage } from "./pages/OverviewPage";
import { GnssPage } from "./pages/GnssPage";
import { ValuationPage } from "./pages/ValuationPage";
import { BayesianPage } from "./pages/BayesianPage";
import { EntropyPage } from "./pages/EntropyPage";

const PAGE_BG = "#0a0a0a";

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
        justifyContent: "flex-end",
        padding: "10px 24px",
        background: "#111",
        borderBottom: "1px solid #222",
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
          background: PAGE_BG,
          color: "#e0e0e0",
          fontFamily: "'Consolas', 'Courier New', monospace",
        }}
      >
        <Sidebar />

        <div style={{ flex: 1, display: "flex", flexDirection: "column", minWidth: 0 }}>
          <Header healthy={healthy} />

          <main style={{ flex: 1, padding: "24px 28px", overflowY: "auto" }}>
            <Routes>
              <Route path="/" element={<OverviewPage />} />
              <Route path="/gnss" element={<GnssPage />} />
              <Route path="/valuation" element={<ValuationPage />} />
              <Route path="/bayesian" element={<BayesianPage />} />
              <Route path="/entropy" element={<EntropyPage />} />
            </Routes>
          </main>
        </div>
      </div>
    </HashRouter>
  );
}
