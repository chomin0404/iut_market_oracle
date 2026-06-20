import { lazy, Suspense, useEffect, useState } from "react";
import { HashRouter, Routes, Route, useLocation } from "react-router-dom";
import { checkHealth } from "./api";
import { Sidebar, useIsMobile, NAV_ITEMS } from "./components/Sidebar";
import { Spinner } from "./components/ui";
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
const GraphPage = lazy(() =>
  import("./pages/GraphPage").then((m) => ({ default: m.GraphPage }))
);
const YieldTwinPage = lazy(() =>
  import("./pages/YieldTwinPage").then((m) => ({ default: m.YieldTwinPage }))
);
const StrategyPage = lazy(() =>
  import("./pages/StrategyPage").then((m) => ({ default: m.StrategyPage }))
);

// ---- Page title map (derived from NAV_ITEMS — single source of truth) ----

const PAGE_TITLES: Record<string, string> = Object.fromEntries(
  NAV_ITEMS.map((item) => [item.to, item.label])
);

// ---- Header ----

function Header({
  healthy,
  onMenuClick,
  isMobile,
}: {
  healthy: boolean | null;
  onMenuClick: () => void;
  isMobile: boolean;
}) {
  const location = useLocation();
  const pageTitle = PAGE_TITLES[location.pathname] ?? "Market Oracle";

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
        justifyContent: "space-between",
        padding: "0 24px",
        background: colors.surface0,
        borderBottom: `1px solid ${colors.borderSide}`,
        height: 44,
        flexShrink: 0,
        gap: 12,
      }}
    >
      {/* Left: hamburger (mobile only) + page title */}
      <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
        {isMobile && (
          <button
            onClick={onMenuClick}
            aria-label="Open menu"
            style={{
              background: "none",
              border: "none",
              color: colors.textMuted,
              cursor: "pointer",
              padding: "4px 6px",
              fontSize: 18,
              lineHeight: 1,
              display: "flex",
              alignItems: "center",
            }}
          >
            ☰
          </button>
        )}
        <span
          style={{
            color: colors.textMuted,
            fontSize: 12,
            fontFamily: typography.fontMono,
            letterSpacing: 0.5,
          }}
        >
          {pageTitle}
        </span>
      </div>

      {/* Right: API status indicator */}
      <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
        <span
          style={{
            width: 7,
            height: 7,
            borderRadius: "50%",
            background: dot.color,
            display: "inline-block",
            flexShrink: 0,
          }}
        />
        <span style={{ color: dot.color, fontSize: 11 }}>{dot.label}</span>
      </div>
    </div>
  );
}

// ---- Loading fallback ----

function PageLoader() {
  return (
    <div
      style={{
        display: "flex",
        alignItems: "center",
        gap: 12,
        padding: 32,
        color: colors.textMuted,
        fontSize: 13,
        fontFamily: typography.fontMono,
      }}
    >
      <Spinner />
      <span>Loading…</span>
    </div>
  );
}

// ---- App shell ----

function Shell() {
  const [healthy, setHealthy] = useState<boolean | null>(null);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const isMobile = useIsMobile();

  useEffect(() => {
    checkHealth()
      .then(() => setHealthy(true))
      .catch(() => setHealthy(false));
  }, []);

  // Auto-close sidebar when switching from mobile to desktop
  useEffect(() => {
    if (!isMobile) setSidebarOpen(false);
  }, [isMobile]);

  return (
    <div
      style={{
        display: "flex",
        minHeight: "100vh",
        background: colors.bg,
        color: colors.text,
        fontFamily: typography.fontMono,
      }}
    >
      {/* Sidebar: always rendered on desktop, conditionally open on mobile */}
      {(!isMobile || sidebarOpen) && (
        <Sidebar
          isOpen={sidebarOpen}
          onClose={() => setSidebarOpen(false)}
          isMobile={isMobile}
        />
      )}

      <div style={{ flex: 1, display: "flex", flexDirection: "column", minWidth: 0 }}>
        <Header
          healthy={healthy}
          onMenuClick={() => setSidebarOpen((o) => !o)}
          isMobile={isMobile}
        />

        <main style={{ flex: 1, padding: "24px 28px", overflowY: "auto" }}>
          <Suspense fallback={<PageLoader />}>
            <Routes>
              <Route path="/" element={<OverviewPage />} />
              <Route path="/gnss" element={<GnssPage />} />
              <Route path="/valuation" element={<ValuationPage />} />
              <Route path="/bayesian" element={<BayesianPage />} />
              <Route path="/entropy" element={<EntropyPage />} />
              <Route path="/graph" element={<GraphPage />} />
              <Route path="/yield-twin" element={<YieldTwinPage />} />
              <Route path="/strategy" element={<StrategyPage />} />
            </Routes>
          </Suspense>
        </main>
      </div>
    </div>
  );
}

export default function App() {
  return (
    <HashRouter>
      <Shell />
    </HashRouter>
  );
}
