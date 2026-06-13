import { useNavigate } from "react-router-dom";

interface ModuleCard {
  to: string;
  title: string;
  task: string;
  desc: string;
  color: string;
}

const MODULES: ModuleCard[] = [
  {
    to: "/gnss",
    title: "GNSS Resilience Twin",
    task: "T1300 / T1350 / T1500",
    desc: "10-layer spoofing detection: TESLA chain, IMM-KF, spectral coherence, multi-sensor fusion.",
    color: "#4ade80",
  },
  {
    to: "/valuation",
    title: "Valuation",
    task: "T400",
    desc: "DCF intrinsic value, terminal value, sensitivity table. Reverse-DCF implied growth rate.",
    color: "#60a5fa",
  },
  {
    to: "/bayesian",
    title: "Bayesian Inference",
    task: "T200",
    desc: "Sequential Bayesian update, posterior summary, Bayes factor, credible interval.",
    color: "#f59e0b",
  },
  {
    to: "/entropy",
    title: "Entropy Monitor",
    task: "T1000",
    desc: "Rolling Shannon entropy, KL divergence, entropy rate. Change-point alerts.",
    color: "#a78bfa",
  },
];

export function OverviewPage() {
  const navigate = useNavigate();

  return (
    <div>
      <div style={{ marginBottom: 28 }}>
        <h1
          style={{
            color: "#fff",
            fontSize: 22,
            fontWeight: 700,
            margin: 0,
            letterSpacing: 1,
          }}
        >
          Research Modules
        </h1>
        <p style={{ color: "#555", fontSize: 13, margin: "6px 0 0" }}>
          Select a module to run experiments interactively.
        </p>
      </div>

      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(auto-fill, minmax(280px, 1fr))",
          gap: 16,
        }}
      >
        {MODULES.map((m) => (
          <button
            key={m.to}
            onClick={() => navigate(m.to)}
            style={{
              background: "#161616",
              border: `1px solid #2a2a2a`,
              borderRadius: 8,
              padding: "20px 20px 18px",
              cursor: "pointer",
              textAlign: "left",
              transition: "border-color 0.15s, background 0.15s",
            }}
            onMouseEnter={(e) => {
              (e.currentTarget as HTMLButtonElement).style.borderColor = m.color;
              (e.currentTarget as HTMLButtonElement).style.background = "#1a1a1a";
            }}
            onMouseLeave={(e) => {
              (e.currentTarget as HTMLButtonElement).style.borderColor = "#2a2a2a";
              (e.currentTarget as HTMLButtonElement).style.background = "#161616";
            }}
          >
            <div
              style={{
                color: m.color,
                fontSize: 10,
                fontWeight: 600,
                letterSpacing: 1.5,
                marginBottom: 6,
              }}
            >
              {m.task}
            </div>
            <div
              style={{ color: "#e0e0e0", fontSize: 15, fontWeight: 600, marginBottom: 8 }}
            >
              {m.title}
            </div>
            <div style={{ color: "#666", fontSize: 12, lineHeight: 1.6 }}>{m.desc}</div>
          </button>
        ))}
      </div>
    </div>
  );
}
