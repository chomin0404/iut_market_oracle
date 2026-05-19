import type { RecommendedAction } from "../types";

interface Props {
  action: RecommendedAction;
  reason: string;
  runId?: string;
}

const ACTION_CONFIG: Record<
  RecommendedAction,
  { label: string; bg: string; fg: string; icon: string }
> = {
  nominal: {
    label: "NOMINAL",
    bg: "#1a3a1a",
    fg: "#4ade80",
    icon: "✓",
  },
  monitor: {
    label: "MONITOR",
    bg: "#2a2a00",
    fg: "#facc15",
    icon: "⚑",
  },
  reduce_trust: {
    label: "REDUCE TRUST",
    bg: "#2a1800",
    fg: "#fb923c",
    icon: "⚠",
  },
  switch_source: {
    label: "SWITCH SOURCE",
    bg: "#2a1800",
    fg: "#f97316",
    icon: "↔",
  },
  ground_immediately: {
    label: "GROUND IMMEDIATELY",
    bg: "#3a0000",
    fg: "#f87171",
    icon: "✕",
  },
};

export function StatusBanner({ action, reason, runId }: Props) {
  const cfg = ACTION_CONFIG[action];
  return (
    <div
      style={{
        background: cfg.bg,
        border: `2px solid ${cfg.fg}`,
        borderRadius: 8,
        padding: "12px 20px",
        marginBottom: 16,
        display: "flex",
        alignItems: "flex-start",
        gap: 16,
      }}
    >
      <span style={{ fontSize: 28, color: cfg.fg, lineHeight: 1 }}>
        {cfg.icon}
      </span>
      <div style={{ flex: 1 }}>
        <div
          style={{
            color: cfg.fg,
            fontSize: 18,
            fontWeight: 700,
            letterSpacing: 1,
          }}
        >
          {cfg.label}
        </div>
        <div style={{ color: "#ccc", fontSize: 13, marginTop: 4 }}>{reason}</div>
      </div>
      {runId && (
        <div style={{ color: "#666", fontSize: 11, alignSelf: "flex-end" }}>
          run: {runId}
        </div>
      )}
    </div>
  );
}
