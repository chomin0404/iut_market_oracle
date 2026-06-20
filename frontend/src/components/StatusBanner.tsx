import type { RecommendedAction } from "../types";
import { colors } from "../styles/tokens";

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
    bg: colors.accent.greenDark,
    fg: colors.accent.green,
    icon: "✓",
  },
  monitor: {
    label: "MONITOR",
    bg: colors.accent.amberDark,
    fg: "#facc15",
    icon: "⚑",
  },
  reduce_trust: {
    label: "REDUCE TRUST",
    bg: colors.accent.orangeDark,
    fg: colors.accent.orange,
    icon: "⚠",
  },
  switch_source: {
    label: "SWITCH SOURCE",
    bg: colors.accent.orangeDark,
    fg: "#f97316",
    icon: "↔",
  },
  ground_immediately: {
    label: "GROUND IMMEDIATELY",
    bg: colors.accent.redDark,
    fg: colors.accent.red,
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
        <div style={{ color: colors.text, fontSize: 13, marginTop: 4 }}>{reason}</div>
      </div>
      {runId && (
        <div style={{ color: colors.textDim, fontSize: 11, alignSelf: "flex-end" }}>
          run: {runId}
        </div>
      )}
    </div>
  );
}
