/** Design tokens — single source of truth for all visual constants. */

export const colors = {
  /** Page background */
  bg: "#0a0a0a",
  /** Header / sidebar background */
  surface0: "#111",
  /** Card / panel background */
  surface1: "#161616",
  /** Hover state for cards */
  surface1Hover: "#1a1a1a",
  /** Input background */
  inputBg: "#111",
  /** Default border */
  border: "#2a2a2a",
  /** Hover border */
  borderHover: "#3a3a3a",
  /** Sidebar border */
  borderSide: "#222",
  /** Primary text */
  text: "#e0e0e0",
  /** Muted text (labels, subtitles) */
  textMuted: "#888",
  /** Dimmed text (secondary info) */
  textDim: "#666",
  /** Very dim (disabled / footer) */
  textFaint: "#555",

  accent: {
    green: "#4ade80",
    greenDark: "#1a3a1a",
    blue: "#60a5fa",
    blueDark: "#1e3a5f",
    amber: "#f59e0b",
    amberDark: "#2a2a00",
    purple: "#a78bfa",
    purpleDark: "#2d1a5a",
    red: "#f87171",
    redDark: "#1a0000",
    orange: "#fb923c",
    orangeDark: "#2a1800",
  },
} as const;

export const spacing = {
  xs: 4,
  sm: 8,
  md: 12,
  lg: 16,
  xl: 20,
  xxl: 24,
  xxxl: 28,
} as const;

export const radius = {
  sm: 4,
  md: 6,
  lg: 8,
} as const;

export const typography = {
  fontMono: "'Consolas', 'Courier New', monospace",
  xs: 10,
  sm: 11,
  md: 13,
  lg: 15,
  xl: 18,
  xxl: 22,
  normal: 400,
  semibold: 600,
  bold: 700,
} as const;

// ---- Reusable style objects ----

export const cardStyle = {
  background: colors.surface1,
  border: `1px solid ${colors.border}`,
  borderRadius: radius.lg,
  padding: spacing.xl,
  marginBottom: spacing.lg,
} as const;

export const inputStyle = {
  background: colors.inputBg,
  border: `1px solid ${colors.border}`,
  borderRadius: radius.sm,
  color: colors.text,
  padding: "6px 10px",
  fontSize: typography.md,
  width: "100%",
  fontFamily: typography.fontMono,
  boxSizing: "border-box" as const,
} as const;

export function buttonStyle(color: string) {
  return {
    background: color,
    color: "#000",
    border: "none",
    borderRadius: radius.sm,
    padding: "8px 20px",
    fontSize: typography.md,
    fontWeight: typography.bold,
    cursor: "pointer",
    fontFamily: typography.fontMono,
  } as const;
}
