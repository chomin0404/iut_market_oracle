import { colors } from "../styles/tokens";

export function GnssPage() {
  return (
    <div>
      <div style={{ marginBottom: 28 }}>
        <h1
          style={{
            color: colors.text,
            fontSize: 22,
            fontWeight: 700,
            margin: 0,
            letterSpacing: 1,
          }}
        >
          GNSS Resilience Twin
        </h1>
        <p style={{ color: colors.textFaint, fontSize: 13, margin: "6px 0 0" }}>
          T1300 / T1350 / T1500
        </p>
      </div>

      <div
        style={{
          background: colors.surface1,
          border: `1px solid ${colors.border}`,
          borderRadius: 8,
          padding: "24px 28px",
          maxWidth: 560,
        }}
      >
        <div
          style={{
            color: colors.accent.amber,
            fontSize: 11,
            fontWeight: 600,
            letterSpacing: 1.5,
            marginBottom: 10,
          }}
        >
          MODULE RELOCATED
        </div>
        <p style={{ color: colors.text, fontSize: 14, margin: "0 0 8px", lineHeight: 1.6 }}>
          The GNSS Resilience Twin module has been extracted to a dedicated repository:
        </p>
        <code
          style={{
            display: "block",
            color: colors.accent.green,
            background: colors.bg,
            border: `1px solid ${colors.border}`,
            borderRadius: 4,
            padding: "8px 12px",
            fontSize: 13,
            marginBottom: 12,
          }}
        >
          iut-gnss-resilience-twin
        </code>
        <p style={{ color: colors.textDim, fontSize: 12, margin: 0, lineHeight: 1.6 }}>
          10-layer spoofing detection — TESLA chain, IMM-KF, spectral coherence, multi-sensor
          fusion. Run the standalone API server from that repository to use this module.
        </p>
      </div>
    </div>
  );
}
