/**
 * Shared UI primitives — use these instead of duplicating inline styles across pages.
 */
import type { ReactNode, CSSProperties } from "react";
import { cardStyle, inputStyle, buttonStyle, colors, typography, spacing, radius } from "../styles/tokens";

// ---- Card ----

interface CardProps {
  children: ReactNode;
  style?: CSSProperties;
}
export function Card({ children, style }: CardProps) {
  return <div style={{ ...cardStyle, ...style }}>{children}</div>;
}

// ---- Section label ----

interface SectionLabelProps {
  color: string;
  children: ReactNode;
}
export function SectionLabel({ color, children }: SectionLabelProps) {
  return (
    <div
      style={{
        color,
        fontSize: typography.sm,
        fontWeight: typography.bold,
        marginBottom: spacing.md,
        letterSpacing: 1.5,
      }}
    >
      {children}
    </div>
  );
}

// ---- Field ----

interface FieldProps {
  label: string;
  children: ReactNode;
  error?: string;
}
export function Field({ label, children, error }: FieldProps) {
  return (
    <label style={{ display: "block", marginBottom: spacing.sm }}>
      <div style={{ color: colors.textMuted, fontSize: typography.sm, marginBottom: spacing.xs }}>
        {label}
      </div>
      {children}
      {error && (
        <div style={{ color: colors.accent.red, fontSize: typography.sm, marginTop: spacing.xs }}>
          {error}
        </div>
      )}
    </label>
  );
}

// ---- Input ----

interface InputProps extends React.InputHTMLAttributes<HTMLInputElement> {
  hasError?: boolean;
}
export function Input({ hasError, style, ...props }: InputProps) {
  return (
    <input
      style={{
        ...inputStyle,
        ...(hasError ? { borderColor: colors.accent.red } : {}),
        ...style,
      }}
      {...props}
    />
  );
}

// ---- Textarea ----

interface TextareaProps extends React.TextareaHTMLAttributes<HTMLTextAreaElement> {
  hasError?: boolean;
}
export function Textarea({ hasError, style, ...props }: TextareaProps) {
  return (
    <textarea
      style={{
        ...inputStyle,
        resize: "vertical",
        ...(hasError ? { borderColor: colors.accent.red } : {}),
        ...style,
      }}
      {...props}
    />
  );
}

// ---- Select ----

export function Select(props: React.SelectHTMLAttributes<HTMLSelectElement>) {
  return <select style={inputStyle} {...props} />;
}

// ---- Button ----

interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  accent: string;
  loading?: boolean;
  loadingLabel?: string;
}
export function Button({ accent, loading, loadingLabel, children, style, ...props }: ButtonProps) {
  return (
    <button
      style={{ ...buttonStyle(accent), opacity: loading ? 0.7 : 1, ...style }}
      disabled={loading || props.disabled}
      {...props}
    >
      {loading ? (
        <span style={{ display: "flex", alignItems: "center", gap: 6, justifyContent: "center" }}>
          <Spinner size={13} color="rgba(0,0,0,0.5)" />
          {loadingLabel ?? "Running…"}
        </span>
      ) : children}
    </button>
  );
}

// ---- ErrorBox ----

export function ErrorBox({ message }: { message: string }) {
  return (
    <div
      style={{
        color: colors.accent.red,
        background: colors.accent.redDark,
        border: `1px solid ${colors.accent.red}`,
        borderRadius: radius.md,
        padding: `${spacing.sm}px ${spacing.lg}px`,
        marginBottom: spacing.lg,
        fontSize: typography.md,
      }}
    >
      Error: {message}
    </div>
  );
}

// ---- StatCard ----

interface StatCardProps {
  label: string;
  value: string;
  color?: string;
}
export function StatCard({ label, value, color = colors.accent.green }: StatCardProps) {
  return (
    <div
      style={{
        background: colors.inputBg,
        border: `1px solid ${colors.border}`,
        borderRadius: radius.md,
        padding: `${spacing.sm + 2}px ${spacing.lg}px`,
        flex: "1 1 140px",
      }}
    >
      <div style={{ color: colors.textMuted, fontSize: typography.sm }}>{label}</div>
      <div style={{ color, fontSize: typography.xl, fontWeight: typography.bold }}>{value}</div>
    </div>
  );
}

// ---- StatsRow (wrapping flex row of StatCards) ----

export function StatsRow({ children }: { children: ReactNode }) {
  return (
    <div style={{ display: "flex", gap: spacing.md, flexWrap: "wrap", marginBottom: spacing.lg }}>
      {children}
    </div>
  );
}

// ---- Page heading ----

interface PageHeadingProps {
  children: ReactNode;
  subtitle?: string;
}
export function PageHeading({ children, subtitle }: PageHeadingProps) {
  return (
    <div style={{ marginBottom: spacing.xl }}>
      <h2
        style={{
          color: colors.text,
          fontSize: typography.xl,
          fontWeight: typography.bold,
          margin: 0,
          letterSpacing: 1,
        }}
      >
        {children}
      </h2>
      {subtitle && (
        <p style={{ color: colors.textFaint, fontSize: typography.md, margin: "6px 0 0" }}>
          {subtitle}
        </p>
      )}
    </div>
  );
}

// ---- Spinner ----

export function Spinner({ color = colors.accent.green, size = 20 }: { color?: string; size?: number }) {
  return (
    <div
      style={{
        width: size,
        height: size,
        border: `2px solid ${color}33`,
        borderTopColor: color,
        borderRadius: "50%",
        animation: "spin 0.8s linear infinite",
        display: "inline-block",
        flexShrink: 0,
      }}
    />
  );
}

// ---- FormGrid ----

export function FormGrid({ children }: { children: ReactNode }) {
  return (
    <div
      style={{
        display: "grid",
        gridTemplateColumns: "repeat(auto-fill, minmax(200px, 1fr))",
        gap: spacing.sm + 2,
        marginBottom: spacing.md,
      }}
    >
      {children}
    </div>
  );
}
