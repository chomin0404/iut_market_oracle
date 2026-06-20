import { useEffect } from "react";
import { NavLink } from "react-router-dom";
import { colors } from "../styles/tokens";
import { useIsMobile } from "../hooks/useIsMobile";

export { useIsMobile };

interface NavItem {
  to: string;
  label: string;
  icon: string;
}

export const NAV_ITEMS: NavItem[] = [
  { to: "/", label: "Overview", icon: "⬡" },
  { to: "/valuation", label: "Valuation", icon: "◇" },
  { to: "/bayesian", label: "Bayesian", icon: "◉" },
  { to: "/entropy", label: "Entropy", icon: "∿" },
  { to: "/graph", label: "Graph", icon: "◎" },
  { to: "/yield-twin", label: "Yield Twin", icon: "◈" },
  { to: "/strategy", label: "Strategy", icon: "▲" },
];

interface SidebarProps {
  isOpen: boolean;
  onClose: () => void;
  isMobile: boolean;
}

export function Sidebar({ isOpen, onClose, isMobile }: SidebarProps) {
  // Close on Escape key
  useEffect(() => {
    if (!isMobile) return;
    const handler = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    document.addEventListener("keydown", handler);
    return () => document.removeEventListener("keydown", handler);
  }, [isMobile, onClose]);

  const sidebarStyle: React.CSSProperties = isMobile
    ? {
        position: "fixed",
        top: 0,
        left: 0,
        width: 220,
        height: "100vh",
        zIndex: 200,
        transform: isOpen ? "translateX(0)" : "translateX(-100%)",
        transition: "transform 0.2s ease",
        background: colors.surface0,
        borderRight: `1px solid ${colors.borderSide}`,
        display: "flex",
        flexDirection: "column",
        padding: "16px 0",
      }
    : {
        width: 220,
        minHeight: "100vh",
        background: colors.surface0,
        borderRight: `1px solid ${colors.borderSide}`,
        display: "flex",
        flexDirection: "column",
        padding: "16px 0",
        flexShrink: 0,
      };

  return (
    <>
      {/* Mobile backdrop */}
      {isMobile && isOpen && (
        <div
          onClick={onClose}
          style={{
            position: "fixed",
            inset: 0,
            background: "rgba(0,0,0,0.6)",
            zIndex: 199,
            animation: "fadeIn 0.15s ease",
          }}
        />
      )}

      <nav style={sidebarStyle}>
        <div
          style={{
            padding: "0 20px 20px",
            borderBottom: `1px solid ${colors.borderSide}`,
            marginBottom: 8,
          }}
        >
          <div style={{ color: colors.text, fontWeight: 700, fontSize: 13, letterSpacing: 1 }}>
            IUT MARKET ORACLE
          </div>
          <div style={{ color: colors.textFaint, fontSize: 10, marginTop: 2 }}>
            Quant Research Platform
          </div>
        </div>

        {NAV_ITEMS.map((item) => (
          <NavLink
            key={item.to}
            to={item.to}
            end={item.to === "/"}
            onClick={isMobile ? onClose : undefined}
            style={({ isActive }) => ({
              display: "flex",
              alignItems: "center",
              gap: 10,
              padding: "10px 20px",
              color: isActive ? colors.text : colors.textDim,
              background: isActive ? colors.surface1Hover : "transparent",
              borderLeft: isActive ? `2px solid ${colors.accent.green}` : "2px solid transparent",
              textDecoration: "none",
              fontSize: 13,
              fontFamily: "'Consolas', monospace",
              transition: "color 0.15s, background 0.15s",
            })}
            onMouseEnter={(e) => {
              const el = e.currentTarget as HTMLAnchorElement;
              if (!el.style.background.includes(colors.surface1Hover)) {
                el.style.color = colors.text;
                el.style.background = colors.surface1Hover;
              }
            }}
            onMouseLeave={(e) => {
              const el = e.currentTarget as HTMLAnchorElement;
              if (!el.classList.contains("active") && !el.getAttribute("aria-current")) {
                el.style.color = "";
                el.style.background = "";
              }
            }}
          >
            <span style={{ fontSize: 14, width: 16, textAlign: "center" }}>{item.icon}</span>
            {item.label}
          </NavLink>
        ))}
      </nav>
    </>
  );
}

