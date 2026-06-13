import { NavLink } from "react-router-dom";

interface NavItem {
  to: string;
  label: string;
  icon: string;
}

const NAV_ITEMS: NavItem[] = [
  { to: "/", label: "Overview", icon: "⬡" },
  { to: "/gnss", label: "GNSS Twin", icon: "◈" },
  { to: "/valuation", label: "Valuation", icon: "◇" },
  { to: "/bayesian", label: "Bayesian", icon: "◉" },
  { to: "/entropy", label: "Entropy", icon: "◈" },
];

export function Sidebar() {
  return (
    <nav
      style={{
        width: 220,
        minHeight: "100vh",
        background: "#111",
        borderRight: "1px solid #222",
        display: "flex",
        flexDirection: "column",
        padding: "16px 0",
        flexShrink: 0,
      }}
    >
      <div
        style={{
          padding: "0 20px 20px",
          borderBottom: "1px solid #222",
          marginBottom: 8,
        }}
      >
        <div style={{ color: "#fff", fontWeight: 700, fontSize: 13, letterSpacing: 1 }}>
          IUT MARKET ORACLE
        </div>
        <div style={{ color: "#444", fontSize: 10, marginTop: 2 }}>
          Quant Research Platform
        </div>
      </div>

      {NAV_ITEMS.map((item) => (
        <NavLink
          key={item.to}
          to={item.to}
          end={item.to === "/"}
          style={({ isActive }) => ({
            display: "flex",
            alignItems: "center",
            gap: 10,
            padding: "10px 20px",
            color: isActive ? "#fff" : "#666",
            background: isActive ? "#1a1a1a" : "transparent",
            borderLeft: isActive ? "2px solid #4ade80" : "2px solid transparent",
            textDecoration: "none",
            fontSize: 13,
            fontFamily: "'Consolas', monospace",
            transition: "color 0.15s, background 0.15s",
          })}
        >
          <span style={{ fontSize: 14 }}>{item.icon}</span>
          {item.label}
        </NavLink>
      ))}
    </nav>
  );
}
