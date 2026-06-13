import { useState } from "react";
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from "recharts";
import { postDcf, postReverseDcf } from "../api";
import type { DCFResponse, ReverseDCFResponse } from "../types";

const CARD = {
  background: "#161616",
  border: "1px solid #2a2a2a",
  borderRadius: 8,
  padding: 20,
  marginBottom: 16,
} as const;

const INPUT_STYLE = {
  background: "#111",
  border: "1px solid #333",
  borderRadius: 4,
  color: "#e0e0e0",
  padding: "6px 10px",
  fontSize: 13,
  width: "100%",
  fontFamily: "inherit",
  boxSizing: "border-box" as const,
};

const BTN_STYLE = {
  background: "#4ade80",
  color: "#000",
  border: "none",
  borderRadius: 4,
  padding: "8px 20px",
  fontSize: 13,
  fontWeight: 700,
  cursor: "pointer",
  fontFamily: "inherit",
};

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div style={{ marginBottom: 10 }}>
      <div style={{ color: "#888", fontSize: 11, marginBottom: 4 }}>{label}</div>
      {children}
    </div>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div
      style={{
        background: "#111",
        border: "1px solid #2a2a2a",
        borderRadius: 6,
        padding: "10px 16px",
        flex: "1 1 140px",
      }}
    >
      <div style={{ color: "#888", fontSize: 11 }}>{label}</div>
      <div style={{ color: "#4ade80", fontSize: 18, fontWeight: 700 }}>{value}</div>
    </div>
  );
}

// ---- DCF Form ----

export function ValuationPage() {
  // DCF state
  const [fcffInput, setFcffInput] = useState("100,110,121,133,146");
  const [tgr, setTgr] = useState("0.03");
  const [wacc, setWacc] = useState("0.08");
  const [shares, setShares] = useState("1000");
  const [debt, setDebt] = useState("200");
  const [dcfResult, setDcfResult] = useState<DCFResponse | null>(null);
  const [dcfErr, setDcfErr] = useState<string | null>(null);
  const [dcfLoading, setDcfLoading] = useState(false);

  // Reverse DCF state
  const [mktPrice, setMktPrice] = useState("25.0");
  const [rdShares, setRdShares] = useState("1000");
  const [rdDebt, setRdDebt] = useState("200");
  const [rdWacc, setRdWacc] = useState("0.08");
  const [rdFcffInput, setRdFcffInput] = useState("100,110,121,133,146");
  const [rdResult, setRdResult] = useState<ReverseDCFResponse | null>(null);
  const [rdErr, setRdErr] = useState<string | null>(null);
  const [rdLoading, setRdLoading] = useState(false);

  async function handleDcf() {
    setDcfLoading(true);
    setDcfErr(null);
    try {
      const fcff = fcffInput.split(",").map((v) => parseFloat(v.trim()));
      const result = await postDcf({
        fcff_series: fcff,
        terminal_growth_rate: parseFloat(tgr),
        wacc: parseFloat(wacc),
        shares_outstanding: parseFloat(shares),
        net_debt: parseFloat(debt),
      });
      setDcfResult(result);
    } catch (e) {
      setDcfErr(e instanceof Error ? e.message : String(e));
    } finally {
      setDcfLoading(false);
    }
  }

  async function handleReverseDcf() {
    setRdLoading(true);
    setRdErr(null);
    try {
      const fcff = rdFcffInput.split(",").map((v) => parseFloat(v.trim()));
      const result = await postReverseDcf({
        market_price: parseFloat(mktPrice),
        shares_outstanding: parseFloat(rdShares),
        net_debt: parseFloat(rdDebt),
        wacc: parseFloat(rdWacc),
        explicit_fcff: fcff,
      });
      setRdResult(result);
    } catch (e) {
      setRdErr(e instanceof Error ? e.message : String(e));
    } finally {
      setRdLoading(false);
    }
  }

  // Sensitivity chart data
  const sensData =
    dcfResult?.sensitivity.map((row) => ({
      label: `g=${(row.growth * 100).toFixed(1)}%`,
      value: parseFloat(row.value.toFixed(2)),
    })) ?? [];

  return (
    <div>
      <h2 style={{ color: "#fff", fontSize: 18, fontWeight: 700, margin: "0 0 20px", letterSpacing: 1 }}>
        Valuation — T400
      </h2>

      {/* DCF */}
      <div style={CARD}>
        <div style={{ color: "#60a5fa", fontSize: 12, fontWeight: 600, marginBottom: 14, letterSpacing: 1 }}>
          DISCOUNTED CASH FLOW
        </div>
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(200px, 1fr))", gap: 10 }}>
          <Field label="FCFF series (comma-separated)">
            <input style={INPUT_STYLE} value={fcffInput} onChange={(e) => setFcffInput(e.target.value)} />
          </Field>
          <Field label="Terminal growth rate">
            <input style={INPUT_STYLE} value={tgr} onChange={(e) => setTgr(e.target.value)} />
          </Field>
          <Field label="WACC">
            <input style={INPUT_STYLE} value={wacc} onChange={(e) => setWacc(e.target.value)} />
          </Field>
          <Field label="Shares outstanding (M)">
            <input style={INPUT_STYLE} value={shares} onChange={(e) => setShares(e.target.value)} />
          </Field>
          <Field label="Net debt (M)">
            <input style={INPUT_STYLE} value={debt} onChange={(e) => setDebt(e.target.value)} />
          </Field>
        </div>
        <button style={BTN_STYLE} onClick={() => void handleDcf()} disabled={dcfLoading}>
          {dcfLoading ? "Running…" : "Run DCF"}
        </button>

        {dcfErr && (
          <div style={{ color: "#f87171", fontSize: 12, marginTop: 10 }}>Error: {dcfErr}</div>
        )}

        {dcfResult && (
          <div style={{ marginTop: 16 }}>
            <div style={{ display: "flex", gap: 12, flexWrap: "wrap", marginBottom: 16 }}>
              <Stat label="Intrinsic Value" value={`$${dcfResult.intrinsic_value.toFixed(2)}`} />
              <Stat label="Terminal Value (M)" value={`$${dcfResult.terminal_value.toFixed(0)}M`} />
              <Stat label="PV of FCFF (M)" value={`$${dcfResult.pv_fcff.toFixed(0)}M`} />
            </div>
            {sensData.length > 0 && (
              <>
                <div style={{ color: "#888", fontSize: 11, marginBottom: 8 }}>
                  Sensitivity (intrinsic value by terminal growth rate)
                </div>
                <ResponsiveContainer width="100%" height={160}>
                  <BarChart data={sensData} margin={{ top: 4, right: 8, left: 0, bottom: 4 }}>
                    <XAxis dataKey="label" tick={{ fill: "#666", fontSize: 10 }} axisLine={false} tickLine={false} />
                    <YAxis tick={{ fill: "#666", fontSize: 10 }} axisLine={false} tickLine={false} width={50} />
                    <Tooltip
                      contentStyle={{ background: "#111", border: "1px solid #333", fontSize: 12 }}
                      labelStyle={{ color: "#888" }}
                      itemStyle={{ color: "#60a5fa" }}
                    />
                    <Bar dataKey="value" radius={[3, 3, 0, 0]}>
                      {sensData.map((_, i) => (
                        <Cell key={i} fill={i === Math.floor(sensData.length / 2) ? "#60a5fa" : "#1e3a5f"} />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </>
            )}
          </div>
        )}
      </div>

      {/* Reverse DCF */}
      <div style={CARD}>
        <div style={{ color: "#60a5fa", fontSize: 12, fontWeight: 600, marginBottom: 14, letterSpacing: 1 }}>
          REVERSE DCF — IMPLIED GROWTH RATE
        </div>
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(200px, 1fr))", gap: 10 }}>
          <Field label="Market price per share ($)">
            <input style={INPUT_STYLE} value={mktPrice} onChange={(e) => setMktPrice(e.target.value)} />
          </Field>
          <Field label="Shares outstanding (M)">
            <input style={INPUT_STYLE} value={rdShares} onChange={(e) => setRdShares(e.target.value)} />
          </Field>
          <Field label="Net debt (M)">
            <input style={INPUT_STYLE} value={rdDebt} onChange={(e) => setRdDebt(e.target.value)} />
          </Field>
          <Field label="WACC">
            <input style={INPUT_STYLE} value={rdWacc} onChange={(e) => setRdWacc(e.target.value)} />
          </Field>
          <Field label="Explicit FCFF (comma-separated)">
            <input style={INPUT_STYLE} value={rdFcffInput} onChange={(e) => setRdFcffInput(e.target.value)} />
          </Field>
        </div>
        <button style={BTN_STYLE} onClick={() => void handleReverseDcf()} disabled={rdLoading}>
          {rdLoading ? "Running…" : "Solve Implied Growth"}
        </button>

        {rdErr && (
          <div style={{ color: "#f87171", fontSize: 12, marginTop: 10 }}>Error: {rdErr}</div>
        )}

        {rdResult && (
          <div style={{ display: "flex", gap: 12, flexWrap: "wrap", marginTop: 16 }}>
            <Stat label="Implied Growth Rate" value={`${(rdResult.implied_growth_rate * 100).toFixed(2)}%`} />
            <Stat label="Terminal Value (M)" value={`$${rdResult.terminal_value.toFixed(0)}M`} />
            <Stat label="PV Explicit FCFF (M)" value={`$${rdResult.pv_explicit.toFixed(0)}M`} />
          </div>
        )}
      </div>
    </div>
  );
}
