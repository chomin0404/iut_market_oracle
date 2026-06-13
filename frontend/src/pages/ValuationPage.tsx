import { useState } from "react";
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from "recharts";
import { postDcf, postReverseDcf } from "../api";
import type { DCFResponse, ReverseDCFResponse } from "../types";
import {
  Card,
  SectionLabel,
  Field,
  Input,
  Button,
  ErrorBox,
  StatCard,
  StatsRow,
  PageHeading,
  FormGrid,
} from "../components/ui";
import { colors } from "../styles/tokens";

// ---- DCF Panel ----

function DcfPanel() {
  const [initialFcf, setInitialFcf] = useState("100");
  const [growthRate, setGrowthRate] = useState("0.10");
  const [discountRate, setDiscountRate] = useState("0.08");
  const [forecastYears, setForecastYears] = useState("5");
  const [terminalGrowth, setTerminalGrowth] = useState("0.03");
  const [result, setResult] = useState<DCFResponse | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  async function handleSubmit() {
    setLoading(true);
    setErr(null);
    try {
      const res = await postDcf({
        initial_fcf: parseFloat(initialFcf),
        growth_rate: parseFloat(growthRate),
        discount_rate: parseFloat(discountRate),
        forecast_years: parseInt(forecastYears, 10),
        terminal_growth_rate: parseFloat(terminalGrowth),
      });
      setResult(res);
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }

  const chartData =
    result?.projected_fcfs.map((fcf, i) => ({
      year: `Y${i + 1}`,
      projected: parseFloat(fcf.toFixed(1)),
      discounted: parseFloat(result.discounted_fcfs[i].toFixed(1)),
    })) ?? [];

  return (
    <Card>
      <SectionLabel color={colors.accent.blue}>DISCOUNTED CASH FLOW</SectionLabel>
      <FormGrid>
        <Field label="Initial FCF (M)">
          <Input value={initialFcf} onChange={(e) => setInitialFcf(e.target.value)} />
        </Field>
        <Field label="Growth rate (e.g. 0.10)">
          <Input value={growthRate} onChange={(e) => setGrowthRate(e.target.value)} />
        </Field>
        <Field label="Discount rate / WACC (e.g. 0.08)">
          <Input value={discountRate} onChange={(e) => setDiscountRate(e.target.value)} />
        </Field>
        <Field label="Forecast years">
          <Input
            type="number"
            min={1}
            max={20}
            value={forecastYears}
            onChange={(e) => setForecastYears(e.target.value)}
          />
        </Field>
        <Field label="Terminal growth rate (e.g. 0.03)">
          <Input value={terminalGrowth} onChange={(e) => setTerminalGrowth(e.target.value)} />
        </Field>
      </FormGrid>

      <Button accent={colors.accent.blue} loading={loading} onClick={() => void handleSubmit()}>
        Run DCF
      </Button>

      {err && <ErrorBox message={err} />}

      {result && (
        <>
          <StatsRow>
            <StatCard
              label="Enterprise Value (M)"
              value={`$${result.enterprise_value.toFixed(2)}M`}
              color={colors.accent.blue}
            />
            <StatCard
              label="Terminal Value (M)"
              value={`$${result.terminal_value.toFixed(2)}M`}
              color={colors.accent.blue}
            />
            <StatCard
              label="Disc. Terminal Value (M)"
              value={`$${result.discounted_terminal_value.toFixed(2)}M`}
              color={colors.accent.blue}
            />
          </StatsRow>
          <div style={{ color: colors.textMuted, fontSize: 11, marginBottom: 8 }}>
            Projected vs discounted FCF by year
          </div>
          <ResponsiveContainer width="100%" height={160}>
            <BarChart data={chartData} margin={{ top: 4, right: 8, left: 0, bottom: 4 }}>
              <XAxis
                dataKey="year"
                tick={{ fill: colors.textDim, fontSize: 10 }}
                axisLine={false}
                tickLine={false}
              />
              <YAxis
                tick={{ fill: colors.textDim, fontSize: 10 }}
                axisLine={false}
                tickLine={false}
                width={50}
              />
              <Tooltip
                contentStyle={{ background: colors.surface0, border: `1px solid ${colors.border}`, fontSize: 12 }}
                labelStyle={{ color: colors.textMuted }}
              />
              <Bar dataKey="projected" name="Projected FCF" radius={[3, 3, 0, 0]}>
                {chartData.map((_, i) => (
                  <Cell key={i} fill={colors.accent.blueDark} />
                ))}
              </Bar>
              <Bar dataKey="discounted" name="Discounted FCF" radius={[3, 3, 0, 0]}>
                {chartData.map((_, i) => (
                  <Cell key={i} fill={colors.accent.blue} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </>
      )}
    </Card>
  );
}

// ---- Reverse DCF Panel ----

function ReverseDcfPanel() {
  const [targetEv, setTargetEv] = useState("800");
  const [initialFcf, setInitialFcf] = useState("50");
  const [discountRate, setDiscountRate] = useState("0.10");
  const [forecastYears, setForecastYears] = useState("5");
  const [terminalGrowth, setTerminalGrowth] = useState("0.025");
  const [result, setResult] = useState<ReverseDCFResponse | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  async function handleSubmit() {
    setLoading(true);
    setErr(null);
    try {
      const res = await postReverseDcf({
        target_enterprise_value: parseFloat(targetEv),
        initial_fcf: parseFloat(initialFcf),
        discount_rate: parseFloat(discountRate),
        forecast_years: parseInt(forecastYears, 10),
        terminal_growth_rate: parseFloat(terminalGrowth),
      });
      setResult(res);
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }

  return (
    <Card>
      <SectionLabel color={colors.accent.blue}>REVERSE DCF — IMPLIED GROWTH RATE</SectionLabel>
      <FormGrid>
        <Field label="Target enterprise value (M)">
          <Input value={targetEv} onChange={(e) => setTargetEv(e.target.value)} />
        </Field>
        <Field label="Initial FCF (M)">
          <Input value={initialFcf} onChange={(e) => setInitialFcf(e.target.value)} />
        </Field>
        <Field label="Discount rate / WACC">
          <Input value={discountRate} onChange={(e) => setDiscountRate(e.target.value)} />
        </Field>
        <Field label="Forecast years">
          <Input
            type="number"
            min={1}
            max={20}
            value={forecastYears}
            onChange={(e) => setForecastYears(e.target.value)}
          />
        </Field>
        <Field label="Terminal growth rate">
          <Input value={terminalGrowth} onChange={(e) => setTerminalGrowth(e.target.value)} />
        </Field>
      </FormGrid>

      <Button accent={colors.accent.blue} loading={loading} onClick={() => void handleSubmit()}>
        Solve Implied Growth
      </Button>

      {err && <ErrorBox message={err} />}

      {result && (
        <StatsRow>
          <StatCard
            label="Implied Growth Rate"
            value={`${(result.implied_growth_rate * 100).toFixed(3)}%`}
            color={colors.accent.blue}
          />
        </StatsRow>
      )}
    </Card>
  );
}

// ---- Page ----

export function ValuationPage() {
  return (
    <div>
      <PageHeading subtitle="T400 — DCF intrinsic value, terminal value, implied growth rate">
        Valuation
      </PageHeading>
      <DcfPanel />
      <ReverseDcfPanel />
    </div>
  );
}
