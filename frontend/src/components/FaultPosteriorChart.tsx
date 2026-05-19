import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  ReferenceLine,
} from "recharts";
import type { EpochReport } from "../types";

interface Props {
  epochs: EpochReport[];
  alertEpochs: number[];
}

interface ChartRow {
  epoch: number;
  nominal: number;
  multipath: number;
  hardware_fault: number;
  spoofing: number;
  confidence: number;
}

const COLORS = {
  nominal: "#4ade80",
  multipath: "#facc15",
  hardware_fault: "#fb923c",
  spoofing: "#f87171",
};

export function FaultPosteriorChart({ epochs, alertEpochs }: Props) {
  const data: ChartRow[] = epochs.map((e) => ({
    epoch: e.epoch,
    nominal: parseFloat((e.fault_posterior.nominal * 100).toFixed(1)),
    multipath: parseFloat((e.fault_posterior.multipath * 100).toFixed(1)),
    hardware_fault: parseFloat((e.fault_posterior.hardware_fault * 100).toFixed(1)),
    spoofing: parseFloat((e.fault_posterior.spoofing * 100).toFixed(1)),
    confidence: parseFloat((e.confidence * 100).toFixed(1)),
  }));

  const alertSet = new Set(alertEpochs);

  return (
    <div style={{ background: "#111", borderRadius: 8, padding: 16 }}>
      <h3 style={{ color: "#ddd", margin: "0 0 12px", fontSize: 14 }}>
        Fault Posterior per Epoch [%]
      </h3>
      <ResponsiveContainer width="100%" height={260}>
        <AreaChart data={data} margin={{ top: 4, right: 16, left: 0, bottom: 4 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#333" />
          <XAxis
            dataKey="epoch"
            stroke="#666"
            tick={{ fill: "#888", fontSize: 11 }}
            label={{ value: "Epoch", position: "insideBottomRight", offset: -4, fill: "#666", fontSize: 11 }}
          />
          <YAxis
            stroke="#666"
            tick={{ fill: "#888", fontSize: 11 }}
            domain={[0, 100]}
            tickFormatter={(v: number) => `${v}%`}
          />
          <Tooltip
            contentStyle={{ background: "#1a1a1a", border: "1px solid #444", fontSize: 12 }}
            formatter={(value: number, name: string) => [`${value}%`, name]}
          />
          <Legend
            wrapperStyle={{ fontSize: 12, color: "#aaa" }}
          />
          {alertEpochs.map((ep) => (
            <ReferenceLine
              key={ep}
              x={ep}
              stroke={alertSet.has(ep) ? "#f87171" : undefined}
              strokeDasharray="4 2"
              strokeOpacity={0.6}
            />
          ))}
          <Area
            type="monotone"
            dataKey="nominal"
            stackId="1"
            stroke={COLORS.nominal}
            fill={COLORS.nominal}
            fillOpacity={0.7}
          />
          <Area
            type="monotone"
            dataKey="multipath"
            stackId="1"
            stroke={COLORS.multipath}
            fill={COLORS.multipath}
            fillOpacity={0.7}
          />
          <Area
            type="monotone"
            dataKey="hardware_fault"
            stackId="1"
            stroke={COLORS.hardware_fault}
            fill={COLORS.hardware_fault}
            fillOpacity={0.7}
          />
          <Area
            type="monotone"
            dataKey="spoofing"
            stackId="1"
            stroke={COLORS.spoofing}
            fill={COLORS.spoofing}
            fillOpacity={0.7}
          />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
}
