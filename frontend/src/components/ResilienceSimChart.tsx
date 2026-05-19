import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Cell,
  ResponsiveContainer,
} from "recharts";
import type { ResilienceTwinReport } from "../types";

interface Props {
  report: ResilienceTwinReport;
}

const CLASS_ORDER = ["nominal", "multipath", "hardware_fault", "spoofing"];
const CLASS_COLORS: Record<string, string> = {
  nominal: "#4ade80",
  multipath: "#facc15",
  hardware_fault: "#fb923c",
  spoofing: "#f87171",
};

export function ResilienceSimChart({ report }: Props) {
  const accuracyData = CLASS_ORDER.map((cls) => ({
    name: cls.replace("_", "\n"),
    accuracy: parseFloat(((report.per_class_accuracy[cls] ?? 0) * 100).toFixed(1)),
    color: CLASS_COLORS[cls] ?? "#888",
  }));

  // Confusion matrix display
  const cmLabels = ["NOM", "MP", "HW", "SPOOF"];

  return (
    <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16 }}>
      {/* Per-class accuracy bar chart */}
      <div style={{ background: "#111", borderRadius: 8, padding: 16 }}>
        <h3 style={{ color: "#ddd", margin: "0 0 4px", fontSize: 14 }}>
          Per-class Accuracy [%]
        </h3>
        <div style={{ color: "#888", fontSize: 11, marginBottom: 12 }}>
          AUC = {report.auc.toFixed(3)} &nbsp;|&nbsp; DR ={" "}
          {(report.p_detection * 100).toFixed(1)}% &nbsp;|&nbsp; FAR ={" "}
          {(report.p_false_alarm * 100).toFixed(1)}% &nbsp;|&nbsp; Conf ={" "}
          {(report.mean_confidence * 100).toFixed(1)}%
        </div>
        <ResponsiveContainer width="100%" height={200}>
          <BarChart
            data={accuracyData}
            margin={{ top: 4, right: 8, left: 0, bottom: 4 }}
          >
            <CartesianGrid strokeDasharray="3 3" stroke="#333" />
            <XAxis
              dataKey="name"
              stroke="#666"
              tick={{ fill: "#888", fontSize: 10 }}
            />
            <YAxis
              stroke="#666"
              tick={{ fill: "#888", fontSize: 11 }}
              domain={[0, 100]}
              tickFormatter={(v: number) => `${v}%`}
            />
            <Tooltip
              contentStyle={{
                background: "#1a1a1a",
                border: "1px solid #444",
                fontSize: 12,
              }}
              formatter={(value: number) => [`${value}%`, "Accuracy"]}
            />
            <Bar dataKey="accuracy" radius={[4, 4, 0, 0]}>
              {accuracyData.map((entry, index) => (
                <Cell key={index} fill={entry.color} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* Confusion matrix */}
      <div style={{ background: "#111", borderRadius: 8, padding: 16 }}>
        <h3 style={{ color: "#ddd", margin: "0 0 4px", fontSize: 14 }}>
          Confusion Matrix
        </h3>
        <div style={{ color: "#888", fontSize: 11, marginBottom: 12 }}>
          n_mc = {report.n_mc} trials (rows = ground truth, cols = predicted)
        </div>
        <table
          style={{
            borderCollapse: "collapse",
            width: "100%",
            fontSize: 12,
          }}
        >
          <thead>
            <tr>
              <th style={{ color: "#666", padding: "4px 8px" }}></th>
              {cmLabels.map((l) => (
                <th
                  key={l}
                  style={{ color: "#aaa", padding: "4px 8px", fontWeight: 600 }}
                >
                  {l}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {report.confusion_matrix.map((row, ri) => {
              const total = row.reduce((a, b) => a + b, 0);
              return (
                <tr key={ri}>
                  <td
                    style={{
                      color: CLASS_COLORS[CLASS_ORDER[ri]] ?? "#aaa",
                      padding: "4px 8px",
                      fontWeight: 600,
                    }}
                  >
                    {cmLabels[ri]}
                  </td>
                  {row.map((val, ci) => {
                    const pct = total > 0 ? val / total : 0;
                    const isDiag = ri === ci;
                    return (
                      <td
                        key={ci}
                        style={{
                          padding: "4px 8px",
                          textAlign: "center",
                          color: isDiag ? "#fff" : "#666",
                          background: isDiag
                            ? `rgba(74,222,128,${0.1 + pct * 0.5})`
                            : val > 0
                              ? `rgba(248,113,113,${pct * 0.4})`
                              : "transparent",
                          borderRadius: 3,
                        }}
                      >
                        {val}
                      </td>
                    );
                  })}
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}
