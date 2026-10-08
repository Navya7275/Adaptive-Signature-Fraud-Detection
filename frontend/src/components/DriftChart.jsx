import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  CartesianGrid,
  ReferenceLine,
  Area,
  AreaChart,
} from "recharts";
import { useTheme } from "../context/ThemeContext";

export default function DriftChart({ signatures = [], threshold = 0.8, height = 320 }) {
  const { theme } = useTheme();
  const isDark = theme === "dark";

  const data = signatures
    .filter((s) => s.similarity_score != null)
    .map((s) => ({
      date: s.capture_date,
      score: s.similarity_score,
      tremor: s.tremor_index,
    }));

  const gridColor = isDark ? "#26262e" : "#e5e7eb";
  const axisColor = isDark ? "#6b7280" : "#9ca3af";

  return (
    <div style={{ width: "100%", height }}>
      <ResponsiveContainer>
        <AreaChart data={data} margin={{ top: 10, right: 20, left: 0, bottom: 10 }}>
          <defs>
            <linearGradient id="scoreGradient" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor="#6366f1" stopOpacity={0.3} />
              <stop offset="95%" stopColor="#6366f1" stopOpacity={0} />
            </linearGradient>
          </defs>
          <CartesianGrid strokeDasharray="3 3" stroke={gridColor} />
          <XAxis
            dataKey="date"
            stroke={axisColor}
            fontSize={11}
            tickFormatter={(v) => {
              const d = new Date(v);
              return d.toLocaleDateString(undefined, { month: "short", year: "2-digit" });
            }}
          />
          <YAxis stroke={axisColor} fontSize={11} domain={[0, 1]} />
          <Tooltip
            contentStyle={{
              backgroundColor: isDark ? "#141419" : "#ffffff",
              border: `1px solid ${gridColor}`,
              borderRadius: 12,
              fontSize: 12,
              boxShadow: "0 10px 30px rgba(0,0,0,0.15)",
            }}
            labelStyle={{ color: isDark ? "#e5e7eb" : "#111827", fontWeight: 600 }}
            formatter={(value, name) => [
              typeof value === "number" ? value.toFixed(3) : value,
              name === "score" ? "Similarity" : "Tremor",
            ]}
          />
          <ReferenceLine
            y={threshold}
            stroke="#ef4444"
            strokeDasharray="4 4"
            strokeWidth={1.5}
            label={{
              value: `Threshold ${threshold.toFixed(2)}`,
              fontSize: 10,
              fill: "#ef4444",
              position: "right",
            }}
          />
          <Area
            type="monotone"
            dataKey="score"
            stroke="#6366f1"
            strokeWidth={2.5}
            fill="url(#scoreGradient)"
            dot={{ r: 3, fill: "#6366f1", strokeWidth: 0 }}
            activeDot={{ r: 5, stroke: "#6366f1", strokeWidth: 2, fill: "#fff" }}
          />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
}