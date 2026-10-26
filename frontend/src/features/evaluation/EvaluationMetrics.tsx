import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { api, BenchmarkResult } from "../../lib/api";
import { ShieldCheck, BarChart3, Info, CheckCircle, TrendingDown } from "lucide-react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
} from "recharts";

const PALETTE = ["#818cf8", "#34d399", "#38bdf8", "#fbbf24", "#f472b6", "#a78bfa"];

export default function EvaluationMetrics() {
  const { data: benchmarks = [], isLoading } = useQuery<BenchmarkResult[]>({
    queryKey: ["benchmarks"],
    queryFn: api.benchmarks,
  });

  const [metricTab, setMetricTab] = useState<"mse" | "ssim" | "psnr">("mse");
  const horizons = Array.from({ length: 10 }, (_, i) => i + 1);

  return (
    <div className="space-y-6 max-w-7xl mx-auto">
      {/* Header */}
      <div>
        <div className="flex items-center gap-2 mb-1">
          <span className="px-2.5 py-0.5 text-xs font-mono rounded bg-indigo-950/60 border border-indigo-800/40 text-indigo-300">
            Held-Out Test Benchmarks
          </span>
          <span className="text-xs text-slate-400">· 100 Strictly Disjoint KTH Test Windows</span>
        </div>
        <h1 className="text-2xl md:text-3xl font-bold tracking-tight text-slate-100">
          Evaluation Metrics & Scientific Benchmarks
        </h1>
        <p className="text-slate-400 text-sm mt-1">
          Standardized held-out test evaluation on KTH-64 (Persons 17-25, completely unobserved during training).
          Strict autoregressive rollout (T_in=10 → T_out=10) with zero ground-truth leakage.
        </p>
      </div>

      {/* Verified Benchmark Table */}
      <div className="rounded-xl border border-slate-800/80 bg-slate-900/60 overflow-hidden shadow-lg">
        <div className="px-5 py-3.5 border-b border-slate-800/80 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <ShieldCheck className="w-4 h-4 text-emerald-400" />
            <h2 className="text-sm font-bold text-slate-100">
              Official Benchmark Leaderboard (Held-Out Test Split)
            </h2>
          </div>
          <span className="text-xs font-mono text-slate-400">
            Evaluated on: NVIDIA GeForce RTX 3050 Laptop GPU
          </span>
        </div>

        {isLoading ? (
          <div className="p-8 text-center text-xs text-slate-400 font-mono">Loading benchmarks…</div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-xs text-left">
              <thead className="bg-slate-950/60 text-slate-400 font-mono uppercase text-[11px] border-b border-slate-800/60">
                <tr>
                  <th className="px-4 py-3">Model</th>
                  <th className="px-3 py-3">Architecture</th>
                  <th className="px-3 py-3">Checkpoint Fingerprint</th>
                  <th className="px-3 py-3 font-semibold text-slate-200">Test MSE ↓</th>
                  <th className="px-3 py-3 font-semibold text-slate-200">Test MAE ↓</th>
                  <th className="px-3 py-3 font-semibold text-slate-200">Test PSNR ↑</th>
                  <th className="px-3 py-3 font-semibold text-slate-200">Test SSIM ↑</th>
                  <th className="px-3 py-3">Latency</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800/50 font-mono">
                {benchmarks.map((b) => {
                  const isQPred50 = b.model_id === "qpredrnn_50ep";
                  const isPilotQPred = b.model_id === "qpredrnn_pilot";
                  const isPredRNN = b.model_id === "predrnn_v2_main";

                  return (
                    <tr
                      key={b.model_id}
                      className={`hover:bg-slate-800/30 transition-colors ${
                        isQPred50 ? "bg-indigo-950/15" : ""
                      }`}
                    >
                      <td className="px-4 py-3 font-sans font-bold text-slate-100">
                        {b.display_name}
                        {isQPred50 && (
                          <span className="ml-2 px-2 py-0.5 rounded text-[10px] font-mono bg-indigo-950 text-indigo-300 border border-indigo-700/60">
                            Primary Candidate
                          </span>
                        )}
                      </td>
                      <td className="px-3 py-3 text-slate-300 font-sans">
                        {b.architecture}
                      </td>
                      <td className="px-3 py-3 text-slate-400 text-[10px]">
                        {b.checkpoint_sha256 ? `${b.checkpoint_sha256.slice(0, 10)}…` : "Analytical"}
                      </td>
                      <td className="px-3 py-3 font-bold text-slate-100">
                        {b.metrics.mse.toFixed(6)}
                      </td>
                      <td className="px-3 py-3 text-slate-300">
                        {b.metrics.mae.toFixed(6)}
                      </td>
                      <td className="px-3 py-3 text-slate-300">
                        {b.metrics.psnr.toFixed(2)} dB
                      </td>
                      <td className={`px-3 py-3 font-bold ${isQPred50 || isPilotQPred ? "text-cyan-400" : "text-slate-200"}`}>
                        {b.metrics.ssim.toFixed(4)}
                      </td>
                      <td className="px-3 py-3 text-slate-400">
                        {b.metrics.latency_ms_per_window.toFixed(1)} ms/clip
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {/* Degradation Chart */}
      <div className="p-5 rounded-xl border border-slate-800/80 bg-slate-900/60 space-y-4">
        <div className="flex items-center justify-between">
          <div>
            <h3 className="text-sm font-bold text-slate-100">Per-Horizon Lead-Time Degradation Curve</h3>
            <p className="text-xs text-slate-400">
              Evaluates metric behavior across prediction lead times $t \in [1, 10]$ (frames 11 to 20).
            </p>
          </div>
          <div className="flex items-center gap-1.5 p-1 rounded-lg bg-slate-950 border border-slate-800">
            {(["mse", "ssim", "psnr"] as const).map((tab) => (
              <button
                key={tab}
                onClick={() => setMetricTab(tab)}
                className={`px-3 py-1 text-xs rounded font-medium uppercase transition-all ${
                  metricTab === tab ? "bg-indigo-600 text-white font-bold" : "text-slate-400 hover:text-slate-200"
                }`}
              >
                {tab}
              </button>
            ))}
          </div>
        </div>

        <div className="h-64 w-full">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart margin={{ top: 10, right: 20, bottom: 10, left: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(51, 65, 85, 0.4)" />
              <XAxis
                type="number"
                dataKey="t"
                domain={[1, 10]}
                tick={{ fill: "#94a3b8", fontSize: 11 }}
                label={{ value: "Forecast Lead Time Horizon (t)", position: "insideBottom", offset: -5, fill: "#94a3b8", fontSize: 11 }}
              />
              <YAxis
                tick={{ fill: "#94a3b8", fontSize: 11 }}
                domain={metricTab === "ssim" ? [0.65, 1.0] : ["auto", "auto"]}
              />
              <Tooltip
                contentStyle={{
                  backgroundColor: "rgba(15, 23, 42, 0.95)",
                  borderColor: "rgba(51, 65, 85, 0.8)",
                  borderRadius: 8,
                  fontSize: 12,
                }}
              />
              <Legend wrapperStyle={{ fontSize: 11, paddingTop: 10 }} />
              {benchmarks.map((b, i) => {
                const data = horizons.map((t) => ({
                  t,
                  val: b.per_horizon[metricTab][t - 1],
                }));
                return (
                  <Line
                    key={b.model_id}
                    data={data}
                    type="monotone"
                    dataKey="val"
                    stroke={PALETTE[i % PALETTE.length]}
                    strokeWidth={b.model_id.includes("qpred") ? 2.8 : 1.8}
                    dot={{ r: 2.5 }}
                    name={b.display_name}
                  />
                );
              })}
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Metric Definitions & Methodology */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div className="p-5 rounded-xl border border-slate-800/80 bg-slate-900/60 space-y-3">
          <h3 className="text-sm font-bold text-slate-100 flex items-center gap-2">
            <Info className="w-4 h-4 text-cyan-400" /> Metric Formulations
          </h3>
          <div className="space-y-3 text-xs">
            <div>
              <div className="font-semibold text-cyan-300 font-mono">Mean Squared Error (MSE)</div>
              <div className="text-slate-400 text-[11px]">(1/N) * sum((Y_pred - Y_true)^2) · Lower is better · Pixel values in [0, 1].</div>
            </div>
            <div>
              <div className="font-semibold text-cyan-300 font-mono">Peak Signal-to-Noise Ratio (PSNR)</div>
              <div className="text-slate-400 text-[11px]">10 * log10(1.0^2 / MSE) · Higher is better · Decibels (dB).</div>
            </div>
            <div>
              <div className="font-semibold text-cyan-300 font-mono">Structural Similarity Index (SSIM)</div>
              <div className="text-slate-400 text-[11px]">Gaussian window sigma=1.5, data range 1.0 via skimage. Range [0, 1].</div>
            </div>
          </div>
        </div>

        <div className="p-5 rounded-xl border border-slate-800/80 bg-slate-900/60 space-y-3">
          <h3 className="text-sm font-bold text-slate-100 flex items-center gap-2">
            <ShieldCheck className="w-4 h-4 text-emerald-400" /> Scientific Methodology Standard
          </h3>
          <ul className="text-xs text-slate-300 space-y-2 list-disc pl-4 leading-relaxed">
            <li>
              <strong>Held-Out Subject Split:</strong> Persons 17-25 are strictly held out. Zero data leakage across training, validation, and test splits.
            </li>
            <li>
              <strong>Autoregressive Rollout:</strong> Context frames $t \in [1, 10]$ are encoded. All future frames $t \in [11, 20]$ are predicted sequentially from model predictions.
            </li>
            <li>
              <strong>Simulator Transparency:</strong> QPredRNN quantum modulation is simulated classically on CPU via PennyLane <code className="text-violet-300">default.qubit</code>. No physical quantum hardware execution or quantum advantage is claimed.
            </li>
          </ul>
        </div>
      </div>
    </div>
  );
}