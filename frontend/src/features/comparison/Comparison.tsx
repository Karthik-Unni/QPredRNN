import { useState, useEffect } from "react";
import { useQuery } from "@tanstack/react-query";
import { api, Run, PredictResult } from "../../lib/api";
import { Play, Loader2, AlertCircle, CheckCircle, Sliders, Layers } from "lucide-react";
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend } from "recharts";

function Frame({ b64, label, sub }: { b64: string; label: string; sub?: string }) {
  return (
    <div className="text-center space-y-1">
      <img
        src={`data:image/png;base64,${b64}`}
        alt={label}
        className="w-full rounded border border-slate-700/80 bg-slate-950 aspect-square object-contain"
        style={{ imageRendering: "pixelated" }}
      />
      <div className="text-xs font-semibold text-slate-200 truncate">{label}</div>
      {sub && <div className="text-[10px] text-slate-400 font-mono">{sub}</div>}
    </div>
  );
}

const PALETTE = ["#818cf8", "#34d399", "#38bdf8", "#fbbf24", "#f472b6", "#a78bfa"];

export default function Comparison() {
  const { data: runs = [] } = useQuery<Run[]>({ queryKey: ["runs"], queryFn: api.runs });
  
  // Available comparison items including analytical baselines
  const allComparisonOptions = [
    {
      id: "kth_qpredrnn_local_50ep",
      name: "QPredRNN — 50-Epoch Local Run",
      category: "primary",
      arch: "QPredRNN (VQC + ST-LSTM)",
      hasBest: true,
    },
    {
      id: "kth_predrnn_v2_local_50ep",
      name: "PredRNN-V2 — 50-Epoch Local Run",
      category: "primary",
      arch: "PredRNN-V2 (Dual-Memory ST-LSTM)",
      hasBest: true,
    },
    {
      id: "kth_classical_tpu_50ep",
      name: "ConvLSTM — 50-Epoch TPU Run",
      category: "primary",
      arch: "ConvLSTM (2-Layer Recurrent)",
      hasBest: true,
    },
    {
      id: "kth_classical_final_50ep",
      name: "ConvLSTM — Final 50-Epoch Full-Dataset Run",
      category: "primary",
      arch: "ConvLSTM (Full Dataset)",
      hasBest: true,
    },
    {
      id: "kth_predrnn_v2_overnight_gpu",
      name: "PredRNN-V2 — 10-Epoch Overnight Run",
      category: "pilot",
      arch: "PredRNN-V2 (ST-LSTM)",
      hasBest: true,
    },
    {
      id: "kth_classical_pilot_gpu",
      name: "ConvLSTM — 10-Epoch Pilot Run",
      category: "pilot",
      arch: "ConvLSTM",
      hasBest: true,
    },
    {
      id: "kth_qpredrnn_pilot_gpu",
      name: "QPredRNN — 10-Epoch Pilot Run",
      category: "pilot",
      arch: "QPredRNN (VQC + ST-LSTM)",
      hasBest: true,
    },
    {
      id: "baseline_persistence",
      name: "Persistence Baseline (Static)",
      category: "baseline",
      arch: "Non-parametric (Repeat t=10)",
      hasBest: false,
    },
    {
      id: "baseline_mean",
      name: "Mean Context Baseline",
      category: "baseline",
      arch: "Non-parametric (Temporal Mean)",
      hasBest: false,
    },
    {
      id: "kth_classical_gpu",
      name: "ConvLSTM — Early 3-Epoch Run",
      category: "experimental",
      arch: "ConvLSTM",
      hasBest: false,
    },
    {
      id: "kth_qpredrnn_smoke_gpu",
      name: "QPredRNN — GPU Smoke Test",
      category: "smoke",
      arch: "QPredRNN",
      hasBest: false,
    },
  ];

  const [selectedIds, setSelectedIds] = useState<string[]>([
    "kth_qpredrnn_local_50ep",
    "kth_predrnn_v2_local_50ep",
    "kth_classical_tpu_50ep",
    "baseline_persistence",
  ]);
  const [categoryFilter, setCategoryFilter] = useState<string>("all");
  const [split, setSplit] = useState<"test" | "val" | "train">("test");
  const [clipIdx, setClipIdx] = useState<number>(0);
  const [horizonIdx, setHorizonIdx] = useState<number>(4); // default 5th frame (horizon 5)
  const [metricTab, setMetricTab] = useState<"mse" | "ssim">("mse");

  const [results, setResults] = useState<Record<string, PredictResult>>({});
  const [loading, setLoading] = useState<Record<string, boolean>>({});
  const [errors, setErrors] = useState<Record<string, string>>({});

  const toggleSelect = (id: string) => {
    setSelectedIds((prev) =>
      prev.includes(id) ? prev.filter((x) => x !== id) : [...prev, id]
    );
  };

  const runComparison = async () => {
    for (const id of selectedIds) {
      setLoading((l) => ({ ...l, [id]: true }));
      setErrors((e) => ({ ...e, [id]: "" }));
      try {
        const res = await api.predict({
          run_id: id,
          checkpoint: "best",
          dataset: "kth",
          split: split,
          clip_index: clipIdx,
          device: "cuda",
        });
        setResults((r) => ({ ...r, [id]: res }));
      } catch (e) {
        setErrors((er) => ({ ...er, [id]: e instanceof Error ? e.message : String(e) }));
      } finally {
        setLoading((l) => ({ ...l, [id]: false }));
      }
    }
  };

  // Run automatically on first load if results are empty
  useEffect(() => {
    if (Object.keys(results).length === 0) {
      runComparison();
    }
  }, []);

  const isAnyLoading = Object.values(loading).some(Boolean);
  const horizons = Array.from({ length: 10 }, (_, i) => i + 1);

  const filteredOptions = allComparisonOptions.filter((opt) => {
    if (categoryFilter === "all") return true;
    if (categoryFilter === "benchmark") return opt.category === "primary" || opt.category === "baseline";
    if (categoryFilter === "pilot") return opt.category === "pilot";
    if (categoryFilter === "smoke") return opt.category === "smoke" || opt.category === "experimental";
    return true;
  });

  return (
    <div className="space-y-6 max-w-7xl mx-auto">
      {/* Header */}
      <div>
        <div className="flex items-center gap-2 mb-1">
          <span className="px-2.5 py-0.5 text-xs font-mono rounded bg-indigo-950/60 border border-indigo-800/40 text-indigo-300">
            Fair Benchmark Suite
          </span>
          <span className="text-xs text-slate-400">· Strictly Identical Input & Split Conditions</span>
        </div>
        <h1 className="text-2xl md:text-3xl font-bold tracking-tight text-slate-100">
          Cross-Architecture Model Comparison
        </h1>
        <p className="text-slate-400 text-sm mt-1">
          Compare QPredRNN, PredRNN-V2, ConvLSTM, and non-parametric baselines on the exact same temporal test window.
        </p>
      </div>

      {/* Control Panel */}
      <div className="p-5 rounded-xl bg-slate-900/60 border border-slate-800/80 space-y-4">
        {/* Category Filters */}
        <div className="flex flex-wrap items-center justify-between gap-3 pb-3 border-b border-slate-800/60">
          <div className="flex items-center gap-1.5 text-xs">
            <span className="text-slate-400 font-mono mr-1">Filter Models:</span>
            {[
              { id: "all", label: "All Candidates" },
              { id: "benchmark", label: "Primary Benchmarks & Baselines" },
              { id: "pilot", label: "Pilot Studies" },
              { id: "smoke", label: "Smoke & Experimental" },
            ].map((f) => (
              <button
                key={f.id}
                onClick={() => setCategoryFilter(f.id)}
                className={`px-2.5 py-1 rounded text-xs transition-all ${
                  categoryFilter === f.id
                    ? "bg-indigo-600 text-white font-semibold"
                    : "bg-slate-800/60 text-slate-400 hover:text-slate-200"
                }`}
              >
                {f.label}
              </button>
            ))}
          </div>

          <div className="flex items-center gap-2">
            <button
              onClick={() => setSelectedIds(["kth_qpredrnn_local_50ep", "kth_predrnn_v2_local_50ep", "kth_classical_tpu_50ep", "baseline_persistence"])}
              className="text-xs text-cyan-400 hover:underline font-mono"
            >
              Select 50-Epoch Primary Trio + Baseline
            </button>
          </div>
        </div>

        {/* Model Chips */}
        <div>
          <div className="text-[11px] font-mono text-slate-400 uppercase tracking-wider mb-2">
            Select Models to Compare ({selectedIds.length} active):
          </div>
          <div className="flex flex-wrap gap-2">
            {filteredOptions.map((opt, i) => {
              const sel = selectedIds.includes(opt.id);
              const color = PALETTE[i % PALETTE.length];
              return (
                <button
                  key={opt.id}
                  onClick={() => toggleSelect(opt.id)}
                  className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-all flex items-center gap-1.5 ${
                    sel
                      ? "bg-slate-800 border-2 text-slate-100 shadow-sm"
                      : "bg-slate-950/40 border border-slate-800 text-slate-400 hover:border-slate-700 hover:text-slate-200"
                  }`}
                  style={{ borderColor: sel ? color : undefined }}
                >
                  <span
                    className="w-2 h-2 rounded-full"
                    style={{ backgroundColor: color }}
                  />
                  <span>{opt.name}</span>
                  {sel && <CheckCircle className="w-3 h-3 text-emerald-400" />}
                </button>
              );
            })}
          </div>
        </div>

        {/* Evaluation Controls */}
        <div className="flex flex-wrap items-center gap-4 pt-2">
          <div className="flex items-center gap-2">
            <span className="text-xs text-slate-400 font-mono">Dataset Split:</span>
            <select
              value={split}
              onChange={(e) => setSplit(e.target.value as any)}
              className="px-3 py-1.5 rounded-lg bg-slate-950 border border-slate-800 text-xs text-slate-200 font-mono"
            >
              <option value="test">Held-Out Test (Persons 17-25)</option>
              <option value="val">Validation (Indoor d4)</option>
              <option value="train">Train Split</option>
            </select>
          </div>

          <div className="flex items-center gap-2">
            <span className="text-xs text-slate-400 font-mono">Clip Index:</span>
            <input
              type="number"
              min={0}
              value={clipIdx}
              onChange={(e) => setClipIdx(Math.max(0, parseInt(e.target.value) || 0))}
              className="w-20 px-3 py-1.5 rounded-lg bg-slate-950 border border-slate-800 text-xs text-slate-200 font-mono"
            />
          </div>

          <button
            onClick={runComparison}
            disabled={selectedIds.length === 0 || isAnyLoading}
            className={`px-5 py-2 rounded-lg text-xs font-bold transition-all flex items-center gap-2 shadow-lg ${
              selectedIds.length > 0 && !isAnyLoading
                ? "bg-indigo-600 hover:bg-indigo-500 text-white shadow-indigo-600/20 cursor-pointer"
                : "bg-slate-800 text-slate-400 cursor-not-allowed"
            }`}
          >
            {isAnyLoading ? <Loader2 className="w-4 h-4 animate-spin" /> : <Play className="w-4 h-4" />}
            Run Benchmark Comparison
          </button>
        </div>
      </div>

      {/* Errors container if any */}
      {Object.entries(errors).filter(([, v]) => v).length > 0 && (
        <div className="space-y-2">
          {Object.entries(errors)
            .filter(([, v]) => v)
            .map(([id, msg]) => (
              <div
                key={id}
                className="p-3 rounded-lg bg-red-950/40 border border-red-800/60 text-xs text-red-200 flex items-center gap-2"
              >
                <AlertCircle className="w-4 h-4 text-red-400 shrink-0" />
                <span>
                  <strong>{id}:</strong> {msg}
                </span>
              </div>
            ))}
        </div>
      )}

      {/* Comparison Results */}
      {Object.keys(results).length > 0 && (
        <div className="space-y-6">
          {/* Metrics Summary Table */}
          <div className="rounded-xl border border-slate-800/80 bg-slate-900/60 overflow-hidden shadow-lg">
            <div className="px-5 py-3.5 border-b border-slate-800/80 flex items-center justify-between">
              <div>
                <h2 className="text-sm font-bold text-slate-100">Evaluated Clip Metrics Summary</h2>
                <p className="text-xs text-slate-400">{Object.values(results)[0]?.seq_desc}</p>
              </div>
              <span className="text-xs font-mono text-cyan-400">Lower MSE/MAE is better · Higher PSNR/SSIM is better</span>
            </div>
            <div className="overflow-x-auto">
              <table className="w-full text-xs text-left">
                <thead className="bg-slate-950/60 text-slate-400 font-mono uppercase text-[11px] border-b border-slate-800/60">
                  <tr>
                    <th className="px-4 py-3">Model</th>
                    <th className="px-3 py-3">Checkpoint</th>
                    <th className="px-3 py-3 font-semibold text-slate-200">Test MSE ↓</th>
                    <th className="px-3 py-3 font-semibold text-slate-200">MAE ↓</th>
                    <th className="px-3 py-3 font-semibold text-slate-200">PSNR ↑</th>
                    <th className="px-3 py-3 font-semibold text-slate-200">SSIM ↑</th>
                    <th className="px-3 py-3">Latency</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-800/50 font-mono">
                  {Object.entries(results).map(([id, r]) => {
                    const allMSE = Object.values(results).map((x) => x.metrics.mse);
                    const allSSIM = Object.values(results).map((x) => x.metrics.ssim);
                    const isBestMSE = r.metrics.mse === Math.min(...allMSE);
                    const isBestSSIM = r.metrics.ssim === Math.max(...allSSIM);
                    const opt = allComparisonOptions.find((o) => o.id === id);
                    const displayName = opt?.name ?? id;

                    return (
                      <tr key={id} className="hover:bg-slate-800/30 transition-colors">
                        <td className="px-4 py-3 font-sans font-semibold text-slate-100">
                          {displayName}
                        </td>
                        <td className="px-3 py-3 text-slate-400">
                          {r.checkpoint_epoch ? `Epoch ${r.checkpoint_epoch}` : r.checkpoint}
                        </td>
                        <td className={`px-3 py-3 font-bold ${isBestMSE ? "text-emerald-400 bg-emerald-950/20" : "text-slate-200"}`}>
                          {r.metrics.mse.toFixed(6)} {isBestMSE && "★"}
                        </td>
                        <td className="px-3 py-3 text-slate-300">
                          {r.metrics.mae.toFixed(6)}
                        </td>
                        <td className="px-3 py-3 text-slate-300">
                          {r.metrics.psnr.toFixed(2)} dB
                        </td>
                        <td className={`px-3 py-3 font-bold ${isBestSSIM ? "text-cyan-400 bg-cyan-950/20" : "text-slate-200"}`}>
                          {r.metrics.ssim.toFixed(4)} {isBestSSIM && "★"}
                        </td>
                        <td className="px-3 py-3 text-slate-400">
                          {r.latency_ms.toFixed(1)} ms
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>

          {/* Visual Strip Comparison Across Horizon Steps */}
          <div className="p-5 rounded-xl border border-slate-800/80 bg-slate-900/60 space-y-4">
            <div className="flex flex-wrap items-center justify-between gap-3">
              <div>
                <h3 className="text-sm font-bold text-slate-100">Visual Rollout Comparison</h3>
                <p className="text-xs text-slate-400">
                  Scrub across forecast horizon steps to visually compare predictions and magma error heatmaps.
                </p>
              </div>
              <div className="flex items-center gap-3">
                <span className="text-xs font-mono text-cyan-300 font-bold">
                  Step t = {10 + horizonIdx + 1} (Horizon {horizonIdx + 1})
                </span>
                <input
                  type="range"
                  min={0}
                  max={9}
                  value={horizonIdx}
                  onChange={(e) => setHorizonIdx(parseInt(e.target.value))}
                  className="w-32 cursor-pointer accent-cyan-500"
                />
              </div>
            </div>

            {/* Grid of Predicted Frames */}
            <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-6 gap-3 pt-2">
              {/* Context Frame */}
              {Object.values(results)[0] && (
                <Frame
                  b64={Object.values(results)[0].input_frames[9]}
                  label="Last Context (t=10)"
                  sub="Observation"
                />
              )}
              {/* Ground Truth Frame */}
              {Object.values(results)[0] && (
                <Frame
                  b64={Object.values(results)[0].gt_frames[horizonIdx]}
                  label={`Ground Truth (t=${11 + horizonIdx})`}
                  sub="Target"
                />
              )}
              {/* Model Predictions */}
              {Object.entries(results).map(([id, r]) => {
                const opt = allComparisonOptions.find((o) => o.id === id);
                return (
                  <Frame
                    key={id}
                    b64={r.pred_frames[horizonIdx]}
                    label={opt?.name ?? id}
                    sub={`MSE: ${r.metrics.per_frame.mse[horizonIdx].toFixed(5)}`}
                  />
                );
              })}
            </div>

            {/* Grid of Error Heatmaps */}
            <div>
              <div className="text-[11px] font-mono text-slate-400 uppercase tracking-wider mb-2 pt-2">
                Absolute Error Heatmaps (Magma Colormap):
              </div>
              <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-6 gap-3">
                <div className="hidden sm:block" />
                <div className="hidden sm:block" />
                {Object.entries(results).map(([id, r]) => (
                  <Frame
                    key={`err-${id}`}
                    b64={r.error_frames[horizonIdx]}
                    label={`Error: ${id.replace("kth_", "")}`}
                    sub={`SSIM: ${r.metrics.per_frame.ssim[horizonIdx].toFixed(3)}`}
                  />
                ))}
              </div>
            </div>
          </div>

          {/* Degradation Curves */}
          <div className="p-5 rounded-xl border border-slate-800/80 bg-slate-900/60 space-y-4">
            <div className="flex items-center justify-between">
              <div>
                <h3 className="text-sm font-bold text-slate-100">Per-Horizon Metric Degradation Curve</h3>
                <p className="text-xs text-slate-400">
                  Tracks how error accumulates as prediction extends further into the future (lead times 1 to 10).
                </p>
              </div>
              <div className="flex items-center gap-1.5 p-1 rounded-lg bg-slate-950 border border-slate-800">
                <button
                  onClick={() => setMetricTab("mse")}
                  className={`px-3 py-1 text-xs rounded font-medium transition-all ${
                    metricTab === "mse" ? "bg-indigo-600 text-white font-bold" : "text-slate-400 hover:text-slate-200"
                  }`}
                >
                  MSE Curve
                </button>
                <button
                  onClick={() => setMetricTab("ssim")}
                  className={`px-3 py-1 text-xs rounded font-medium transition-all ${
                    metricTab === "ssim" ? "bg-indigo-600 text-white font-bold" : "text-slate-400 hover:text-slate-200"
                  }`}
                >
                  SSIM Curve
                </button>
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
                    label={{ value: "Forecast Horizon Step (t)", position: "insideBottom", offset: -5, fill: "#94a3b8", fontSize: 11 }}
                  />
                  <YAxis
                    tick={{ fill: "#94a3b8", fontSize: 11 }}
                    domain={metricTab === "ssim" ? [0.6, 1.0] : ["auto", "auto"]}
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
                  {Object.entries(results).map(([id, r], i) => {
                    const data = horizons.map((t) => ({
                      t,
                      val: metricTab === "mse" ? r.metrics.per_frame.mse[t - 1] : r.metrics.per_frame.ssim[t - 1],
                    }));
                    const opt = allComparisonOptions.find((o) => o.id === id);
                    return (
                      <Line
                        key={id}
                        data={data}
                        type="monotone"
                        dataKey="val"
                        stroke={PALETTE[i % PALETTE.length]}
                        strokeWidth={2.5}
                        dot={{ r: 3 }}
                        name={opt?.name ?? id}
                      />
                    );
                  })}
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}