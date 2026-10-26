import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { api, Run, HistoryEntry } from "../../lib/api";
import { RefreshCw, CheckCircle, TrendingDown, Layers, Award } from "lucide-react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
  ReferenceLine,
} from "recharts";

export default function TrainingMonitor() {
  const { data: runs = [], refetch: refetchRuns } = useQuery<Run[]>({
    queryKey: ["runs"],
    queryFn: api.runs,
  });

  const runsWithHistory = runs.filter((r) => r.has_history);

  const [primaryRunId, setPrimaryRunId] = useState<string>("kth_qpredrnn_local_50ep");
  const [compareRunId, setCompareRunId] = useState<string>("kth_predrnn_v2_overnight_gpu");
  const [logScale, setLogScale] = useState(false);
  const [viewMetric, setViewMetric] = useState<"loss" | "decouple" | "rss">("loss");

  const { data: primaryHistory = [], isLoading: primaryLoading } = useQuery<HistoryEntry[]>({
    queryKey: ["history", primaryRunId],
    queryFn: () => api.history(primaryRunId),
    enabled: !!primaryRunId,
  });

  const { data: compareHistory = [] } = useQuery<HistoryEntry[]>({
    queryKey: ["history", compareRunId],
    queryFn: () => api.history(compareRunId),
    enabled: !!compareRunId && compareRunId !== primaryRunId,
  });

  const primaryRun = runs.find((r) => r.id === primaryRunId);
  const compareRun = runs.find((r) => r.id === compareRunId);

  // Best validation calculation
  const bestVal = primaryHistory.length
    ? Math.min(...primaryHistory.map((h) => h.val_loss))
    : null;
  const bestEpoch = primaryHistory.find((h) => h.val_loss === bestVal)?.epoch ?? null;

  // Combine history data for charts
  const maxEpochs = Math.max(
    primaryHistory.length,
    compareRunId !== primaryRunId ? compareHistory.length : 0
  );
  const chartData = Array.from({ length: maxEpochs }, (_, i) => {
    const ep = i + 1;
    const p = primaryHistory.find((h) => h.epoch === ep);
    const c = compareHistory.find((h) => h.epoch === ep);
    return {
      epoch: ep,
      primary_train: p?.train_loss,
      primary_val: p?.val_loss,
      primary_decouple: p?.decouple_loss,
      primary_r_eta: p?.r_eta,
      primary_eta: p?.eta,
      compare_train: c?.train_loss,
      compare_val: c?.val_loss,
      compare_decouple: c?.decouple_loss,
      compare_r_eta: c?.r_eta,
    };
  });

  return (
    <div className="space-y-6 max-w-7xl mx-auto">
      {/* Header */}
      <div className="flex flex-wrap items-center justify-between gap-4">
        <div>
          <div className="flex items-center gap-2 mb-1">
            <span className="px-2.5 py-0.5 text-xs font-mono rounded bg-indigo-950/60 border border-indigo-800/40 text-indigo-300">
              Verified Training Records
            </span>
            <span className="text-xs text-slate-400">· Real Epoch Logs from history.json</span>
          </div>
          <h1 className="text-2xl md:text-3xl font-bold tracking-tight text-slate-100">
            Historical Training Convergence Monitor
          </h1>
          <p className="text-slate-400 text-sm mt-1">
            Audited convergence curves, memory decoupling cosine losses, and reverse scheduled sampling schedules (r_eta, eta).
          </p>
        </div>

        <div className="flex items-center gap-3">
          <label className="flex items-center gap-2 text-xs text-slate-300 cursor-pointer">
            <input
              type="checkbox"
              checked={logScale}
              onChange={(e) => setLogScale(e.target.checked)}
              className="rounded accent-indigo-500"
            />
            Log-Scale Y-Axis
          </label>
          <button
            onClick={() => refetchRuns()}
            className="px-3 py-1.5 rounded-lg bg-slate-900 border border-slate-800 text-xs text-slate-300 hover:text-white transition-all flex items-center gap-1.5"
          >
            <RefreshCw className="w-3.5 h-3.5" /> Refresh
          </button>
        </div>
      </div>

      {/* Run Selection & Telemetry Bar */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {/* Primary Run Selector */}
        <div className="p-4 rounded-xl bg-slate-900/60 border border-indigo-700/60 space-y-2">
          <div className="text-[11px] font-mono text-indigo-400 uppercase tracking-wider">
            Primary Research Run:
          </div>
          <select
            value={primaryRunId}
            onChange={(e) => setPrimaryRunId(e.target.value)}
            className="w-full px-3 py-2 rounded-lg bg-slate-950 border border-slate-800 text-xs text-slate-100 font-medium"
          >
            {runsWithHistory.map((r) => (
              <option key={r.id} value={r.id}>
                {r.display_name ?? r.name} ({r.history_entries} epochs)
              </option>
            ))}
          </select>
          <div className="text-[11px] text-slate-400 font-mono flex items-center justify-between pt-1">
            <span>Logged Epochs: {primaryHistory.length}</span>
            <span className="text-emerald-400 font-bold">
              Best MSE: {bestVal != null ? bestVal.toFixed(6) : "—"} (ep {bestEpoch})
            </span>
          </div>
        </div>

        {/* Comparison Run Selector */}
        <div className="p-4 rounded-xl bg-slate-900/60 border border-slate-800/80 space-y-2">
          <div className="text-[11px] font-mono text-slate-400 uppercase tracking-wider">
            Compare Against Baseline:
          </div>
          <select
            value={compareRunId}
            onChange={(e) => setCompareRunId(e.target.value)}
            className="w-full px-3 py-2 rounded-lg bg-slate-950 border border-slate-800 text-xs text-slate-100 font-medium"
          >
            <option value="">None (Single Run View)</option>
            {runsWithHistory
              .filter((r) => r.id !== primaryRunId)
              .map((r) => (
                <option key={r.id} value={r.id}>
                  {r.display_name ?? r.name} ({r.history_entries} epochs)
                </option>
              ))}
          </select>
          <div className="text-[11px] text-slate-400 font-mono pt-1">
            {compareRun ? (
              <span>Comparing against: {compareRun.display_name ?? compareRun.name}</span>
            ) : (
              <span>Select a second run for overlay comparison</span>
            )}
          </div>
        </div>

        {/* Best Performance Highlight Card */}
        <div className="p-4 rounded-xl bg-emerald-950/20 border border-emerald-800/40 flex flex-col justify-between">
          <div>
            <div className="flex items-center justify-between">
              <span className="text-[11px] font-mono text-emerald-400 uppercase tracking-wider flex items-center gap-1">
                <Award className="w-3.5 h-3.5" /> Best Recorded Validation
              </span>
              <span className="text-xs font-mono text-slate-400">Epoch {bestEpoch ?? "—"}</span>
            </div>
            <div className="text-2xl font-bold font-mono text-slate-100 mt-1">
              {bestVal != null ? bestVal.toFixed(6) : "—"}
            </div>
          </div>
          <p className="text-[11px] text-slate-400">
            Saved as <code className="text-emerald-300">checkpoint_best.pt</code> in {primaryRunId}
          </p>
        </div>
      </div>

      {/* Metric Selector Tabs */}
      <div className="flex items-center gap-2 border-b border-slate-800/80 pb-3">
        <button
          onClick={() => setViewMetric("loss")}
          className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition-all ${
            viewMetric === "loss"
              ? "bg-indigo-600 text-white shadow-sm"
              : "bg-slate-900/60 text-slate-400 hover:text-slate-200"
          }`}
        >
          Train & Validation Loss
        </button>
        <button
          onClick={() => setViewMetric("decouple")}
          className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition-all ${
            viewMetric === "decouple"
              ? "bg-indigo-600 text-white shadow-sm"
              : "bg-slate-900/60 text-slate-400 hover:text-slate-200"
          }`}
        >
          Memory Decoupling Loss (L_decouple)
        </button>
        <button
          onClick={() => setViewMetric("rss")}
          className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition-all ${
            viewMetric === "rss"
              ? "bg-indigo-600 text-white shadow-sm"
              : "bg-slate-900/60 text-slate-400 hover:text-slate-200"
          }`}
        >
          Reverse Scheduled Sampling (r_eta, eta)
        </button>
      </div>

      {/* Main Chart */}
      <div className="p-5 rounded-xl border border-slate-800/80 bg-slate-900/60 space-y-4 shadow-lg">
        <div className="flex items-center justify-between">
          <h2 className="text-sm font-bold text-slate-100">
            {viewMetric === "loss" && "Convergence Curves: Training vs Validation MSE"}
            {viewMetric === "decouple" && "Decoupling Cosine Distance Between Cell & Spatiotemporal Memory"}
            {viewMetric === "rss" && "Curriculum Learning: Scheduled Sampling Probability Schedules"}
          </h2>
          <span className="text-xs font-mono text-slate-400">
            {primaryHistory.length} Epochs Displayed
          </span>
        </div>

        <div className="h-80 w-full">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={chartData} margin={{ top: 10, right: 20, bottom: 10, left: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(51, 65, 85, 0.4)" />
              <XAxis
                dataKey="epoch"
                tick={{ fill: "#94a3b8", fontSize: 11 }}
                label={{ value: "Epoch Number", position: "insideBottom", offset: -5, fill: "#94a3b8", fontSize: 11 }}
              />
              <YAxis
                scale={logScale ? "log" : "auto"}
                tick={{ fill: "#94a3b8", fontSize: 11 }}
                domain={logScale ? ["auto", "auto"] : [0, "auto"]}
              />
              <Tooltip
                contentStyle={{
                  backgroundColor: "rgba(15, 23, 42, 0.95)",
                  borderColor: "rgba(51, 65, 85, 0.8)",
                  borderRadius: 8,
                  fontSize: 12,
                }}
              />
              <Legend wrapperStyle={{ fontSize: 11, paddingTop: 8 }} />

              {bestEpoch && viewMetric === "loss" && (
                <ReferenceLine
                  x={bestEpoch}
                  stroke="#34d399"
                  strokeDasharray="4 4"
                  label={{ value: `Best Ep ${bestEpoch}`, fill: "#34d399", fontSize: 10, position: "top" }}
                />
              )}

              {viewMetric === "loss" && (
                <>
                  <Line
                    dataKey="primary_val"
                    name={`${primaryRun?.display_name ?? primaryRunId} (Val MSE)`}
                    stroke="#818cf8"
                    strokeWidth={2.5}
                    dot={{ r: 2.5 }}
                  />
                  <Line
                    dataKey="primary_train"
                    name={`${primaryRun?.display_name ?? primaryRunId} (Train MSE)`}
                    stroke="#818cf8"
                    strokeWidth={1.5}
                    strokeDasharray="4 2"
                    dot={false}
                    strokeOpacity={0.6}
                  />
                  {compareRunId && (
                    <>
                      <Line
                        dataKey="compare_val"
                        name={`${compareRun?.display_name ?? compareRunId} (Val MSE)`}
                        stroke="#34d399"
                        strokeWidth={2.5}
                        dot={{ r: 2.5 }}
                      />
                      <Line
                        dataKey="compare_train"
                        name={`${compareRun?.display_name ?? compareRunId} (Train MSE)`}
                        stroke="#34d399"
                        strokeWidth={1.5}
                        strokeDasharray="4 2"
                        dot={false}
                        strokeOpacity={0.6}
                      />
                    </>
                  )}
                </>
              )}

              {viewMetric === "decouple" && (
                <>
                  <Line
                    dataKey="primary_decouple"
                    name={`${primaryRun?.display_name ?? primaryRunId} (Decouple Loss)`}
                    stroke="#a78bfa"
                    strokeWidth={2.5}
                    dot={{ r: 2.5 }}
                  />
                  {compareRunId && (
                    <Line
                      dataKey="compare_decouple"
                      name={`${compareRun?.display_name ?? compareRunId} (Decouple Loss)`}
                      stroke="#fbbf24"
                      strokeWidth={2}
                      dot={{ r: 2 }}
                    />
                  )}
                </>
              )}

              {viewMetric === "rss" && (
                <>
                  <Line
                    dataKey="primary_r_eta"
                    name="Reverse Scheduled Sampling r_eta (Exponential Increase)"
                    stroke="#38bdf8"
                    strokeWidth={2.5}
                    dot={{ r: 2 }}
                  />
                  <Line
                    dataKey="primary_eta"
                    name="Scheduled Sampling eta (Linear Decay)"
                    stroke="#f472b6"
                    strokeWidth={2}
                    strokeDasharray="4 2"
                    dot={false}
                  />
                </>
              )}
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Run Summary Table */}
      <div className="rounded-xl border border-slate-800/80 bg-slate-900/60 overflow-hidden shadow-lg">
        <div className="px-5 py-3.5 border-b border-slate-800/80">
          <h2 className="text-sm font-bold text-slate-100">Discovered Training Run Progression</h2>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-xs text-left">
            <thead className="bg-slate-950/60 text-slate-400 font-mono uppercase text-[11px] border-b border-slate-800/60">
              <tr>
                <th className="px-4 py-3">Experiment Run</th>
                <th className="px-3 py-3">Architecture</th>
                <th className="px-3 py-3">Category</th>
                <th className="px-3 py-3 font-semibold text-slate-200">Best Val MSE</th>
                <th className="px-3 py-3">Logged Epochs</th>
                <th className="px-3 py-3">Checkpoints</th>
                <th className="px-3 py-3">Status</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/50 font-mono">
              {runsWithHistory.map((r) => {
                const isSelected = r.id === primaryRunId;
                return (
                  <tr
                    key={r.id}
                    className={`hover:bg-slate-800/30 transition-colors ${
                      isSelected ? "bg-indigo-950/20" : ""
                    }`}
                  >
                    <td className="px-4 py-3 font-sans font-semibold text-slate-100">
                      {r.display_name ?? r.name}
                    </td>
                    <td className="px-3 py-3 text-slate-300 font-sans">{r.model_type}</td>
                    <td className="px-3 py-3 text-slate-400 uppercase text-[10px]">{r.category ?? "run"}</td>
                    <td className="px-3 py-3 font-bold text-emerald-400">
                      {r.best_val_mse != null ? r.best_val_mse.toFixed(6) : "—"}
                    </td>
                    <td className="px-3 py-3 text-slate-200">
                      {r.last_epoch != null ? `${r.last_epoch} epochs` : `${r.history_entries} logged`}
                    </td>
                    <td className="px-3 py-3 text-slate-400">
                      {r.has_best_ckpt ? "best.pt" : ""}
                      {r.has_last_ckpt ? " + last.pt" : ""}
                    </td>
                    <td className="px-3 py-3">
                      <span className="inline-flex items-center gap-1 text-emerald-400 text-[11px]">
                        <CheckCircle className="w-3.5 h-3.5" />
                        {r.status ?? "Completed"}
                      </span>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}