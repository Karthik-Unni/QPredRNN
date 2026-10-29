import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import {
  api,
  Run,
  HistoryEntry,
  LiveTrainingStatus,
  LiveRunHistory,
} from "../../lib/api";
import {
  RefreshCw,
  CheckCircle,
  TrendingDown,
  Layers,
  Award,
  Activity,
  Clock,
  AlertTriangle,
  Loader,
  Cpu,
} from "lucide-react";
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

// ─── Live Training Status Panel ─────────────────────────────────────────────

function StatusBadge({ status }: { status: LiveTrainingStatus["status"] }) {
  switch (status) {
    case "RUNNING":
      return (
        <span className="flex items-center gap-1 px-2 py-0.5 rounded-full bg-emerald-900/50 border border-emerald-600/40 text-emerald-300 text-[11px] font-mono animate-pulse">
          <Activity className="w-3 h-3" /> RUNNING
        </span>
      );
    case "COMPLETED":
      return (
        <span className="flex items-center gap-1 px-2 py-0.5 rounded-full bg-blue-900/50 border border-blue-600/40 text-blue-300 text-[11px] font-mono">
          <CheckCircle className="w-3 h-3" /> COMPLETED
        </span>
      );
    case "INITIALIZING":
      return (
        <span className="flex items-center gap-1 px-2 py-0.5 rounded-full bg-yellow-900/40 border border-yellow-700/40 text-yellow-300 text-[11px] font-mono">
          <Loader className="w-3 h-3 animate-spin" /> INITIALIZING
        </span>
      );
    default:
      return (
        <span className="flex items-center gap-1 px-2 py-0.5 rounded-full bg-slate-800/60 border border-slate-700 text-slate-400 text-[11px] font-mono">
          NOT STARTED
        </span>
      );
  }
}

function ProgressBar({ completed, target }: { completed: number; target: number }) {
  const pct = Math.min(100, Math.round((completed / target) * 100));
  return (
    <div className="w-full">
      <div className="flex justify-between text-[10px] text-slate-400 font-mono mb-1">
        <span>{completed} / {target} epochs</span>
        <span>{pct}%</span>
      </div>
      <div className="h-1.5 rounded-full bg-slate-800 overflow-hidden">
        <div
          className="h-full rounded-full bg-gradient-to-r from-indigo-600 to-violet-500 transition-all duration-1000"
          style={{ width: `${pct}%` }}
        />
      </div>
    </div>
  );
}

function LiveRunCard({ run }: { run: LiveTrainingStatus }) {
  const { data: runHistory } = useQuery<LiveRunHistory>({
    queryKey: ["training-live-history", run.run_id],
    queryFn: () => api.trainingLiveRun(run.run_id),
    enabled: run.status === "RUNNING" || run.status === "COMPLETED",
    refetchInterval: run.status === "RUNNING" ? 30000 : false,
  });

  const history = runHistory?.history ?? [];
  const lastFive = history.slice(-5);
  const miniChartData = lastFive.map((h) => ({
    epoch: h.epoch,
    val: h.val_loss,
    train: h.train_loss,
  }));

  return (
    <div
      className={`rounded-xl border p-4 space-y-3 ${
        run.status === "RUNNING"
          ? "bg-emerald-950/15 border-emerald-700/40"
          : run.status === "COMPLETED"
          ? "bg-blue-950/15 border-blue-700/40"
          : "bg-slate-900/60 border-slate-800"
      }`}
    >
      {/* Header */}
      <div className="flex items-start justify-between gap-2">
        <div>
          <div className="text-xs font-semibold text-slate-100 leading-tight">{run.display_name}</div>
          <div className="text-[11px] text-slate-400 font-mono mt-0.5">{run.run_id}</div>
        </div>
        <StatusBadge status={run.status} />
      </div>

      {/* Progress */}
      <ProgressBar completed={run.completed_epochs} target={run.target_epochs} />

      {/* Metrics Row */}
      <div className="grid grid-cols-3 gap-2">
        <div className="text-center">
          <div className="text-[10px] text-slate-500 font-mono uppercase">Train Loss</div>
          <div className="text-xs font-mono text-slate-200 mt-0.5">
            {run.train_loss != null ? run.train_loss.toFixed(6) : "—"}
          </div>
        </div>
        <div className="text-center">
          <div className="text-[10px] text-slate-500 font-mono uppercase">Val Loss</div>
          <div className="text-xs font-mono text-emerald-300 mt-0.5">
            {run.val_loss != null ? run.val_loss.toFixed(6) : "—"}
          </div>
        </div>
        <div className="text-center">
          <div className="text-[10px] text-slate-500 font-mono uppercase">Best Val</div>
          <div className="text-xs font-mono text-violet-300 mt-0.5">
            {run.best_val_loss != null ? `${run.best_val_loss.toFixed(6)} (ep ${run.best_val_epoch})` : "—"}
          </div>
        </div>
      </div>

      {/* Mini sparkline if data available */}
      {miniChartData.length > 1 && (
        <div className="h-14">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={miniChartData} margin={{ top: 2, right: 2, left: 2, bottom: 2 }}>
              <Line type="monotone" dataKey="val" stroke="#6ee7b7" strokeWidth={1.5} dot={false} />
              <Line type="monotone" dataKey="train" stroke="#818cf8" strokeWidth={1} dot={false} strokeDasharray="3 2" />
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}

      {run.status === "INITIALIZING" && (
        <div className="text-[11px] text-yellow-400/80 font-mono flex items-center gap-1.5">
          <Loader className="w-3 h-3 animate-spin" /> Building dataset loader, first epoch in progress…
        </div>
      )}

      {run.status === "NOT_STARTED" && (
        <div className="text-[11px] text-slate-500 font-mono">Run directory not yet created.</div>
      )}

      {run.has_best_ckpt && (
        <div className="text-[11px] text-emerald-400/70 font-mono flex items-center gap-1">
          <CheckCircle className="w-3 h-3" /> checkpoint_best.pt saved
        </div>
      )}
    </div>
  );
}

function LiveTrainingPanel() {
  const { data: liveRuns = [], isLoading, refetch } = useQuery<LiveTrainingStatus[]>({
    queryKey: ["training-live"],
    queryFn: api.trainingLive,
    refetchInterval: 30000, // poll every 30s
  });

  const anyRunning = liveRuns.some((r) => r.status === "RUNNING" || r.status === "INITIALIZING");

  return (
    <div className="rounded-2xl border border-indigo-800/30 bg-indigo-950/10 p-5 space-y-4">
      <div className="flex items-center justify-between">
        <div>
          <div className="flex items-center gap-2">
            <Cpu className="w-4 h-4 text-indigo-400" />
            <h2 className="text-sm font-bold text-slate-100 tracking-wide">
              Live Training — New Full-Dataset 50-Epoch Runs
            </h2>
            {anyRunning && (
              <span className="flex items-center gap-1 text-[11px] text-emerald-400 font-mono animate-pulse">
                <Activity className="w-3 h-3" /> ACTIVE
              </span>
            )}
          </div>
          <p className="text-[11px] text-slate-500 mt-0.5 font-mono">
            Each epoch processes all 28,389 training windows (3,549 batches @ bs=8). No artificial batch cap.
          </p>
        </div>
        <button
          onClick={() => refetch()}
          className="flex items-center gap-1.5 px-2.5 py-1 rounded-lg bg-slate-900 border border-slate-800 text-xs text-slate-300 hover:text-white transition-all"
        >
          <RefreshCw className={`w-3.5 h-3.5 ${isLoading ? "animate-spin" : ""}`} />
          Refresh
        </button>
      </div>

      {/* Epoch coverage note */}
      <div className="flex items-start gap-2 p-3 rounded-lg bg-amber-950/20 border border-amber-800/30">
        <AlertTriangle className="w-4 h-4 text-amber-400 shrink-0 mt-0.5" />
        <div className="text-[11px] text-amber-200/80 font-mono leading-relaxed">
          <strong className="text-amber-300">Important — Epoch Semantics:</strong>{" "}
          Previous runs (kth_qpredrnn_local_50ep, kth_predrnn_v2_local_50ep, kth_classical_tpu_50ep) used
          max_train_batches=500 cap. Each "epoch" in those runs processed only 500/28,389 = <strong>1.76%</strong> of the
          training set. These new final runs process the <strong>full 28,389 windows per epoch</strong> — a genuine
          50-epoch full-dataset schedule.
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {isLoading ? (
          <div className="col-span-2 text-center py-6 text-slate-500 text-xs font-mono">
            <Loader className="w-5 h-5 animate-spin mx-auto mb-2" /> Connecting to training monitor…
          </div>
        ) : liveRuns.length === 0 ? (
          <div className="col-span-2 text-center py-6 text-slate-500 text-xs font-mono">
            No tracked runs found. Launch training with the commands below.
          </div>
        ) : (
          liveRuns.map((run) => <LiveRunCard key={run.run_id} run={run} />)
        )}
      </div>

      {/* Launch commands */}
      <div className="mt-2 p-3 rounded-lg bg-slate-950/60 border border-slate-800/60 space-y-2">
        <div className="text-[11px] text-slate-400 font-mono uppercase tracking-wider">Launch Commands (if not running)</div>
        <div className="text-[11px] font-mono text-slate-300 space-y-1">
          <div className="text-slate-500">{`# ConvLSTM Final 50-Epoch Run`}</div>
          <code className="text-indigo-300 break-all">
            {`& "C:\\venvs\\qpred-gpu\\Scripts\\python.exe" -m qpred.train --config configs/kth_classical_final_50ep.yaml`}
          </code>
          <div className="text-slate-500 mt-1">{`# PredRNN-V2 Final 50-Epoch Run (after ConvLSTM completes)`}</div>
          <code className="text-violet-300 break-all">
            {`& "C:\\venvs\\qpred-gpu\\Scripts\\python.exe" -m qpred.train --config configs/kth_predrnn_v2_final_50ep.yaml`}
          </code>
        </div>
      </div>
    </div>
  );
}

// ─── Main Component ──────────────────────────────────────────────────────────

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
              Training Monitor
            </span>
            <span className="text-xs text-slate-400">· Live + Historical Training Records</span>
          </div>
          <h1 className="text-2xl md:text-3xl font-bold tracking-tight text-slate-100">
            Training Convergence Monitor
          </h1>
          <p className="text-slate-400 text-sm mt-1">
            Live status of new full-dataset runs + audited historical convergence curves.
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

      {/* Live Training Panel */}
      <LiveTrainingPanel />

      {/* Historical Run Selection */}
      <div>
        <div className="flex items-center gap-2 mb-3">
          <TrendingDown className="w-4 h-4 text-slate-400" />
          <h2 className="text-sm font-bold text-slate-200">Historical Convergence — Existing Runs</h2>
          <span className="text-[11px] text-slate-500 font-mono">(capped 500-batch/epoch training — see caveat above)</span>
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
              ? "bg-violet-600 text-white shadow-sm"
              : "bg-slate-900/60 text-slate-400 hover:text-slate-200"
          }`}
        >
          Decoupling Loss
        </button>
        <button
          onClick={() => setViewMetric("rss")}
          className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition-all ${
            viewMetric === "rss"
              ? "bg-emerald-700 text-white shadow-sm"
              : "bg-slate-900/60 text-slate-400 hover:text-slate-200"
          }`}
        >
          RSS Schedule (r_eta / eta)
        </button>
      </div>

      {/* Chart */}
      <div className="p-5 rounded-2xl bg-slate-900/50 border border-slate-800/60">
        {primaryLoading ? (
          <div className="h-64 flex items-center justify-center text-slate-500 text-sm font-mono">
            <Loader className="w-5 h-5 animate-spin mr-2" /> Loading history…
          </div>
        ) : primaryHistory.length === 0 ? (
          <div className="h-64 flex items-center justify-center text-slate-500 text-sm font-mono">
            No training history found for this run.
          </div>
        ) : (
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={chartData} margin={{ top: 4, right: 16, left: 0, bottom: 4 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
              <XAxis
                dataKey="epoch"
                stroke="#475569"
                tick={{ fontSize: 10, fill: "#94a3b8" }}
                label={{ value: "Epoch", position: "insideBottom", offset: -4, fontSize: 10, fill: "#64748b" }}
              />
              <YAxis
                stroke="#475569"
                tick={{ fontSize: 10, fill: "#94a3b8" }}
                scale={logScale ? "log" : "auto"}
                domain={logScale ? ["auto", "auto"] : [0, "auto"]}
                tickFormatter={(v) => (typeof v === "number" ? v.toFixed(4) : v)}
              />
              <Tooltip
                contentStyle={{ background: "#0f172a", border: "1px solid #334155", borderRadius: 8, fontSize: 11 }}
                labelStyle={{ color: "#94a3b8" }}
                formatter={(v: unknown) => [typeof v === "number" ? v.toFixed(6) : String(v ?? ""), ""]}
              />
              <Legend wrapperStyle={{ fontSize: 11, paddingTop: 8 }} />

              {viewMetric === "loss" && (
                <>
                  <Line type="monotone" dataKey="primary_train" name={`${primaryRun?.display_name ?? primaryRunId} Train`} stroke="#6366f1" strokeWidth={2} dot={false} />
                  <Line type="monotone" dataKey="primary_val" name={`${primaryRun?.display_name ?? primaryRunId} Val`} stroke="#34d399" strokeWidth={2} dot={false} />
                  {compareRunId && compareRunId !== primaryRunId && (
                    <>
                      <Line type="monotone" dataKey="compare_train" name={`${compareRun?.display_name ?? compareRunId} Train`} stroke="#a78bfa" strokeWidth={1.5} strokeDasharray="5 3" dot={false} />
                      <Line type="monotone" dataKey="compare_val" name={`${compareRun?.display_name ?? compareRunId} Val`} stroke="#fb923c" strokeWidth={1.5} strokeDasharray="5 3" dot={false} />
                    </>
                  )}
                  {bestEpoch && (
                    <ReferenceLine x={bestEpoch} stroke="#34d399" strokeDasharray="4 2" label={{ value: `Best ep${bestEpoch}`, fontSize: 9, fill: "#34d399", position: "top" }} />
                  )}
                </>
              )}

              {viewMetric === "decouple" && (
                <>
                  <Line type="monotone" dataKey="primary_decouple" name={`${primaryRun?.display_name ?? primaryRunId} Decouple`} stroke="#a78bfa" strokeWidth={2} dot={false} />
                  {compareRunId && compareRunId !== primaryRunId && (
                    <Line type="monotone" dataKey="compare_decouple" name={`${compareRun?.display_name ?? compareRunId} Decouple`} stroke="#fb923c" strokeWidth={1.5} strokeDasharray="5 3" dot={false} />
                  )}
                </>
              )}

              {viewMetric === "rss" && (
                <>
                  <Line type="monotone" dataKey="primary_r_eta" name="r_eta (context ratio)" stroke="#34d399" strokeWidth={2} dot={false} />
                  <Line type="monotone" dataKey="primary_eta" name="eta (future ratio)" stroke="#f59e0b" strokeWidth={2} dot={false} />
                  {compareRunId && compareRunId !== primaryRunId && (
                    <Line type="monotone" dataKey="compare_r_eta" name={`${compareRun?.display_name ?? compareRunId} r_eta`} stroke="#a78bfa" strokeWidth={1.5} strokeDasharray="5 3" dot={false} />
                  )}
                </>
              )}
            </LineChart>
          </ResponsiveContainer>
        )}
      </div>

      {/* Stats Row */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        {[
          {
            label: "Total Runs Recorded",
            value: runsWithHistory.length,
            icon: <Layers className="w-4 h-4 text-indigo-400" />,
            color: "indigo",
          },
          {
            label: "Epochs Logged",
            value: primaryHistory.length,
            icon: <Clock className="w-4 h-4 text-slate-400" />,
            color: "slate",
          },
          {
            label: "Best Val MSE",
            value: bestVal != null ? bestVal.toFixed(6) : "—",
            icon: <Award className="w-4 h-4 text-emerald-400" />,
            color: "emerald",
          },
          {
            label: "Best Epoch",
            value: bestEpoch ?? "—",
            icon: <CheckCircle className="w-4 h-4 text-violet-400" />,
            color: "violet",
          },
        ].map((s) => (
          <div
            key={s.label}
            className={`p-3 rounded-xl border border-${s.color}-800/30 bg-${s.color}-950/10 flex items-start gap-2.5`}
          >
            <div className="mt-0.5 shrink-0">{s.icon}</div>
            <div>
              <div className="text-[10px] text-slate-500 font-mono uppercase tracking-wide">{s.label}</div>
              <div className="text-base font-bold font-mono text-slate-100 mt-0.5">{s.value}</div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}