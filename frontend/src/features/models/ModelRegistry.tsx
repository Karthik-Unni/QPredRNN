import { useQuery } from "@tanstack/react-query";
import { api, CanonicalModel } from "../../lib/api";
import { useNavigate } from "react-router-dom";
import { Cpu, Atom, Layers, FileCode, CheckCircle, ArrowRight } from "lucide-react";

export default function ModelRegistry() {
  const navigate = useNavigate();
  const { data: models, isLoading } = useQuery<CanonicalModel[]>({
    queryKey: ["canonical_models"],
    queryFn: api.models,
  });

  return (
    <div className="space-y-8 max-w-7xl mx-auto">
      {/* Header */}
      <div>
        <div className="flex items-center gap-2 mb-1">
          <span className="px-2.5 py-0.5 text-xs font-mono rounded bg-cyan-950/60 border border-cyan-800/40 text-cyan-300">
            Canonical Architecture Registry
          </span>
          <span className="text-xs text-slate-400">· Decoupled from Experiment Runs</span>
        </div>
        <h1 className="text-2xl md:text-3xl font-bold tracking-tight text-slate-100">
          Model Architecture Catalog
        </h1>
        <p className="text-slate-400 text-sm mt-1 max-w-3xl">
          Scientific registry of all verified deep learning architectures and forecasting baselines.
          Each architecture topology is maintained independently from specific training runs and checkpoints.
        </p>
      </div>

      {isLoading ? (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {[1, 2, 3, 4].map((i) => (
            <div key={i} className="h-48 rounded-xl bg-slate-900/40 border border-slate-800/50 animate-pulse" />
          ))}
        </div>
      ) : (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {models?.map((m) => (
            <div
              key={m.id}
              className="rounded-xl p-6 bg-slate-900/60 border border-slate-800/80 hover:border-slate-700/80 transition-all flex flex-col justify-between"
            >
              <div className="space-y-4">
                <div className="flex items-start justify-between gap-4">
                  <div>
                    <div className="flex items-center gap-2">
                      <h2 className="text-lg font-bold text-slate-100">{m.display_name}</h2>
                      {m.quantum_enabled ? (
                        <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded text-xs font-mono bg-violet-950/60 text-violet-300 border border-violet-800/40">
                          <Atom className="w-3 h-3" /> Quantum VQC
                        </span>
                      ) : (
                        <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded text-xs font-mono bg-slate-800/60 text-slate-300 border border-slate-700/40">
                          <Layers className="w-3 h-3" /> Classical
                        </span>
                      )}
                    </div>
                    <p className="text-xs font-mono text-cyan-400 mt-0.5">{m.family}</p>
                  </div>
                  <span className="inline-flex items-center gap-1 text-xs text-emerald-400 font-mono">
                    <CheckCircle className="w-3.5 h-3.5" /> Verified
                  </span>
                </div>

                <p className="text-xs text-slate-300 leading-relaxed">{m.description}</p>

                {/* Specs Grid */}
                <div className="grid grid-cols-2 sm:grid-cols-3 gap-2.5 text-xs font-mono">
                  <div className="p-2.5 rounded-lg bg-slate-950/60 border border-slate-800/60">
                    <span className="text-slate-400 block text-[10px] uppercase">Parameters</span>
                    <span className="text-slate-100 font-semibold">
                      {m.parameter_count > 0 ? m.parameter_count.toLocaleString() : "0 (Analytical)"}
                    </span>
                  </div>
                  <div className="p-2.5 rounded-lg bg-slate-950/60 border border-slate-800/60">
                    <span className="text-slate-400 block text-[10px] uppercase">Frames (In → Out)</span>
                    <span className="text-slate-100 font-semibold">{m.input_frames} → {m.output_frames}</span>
                  </div>
                  <div className="p-2.5 rounded-lg bg-slate-950/60 border border-slate-800/60 col-span-2 sm:col-span-1">
                    <span className="text-slate-400 block text-[10px] uppercase">Execution Path</span>
                    <span className="text-cyan-300 font-semibold truncate block">
                      {m.quantum_enabled ? "PennyLane VQC" : "PyTorch CUDA"}
                    </span>
                  </div>
                </div>

                {/* Quantum details if enabled */}
                {m.quantum_enabled && (
                  <div className="p-3 rounded-lg bg-violet-950/30 border border-violet-800/40 text-xs space-y-1 font-mono text-violet-200">
                    <div className="flex items-center justify-between text-[11px]">
                      <span>Register: {m.n_qubits} Qubits</span>
                      <span>Depth: {m.circuit_depth} Layers</span>
                      <span>Backend: {m.simulator_backend}</span>
                    </div>
                    <div className="text-[10px] text-slate-400 font-sans">
                      * Statevector simulation via PennyLane CPU interface. No physical quantum advantage claimed.
                    </div>
                  </div>
                )}

                {/* File Reference */}
                <div className="flex items-center gap-1.5 text-xs text-slate-400 font-mono">
                  <FileCode className="w-3.5 h-3.5 text-slate-400" />
                  <span>{m.implementation_file}</span>
                </div>
              </div>

              {/* Action Buttons */}
              <div className="pt-5 mt-4 border-t border-slate-800/60 flex items-center justify-between gap-3">
                <button
                  onClick={() => navigate(`/forecast?run=${m.default_run_id}`)}
                  className="px-3.5 py-1.5 rounded-lg text-xs font-semibold bg-cyan-600/20 hover:bg-cyan-600/30 text-cyan-300 border border-cyan-500/30 transition-all flex items-center gap-1"
                >
                  Inspect in Forecast Explorer <ArrowRight className="w-3.5 h-3.5" />
                </button>
                <button
                  onClick={() => navigate(`/comparison`)}
                  className="px-3 py-1.5 rounded-lg text-xs font-semibold text-slate-400 hover:text-slate-200 transition-all"
                >
                  Benchmark View
                </button>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
