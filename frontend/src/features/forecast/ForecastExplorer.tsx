import { useState, useCallback, useEffect } from "react";
import { useQuery, useMutation } from "@tanstack/react-query";
import { api, PredictResult, PredictRequest } from "../../lib/api";
import {
  Play, SkipBack, SkipForward, ChevronLeft, ChevronRight,
  Loader2, AlertCircle, BarChart2
} from "lucide-react";
import {
  LineChart, Line, XAxis, YAxis, Tooltip, CartesianGrid,
  ResponsiveContainer, ReferenceLine, Legend,
} from "recharts";

// ── Compact B64 frame display ────────────────────────────────────────────────
function Frame({ b64, label, colorBorder }: { b64: string; label: string; colorBorder?: string }) {
  return (
    <div style={{ textAlign: "center" }}>
      <img src={`data:image/png;base64,${b64}`} alt={label}
        style={{ width: "100%", imageRendering: "pixelated", border: `2px solid ${colorBorder ?? "rgb(38 45 72)"}`, borderRadius: 4 }} />
      <div style={{ fontSize: 10, color: "rgb(100 116 139)", marginTop: 2 }}>{label}</div>
    </div>
  );
}

// ── Filmstrip row ────────────────────────────────────────────────────────────
function FilmStrip({ frames, label, color, startT = 1 }: {
  frames: string[]; label: string; color: string; startT?: number }) {
  return (
    <div style={{ marginBottom: 16 }}>
      <div style={{ fontSize: 11, fontWeight: 700, color, textTransform: "uppercase",
        letterSpacing: "0.06em", marginBottom: 6, display: "flex", alignItems: "center", gap: 6 }}>
        <div style={{ width: 3, height: 14, background: color, borderRadius: 2 }} />
        {label}
      </div>
      <div style={{ display: "flex", gap: 4, overflowX: "auto", paddingBottom: 4 }}>
        {frames.map((b64, i) => (
          <div key={i} style={{ flex: "0 0 auto", width: 80 }}>
            <Frame b64={b64} label={`t=${startT + i}`} colorBorder={color + "60"} />
          </div>
        ))}
      </div>
    </div>
  );
}

// ── Main forecast page ───────────────────────────────────────────────────────
export default function ForecastExplorer() {
  const { data: runs = [] } = useQuery({ queryKey: ["runs"], queryFn: api.runs });

  const [runId, setRunId] = useState("kth_qpredrnn_local_50ep");
  const [ckpt, setCkpt] = useState<"best" | "last">("best");
  const [dataset, setDataset] = useState<"kth" | "toy">("kth");
  const [split, setSplit] = useState<"train" | "val" | "test">("test");
  const [clipIdx, setClipIdx] = useState(0);
  const [device, setDevice] = useState<"cuda" | "cpu">("cuda");
  const [horizon, setHorizon] = useState(0);

  const { data: splitInfo } = useQuery({
    queryKey: ["splitInfo", dataset, split],
    queryFn: () => api.splitInfo(dataset, split),
    enabled: !!dataset,
    retry: 1,
  });

  const { mutate, data: result, isPending, error, reset } = useMutation({
    mutationFn: (req: PredictRequest) => api.predict(req),
    onSuccess: () => setHorizon(0),
  });

  // Auto-run prediction on mount
  useEffect(() => {
    mutate({
      run_id: "kth_qpredrnn_local_50ep",
      checkpoint: "best",
      dataset: "kth",
      split: "test",
      clip_index: 0,
      device: "cuda",
    });
  }, [mutate]);

  const selectedRun = runs.find((r) => r.id === runId);

  const handlePredict = useCallback(() => {
    if (!runId) return;
    mutate({ run_id: runId, checkpoint: ckpt, dataset, split, clip_index: clipIdx, device });
  }, [runId, ckpt, dataset, split, clipIdx, device, mutate]);

  const errMsg = error instanceof Error ? error.message : String(error ?? "");

  // Degradation chart data
  const chartData = result?.metrics.per_frame.mse.map((v, i) => ({
    t: i + 1,
    mse: +v.toFixed(6),
    mae: +(result.metrics.per_frame.mae[i] ?? 0).toFixed(6),
    psnr: +(result.metrics.per_frame.psnr[i] ?? 0).toFixed(2),
  }));

  const labelStyle = {
    fontSize: 11,
    fontWeight: 600,
    color: "rgb(100 116 139)",
    textTransform: "uppercase" as const,
    letterSpacing: "0.05em",
    marginBottom: 4,
  };
  const selectStyle = {
    background: "rgb(15 17 32)",
    border: "1px solid rgb(38 45 72)",
    borderRadius: 8,
    padding: "6px 10px",
    color: "rgb(226 232 240)",
    fontSize: 13,
    width: "100%",
  };

  return (
    <div style={{ maxWidth: 1200 }}>
      <div style={{ marginBottom: 20 }}>
        <h1 style={{ fontSize: 20, fontWeight: 700, margin: 0 }}>Forecast Explorer</h1>
        <p style={{ color: "rgb(100 116 139)", marginTop: 4, fontSize: 13 }}>
          Load a checkpoint or analytical baseline, select a KTH clip, and run autoregressive video prediction.
        </p>
      </div>

      {/* ── Quick preset pills ─── */}
      <div style={{ display: "flex", flexWrap: "wrap", gap: 8, marginBottom: 16 }}>
        <span style={{ fontSize: 12, color: "rgb(100 116 139)", alignSelf: "center", marginRight: 4 }}>
          Quick Select:
        </span>
        {[
          { id: "kth_qpredrnn_local_50ep", label: "🌌 QPredRNN (50ep Local)", col: "#818cf8" },
          { id: "kth_predrnn_v2_overnight_gpu", label: "⚡ PredRNN-V2 (10ep)", col: "#34d399" },
          { id: "kth_classical_pilot_gpu", label: "📦 ConvLSTM (10ep)", col: "#60a5fa" },
          { id: "baseline_persistence", label: "⏹ Persistence Baseline", col: "#fbbf24" },
          { id: "baseline_mean", label: "📊 Mean Context", col: "#f472b6" },
        ].map((preset) => (
          <button
            key={preset.id}
            onClick={() => {
              setRunId(preset.id);
              mutate({ run_id: preset.id, checkpoint: ckpt, dataset, split, clip_index: clipIdx, device });
            }}
            style={{
              padding: "5px 12px",
              borderRadius: 8,
              fontSize: 12,
              fontWeight: runId === preset.id ? 700 : 500,
              background: runId === preset.id ? `${preset.col}25` : "rgb(20 23 40)",
              border: `1px solid ${runId === preset.id ? preset.col : "rgb(38 45 72)"}`,
              color: runId === preset.id ? preset.col : "rgb(203 213 225)",
              cursor: "pointer",
            }}
          >
            {preset.label}
          </button>
        ))}
      </div>

      {/* ── Controls panel ─── */}
      <div style={{ background: "rgb(20 23 40)", border: "1px solid rgb(38 45 72)", borderRadius: 12, padding: 20, marginBottom: 20 }}>
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit,minmax(160px,1fr))", gap: 16, marginBottom: 16 }}>
          <div>
            <div style={labelStyle}>Experiment Run</div>
            <select style={selectStyle} value={runId} onChange={e => { setRunId(e.target.value); reset(); }}>
              <optgroup label="Primary Benchmarks">
                <option value="kth_qpredrnn_local_50ep">QPredRNN — 50-Epoch Local Run</option>
                <option value="kth_predrnn_v2_overnight_gpu">PredRNN-V2 — 10-Epoch Overnight</option>
              </optgroup>
              <optgroup label="Pilot Studies">
                <option value="kth_qpredrnn_pilot_gpu">QPredRNN — 10-Epoch Pilot</option>
                <option value="kth_classical_pilot_gpu">ConvLSTM — 10-Epoch Pilot</option>
              </optgroup>
              <optgroup label="Analytical Baselines">
                <option value="baseline_persistence">Persistence Baseline (Last Frame)</option>
                <option value="baseline_mean">Mean Context Baseline</option>
              </optgroup>
              <optgroup label="Other / Smoke Tests">
                {runs.filter(r => !["kth_qpredrnn_local_50ep", "kth_predrnn_v2_overnight_gpu", "kth_qpredrnn_pilot_gpu", "kth_classical_pilot_gpu"].includes(r.id)).map(r => (
                  <option key={r.id} value={r.id}>
                    {r.display_name ?? r.name} {r.has_best_ckpt ? "✓" : ""}
                  </option>
                ))}
              </optgroup>
            </select>
          </div>
          <div>
            <div style={labelStyle}>Checkpoint</div>
            <select style={selectStyle} value={ckpt} onChange={e => setCkpt(e.target.value as "best"|"last")}>
              <option value="best">Best {selectedRun?.has_best_ckpt ? "✓" : "(missing)"}</option>
              <option value="last">Last {selectedRun?.has_last_ckpt ? "✓" : "(missing)"}</option>
            </select>
          </div>
          <div>
            <div style={labelStyle}>Dataset</div>
            <select style={selectStyle} value={dataset} onChange={e => setDataset(e.target.value as "kth"|"toy")}>
              <option value="kth">KTH-64</option>
              <option value="toy">Toy Dataset</option>
            </select>
          </div>
          <div>
            <div style={labelStyle}>Split</div>
            <select style={selectStyle} value={split} onChange={e => setSplit(e.target.value as "train"|"val"|"test")}>
              <option value="test">Test (held-out)</option>
              <option value="val">Validation</option>
              <option value="train">Train</option>
            </select>
          </div>
          <div>
            <div style={labelStyle}>Device</div>
            <select style={selectStyle} value={device} onChange={e => setDevice(e.target.value as "cuda"|"cpu")}>
              <option value="cuda">CUDA (GPU)</option>
              <option value="cpu">CPU</option>
            </select>
          </div>
          <div>
            <div style={labelStyle}>
              Clip Index {splitInfo ? `(0–${splitInfo.num_clips - 1})` : ""}
            </div>
            <input type="number" min={0} max={(splitInfo?.num_clips ?? 1) - 1}
              value={clipIdx} onChange={e => setClipIdx(Number(e.target.value))}
              style={{ ...selectStyle }} />
          </div>
        </div>
        {splitInfo && (
          <div style={{ fontSize: 11, color: "rgb(100 116 139)", marginBottom: 12 }}>
            {splitInfo.num_clips.toLocaleString()} temporal clips · {splitInfo.num_sequences} sequences
          </div>
        )}
        <button onClick={handlePredict} disabled={!runId || isPending}
          style={{
            background: !runId || isPending ? "rgb(38 45 72)" : "rgb(99 102 241)",
            color: "white", border: "none", borderRadius: 8, padding: "10px 24px",
            fontSize: 13, fontWeight: 700, cursor: runId && !isPending ? "pointer" : "not-allowed",
            display: "flex", alignItems: "center", gap: 8,
          }}>
          {isPending ? <><Loader2 size={14} style={{ animation: "spin 1s linear infinite" }} /> Running Prediction…</>
            : <><Play size={14} /> Run Prediction Rollout</>}
        </button>
      </div>

      {/* ── Error ─── */}
      {error && (
        <div style={{ background: "rgba(248 113 113 / 0.1)", border: "1px solid rgba(248 113 113 / 0.4)",
          borderRadius: 10, padding: 16, marginBottom: 20, display: "flex", gap: 10 }}>
          <AlertCircle size={16} color="rgb(248 113 113)" style={{ flexShrink: 0, marginTop: 1 }} />
          <div>
            <div style={{ fontWeight: 600, color: "rgb(248 113 113)" }}>Prediction Failed</div>
            <div style={{ fontSize: 12, color: "rgb(226 232 240)", marginTop: 4 }}>{errMsg}</div>
          </div>
        </div>
      )}

      {/* ── Results ─── */}
      {result && (
        <>
          {/* Summary bar */}
          <div style={{ background: "rgb(20 23 40)", border: "1px solid rgb(38 45 72)",
            borderRadius: 10, padding: "12px 20px", marginBottom: 16,
            display: "flex", flexWrap: "wrap", gap: 20, alignItems: "center" }}>
            <span style={{ fontSize: 12, color: "rgb(100 116 139)" }}>{result.seq_desc}</span>
            <div style={{ display: "flex", gap: 20, marginLeft: "auto" }}>
              {[
                { k: "MSE", v: result.metrics.mse.toFixed(6) },
                { k: "MAE", v: result.metrics.mae.toFixed(6) },
                { k: "PSNR", v: result.metrics.psnr.toFixed(2) + " dB" },
                { k: "SSIM", v: result.metrics.ssim.toFixed(4) },
                { k: "Latency", v: result.latency_ms + " ms" },
              ].map(({ k, v }) => (
                <div key={k} style={{ textAlign: "center" }}>
                  <div style={{ fontSize: 14, fontWeight: 700, fontFamily: "monospace", color: "rgb(34 211 238)" }}>{v}</div>
                  <div style={{ fontSize: 10, color: "rgb(100 116 139)", textTransform: "uppercase" }}>{k}</div>
                </div>
              ))}
            </div>
          </div>

          {/* Filmstrips */}
          <div style={{ background: "rgb(20 23 40)", border: "1px solid rgb(38 45 72)",
            borderRadius: 12, padding: 20, marginBottom: 20 }}>
            <FilmStrip frames={result.input_frames} label="Observed Context" color="rgb(99 102 241)" startT={1} />
            <FilmStrip frames={result.gt_frames} label="Ground Truth Future" color="rgb(52 211 153)" startT={11} />
            <FilmStrip frames={result.pred_frames} label="Model Prediction" color="rgb(167 139 250)" startT={11} />
            <FilmStrip frames={result.error_frames} label="|GT − Pred| Error (magma)" color="rgb(248 113 113)" startT={11} />
          </div>

          {/* Frame-by-frame inspector */}
          <div style={{ background: "rgb(20 23 40)", border: "1px solid rgb(38 45 72)",
            borderRadius: 12, padding: 20, marginBottom: 20 }}>
            <div style={{ fontSize: 13, fontWeight: 600, marginBottom: 14, display: "flex", alignItems: "center", gap: 8 }}>
              <BarChart2 size={15} /> Frame-by-Frame Inspector
            </div>
            {/* Horizon controls */}
            <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 16 }}>
              <button onClick={() => setHorizon(h => Math.max(0, h-1))} disabled={horizon === 0}
                style={{ background: "rgb(38 45 72)", border: "none", borderRadius: 6, padding: "6px 10px", cursor: "pointer", color: "white" }}>
                <ChevronLeft size={14} />
              </button>
              <input type="range" min={0} max={result.pred_frames.length - 1}
                value={horizon} onChange={e => setHorizon(Number(e.target.value))}
                style={{ flex: 1, accentColor: "rgb(99 102 241)" }} />
              <button onClick={() => setHorizon(h => Math.min(result.pred_frames.length - 1, h+1))}
                disabled={horizon === result.pred_frames.length - 1}
                style={{ background: "rgb(38 45 72)", border: "none", borderRadius: 6, padding: "6px 10px", cursor: "pointer", color: "white" }}>
                <ChevronRight size={14} />
              </button>
              <span style={{ fontSize: 12, fontFamily: "monospace", color: "rgb(34 211 238)", minWidth: 60 }}>
                t={11 + horizon} / {10 + result.pred_frames.length}
              </span>
            </div>
            <div style={{ display: "grid", gridTemplateColumns: "repeat(4,1fr)", gap: 12, marginBottom: 16 }}>
              <Frame b64={result.input_frames[result.input_frames.length - 1]} label="Last Context (t=10)" colorBorder="rgb(99 102 241)" />
              <Frame b64={result.gt_frames[horizon]} label={`GT t=${11+horizon}`} colorBorder="rgb(52 211 153)" />
              <Frame b64={result.pred_frames[horizon]} label={`Pred t=${11+horizon}`} colorBorder="rgb(167 139 250)" />
              <Frame b64={result.error_frames[horizon]} label={`Error t=${11+horizon}`} colorBorder="rgb(248 113 113)" />
            </div>
            {/* Per-frame metrics */}
            <div style={{ display: "grid", gridTemplateColumns: "repeat(4,1fr)", gap: 12 }}>
              {[
                { k: "MSE", v: result.metrics.per_frame.mse[horizon]?.toFixed(6) },
                { k: "MAE", v: result.metrics.per_frame.mae[horizon]?.toFixed(6) },
                { k: "PSNR", v: result.metrics.per_frame.psnr[horizon]?.toFixed(2) + " dB" },
                { k: "SSIM", v: result.metrics.per_frame.ssim[horizon]?.toFixed(4) },
              ].map(({ k, v }) => (
                <div key={k} style={{ background: "rgb(15 17 32)", borderRadius: 8, padding: "12px 16px", border: "1px solid rgb(38 45 72)" }}>
                  <div style={{ fontSize: 18, fontWeight: 700, fontFamily: "monospace", color: "rgb(226 232 240)" }}>{v}</div>
                  <div style={{ fontSize: 10, color: "rgb(100 116 139)", textTransform: "uppercase", marginTop: 4 }}>{k}</div>
                </div>
              ))}
            </div>
          </div>

          {/* Degradation chart */}
          {chartData && (
            <div style={{ background: "rgb(20 23 40)", border: "1px solid rgb(38 45 72)", borderRadius: 12, padding: 20 }}>
              <div style={{ fontSize: 13, fontWeight: 600, marginBottom: 16 }}>Per-Horizon Degradation</div>
              <ResponsiveContainer width="100%" height={220}>
                <LineChart data={chartData} margin={{ top: 4, right: 16, bottom: 4, left: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgb(38 45 72)" />
                  <XAxis dataKey="t" tick={{ fill: "rgb(100 116 139)", fontSize: 10 }} label={{ value: "Future Step", fill: "rgb(100 116 139)", fontSize: 11, position: "insideBottom", offset: -4 }} />
                  <YAxis tick={{ fill: "rgb(100 116 139)", fontSize: 10 }} width={65} />
                  <Tooltip contentStyle={{ background: "rgb(20 23 40)", border: "1px solid rgb(38 45 72)", borderRadius: 8, fontSize: 12 }} />
                  <Legend wrapperStyle={{ fontSize: 12, color: "rgb(100 116 139)" }} />
                  <ReferenceLine x={horizon + 1} stroke="rgb(167 139 250)" strokeDasharray="4 2" />
                  <Line type="monotone" dataKey="mse" stroke="rgb(248 113 113)" strokeWidth={2} dot={false} name="MSE" />
                  <Line type="monotone" dataKey="mae" stroke="rgb(251 191 36)" strokeWidth={2} dot={false} name="MAE" />
                </LineChart>
              </ResponsiveContainer>
            </div>
          )}
        </>
      )}

      {/* Loading skeleton */}
      {isPending && (
        <div style={{ display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center",
          padding: 60, color: "rgb(100 116 139)", gap: 12 }}>
          <Loader2 size={32} style={{ animation: "spin 1s linear infinite", color: "rgb(99 102 241)" }} />
          <div style={{ fontSize: 14 }}>Running inference…</div>
          <div style={{ fontSize: 12 }}>Loading checkpoint and generating {10} future frames</div>
        </div>
      )}

      {!result && !isPending && !error && (
        <div style={{ textAlign: "center", padding: 60, color: "rgb(100 116 139)" }}>
          <div style={{ fontSize: 14, marginBottom: 8 }}>Select an experiment and click Run Prediction Rollout</div>
          <div style={{ fontSize: 12 }}>Frames from the selected KTH clip will appear here</div>
        </div>
      )}

      <style>{`@keyframes spin { from { transform: rotate(0deg); } to { transform: rotate(360deg); } }`}</style>
    </div>
  );
}