import { NavLink, useLocation } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import { api } from "../../lib/api";
import {
  LayoutDashboard, Search, GitCompare, Activity, BarChart2,
  FlaskConical, Download, Cpu, ChevronRight, Atom, Wifi, WifiOff,
} from "lucide-react";

const nav = [
  { to: "/",           icon: LayoutDashboard, label: "Overview" },
  { to: "/models",      icon: Cpu,             label: "Model Catalog" },
  { to: "/forecast",   icon: Search,          label: "Forecast Explorer" },
  { to: "/comparison", icon: GitCompare,       label: "Model Comparison" },
  { to: "/training",   icon: Activity,         label: "Training Monitor" },
  { to: "/evaluation", icon: BarChart2,        label: "Evaluation Metrics" },
  { to: "/experiments",icon: FlaskConical,     label: "Experiments & Checkpoints" },
  { to: "/export",     icon: Download,         label: "Reports & Export" },
];

export default function Shell({ children }: { children: React.ReactNode }) {
  const loc = useLocation();
  const current = nav.find(n => n.to === loc.pathname)?.label ?? "QPredRNN";

  const { data: sys, isError: sysErr } = useQuery({
    queryKey: ["system"],
    queryFn: api.system,
    refetchInterval: 30_000,
    retry: 1,
  });
  const { isError: healthErr } = useQuery({
    queryKey: ["health"],
    queryFn: api.health,
    refetchInterval: 10_000,
    retry: 1,
  });
  const connected = !healthErr && !sysErr;

  return (
    <div style={{ display: "grid", gridTemplateColumns: "220px 1fr", gridTemplateRows: "52px 1fr", minHeight: "100vh" }}>
      {/* ── Sidebar ───────────────────────────────────────────────────── */}
      <aside style={{
        gridRow: "1 / 3", background: "rgb(15 17 32)",
        borderRight: "1px solid rgb(38 45 72)", display: "flex", flexDirection: "column",
        padding: "0",
      }}>
        {/* Brand */}
        <div style={{
          padding: "16px 20px", display: "flex", alignItems: "center", gap: 10,
          borderBottom: "1px solid rgb(38 45 72)",
        }}>
          <div style={{
            width: 32, height: 32, borderRadius: 8,
            background: "linear-gradient(135deg,rgb(99 102 241),rgb(34 211 238))",
            display: "flex", alignItems: "center", justifyContent: "center",
          }}>
            <Atom size={18} color="white" />
          </div>
          <div>
            <div style={{ fontWeight: 700, fontSize: 14, color: "rgb(226 232 240)", letterSpacing: "-0.02em" }}>QPredRNN</div>
            <div style={{ fontSize: 10, color: "rgb(100 116 139)" }}>Research Studio</div>
          </div>
        </div>

        {/* Nav */}
        <nav style={{ flex: 1, padding: "12px 8px" }}>
          {nav.map(({ to, icon: Icon, label }) => (
            <NavLink key={to} to={to} end={to === "/"} style={({ isActive }) => ({
              display: "flex", alignItems: "center", gap: 10, padding: "8px 12px",
              borderRadius: 8, marginBottom: 2, textDecoration: "none", fontSize: 13,
              fontWeight: isActive ? 600 : 400,
              color: isActive ? "rgb(226 232 240)" : "rgb(100 116 139)",
              background: isActive ? "rgba(99 102 241 / 0.15)" : "transparent",
              borderLeft: isActive ? "2px solid rgb(99 102 241)" : "2px solid transparent",
              transition: "all 0.15s",
            })}>
              <Icon size={15} />
              <span>{label}</span>
            </NavLink>
          ))}
        </nav>

        {/* System status */}
        <div style={{ padding: "12px 16px", borderTop: "1px solid rgb(38 45 72)", fontSize: 11 }}>
          <div style={{ display: "flex", alignItems: "center", gap: 6, marginBottom: 6 }}>
            {connected
              ? <Wifi size={12} color="rgb(52 211 153)" />
              : <WifiOff size={12} color="rgb(248 113 113)" />}
            <span style={{ color: connected ? "rgb(52 211 153)" : "rgb(248 113 113)" }}>
              {connected ? "API Connected" : "API Offline"}
            </span>
          </div>
          {sys?.cuda_available && (
            <div style={{ color: "rgb(100 116 139)", display: "flex", gap: 4 }}>
              <Cpu size={11} style={{ marginTop: 1 }} />
              <span>{sys.gpu_name?.split(" ").slice(-2).join(" ") ?? "GPU"}</span>
            </div>
          )}
          {sys && (
            <div style={{ color: "rgb(100 116 139)", marginTop: 4 }}>
              VRAM {sys.vram_used_mb ?? "?"}MB / {sys.vram_total_mb ?? "?"}MB
            </div>
          )}
        </div>
      </aside>

      {/* ── Topbar ───────────────────────────────────────────────────── */}
      <header style={{
        gridColumn: 2, background: "rgb(15 17 32)",
        borderBottom: "1px solid rgb(38 45 72)",
        display: "flex", alignItems: "center", padding: "0 24px", gap: 8,
      }}>
        <span style={{ color: "rgb(100 116 139)", fontSize: 12 }}>QPredRNN</span>
        <ChevronRight size={12} color="rgb(100 116 139)" />
        <span style={{ fontSize: 13, fontWeight: 600, color: "rgb(226 232 240)" }}>{current}</span>
        <div style={{ marginLeft: "auto", display: "flex", gap: 8 }}>
          {sys?.pytorch && (
            <span style={{
              fontSize: 11, padding: "2px 8px", borderRadius: 999,
              background: "rgba(99 102 241 / 0.15)", color: "rgb(165 180 252)",
              border: "1px solid rgba(99 102 241 / 0.3)",
            }}>
              PyTorch {sys.pytorch.split("+")[0]}
            </span>
          )}
        </div>
      </header>

      {/* ── Main ─────────────────────────────────────────────────────── */}
      <main style={{ gridColumn: 2, padding: "24px", overflowY: "auto", background: "rgb(9 10 20)" }}>
        {children}
      </main>
    </div>
  );
}